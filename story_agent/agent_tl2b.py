"""TL2B v0.87.0: isolated compare reader and SA2-compatible audited transport.

No answer validation or semantic grading. All model IO uses the shared ledger.
"""
from collections import Counter
import json
from pathlib import Path
import re
import threading
import time

import httpx
from langchain.agents import create_agent

from story_agent.budget import BudgetStopped, TokenLedger
from story_agent.strong_agent_sa1 import SYSTEM_PROMPT as SA1_PROMPT, emit, read_json, save
from story_agent.strong_agent_sa2 import DB, MODEL, live_model
from story_agent.strong_agent_sa2_v822 import ObservedWire, ToolGuard, estimate_missing

VERSION = 'v0.87.0'
PREFIX = 'TL2B:' + VERSION + ':'
APPENDIX = '任务文本（search_quests、read_quest）中的叙述与系统说明可作为确定的设定；人物台词只表示该人物这样说，引用时写明说话人，所指对象要依上下文确定。需要判断两处材料是否说的是同一回事时，可以交给对照读者单独比较。'
SYSTEM_PROMPT = SA1_PROMPT + APPENDIX
DESCRIPTION = '对照读者：当你需要判断两三本书（或任务文本）里的两处材料是否说的是同一人物、事件或概念，或它们是什么关系时调用。请在问题里写明要比较的对象、所在书名和大致行号。它会读原文后单独作答。'
COMPARE_PROMPT = '你是对照读者。你会收到一个写明比较对象的问题和两三本书（或任务文本）。请用工具读这些材料的相关原文，只依据原文回答：两处说的是否同一人物、事件或概念，或是什么关系——相同、相容、冲突、只是主题相近、无关，可以不止一种；说明依据的书名与行号；分清原文明说的和你的推断，原文没写的就说没写；同一作者或同一系列的前后篇如有补充或修正，要指出来。不使用你自己已知的原神设定。用中文自由作答，不超过600字。'
QUESTIONS = (
    '201039《审议之庭-至高领主共识评议会记录》第32行说到人类与燃素；1054《哈南帕查记事》第9—15行讲贤者带来燃素。两处说的是不是同一件事？双方各自是什么立场？',
    '201217《某位高贵之人的笔记·其二》第4行与201219《某位高贵之人的笔记·其四》第4—6行都谈到亥珀波瑞亚的覆灭与审判。两处对覆灭的原因和审判的时机说法是否一致？',
)
CHILD_TOOLS = {'read_book', 'read_quest'}
MAIN_TOOLS = {'book_map', 'search_books', 'read_book', 'search_quests',
              'read_quest', 'list_books', 'similar_passages', 'task'}


def build_compare(model, tools):
    selected = [t for t in tools if t.name in CHILD_TOOLS]
    if {t.name for t in selected} != CHILD_TOOLS or len(selected) != 2:
        raise BudgetStopped('Compare requires the two TL1B read tools')
    return create_agent(model, tools=selected, system_prompt=COMPARE_PROMPT, name='compare')


def build_main(model, tools):
    """CompiledSubAgent is Deep Agents' public custom-subagent mechanism."""
    from deepagents import create_deep_agent
    from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
    register_harness_profile('deepseek:' + MODEL, HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools={'ls', 'read_file', 'write_file', 'edit_file', 'delete', 'glob', 'grep', 'execute'},
    ))
    return create_deep_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT,
        subagents=[{'name': 'compare', 'description': DESCRIPTION,
                    'runnable': build_compare(model, tools)}])


def accounted(out, topic=None):
    totals = Counter(requests=0, prompt_tokens=0, completion_tokens=0, total_tokens=0,
                     prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=0,
                     reasoning_tokens=0, measured_budget_tokens=0, estimated_requests=0,
                     estimated_budget_tokens=0, estimated_raw_tokens=0)
    root = Path(out)
    pattern = f'{topic}/calls/*/response.json' if topic else '*/calls/*/response.json'
    for path in root.glob(pattern):
        u = read_json(path)['usage']
        keys = ('prompt_tokens', 'completion_tokens', 'total_tokens',
                'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens')
        if any(type(u.get(k)) is not int or u[k] < 0 for k in keys):
            raise BudgetStopped('Missing provider cache accounting')
        if (u['prompt_tokens'] != u['prompt_cache_hit_tokens'] + u['prompt_cache_miss_tokens']
                or u['total_tokens'] != u['prompt_tokens'] + u['completion_tokens']):
            raise BudgetStopped('Invalid provider usage')
        for k in keys:
            totals[k] += u[k]
        reasoning = u.get('completion_tokens_details', {}).get('reasoning_tokens')
        if type(reasoning) is not int or not 0 <= reasoning <= u['completion_tokens']:
            raise BudgetStopped('Missing reasoning accounting')
        totals['reasoning_tokens'] += reasoning
        totals['requests'] += 1
        totals['measured_budget_tokens'] += u['prompt_cache_miss_tokens'] + u['completion_tokens']
    for path in (root / 'estimates').glob('*.json'):
        e = read_json(path)
        if topic and e['topic'] != topic:
            continue
        totals['estimated_requests'] += 1
        totals['estimated_budget_tokens'] += e['budget_tokens']
        totals['estimated_raw_tokens'] += e['total_tokens']
    totals['budget_tokens'] = totals['measured_budget_tokens'] + totals['estimated_budget_tokens']
    totals['attempts'] = totals['requests'] + totals['estimated_requests']
    totals['accounted_raw_tokens'] = totals['total_tokens'] + totals['estimated_raw_tokens']
    return dict(totals)


class CompareGuard(ToolGuard):
    def observe(self, message):
        # Deep Agents returns a custom subagent's free text as task's result.
        if message.type == 'tool' and self.calls[message.tool_call_id]['name'] == 'task':
            self.returns += 1
            self.faults = self.faults + 1 if message.status == 'error' else 0
            if self.faults >= 2:
                raise BudgetStopped('Consecutive tool infrastructure faults')
            return
        super().observe(message)


class AuditedDeepSeek(httpx.BaseTransport):
    def __init__(self, out, topic, key, cap=20_000, ledger=None, inner=None,
                 resume=False, sleeper=time.sleep):
        self.out, self.topic, self.key, self.cap = Path(out), topic, key, cap
        self.ledger = ledger or TokenLedger(DB)
        self.wire = ObservedWire(key, inner)
        self.sleeper, self.lock = sleeper, threading.Lock()
        self.run_id = PREFIX + topic
        self.guard, self.observed = CompareGuard(), 0
        self.replay = sorted((self.out / topic / 'calls').glob('*/response.json')) if resume else []
        self.replayed = 0

    def safe(self, value):
        if self.key in json.dumps(value, ensure_ascii=False):
            raise BudgetStopped('Credential echo rejected before persistence')
        return value

    def error_text(self, value):
        value = str(value).replace(self.key, '[REDACTED]')
        value = re.sub(r'https?://\S+', '[URL]', value)
        value = re.sub(r'(?i)bearer\s+\S+|sk-[A-Za-z0-9_-]+', '[REDACTED]', value)
        return value[:8192]

    def gate(self):
        summary = self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts']) - {'settled'}:
            raise BudgetStopped('Shared ledger must be accounting-only and settled')
        totals = accounted(self.out)
        # Other stages have separate evidence roots, so compare only this stage's topics.
        with self.ledger.connect() as db:
            topics = [p.name for p in self.out.iterdir() if p.is_dir() and (p/'calls').exists()]
            rows = [row for topic in topics for row in db.execute(
                'SELECT status,actual FROM calls WHERE run_id=?', (PREFIX+topic,)).fetchall()]
        if len(rows) != totals['attempts'] or sum(r[1] or 0 for r in rows) != totals['accounted_raw_tokens']:
            raise BudgetStopped('Task ledger and evidence disagree')
        if totals['budget_tokens'] >= self.cap:
            raise BudgetStopped('Uncached input plus output cap reached')
        return totals

    def observe_history(self, payload):
        from langchain_core.messages import AIMessage, ToolMessage
        # Wire history is the durable source, including child tool results.
        # Different invocations have separate histories; key by first user turn.
        history_key = json.dumps([m for m in payload['messages'] if m['role']=='user'][:1], ensure_ascii=False)
        if not hasattr(self, 'histories'):
            self.histories = {}
        seen, guard = self.histories.setdefault(history_key, (set(), CompareGuard()))
        for i, m in enumerate(payload['messages']):
            key = (i, json.dumps(m, sort_keys=True, ensure_ascii=False))
            if key in seen:
                continue
            seen.add(key)
            if m['role'] == 'assistant':
                calls = [{'id': t['id'], 'name': t['function']['name'],
                          'args': json.loads(t['function']['arguments'])} for t in m.get('tool_calls', [])]
                guard.observe(AIMessage(content=m.get('content') or '', tool_calls=calls))
            elif m['role'] == 'tool':
                guard.observe(ToolMessage(content=m['content'], tool_call_id=m['tool_call_id']))
                emit(tool_returned=guard.calls[m['tool_call_id']]['name'],
                     returned_characters=len(m['content']), invalid_calls=sum(guard.invalid.values()))

    def handle_request(self, request):
        with self.lock:
            return self._request(request)

    def _request(self, request):
        payload = self.safe(json.loads(request.content))
        names = {t['function']['name'] for t in payload.get('tools', [])}
        if (request.method != 'POST' or request.url.scheme != 'https'
                or request.url.host != 'api.deepseek.com' or not request.url.path.endswith('/chat/completions')
                or payload.get('model') != MODEL or payload.get('thinking') != {'type': 'enabled'}
                or payload.get('reasoning_effort') != 'max' or 'temperature' in payload
                or payload.get('max_tokens', payload.get('max_completion_tokens')) != 49152
                or payload.get('stream') or names not in (CHILD_TOOLS, MAIN_TOOLS)):
            raise BudgetStopped('Unexpected model or tools')
        self.observe_history(payload)
        if self.replayed < len(self.replay):
            path = self.replay[self.replayed]
            if request.content != (path.parent/'request-body.bin').read_bytes():
                raise BudgetStopped('Resume request bytes differ from settled request')
            self.replayed += 1
            emit(replayed_settled=self.replayed, paid=False)
            return httpx.Response(200, json=read_json(path))
        for attempt in range(2):
            totals = self.gate()
            number = totals['attempts'] + 1
            directory = self.out / self.topic / 'calls' / f'{number:03d}'
            save(directory/'request.json', payload)
            (directory/'request-body.bin').write_bytes(request.content)
            ident = self.ledger.reserve(self.run_id, MODEL, 2*len(request.content)+2048+49152)
            save(directory/'reservation.json', {'id': ident})
            emit(request_started=number, topic=self.topic, budget_tokens=totals['budget_tokens'])
            try:
                response = self.wire.handle_request(request)
            except (httpx.TransportError, OSError) as exc:
                error = dict(self.wire.error or {}, http_status=None, request_id=None,
                             error_type=type(exc).__name__, error=self.error_text(exc))
                save(directory/'error.json', error)
                try:
                    self.ledger.settle(ident, None, None)
                except BudgetStopped:
                    pass  # The durable uncertain row is reconciled explicitly below.
                if not error.get('retry_eligible'):
                    raise BudgetStopped('Partial response; no automatic resend') from None
                prior = sorted((self.out/self.topic/'calls').glob('*/response.json'))
                if prior:
                    p = prior[-1]
                    estimate = estimate_missing(payload, read_json(p.parent/'request.json'), read_json(p)['usage'])
                else:
                    prompt = 2*len(request.content)+2048
                    estimate = {'prompt_tokens': prompt, 'completion_tokens': 49152,
                                'total_tokens': prompt+49152, 'prompt_cache_miss_tokens': prompt,
                                'budget_tokens': prompt+49152, 'estimated': True,
                                'method': 'First request has no cached prefix: UTF8 upper bound plus maximum output'}
                estimate.update(topic=self.topic, call_id=ident)
                save(self.out/'estimates'/f'{ident}.json', estimate)
                with self.ledger.connect() as db:
                    db.execute('CREATE TABLE IF NOT EXISTS tl2b_estimated_usage (call_id TEXT PRIMARY KEY, authority TEXT, estimate_json TEXT)')
                    db.execute('INSERT INTO tl2b_estimated_usage VALUES (?,?,?)',
                               (ident, 'D-20261106-007/008', json.dumps(estimate)))
                    db.execute("UPDATE calls SET status='settled',actual=?,prompt_tokens=?,completion_tokens=? WHERE id=? AND status='uncertain'",
                               (estimate['total_tokens'], estimate['prompt_tokens'], 49152, ident))
                if attempt:
                    raise BudgetStopped('Two consecutive zero-byte transport failures') from None
                self.gate()
                emit(retry_wait_seconds=60)
                self.sleeper(30)
                emit(retry_wait_seconds=30)
                self.sleeper(30)
                continue
            raw = response.content
            # Persist error evidence before parse/settlement; no authentication headers.
            try:
                body = response.json()
            except (ValueError, UnicodeError):
                body = None
            if not response.is_success or not isinstance(body, dict) or not isinstance(body.get('usage'), dict):
                save(directory/'error.json', {'http_status': response.status_code,
                    'request_id': response.headers.get('x-request-id') or response.headers.get('request-id'),
                    'received_bytes': len(raw), 'error_type': 'HTTPStatusError' if not response.is_success else 'InvalidResponse',
                    'error_body': self.error_text(raw.decode('utf-8', errors='replace')),
                    'unknown_raw_token_reservation': 2*len(request.content)+2048+49152,
                    'unknown_budget_token_reservation': 2*len(request.content)+2048+49152})
                self.ledger.settle(ident, None, response.status_code)
                raise BudgetStopped('HTTP/parse failure; usage unknown, no automatic resend')
            body = self.safe(body)
            save(directory/'response.json', body)
            self.ledger.settle(ident, body['usage'], response.status_code)
            totals = accounted(self.out)
            save(directory/'usage.json', {'usage': body['usage'], 'stage_usage': totals})
            try:
                choice = body['choices'][0]
                message = choice['message']
                save(directory/'message.json', message)
                reasoning = message.get('reasoning_content')
                if not isinstance(reasoning, str) or not reasoning.strip():
                    raise BudgetStopped('Missing reasoning output')
                (directory/'reasoning.txt').write_text(reasoning, encoding='utf-8')
                (directory/'content.txt').write_text(message.get('content') or '', encoding='utf-8')
                if choice.get('finish_reason') not in ('stop', 'tool_calls'):
                    raise BudgetStopped('Non-completed response; no feedback or retry')
            except (KeyError, IndexError, TypeError):
                raise BudgetStopped('Malformed model response') from None
            emit(request_settled=number, topic=self.topic, usage=body['usage'], budget_tokens=totals['budget_tokens'])
            return response

    def close(self):
        self.wire.close()
