"""TL4 v0.89.0: authorized prompt, read_text child, monitoring and D-017 IO."""
import json
import time
from datetime import datetime, timezone

import httpx
from langchain.agents import create_agent
from story_agent import agent_tl3 as frozen
from story_agent import agent_tl2b as base
from story_agent.agent_tl3_d017 import estimate
from story_agent.budget import BudgetStopped
from story_agent.strong_agent_sa1 import emit, read_json, save

VERSION = 'v0.89.0'
PREFIX = 'TL4:' + VERSION + ':'
MODEL = base.MODEL
accounted = base.accounted
DESCRIPTION, COMPARE_PROMPT = frozen.DESCRIPTION, frozen.COMPARE_PROMPT
OLD = '把你打算写成定论、强对应或因果的几条结论交给对照读者核对'
NEW = '把你打算标为【明说】或【推导】的结论交给对照读者核对'
APPENDIX = '书籍和任务以外的游戏文本（角色故事、角色语音、邀约事件、闲聊对话，以及材料、武器、圣遗物、生物、装扮等物品的描述）可以用 search_texts 字面搜索、用 read_text 按行读取。报告里每条结论前标级别：【明说】原文直接写着（人物说的、猜的或主张的，写成“某某说”，不算明说的事实）；【推导】由原文推出、但原文没有直接写，写出依据的每处原文与行号，还有其他说得通的解释就一并写出；【猜测】明说和推导都给不出答案、并写明查过哪里之后，可以大胆猜：写出引发猜测的原文（名称与行号），以及什么样的原文能证实或推翻它，不能与原文矛盾。只是意象或母题相近的，直接写“只是主题相近”。任何一步用到猜测的结论只能是猜测；猜测单列一节，不作其他结论的前提。先查题面里的说法原文是否真有，没有就直说。题面里的称谓搜不到时，先想用户可能指的是什么（错字、近音、别名），换写法去搜，但事实只能来自原文。把题目拆成几个小问题：每个小问题要么有标了级别的回答，要么写明查过哪些词、哪些文本而没找到；连续换几种查法都没有新的相关原文，就可以停下交稿。'
SYSTEM_PROMPT = frozen.SYSTEM_PROMPT.replace(OLD, NEW) + APPENDIX
CHILD_TOOLS = base.CHILD_TOOLS | {'read_text'}
MAIN_TOOLS = base.MAIN_TOOLS | {'search_texts', 'read_text'}


def build_compare(model, tools):
    selected = [t for t in tools if t.name in CHILD_TOOLS]
    if {t.name for t in selected} != CHILD_TOOLS or len(selected) != 3:
        raise BudgetStopped('Compare requires three read tools')
    return create_agent(model, tools=selected, system_prompt=COMPARE_PROMPT, name='compare')


def build_main(model, tools):
    from deepagents import create_deep_agent
    from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
    register_harness_profile('deepseek:' + MODEL, HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools={'ls', 'read_file', 'write_file', 'edit_file', 'delete', 'glob', 'grep', 'execute'}))
    return create_deep_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT,
        subagents=[{'name': 'compare', 'description': DESCRIPTION,
                    'runnable': build_compare(model, tools)}])


class SmokeLimit(BudgetStopped):
    """The authorized stage-1(b) two-main-response cutoff, not a failure."""


class Monitor:
    def __init__(self, directory, topic, sleeper=time.sleep):
        self.directory, self.topic, self.sleeper = directory, topic, sleeper
        self.released, self.sequence = False, 0

    def check_stop(self):
        path = self.directory / 'STOP'
        if path.exists():
            reason = path.read_text(encoding='utf-8')
            if not (self.directory/'failure.json').exists():
                save(self.directory/'failure.json', {'reason': reason, 'status': '胡编停止'})
            raise BudgetStopped('STOP: ' + reason)

    def observe(self, message, call_number, child=False, replay=False):
        if child:
            return
        self.sequence += 1
        trace = self.directory / 'monitor' / 'responses.jsonl'
        if not replay:
            trace.parent.mkdir(parents=True, exist_ok=True)
            with trace.open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps({'sequence': self.sequence, 'call': call_number,
                    'message': message}, ensure_ascii=False)+'\n')
        if self.topic != 'T1' or self.released:
            return
        path = self.directory/'monitor'/f'verdict-{self.sequence:03d}.json'
        emit(monitor_waiting=self.sequence, response_call=call_number, verdict_file=str(path))
        while not path.exists():
            self.check_stop()
            self.sleeper(1)
        value = read_json(path)
        if (value.get('verdict') not in ('continue', 'release', 'stop')
                or not isinstance(value.get('reason'), str) or not value['reason'].strip()
                or not isinstance(value.get('evidence'), str) or not value['evidence'].strip()):
            raise BudgetStopped('Invalid monitor verdict')
        if not replay:
            with (path.parent/'log.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(dict(value, time=datetime.now(timezone.utc).isoformat(),
                    topic=self.topic, sequence=self.sequence, call=call_number), ensure_ascii=False)+'\n')
        self.released = value['verdict'] == 'release'
        if value['verdict'] == 'stop':
            (self.directory/'STOP').write_text(value['reason'], encoding='utf-8')
        self.check_stop()


class AuditedDeepSeek(base.AuditedDeepSeek):
    def __init__(self, *args, max_main_responses=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = PREFIX + self.out.name + ':' + self.topic
        self.monitor = Monitor(self.out/self.topic, self.topic, self.sleeper)
        self.max_main_responses = max_main_responses

    def gate(self):
        self.monitor.check_stop()
        summary = self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts']) - {'settled'}:
            raise BudgetStopped('Shared ledger must be accounting-only and settled')
        totals = accounted(self.out)
        with self.ledger.connect() as db:
            topics = [p.name for p in self.out.iterdir() if p.is_dir() and (p/'calls').exists()]
            rows = [r for topic in topics for r in db.execute(
                'SELECT status,actual FROM calls WHERE run_id=?',
                (PREFIX+self.out.name+':'+topic,)).fetchall()]
        if len(rows) != totals['attempts'] or sum(r[1] or 0 for r in rows) != totals['accounted_raw_tokens']:
            raise BudgetStopped('Task ledger and evidence disagree')
        if totals['budget_tokens'] > self.cap:
            raise BudgetStopped('Uncached input plus output cap exceeded')
        return totals

    def reconcile(self, identifier, value):
        value = dict(value, authority='D-20261106-017; codex/106',
                     topic=self.topic, call_id=identifier)
        with self.ledger.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT run_id,status,actual FROM calls WHERE id=?', (identifier,)).fetchone()
            if not row or tuple(row) != (self.run_id, 'uncertain', None):
                raise BudgetStopped('Unexpected TL4 reconciliation target')
            db.execute('CREATE TABLE IF NOT EXISTS tl4_estimated_usage (call_id TEXT PRIMARY KEY, before_json TEXT, estimate_json TEXT)')
            db.execute('INSERT INTO tl4_estimated_usage VALUES (?,?,?)',
                       (identifier, json.dumps(list(row)), json.dumps(value, ensure_ascii=False)))
            db.execute("UPDATE calls SET status='settled',actual=?,prompt_tokens=?,completion_tokens=? WHERE id=? AND status='uncertain'",
                       (value['total_tokens'], value['prompt_tokens'], value['completion_tokens'], identifier))
            save(self.out/'estimates'/f'{identifier}.json', value)

    def _request(self, request):
        payload = self.safe(json.loads(request.content))
        names = {t['function']['name'] for t in payload.get('tools', [])}
        if (request.method != 'POST' or request.url.scheme != 'https'
                or request.url.host != 'api.deepseek.com' or not request.url.path.endswith('/chat/completions')
                or payload.get('model') != MODEL or payload.get('thinking') != {'type': 'enabled'}
                or payload.get('reasoning_effort') != 'max' or 'temperature' in payload
                or payload.get('max_tokens', payload.get('max_completion_tokens')) != 49152
                or payload.get('stream') or names not in (set(), CHILD_TOOLS, MAIN_TOOLS)):
            raise BudgetStopped('Unexpected model or tools')
        child = names == CHILD_TOOLS
        self.monitor.check_stop()
        if (not child and self.max_main_responses is not None
                and self.monitor.sequence >= self.max_main_responses):
            raise SmokeLimit('Two main responses completed')
        self.observe_history(payload)
        if self.replayed < len(self.replay):
            path = self.replay[self.replayed]
            if request.content != (path.parent/'request-body.bin').read_bytes():
                raise BudgetStopped('Resume request bytes differ from settled request')
            self.replayed += 1
            body = read_json(path)
            self.monitor.observe(body['choices'][0]['message'], int(path.parent.name), child, replay=True)
            return httpx.Response(200, json=body)
        for attempt in range(2):
            totals = self.gate()
            number = totals['attempts']+1
            directory = self.out/self.topic/'calls'/f'{number:03d}'
            save(directory/'request.json', payload)
            (directory/'request-body.bin').write_bytes(request.content)
            ident = self.ledger.reserve(self.run_id, MODEL, 2*len(request.content)+2048+49152)
            save(directory/'reservation.json', {'id': ident})
            emit(request_started=number, topic=self.topic, budget_tokens=totals['budget_tokens'])
            try:
                response = self.wire.handle_request(request)
            except (httpx.TransportError, OSError) as exc:
                save(directory/'error.json', dict(self.wire.error or {}, error_type=type(exc).__name__,
                                                 error=self.error_text(exc)))
                try:
                    self.ledger.settle(ident, None, None)
                except BudgetStopped:
                    pass
                self.reconcile(ident, estimate(self.out, self.topic, payload, request.content))
                if attempt:
                    raise BudgetStopped('Two consecutive unusable transport responses') from None
                self.gate()
                emit(retry_wait_seconds=60)
                self.sleeper(30)
                emit(retry_wait_seconds=30)
                self.sleeper(30)
                continue
            try:
                body = response.json()
            except (ValueError, UnicodeError):
                body = None
            if not response.is_success or not isinstance(body, dict) or not isinstance(body.get('usage'), dict):
                save(directory/'error.json', {'http_status': response.status_code,
                    'error_type': 'InvalidResponse', 'received_bytes': len(response.content),
                    'error_body': self.error_text(response.content.decode('utf-8', errors='replace'))})
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
            self.monitor.observe(message, number, child)
            return response
