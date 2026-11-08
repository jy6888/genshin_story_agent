"""TL4b v0.91.1: source gate, six-tool compare, exact authorized prompts."""
import json
import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from story_agent import agent_tl4 as frozen
from story_agent.agent_tl3_d017 import estimate
from story_agent.budget import BudgetStopped
from story_agent.strong_agent_sa1 import emit, read_json, save
from story_agent.tl4b_delegation_v911 import VERSION, delegation_error

PREFIX = 'TL4b:' + VERSION + ':'
MODEL = frozen.MODEL
accounted = frozen.accounted
DESCRIPTION = frozen.DESCRIPTION
OLD = '请它查原文有没有推翻或修正这条结论，结论的强度有没有超出原文。'
NEW = '请它查原文有没有推翻或修正这条结论，结论的强度有没有超出原文，以及这些原文和同系列、同人物的其他文本还能推出哪些与题目有关、而你没有写的结论（例如后一篇推翻或修正了前一篇）。'
APPENDIX = '需要查别处有没有相关或相反的说法时，可以用 search_books、search_quests、search_texts 搜索。被要求核对结论时，除了逐条判断，还要指出所给原文以及同系列、同人物的其他文本还能推出、而清单里没写的与题目有关的结论，写明依据的名称与行号。'
assert frozen.SYSTEM_PROMPT.count(OLD) == 1
SYSTEM_PROMPT = frozen.SYSTEM_PROMPT.replace(OLD, NEW, 1)
COMPARE_PROMPT = frozen.COMPARE_PROMPT + APPENDIX
CHILD_TOOLS = frozen.CHILD_TOOLS | {'search_books', 'search_quests', 'search_texts'}
MAIN_TOOLS = frozen.MAIN_TOOLS


class DelegationGuard(AgentMiddleware):
    """Intercept task before the framework starts compare or invokes a model."""
    def __init__(self):
        self.events = []

    def refusal(self, request):
        call = request.tool_call
        args = call.get('args', {})
        if call['name'] != 'task' or args.get('subagent_type') != 'compare':
            return None
        description = args.get('description', '')
        error = delegation_error(description)
        self.events.append({'tool_call_id': call['id'], 'description': description,
                            'blocked': error is not None})
        if error:
            return ToolMessage(content=error, tool_call_id=call['id'], name='task')
        return None

    def wrap_tool_call(self, request, handler):
        result = self.refusal(request)
        return result if result is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        result = self.refusal(request)
        return result if result is not None else await handler(request)


def build_compare(model, tools):
    selected = [t for t in tools if t.name in CHILD_TOOLS]
    if {t.name for t in selected} != CHILD_TOOLS or len(selected) != 6:
        raise BudgetStopped('Compare requires six authorized read/search tools')
    return create_agent(model, tools=selected, system_prompt=COMPARE_PROMPT, name='compare')


def build_main(model, tools, guard=None):
    from deepagents import create_deep_agent
    from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
    register_harness_profile('deepseek:' + MODEL, HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools={'ls', 'read_file', 'write_file', 'edit_file', 'delete', 'glob', 'grep', 'execute'}))
    return create_deep_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT,
        middleware=[guard if guard is not None else DelegationGuard()],
        subagents=[{'name': 'compare', 'description': DESCRIPTION,
                    'runnable': build_compare(model, tools)}])


class AuditedDeepSeek(frozen.AuditedDeepSeek):
    """Keep TL4's IO, D-017 and actual-only budget policy; isolate run IDs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = PREFIX + self.out.name + ':' + self.topic

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
        value = dict(value, authority='D-20261106-017; codex/107 §1',
                     topic=self.topic, call_id=identifier)
        with self.ledger.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT run_id,status,actual FROM calls WHERE id=?', (identifier,)).fetchone()
            if not row or tuple(row) != (self.run_id, 'uncertain', None):
                raise BudgetStopped('Unexpected TL4b reconciliation target')
            db.execute('CREATE TABLE IF NOT EXISTS tl4b_estimated_usage (call_id TEXT PRIMARY KEY, before_json TEXT, estimate_json TEXT)')
            db.execute('INSERT INTO tl4b_estimated_usage VALUES (?,?,?)',
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
                or payload.get('stream') or names not in (CHILD_TOOLS, MAIN_TOOLS)):
            raise BudgetStopped('Unexpected model or tools')
        child = names == CHILD_TOOLS
        self.monitor.check_stop()
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
