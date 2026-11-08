"""PT0: frozen TL4b D-017 lifecycle, explicit PT0 schemas and namespace."""
import json
import httpx
from story_agent import agent_tl4b_v912 as previous
from story_agent.agent_pt0 import MODEL, REVIEW_TOOLS, VERSION
from story_agent.agent_tl3_d017 import estimate
from story_agent.budget import BudgetStopped
from story_agent.strong_agent_sa1 import emit, read_json, save
from story_agent.tl4b_wire_v914 import HistoryAudit
accounted = previous.accounted
PREFIX = 'PT0:' + VERSION + ':'

class AuditedDeepSeek(HistoryAudit, previous.AuditedDeepSeek):
    def __init__(self, *args, allowed_tool_sets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = PREFIX + self.out.name + ':' + self.topic
        main = previous.MAIN_TOOLS | {'request_external'}
        self.allowed_tool_sets = allowed_tool_sets or (REVIEW_TOOLS, main,
            main | {'search_reference', 'read_reference'}, {'search_reference', 'read_reference'})

    def observe_history(self, payload):
        # Audit a JSON envelope for the mandated plain-text rejection, while
        # preserving actual tool results, prompts and request bytes exactly.
        audit = json.loads(json.dumps(payload, ensure_ascii=False))
        calls = {}
        for msg in audit['messages']:
            for call in msg.get('tool_calls', []):
                calls[call['id']] = call['function']['name']
            if msg['role'] == 'tool' and calls.get(msg['tool_call_id']) == 'request_external':
                msg['content'] = json.dumps({'external_result': msg['content']}, ensure_ascii=False)
        HistoryAudit.observe_history(self, audit)

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
        value = dict(value, authority='D-20261106-017; codex/108 §1',
                     topic=self.topic, call_id=identifier)
        with self.ledger.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT run_id,status,actual FROM calls WHERE id=?', (identifier,)).fetchone()
            if not row or tuple(row) != (self.run_id, 'uncertain', None):
                raise BudgetStopped('Unexpected TL4b reconciliation target')
            db.execute('CREATE TABLE IF NOT EXISTS pt0_estimated_usage (call_id TEXT PRIMARY KEY, before_json TEXT, estimate_json TEXT)')
            db.execute('INSERT INTO pt0_estimated_usage VALUES (?,?,?)',
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
                or payload.get('stream') or names not in self.allowed_tool_sets):
            raise BudgetStopped('Unexpected model or tools')
        child = names == REVIEW_TOOLS
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
