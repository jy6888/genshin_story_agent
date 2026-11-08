"""D-017 transport recovery; prompts, tools and model parameters stay frozen."""
import json
import time
import httpx

from story_agent.agent_tl3 import AuditedDeepSeek as FrozenTransport
from story_agent.agent_tl2b import CHILD_TOOLS, MAIN_TOOLS, MODEL, accounted, estimate_missing
from story_agent.budget import BudgetStopped
from story_agent.strong_agent_sa1 import read_json, save, emit

AUTHORITY = 'D-20261106-017; codex/104 section 9'


def scope(payload):
    return (payload['tools'], next(m['content'] for m in payload['messages'] if m['role']=='user'))


def estimate(out, topic, payload, request_bytes):
    candidates = sorted((out/topic/'calls').glob('*/response.json'))
    for p in reversed(candidates):
        previous = read_json(p.parent/'request.json')
        if scope(previous) == scope(payload):
            value = estimate_missing(payload, previous, read_json(p)['usage'])
            value['previous_same_scope_response'] = str(p)
            break
    else:
        prompt = 2*len(request_bytes)+2048
        value = {'prompt_tokens': prompt, 'completion_tokens': 49152,
            'total_tokens': prompt+49152, 'prompt_cache_miss_tokens': prompt,
            'budget_tokens': prompt+49152, 'estimated': True,
            'method': 'No settled same-scope prefix: input=2*entire UTF8 request bytes+2048; output=49152'}
    value['authority'] = AUTHORITY
    return value


def reconcile(ledger, out, topic, identifier, value):
    """Preserve original uncertain row and explicit authorization in SQLite."""
    value = dict(value, topic=topic, call_id=identifier)
    target = out/'estimates'/f'{identifier}.json'
    with ledger.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT run_id,status,actual,http_status,reserved FROM calls WHERE id=?', (identifier,)).fetchone()
        if not row or row[0] != 'TL3:v0.88.0:'+topic or row[1:3] != ('uncertain', None):
            raise BudgetStopped('Unexpected D-017 reconciliation target')
        db.execute('CREATE TABLE IF NOT EXISTS tl3_d017_estimated_usage (call_id TEXT PRIMARY KEY, authority TEXT, before_json TEXT, estimate_json TEXT)')
        db.execute('INSERT INTO tl3_d017_estimated_usage VALUES (?,?,?,?)',
                   (identifier, AUTHORITY, json.dumps(list(row)), json.dumps(value, ensure_ascii=False)))
        db.execute("UPDATE calls SET status='settled',actual=?,prompt_tokens=?,completion_tokens=? WHERE id=? AND status='uncertain'",
                   (value['total_tokens'], value['prompt_tokens'], 49152, identifier))
        save(target, value)
    return value


class AuditedDeepSeek(FrozenTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        failed = self.out/'NQ1B/calls/058/request-body.bin'
        self.expected_retry = failed.read_bytes() if self.topic=='NQ1B' and kwargs.get('resume') and failed.exists() else None

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
        offset = 0
        if self.expected_retry is not None:
            if request.content != self.expected_retry:
                raise BudgetStopped('Authorized 058 retry request bytes differ')
            offset = 1  # 058 was failure one; a failed 059 is failure two.
            elapsed = time.time()-(self.out/'NQ1B/calls/058/error.json').stat().st_mtime
            remaining = max(0, 60-elapsed)
            while remaining:
                pause = min(30, remaining)
                emit(retry_wait_seconds=remaining)
                self.sleeper(pause)
                remaining -= pause
            self.expected_retry = None
        for attempt in range(offset, 2):
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
                error = dict(self.wire.error or {}, http_status=None, request_id=None,
                    error_type=type(exc).__name__, error=self.error_text(exc), d017_retry_eligible=True)
                save(directory/'error.json', error)
                try: self.ledger.settle(ident, None, None)
                except BudgetStopped: pass
                value = reconcile(self.ledger, self.out, self.topic, ident,
                                  estimate(self.out, self.topic, payload, request.content))
                emit(failure_estimated=True, topic=self.topic, budget_tokens=value['budget_tokens'])
                if attempt:
                    raise BudgetStopped('Two consecutive unusable transport responses') from None
                self.gate()
                emit(retry_wait_seconds=60)
                self.sleeper(30)
                emit(retry_wait_seconds=30)
                self.sleeper(30)
                continue
            raw = response.content
            try: body = response.json()
            except (ValueError, UnicodeError): body = None
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
