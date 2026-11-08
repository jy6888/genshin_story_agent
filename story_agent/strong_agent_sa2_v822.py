"""SA2 v0.82.2: authorized estimated reconciliation and one transport retry."""
from collections import Counter
import json
import math
import re
import time
import httpx
from pathlib import Path

from story_agent.strong_agent_sa2 import (
    Books, BudgetStopped, BudgetTransport, DB, MeteredDeepSeek, MODEL, ROOT,
    SYSTEM_PROMPT, TokenLedger, build_agent, credentials, emit, live_model,
    read_json, save, sha,
)

VERSION = 'v0.82.2'
CAP = 500_000
PREFIX = 'SA2:' + VERSION + ':'
BASE = ROOT / 'docs/project/tasks/T-20261106-001/evidence/sa2-v0.82.0'
PRIOR = BASE.parent / 'sa2-v0.82.1'
OUT = BASE.parent / 'sa2-v0.82.2'


def call_paths(out=OUT, topic=None):
    """Old settled requests stay in their original, immutable directory."""
    topics = (topic,) if topic else ('NQ1A', 'NQ1B', 'NQ2A')
    return sorted([p for root in (BASE, PRIOR, Path(out)) for t in topics
                   for p in (root / t / 'calls').glob('*/response.json')],
                  key=lambda p: int(p.parent.name))


def accounted(out=OUT, topic=None):
    totals = Counter(requests=0, total_tokens=0, prompt_tokens=0, completion_tokens=0,
                     prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=0, reasoning_tokens=0,
                     budget_tokens=0)
    for path in call_paths(out, topic):
        u = read_json(path)['usage']
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens',
                    'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens'):
            if type(u.get(key)) is not int or u[key] < 0:
                raise BudgetStopped('Missing provider cache accounting')
            totals[key] += u[key]
        if (u['prompt_tokens'] != u['prompt_cache_hit_tokens'] + u['prompt_cache_miss_tokens']
                or u['total_tokens'] != u['prompt_tokens'] + u['completion_tokens']):
            raise BudgetStopped('Invalid provider usage')
        totals['reasoning_tokens'] += u['completion_tokens_details']['reasoning_tokens']
        totals['budget_tokens'] += u['prompt_cache_miss_tokens'] + u['completion_tokens']
        totals['requests'] += 1
    totals['estimated_requests'] = 0
    totals['estimated_budget_tokens'] = 0
    totals['estimated_raw_tokens'] = 0
    for path in (Path(out) / 'estimates').glob('*.json'):
        e = read_json(path)
        if topic and e['topic'] != topic:
            continue
        totals['estimated_requests'] += 1
        totals['estimated_budget_tokens'] += e['budget_tokens']
        totals['estimated_raw_tokens'] += e['total_tokens']
    totals['measured_budget_tokens'] = totals['budget_tokens']
    totals['budget_tokens'] += totals['estimated_budget_tokens']
    totals['attempts'] = totals['requests'] + totals['estimated_requests']
    totals['accounted_raw_tokens'] = totals['total_tokens'] + totals['estimated_raw_tokens']
    return dict(totals)


def ledger_usage(ledger):
    with ledger.connect() as db:
        rows = db.execute('SELECT status,actual FROM calls WHERE run_id LIKE ?',
                          ('SA2:v0.82.%',)).fetchall()
    return {'requests': len(rows), 'total_tokens': sum(r[1] or 0 for r in rows),
            'status_counts': dict(Counter(r[0] for r in rows))}


def budget_gate(totals):
    if totals['budget_tokens'] >= CAP:
        raise BudgetStopped('Actual uncached-input plus output task cap reached')


def estimate_missing(current, previous, previous_usage):
    """Bound only newly appended input; no provider usage exists for failure.

    Match D-20261106-007: estimated current input minus previous actual input,
    plus maximum output. UTF-8 byte bound follows the existing BudgetTransport,
    and is explicitly an estimate, never a provider token count.
    """
    old_messages = previous['messages']
    if current['messages'][:len(old_messages)] != old_messages or current['tools'] != previous['tools']:
        raise BudgetStopped('Cannot estimate non-prefix request history')
    added = current['messages'][len(old_messages):]
    delta = 2 * len(json.dumps(added, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) + 2048
    prompt = previous_usage['prompt_tokens'] + delta
    return {'prompt_tokens': prompt, 'completion_tokens': 49152, 'total_tokens': prompt + 49152,
            'prompt_cache_miss_tokens': delta, 'assumed_cached_prefix_tokens': previous_usage['prompt_tokens'],
            'budget_tokens': delta + 49152, 'estimated': True,
            'method': 'previous actual prompt + upper bound on appended messages; delta=2*UTF8 bytes+2048; output=49152',
            'limitation': 'Input tokenizer count and actual cache state unavailable; authorized estimate, not billed usage'}


def reconcile(ledger, identifier, estimate, directory):
    """Explicit provenance inside shared DB; original uncertainty retained."""
    with ledger.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT run_id,status,actual,http_status FROM calls WHERE id=?', (identifier,)).fetchone()
        if not row or row[1:] != ('uncertain', None, None) or not row[0].startswith('SA2:'):
            raise BudgetStopped('Unexpected reconciliation target')
        db.execute('CREATE TABLE IF NOT EXISTS sa2_estimated_usage (call_id TEXT PRIMARY KEY, authority TEXT NOT NULL, before_json TEXT NOT NULL, estimate_json TEXT NOT NULL, evidence TEXT NOT NULL)')
        db.execute('INSERT INTO sa2_estimated_usage VALUES (?,?,?,?,?)',
                   (identifier, 'D-20261106-007', json.dumps(row), json.dumps(estimate, ensure_ascii=False), str(directory)))
        db.execute("UPDATE calls SET status='settled',actual=?,prompt_tokens=?,completion_tokens=? WHERE id=? AND status='uncertain'",
                   (estimate['total_tokens'], estimate['prompt_tokens'], estimate['completion_tokens'], identifier))


class ObservedWire(httpx.BaseTransport):
    def __init__(self, key, inner=None):
        self.key = key
        self.inner = inner or httpx.HTTPTransport(retries=0)
        self.error = None

    def handle_request(self, request):
        self.error = None
        count = 0
        headers_received = False
        try:
            response = self.inner.handle_request(request)
            headers_received = True
            parts = []
            chunks = [response.content] if response.is_stream_consumed else response.iter_raw()
            for chunk in chunks:
                count += len(chunk)
                parts.append(chunk)
            return httpx.Response(response.status_code, headers=response.headers,
                                  content=b''.join(parts), extensions=response.extensions)
        except (httpx.TransportError, OSError) as exc:
            message = str(exc).replace(self.key, '[REDACTED]')
            message = re.sub(r'https?://\S+', '[URL]', message)
            self.error = {'error_type': type(exc).__name__, 'error': message,
                          'received_bytes': count, 'headers_received': headers_received,
                          'retry_eligible': count == 0 and not headers_received}
            raise

    def close(self):
        self.inner.close()


class ResumedDeepSeek(MeteredDeepSeek):
    def __init__(self, out, topic, key, ledger=None, inner=None, sleeper=time.sleep):
        self.wire = ObservedWire(key, inner)
        self.sleeper = sleeper
        super().__init__(out, topic, key, ledger, self.wire)
        self.run_id = PREFIX + topic

    def _request(self, request):
        payload = self.safe(json.loads(request.content))
        if (request.url.scheme != 'https' or request.url.host != 'api.deepseek.com'
                or payload.get('model') != MODEL or payload.get('thinking') != {'type': 'enabled'}
                or payload.get('reasoning_effort') != 'max' or 'temperature' in payload
                or payload.get('max_tokens', payload.get('max_completion_tokens')) != 49152 or payload.get('stream')):
            raise BudgetStopped('Unexpected model/transport settings')
        if {t['function']['name'] for t in payload.get('tools', [])} != {'book_map', 'search_books', 'read_book'}:
            raise BudgetStopped('Unexpected tools')
        for attempt in range(2):
            summary = self.ledger.summary()
            if summary['limit'] is not None or set(summary['status_counts']) - {'settled'}:
                raise BudgetStopped('Shared ledger not fully reconciled')
            totals = accounted(self.out)
            state = ledger_usage(self.ledger)
            if state != {'requests': totals['attempts'], 'total_tokens': totals['accounted_raw_tokens'],
                         'status_counts': {'settled': totals['attempts']}}:
                raise BudgetStopped('Task ledger and measured/estimated evidence mismatch')
            budget_gate(totals)
            directory = self.out / self.topic / 'calls' / f'{state["requests"]+1:03d}'
            save(directory / 'request.json', payload)
            (directory / 'request-body.bin').write_bytes(request.content)
            emit(request_started=state['requests']+1, topic=self.topic, budget_tokens=totals['budget_tokens'], attempt=attempt+1)
            try:
                response = BudgetTransport.handle_request(self, request)
            except BudgetStopped:
                error = self.wire.error
                if error is None:
                    raise
                save(directory / 'network-error.json', self.safe(error))
                if not error['retry_eligible']:
                    raise BudgetStopped('Partial response received; retry forbidden') from None
                paths = call_paths(self.out, self.topic)
                if not paths:
                    raise BudgetStopped('No previous response available for authorized estimate') from None
                prior = paths[-1]
                estimate = estimate_missing(payload, read_json(prior.parent / 'request.json'), read_json(prior)['usage'])
                with self.ledger.connect() as db:
                    uncertain = db.execute("SELECT id FROM calls WHERE run_id=? AND status='uncertain'", (self.run_id,)).fetchall()
                if len(uncertain) != 1:
                    raise BudgetStopped('Ambiguous failed request')
                ident = uncertain[0][0]
                estimate.update(topic=self.topic, call_id=ident, request=str(directory / 'request.json'))
                save(self.out / 'estimates' / (ident + '.json'), estimate)
                reconcile(self.ledger, ident, estimate, directory)
                emit(transport_failure=error, estimated_budget_tokens=estimate['budget_tokens'], retry=attempt == 0)
                if attempt == 1:
                    raise BudgetStopped('Two consecutive zero-byte transport failures') from None
                budget_gate(accounted(self.out))
                emit(retry_wait_seconds=60)
                # Two bounded waits keep the process externally monitorable.
                self.sleeper(30)
                emit(retry_wait_seconds=30)
                self.sleeper(30)
                continue
            body = self.safe(response.json())
            save(directory / 'response.json', body)
            choice = body['choices'][0]
            message = choice['message']
            save(directory / 'message.json', message)
            reasoning = message.get('reasoning_content')
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise BudgetStopped('Missing reasoning output')
            (directory / 'reasoning.txt').write_text(reasoning, encoding='utf-8')
            (directory / 'content.txt').write_text(message.get('content') or '', encoding='utf-8')
            totals = accounted(self.out)
            save(directory / 'usage.json', {'usage': body['usage'], 'task_usage': totals})
            emit(request_settled=state['requests']+1, topic=self.topic, raw_measured_tokens=totals['total_tokens'],
                 budget_tokens=totals['budget_tokens'], estimated_budget_tokens=totals['estimated_budget_tokens'])
            budget_gate(totals)
            if choice.get('finish_reason') not in ('stop', 'tool_calls'):
                raise BudgetStopped('Non-completed API response; no retry')
            return response


class ToolGuard:
    """Observe failures without changing the tool result returned to the model."""
    def __init__(self):
        self.calls = {}
        self.invalid = Counter()
        self.faults = 0
        self.returns = 0

    def observe(self, message):
        if message.type == 'ai':
            for call in message.tool_calls:
                self.calls[call['id']] = call
        if message.type != 'tool':
            return
        self.returns += 1
        call = self.calls[message.tool_call_id]
        if message.status == 'error':
            self.faults += 1
            if self.faults >= 2:
                raise BudgetStopped('Consecutive tool infrastructure faults')
            return
        try:
            value = json.loads(message.content)
        except (ValueError, TypeError):
            raise BudgetStopped('Unparseable tool result') from None
        if not isinstance(value, dict):
            raise BudgetStopped('Unexpected tool result shape')
        self.faults = 0
        if 'error' in value:
            key = json.dumps({'name': call['name'], 'args': call['args']}, ensure_ascii=False, sort_keys=True)
            self.invalid[key] += 1
            if self.invalid[key] >= 5:
                raise BudgetStopped('Same invalid tool call occurred five times in one topic')
