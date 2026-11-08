"""D-019 byte-identical backoff, measured-only caps and successful-call replay."""
from collections import Counter
import json
from pathlib import Path
import time

import httpx
from story_agent.tl5_wire import AuditedDeepSeek as Previous
from story_agent.tl5b_wire_v942 import CompleteReplyContract
from story_agent.agent_tl3_d017 import estimate
from story_agent.budget import BudgetStopped, TokenLedger
from story_agent.strong_agent_sa1 import save, read_json, emit
from story_agent.default_v1 import CONFIGURATION, VERSION, replace_json

DELAYS = (60, 120, 300, 600, 900)


class TransportStopped(BudgetStopped):
    """Exit code 3, resumable after the authorized transport waits."""


def valid_usage(value):
    keys = ('prompt_tokens', 'completion_tokens', 'total_tokens',
            'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens')
    return (isinstance(value, dict) and all(type(value.get(k)) is int and value[k] >= 0 for k in keys)
            and value['total_tokens'] == value['prompt_tokens'] + value['completion_tokens']
            and value['prompt_tokens'] == value['prompt_cache_hit_tokens'] + value['prompt_cache_miss_tokens'])


class AuditedDeepSeek(Previous):
    def __init__(self, directory, key, cap, *, resume=False, inner=None, sleeper=time.sleep):
        directory = Path(directory)
        # Each concurrent question owns a ledger: a pending request in B1 must not
        # prevent B2 or B3 from running. No global ledger policy is changed.
        super().__init__(directory.parent, directory.name, key, cap,
                         ledger=TokenLedger(directory/'ledger.sqlite'), inner=inner,
                         resume=resume, sleeper=sleeper)
        self.directory = directory
        self.run_id = CONFIGURATION + ':' + str(directory.resolve())
        self.contract = CompleteReplyContract()
        self.failures = 0
        self.started = time.monotonic()
        self.pending = read_json(directory/'checkpoint.json') if resume and (directory/'checkpoint.json').exists() else None
        self.access_seen = {(v['scope'],v['tool_call_id']) for v in
            map(json.loads,(directory/'access.jsonl').read_text(encoding='utf-8').splitlines())} if (directory/'access.jsonl').exists() else set()

    def cost(self):
        stages = {}
        errors = Counter()
        waits = 0
        estimates = []
        for folder in sorted((self.directory/'calls').glob('*')):
            if not folder.is_dir():
                continue
            info = read_json(folder/'scope.json')
            stage = stages.setdefault(info['scope'], dict(requests=0, successful_responses=0,
                cached_input=0, uncached_input=0, output=0, budget_tokens=0, empty_reasoning=0))
            stage['requests'] += 1
            if (folder/'usage.json').exists():
                usage = read_json(folder/'usage.json')['usage']
                stage['cached_input'] += usage['prompt_cache_hit_tokens']
                stage['uncached_input'] += usage['prompt_cache_miss_tokens']
                stage['output'] += usage['completion_tokens']
                stage['budget_tokens'] += usage['prompt_cache_miss_tokens'] + usage['completion_tokens']
            if (folder/'contract.json').exists():
                stage['successful_responses'] += 1
                stage['empty_reasoning'] += int(read_json(folder/'contract.json')['reasoning_empty'])
            if (folder/'error.json').exists():
                error = read_json(folder/'error.json')
                errors[error['error_type']] += 1
                waits += error.get('wait_seconds', 0)
        for path in sorted((self.directory/'estimates').glob('*.json')):
            estimates.append(read_json(path))
        fields = ('requests', 'successful_responses', 'cached_input', 'uncached_input', 'output', 'budget_tokens', 'empty_reasoning')
        total = {key: sum(stage[key] for stage in stages.values()) for key in fields}
        total['empty_reasoning_fraction'] = total['empty_reasoning']/total['successful_responses'] if total['successful_responses'] else None
        return dict(configuration=CONFIGURATION, code_version=VERSION, stages=stages, total=total,
                    cap=self.cap, estimated_missing_usage_budget_tokens=sum(e['budget_tokens'] for e in estimates),
                    estimates_count=len(estimates), estimates_in_cap=False, transport_failures=dict(errors),
                    retry_count=sum(1 for p in (self.directory/'calls').glob('*/scope.json') if read_json(p).get('retry', 0)),
                    retry_wait_seconds=waits, failures_this_session=self.failures,
                    resumed=bool(self.pending), elapsed_seconds=round(time.monotonic()-self.started, 2))

    def gate(self):
        self.monitor.check_stop()
        summary = self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts']) - {'settled'}:
            raise BudgetStopped('Question ledger has unsettled requests')
        value = self.cost()
        if value['total']['budget_tokens'] > self.cap:
            raise BudgetStopped('Measured cumulative budget exceeded cap')
        return value

    def trace(self, value):
        with (self.directory/'trace.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, ensure_ascii=False)+'\n')

    def observe_history(self, payload):
        # Preserve the frozen structured history audit. A normal tool result
        # containing disambiguation candidates is feedback, not a structure fault.
        super().observe_history(payload)
        calls = {}
        for message in payload['messages']:
            for call in message.get('tool_calls', []):
                calls[call['id']] = call
            if message['role'] != 'tool':
                continue
            call_id = message['tool_call_id']
            key = (self.monitor.scope, call_id)
            if key in self.access_seen:
                continue
            self.access_seen.add(key)
            call = calls[call_id]['function']
            value = dict(scope=self.monitor.scope, tool_call_id=call_id, name=call['name'],
                         arguments=json.loads(call['arguments']), result=message['content'])
            path = self.directory/'access.jsonl'
            if self.replayed < len(self.replay):
                continue
            with path.open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(value, ensure_ascii=False)+'\n')

    def _settle_failure(self, folder, identifier, payload, request, body, status):
        usage = body.get('usage') if isinstance(body, dict) else None
        if valid_usage(usage):
            self.ledger.settle(identifier, usage, status)
            save(folder/'usage.json', {'usage': usage})
            return
        value = estimate(self.out, self.topic, payload, request.content)
        value.update(topic=self.topic, call_id=identifier, http_status=status,
                     authority='D-20261107-019', estimated=True, included_in_cap=False)
        # The estimate settles bookkeeping only and is never included in caps.
        with self.ledger.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS default_v1_estimates (call_id TEXT PRIMARY KEY, estimate_json TEXT)')
            db.execute('INSERT INTO default_v1_estimates VALUES (?,?)', (identifier, json.dumps(value, ensure_ascii=False)))
            db.execute("UPDATE calls SET status='settled',actual=?,prompt_tokens=?,completion_tokens=?,http_status=? WHERE id=? AND status='pending'",
                       (value['total_tokens'], value['prompt_tokens'], value['completion_tokens'], status, identifier))
        save(self.directory/'estimates'/f'{identifier}.json', value)

    def _request(self, request):
        payload = self.safe(json.loads(request.content))
        names = {t['function']['name'] for t in payload.get('tools', [])}
        if (request.method != 'POST' or request.url.scheme != 'https' or request.url.host != 'api.deepseek.com'
                or not request.url.path.endswith('/chat/completions') or payload.get('model') != 'deepseek-v4-pro'
                or payload.get('thinking') != {'type': 'enabled'} or payload.get('reasoning_effort') != 'max'
                or 'temperature' in payload or payload.get('max_tokens', payload.get('max_completion_tokens')) != 49152
                or payload.get('stream') or names not in self.allowed_tool_sets):
            raise BudgetStopped('Unexpected model or tools')
        self.monitor.scope = self.scope_fn(payload)
        agent = self.contract.agent(payload, self.monitor.scope)
        self.observe_history(payload)
        if self.replayed < len(self.replay):
            path = self.replay[self.replayed]
            if request.content != (path.parent/'request-body.bin').read_bytes():
                raise BudgetStopped('Resume request bytes differ from successful request')
            self.replayed += 1
            body = read_json(path)
            self.accept(body, path.parent, agent, replay=True)
            return httpx.Response(200, json=body)
        if self.pending and self.pending.get('pending_request_sha256'):
            import hashlib
            if hashlib.sha256(request.content).hexdigest() != self.pending['pending_request_sha256']:
                raise BudgetStopped('Resumed pending request bytes differ')
            self.pending = dict(self.pending, pending_request_sha256=None)
        import hashlib
        for retry in range(6):
            self.gate()
            number = len(list((self.directory/'calls').glob('*/scope.json'))) + 1
            folder = self.directory/'calls'/f'{number:03d}'
            save(folder/'scope.json', dict(scope=self.monitor.scope, retry=retry))
            save(folder/'request.json', payload)
            (folder/'request-body.bin').write_bytes(request.content)
            replace_json(self.directory/'checkpoint.json', dict(configuration=CONFIGURATION, code_version=VERSION,
                status='pending', pending_request_sha256=hashlib.sha256(request.content).hexdigest(),
                successful_requests=len(list((self.directory/'calls').glob('*/response.json')))))
            identifier = self.ledger.reserve(self.run_id, 'deepseek-v4-pro', 2*len(request.content)+2048+49152)
            save(folder/'reservation.json', {'id': identifier})
            emit(request_started=number, scope=self.monitor.scope, budget_tokens=self.cost()['total']['budget_tokens'])
            body = None
            status = None
            failure = None
            retryable = False
            try:
                response = self.wire.handle_request(request)
                status = response.status_code
                try:
                    body = response.json()
                except (ValueError, UnicodeError):
                    pass
                if status == 429 or 500 <= status <= 599:
                    failure, retryable = 'HTTP'+str(status), True
                elif status == 200 and (not isinstance(body, dict) or 'choices' not in body or 'usage' not in body):
                    failure, retryable = 'UnusableHTTP200', True
                elif status != 200:
                    failure = 'HTTP'+str(status)
                elif not valid_usage(body.get('usage')):
                    failure = 'MalformedUsage'
            except (httpx.TransportError, OSError) as exc:
                failure, retryable = type(exc).__name__, True
            if failure:
                self.failures += 1
                wait = DELAYS[retry] if retryable and retry < 5 and self.failures <= 10 else 0
                save(folder/'error.json', dict(error_type=failure, http_status=status, retry=retry, wait_seconds=wait))
                self._settle_failure(folder, identifier, payload, request, body, status)
                self.trace(dict(type='transport_failure', call=number, error_type=failure, wait_seconds=wait))
                replace_json(self.directory/'cost.json', self.cost())
                self.gate()
                if not retryable:
                    raise BudgetStopped('Non-retryable response: '+failure)
                if not wait:
                    raise TransportStopped('Transport retries exhausted or failed requests exceeded 10')
                emit(message=f'接口暂时异常，第 {retry+1} 次重发，等待 {wait//60} 分钟')
                remaining = wait
                while remaining:
                    pause = min(30, remaining)
                    self.sleeper(pause)
                    remaining -= pause
                continue
            body = self.safe(body)
            self.ledger.settle(identifier, body['usage'], status)
            save(folder/'usage.json', {'usage': body['usage']})
            # Archive before contract validation; only accepted replies are replayable.
            save(folder/'received.json', body)
            self.accept(body, folder, agent)
            save(folder/'response.json', body)
            self.trace(dict(type='response', call=number, scope=self.monitor.scope,
                            reasoning_empty=not bool(body['choices'][0]['message']['reasoning_content'].strip())))
            replace_json(self.directory/'cost.json', self.cost())
            replace_json(self.directory/'checkpoint.json', dict(configuration=CONFIGURATION, code_version=VERSION,
                 status='accepted', pending_request_sha256=None,
                 successful_requests=len(list((self.directory/'calls').glob('*/response.json')))))
            emit(request_settled=number, scope=self.monitor.scope, budget_tokens=self.cost()['total']['budget_tokens'])
            return response
