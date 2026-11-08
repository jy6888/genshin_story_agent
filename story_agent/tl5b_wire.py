"""TL5b transport: replay100 unchanged; cap and ledger count only new requests."""
import json
from story_agent.tl5_wire_v933 import AuditedDeepSeek as FrozenWire
from story_agent.tl5_wire import accounted
from story_agent.budget import BudgetStopped


def delta(total, baseline):
    return {k: v-baseline.get(k, 0) for k, v in total.items()}


class AuditedDeepSeek(FrozenWire):
    def __init__(self, *args, baseline, first_request=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline = baseline
        self.run_id = 'TL5b:v0.94.0:'+self.out.name+':'+self.topic
        self.first_request = first_request
        self.old_tool_ids = set()
        for path in self.replay[:100]:
            message = json.loads(path.read_text(encoding='utf-8'))['choices'][0]['message']
            self.old_tool_ids.update(c['id'] for c in message.get('tool_calls', []))

    def gate(self):
        self.monitor.check_stop()
        summary = self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts'])-{'settled'}:
            raise BudgetStopped('Shared ledger must be settled')
        new = delta(accounted(self.out), self.baseline)
        with self.ledger.connect() as db:
            rows = db.execute('SELECT status,actual FROM calls WHERE run_id=?', (self.run_id,)).fetchall()
        if len(rows) != new['attempts'] or sum(r[1] or 0 for r in rows) != new['accounted_raw_tokens']:
            raise BudgetStopped('New request ledger/evidence mismatch')
        if new['budget_tokens'] > self.cap:
            raise BudgetStopped('Actual new cumulative budget exceeded')
        return new

    def observe_history(self, payload):
        if self.replayed >= 100:
            for message in payload['messages']:
                if message['role'] != 'tool' or message['tool_call_id'] in self.old_tool_ids:
                    continue
                try: value = json.loads(message['content'])
                except (ValueError, TypeError): continue
                if isinstance(value, dict) and value.get('error'):
                    raise BudgetStopped('New tool error: '+str(value['error']))
        return super().observe_history(payload)

    def _request(self, request):
        if self.offline_boundary is None and self.replayed == 100 and accounted(self.out)['attempts'] == 100:
            if self.first_request is None or request.content != self.first_request.read_bytes():
                raise BudgetStopped('Live request101 differs from offline proof')
        return super()._request(request)
