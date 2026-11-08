"""D-022: retry a measured empty reply once, immediately and byte-identically."""
import copy
import hashlib
from pathlib import Path
from unittest.mock import patch

from story_agent import wire_default_v1 as frozen
from story_agent.default_v953 import CONFIGURATION, VERSION, replace_json
from story_agent.strong_agent_sa1 import read_json, save, emit
from story_agent.budget import BudgetStopped

TransportStopped = frozen.TransportStopped


def complete_empty(body):
    if not isinstance(body, dict) or not frozen.valid_usage(body.get('usage')):
        return False
    try:
        message = body['choices'][0]['message']
        content = message.get('content')
        calls = message.get('tool_calls', [])
        return (content is None or isinstance(content, str) and not content.strip()) and isinstance(calls, list) and not calls
    except (KeyError, IndexError, TypeError, AttributeError):
        return False


class EmptyReply(BudgetStopped):
    def __init__(self, directory):
        self.directory = Path(directory)
        super().__init__('Measured reply has neither tools nor content')


class AuditedDeepSeek(frozen.AuditedDeepSeek):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = CONFIGURATION+':'+str(self.directory.resolve())

    def accept(self, body, directory, agent, replay=False):
        if complete_empty(body):
            raise EmptyReply(directory)
        return super().accept(body, directory, agent, replay)

    def cost(self):
        value = super().cost()
        value.update(configuration=CONFIGURATION, code_version=VERSION,
            complete_empty_reply_count=len(list((self.directory/'calls').glob('*/empty-reply.json'))),
            complete_empty_resend_requests=len(list((self.directory/'calls').glob('*/empty-resend.json'))))
        return value

    def _request(self, request):
        with patch.object(frozen, 'CONFIGURATION', CONFIGURATION), patch.object(frozen, 'VERSION', VERSION):
            return self._request_once_or_resend(request)

    def _request_once_or_resend(self, request):
        replaying = self.replayed < len(self.replay)
        checkpoint = read_json(self.directory/'checkpoint.json') if (self.directory/'checkpoint.json').exists() else {}
        used = checkpoint.get('empty_reply_count', 0) if not replaying and checkpoint.get('status') == 'pending' else 0
        previous_empty = checkpoint.get('empty_reply_call')
        if used >= 2:
            raise BudgetStopped('Measured empty-reply resend exhausted; planner decision required')
        while True:
            before_contract = copy.deepcopy(self.contract.__dict__)
            before = set((self.directory/'calls').glob('*/scope.json'))
            is_resend = bool(used)
            retry_of = previous_empty
            try:
                return super()._request(request)
            except EmptyReply as exc:
                used += 1
                previous_empty = exc.directory.name
                save(exc.directory/'empty-reply.json', dict(configuration=CONFIGURATION, code_version=VERSION,
                    empty_reply_count=used, scope=self.monitor.scope, call=previous_empty,
                    request_sha256=hashlib.sha256(request.content).hexdigest(), usage=read_json(exc.directory/'usage.json')['usage']))
                checkpoint = read_json(self.directory/'checkpoint.json')
                checkpoint.update(empty_reply_count=used, empty_reply_call=previous_empty)
                replace_json(self.directory/'checkpoint.json', checkpoint)
                replace_json(self.directory/'cost.json', self.cost())
                self.trace(dict(type='complete_empty_reply', call=previous_empty, scope=self.monitor.scope,
                                empty_reply_count=used, immediate_resend=used == 1))
                self.gate()
                if used >= 2:
                    raise BudgetStopped('Two measured empty replies for the same request; stopped') from None
                # Failed empty replies must not create a new logical invocation
                # when only accepted responses are replayed on a later resume.
                self.contract.__dict__.clear()
                self.contract.__dict__.update(before_contract)
                emit(message='回复有usage但无正文与工具，原字节立即重发一次', scope=self.monitor.scope)
            finally:
                if is_resend:
                    for path in sorted(set((self.directory/'calls').glob('*/scope.json')) - before):
                        target = path.parent/'empty-resend.json'
                        if not target.exists():
                            save(target, dict(retry_of=retry_of, request_sha256=hashlib.sha256(request.content).hexdigest(),
                                              configuration=CONFIGURATION, code_version=VERSION))
                    replace_json(self.directory/'cost.json', self.cost())
