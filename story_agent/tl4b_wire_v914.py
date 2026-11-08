"""Preserve schema-validation error classification at the lossy wire boundary."""
import json

from langchain_core.messages import AIMessage, ToolMessage
from story_agent.agent_tl2b import CompareGuard
from story_agent.budget import BudgetStopped
from story_agent.strong_agent_sa1 import emit


class ParameterGuard(CompareGuard):
    def __init__(self, required):
        super().__init__()
        self.required = required
        self.schema_errors = []

    def observe(self, message):
        if message.type == 'tool':
            call = self.calls[message.tool_call_id]
            missing = set(self.required.get(call['name'], ())) - call['args'].keys()
            # Only a demonstrably malformed call plus the framework's exact error
            # prefix qualifies. Arbitrary non-JSON/corrupted results still fail.
            prefix = "Error invoking tool '" + call['name'] + "' with kwargs "
            if missing and message.content.startswith(prefix):
                self.returns += 1
                self.faults = 0
                key = json.dumps({'name': call['name'], 'args': call['args']},
                                 ensure_ascii=False, sort_keys=True)
                self.invalid[key] += 1
                self.schema_errors.append({'id': message.tool_call_id,
                                           'missing_required': sorted(missing)})
                if self.invalid[key] >= 5:
                    raise BudgetStopped('Same invalid tool call occurred five times in one topic')
                return
        super().observe(message)


class HistoryAudit:
    """No prompt, schema, tool return, transport, or semantic-gate change."""
    def observe_history(self, payload):
        required = {t['function']['name']: t['function']['parameters'].get('required', ())
                    for t in payload['tools']}
        history_key = json.dumps([m for m in payload['messages'] if m['role']=='user'][:1],
                                 ensure_ascii=False)
        if not hasattr(self, 'histories'):
            self.histories = {}
        seen, guard = self.histories.setdefault(history_key, (set(), ParameterGuard(required)))
        for i, m in enumerate(payload['messages']):
            key = i, json.dumps(m, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            if m['role'] == 'assistant':
                calls = [{'id': t['id'], 'name': t['function']['name'],
                          'args': json.loads(t['function']['arguments'])}
                         for t in m.get('tool_calls', [])]
                guard.observe(AIMessage(content=m.get('content') or '', tool_calls=calls))
            elif m['role'] == 'tool':
                guard.observe(ToolMessage(content=m['content'], tool_call_id=m['tool_call_id']))
                emit(tool_returned=guard.calls[m['tool_call_id']]['name'],
                     returned_characters=len(m['content']), invalid_calls=sum(guard.invalid.values()))
            seen.add(key)
