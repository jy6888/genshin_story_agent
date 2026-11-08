"""Measured carry, prefix-only historical replay, unchanged wire safeguards."""
from unittest.mock import patch
from story_agent import wire_default_v953 as previous,wire_default_v1 as frozen
from story_agent.default_v955 import CONFIGURATION,VERSION,replace_json
from story_agent.strong_agent_sa1 import read_json

TransportStopped=frozen.TransportStopped


class AuditedDeepSeek(previous.AuditedDeepSeek):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        path=self.directory/'recovery.json'
        self.recovery=read_json(path) if path.exists() else {}
        excluded=set(self.recovery.get('excluded_replay_calls',[]))
        self.replay=[p for p in self.replay if p.parent.name not in excluded]
        self.run_id=CONFIGURATION+':'+str(self.directory.resolve())

    def cost(self):
        result=super().cost()
        recovery=getattr(self,'recovery',{})
        result.update(configuration=CONFIGURATION,code_version=VERSION,
            historical_completion_calls_excluded_from_replay=recovery.get('excluded_replay_calls',[]),
            carried_budget_tokens=recovery.get('carried_budget_tokens',0),
            carried_paid_attempts=recovery.get('carried_paid_attempts',0))
        return result

    def _request(self,request):
        with patch.object(previous,'CONFIGURATION',CONFIGURATION),patch.object(previous,'VERSION',VERSION), \
             patch.object(previous,'replace_json',replace_json),patch.object(frozen,'replace_json',replace_json):
            return super()._request(request)
