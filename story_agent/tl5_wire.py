"""Reuse PT0 transport/replay/accounting, isolate TL5 scopes and run IDs."""
import hashlib
import copy
from story_agent import agent_pt0
from story_agent import conflict_check_tl5
from story_agent.pt0_wire_v923 import AuditedDeepSeek as Previous, ReasoningContract, accounted
from story_agent.budget import BudgetStopped
from scripts.run_pt0_v924 import Monitor

PREFIX='TL5:v0.93.0:'

def scope(payload):
    system=payload['messages'][0]['content']
    if system==conflict_check_tl5.PROMPT: return 'conflict'
    if system==agent_pt0.REVIEW_PROMPT: return 'reviewer'
    if system==agent_pt0.previous.COMPARE_PROMPT: return 'compare'
    return 'main'

class Contract(ReasoningContract):
    def agent(self,payload,scope):
        # One allowed formatting feedback adds a second user message to this invocation.
        users=[m['content'] for m in payload['messages'] if m['role']=='user']
        if not users: raise BudgetStopped('Missing user turn')
        key=scope,users[0]
        if not any(m['role']=='assistant' for m in payload['messages']):
            n=self.invocations.get(key,0)+1; self.invocations[key]=n
            self.active[key]=scope+':'+hashlib.sha256(users[0].encode('utf-8')).hexdigest()[:12]+':'+str(n)
        if key not in self.active: raise BudgetStopped('History missing initial request')
        return self.active[key]

class AuditedDeepSeek(Previous):
    def __init__(self,*args,**kwargs):
        main=agent_pt0.previous.MAIN_TOOLS|{'request_external'}
        kwargs.setdefault('allowed_tool_sets',(set(),agent_pt0.REVIEW_TOOLS,
            agent_pt0.REVIEW_TOOLS|{'read_community_graph'},main,main|{'search_reference','read_reference'}))
        super().__init__(*args,scope_fn=scope,monitor_factory=Monitor,**kwargs)
        self.run_id=PREFIX+self.out.name+':'+self.topic
        self.contract=Contract()

    def gate(self):
        self.monitor.check_stop(); summary=self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts'])-{'settled'}:
            raise BudgetStopped('Shared ledger must be settled')
        totals=accounted(self.out)
        with self.ledger.connect() as db:
            rows=db.execute('SELECT status,actual FROM calls WHERE run_id LIKE ?',
                            (PREFIX+self.out.name+':%',)).fetchall()
        if len(rows)!=totals['attempts'] or sum(r[1] or 0 for r in rows)!=totals['accounted_raw_tokens']:
            raise BudgetStopped('TL5 ledger/evidence disagree')
        if totals['budget_tokens']>self.cap: raise BudgetStopped('Actual accumulated budget exceeded')
        return totals

    def observe_history(self,payload):
        # Private graph output is prose in a JSON envelope already; preserve audit behavior.
        audit=copy.deepcopy(payload); audit.setdefault('tools',[])
        return super().observe_history(audit)
