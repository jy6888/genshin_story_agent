"""User override: continue complete tool calls with empty reasoning, retain counts."""
from story_agent.tl5_wire import AuditedDeepSeek as Previous,Contract,accounted
from story_agent.budget import BudgetStopped

PREFIX='TL5:v0.93.3:'
OLD_RUN='TL5:v0.93.2:stage2:T5'

class ToolContinuationContract(Contract):
    def accept(self,body,agent):
        message=body.get('choices',[{}])[0].get('message',{})
        complete=body.get('choices',[{}])[0].get('finish_reason') in ('stop','tool_calls')
        previous=self.streaks.get(agent,0)
        if complete and message.get('reasoning_content')=='' and message.get('tool_calls') and previous>=2:
            # Preserve all format checks and actual streak while bypassing only this stop.
            self.streaks[agent]=0
            try:result,info=super().accept(body,agent)
            except BaseException:
                self.streaks[agent]=previous
                raise
            self.streaks[agent]=previous+1
            info.update(empty_streak=previous+1,complete_tool_continued=True)
            return result,info
        return super().accept(body,agent)

class AuditedDeepSeek(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.run_id=PREFIX+self.out.name+':'+self.topic
        self.contract=ToolContinuationContract()

    def gate(self):
        self.monitor.check_stop();summary=self.ledger.summary();totals=accounted(self.out)
        if summary['limit'] is not None or set(summary['status_counts'])-{'settled'}:raise BudgetStopped('Unsettled ledger')
        with self.ledger.connect() as db:
            rows=[r for ident in (OLD_RUN,PREFIX+self.out.name+':'+self.topic)
                  for r in db.execute('SELECT status,actual FROM calls WHERE run_id=?',(ident,)).fetchall()]
        if len(rows)!=totals['attempts'] or sum(r[1] or 0 for r in rows)!=totals['accounted_raw_tokens']:
            raise BudgetStopped('Old plus new ledger/evidence mismatch')
        if totals['budget_tokens']>self.cap:raise BudgetStopped('Actual cumulative cap exceeded')
        return totals
