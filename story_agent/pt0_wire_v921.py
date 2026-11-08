"""PT0 v0.92.1: §11 carry-in estimate, new request namespace, all-scope trace."""
import json
from pathlib import Path

from story_agent.pt0_wire import AuditedDeepSeek as OriginalWire, accounted
from story_agent.agent_tl4 import Monitor as OriginalMonitor
from story_agent.budget import BudgetStopped

VERSION='v0.92.1'
PREFIX='PT0:'+VERSION+':'
LEGACY_ID='5e619cce-674d-46bc-bf7a-5a536f7a7d28'
LEGACY_RUN='PT0:v0.92.0:stage1:A1'
LEGACY_BUDGET=60_262


class Monitor(OriginalMonitor):
    def observe(self,message,call_number,child=False,replay=False):
        self.check_stop()
        self.sequence+=1
        if not replay:
            path=self.directory/'trace-wire.jsonl'
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a',encoding='utf-8',newline='\n') as stream:
                stream.write(json.dumps({'call':call_number,'scope':'reviewer' if child else 'reference',
                                        'message':message},ensure_ascii=False)+'\n')


class AuditedDeepSeek(OriginalWire):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.run_id=PREFIX+self.out.name+':'+self.topic
        self.monitor=Monitor(self.out/self.topic,self.topic,self.sleeper)

    def gate(self):
        self.monitor.check_stop()
        if (self.out/'STOP').exists():raise BudgetStopped('Stage STOP between requests')
        summary=self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts'])-{'settled'}:
            raise BudgetStopped('Shared ledger must be accounting-only and settled')
        totals=accounted(self.out)
        with self.ledger.connect() as db:
            legacy=db.execute('SELECT run_id,status,actual,reserved FROM calls WHERE id=?',(LEGACY_ID,)).fetchone()
            if legacy!=(LEGACY_RUN,'settled',LEGACY_BUDGET,LEGACY_BUDGET):
                raise BudgetStopped('Authorized interrupted request estimate not reconciled')
            rows=db.execute('SELECT status,actual FROM calls WHERE id=? OR run_id LIKE ?',
                (LEGACY_ID,PREFIX+self.out.name+':%')).fetchall()
        if (len(rows)!=totals['attempts'] or sum(r[1] or 0 for r in rows)!=totals['accounted_raw_tokens']
            or totals['estimated_budget_tokens']<LEGACY_BUDGET):
            raise BudgetStopped('Task ledger and evidence disagree, including carry-in estimate')
        if totals['budget_tokens']>self.cap:
            raise BudgetStopped('Uncached input plus output cap exceeded')
        return totals
