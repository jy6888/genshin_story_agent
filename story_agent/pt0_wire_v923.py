"""D-20261107-011: accept complete empty-reasoning replies, per invocation."""
import hashlib
import json

import httpx
from story_agent.pt0_wire import AuditedDeepSeek as Previous,accounted,MODEL,REVIEW_TOOLS,estimate
from story_agent.budget import BudgetStopped
from story_agent.strong_agent_sa1 import read_json,save,emit

PREFIX='PT0:v0.92.3:'


class ReplayBoundary(Exception):
    """The offline graph reached the first request that would need a provider."""


class ReasoningContract:
    def __init__(self):
        self.invocations={};self.active={};self.streaks={};self.responses=0

    def agent(self,payload,scope):
        users=[m['content'] for m in payload['messages'] if m['role']=='user']
        if len(users)!=1:raise BudgetStopped('Unexpected user-message count')
        key=scope,users[0]
        if not any(m['role']=='assistant' for m in payload['messages']):
            n=self.invocations.get(key,0)+1;self.invocations[key]=n
            self.active[key]=scope+':'+hashlib.sha256(users[0].encode('utf-8')).hexdigest()[:12]+':'+str(n)
        if key not in self.active:raise BudgetStopped('Agent history starts without its first request')
        return self.active[key]

    def accept(self,body,agent):
        try:
            choice=body['choices'][0];m=choice['message'];reasoning=m.get('reasoning_content')
            if not isinstance(reasoning,str):raise BudgetStopped('Malformed reasoning field')
            if choice.get('finish_reason') not in ('stop','tool_calls'):
                raise BudgetStopped('Non-completed response; no feedback or retry')
            content=m.get('content');calls=m.get('tool_calls',[])
            if content is not None and not isinstance(content,str):raise BudgetStopped('Malformed content field')
            if not isinstance(calls,list):raise BudgetStopped('Malformed tool_calls field')
            if not calls and not (content or '').strip():raise BudgetStopped('Reply has neither tools nor content')
            empty=not reasoning.strip();self.responses+=1
            self.streaks[agent]=self.streaks.get(agent,0)+1 if empty else 0
            info={'agent_id':agent,'reasoning_empty':empty,'empty_streak':self.streaks[agent]}
            if empty and self.responses==1:raise BudgetStopped('Topic first reply has empty reasoning')
            if self.streaks[agent]>=3:raise BudgetStopped('Same agent has three consecutive empty reasoning replies')
            return m,info
        except (KeyError,IndexError,TypeError):raise BudgetStopped('Malformed model response') from None


class AuditedDeepSeek(Previous):
    def __init__(self,*args,scope_fn,monitor_factory,carry_run_ids=(),offline_boundary=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.run_id=PREFIX+self.out.name+':'+self.topic
        self.scope_fn=scope_fn;self.carry_run_ids=tuple(carry_run_ids);self.offline_boundary=offline_boundary
        self.contract=ReasoningContract();self.monitor=monitor_factory(self.out/self.topic,self.topic,self.sleeper)

    def gate(self):
        self.monitor.check_stop();summary=self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts'])-{'settled'}:
            raise BudgetStopped('Shared ledger must be accounting-only and settled')
        totals=accounted(self.out)
        ids=(*self.carry_run_ids,*(PREFIX+self.out.name+':'+t for t in ('T1','N1')))
        with self.ledger.connect() as db:
            rows=[r for ident in ids for r in db.execute('SELECT status,actual FROM calls WHERE run_id=?',(ident,)).fetchall()]
        if len(rows)!=totals['attempts'] or sum(r[1] or 0 for r in rows)!=totals['accounted_raw_tokens']:
            raise BudgetStopped('Stage-2 original plus continuation ledger/evidence disagree')
        if totals['budget_tokens']>self.cap:raise BudgetStopped('Uncached input plus output cap exceeded')
        return totals

    def accept(self,body,directory,agent,replay=False):
        m,info=self.contract.accept(body,agent)
        for name,value in (('message.json',m),('contract.json',info)):
            path=directory/name
            if path.exists():
                if read_json(path)!=value:raise BudgetStopped('Replay annotation differs')
            else:save(path,value)
        for name,value in (('reasoning.txt',m['reasoning_content']),('content.txt',m.get('content') or '')):
            path=directory/name
            if path.exists():
                if path.read_text(encoding='utf-8')!=value:raise BudgetStopped('Replay text archive differs')
            else:path.write_text(value,encoding='utf-8')
        self.monitor.annotation=info
        self.monitor.observe(m,int(directory.name),replay=replay)

    def _request(self,request):
        payload=self.safe(json.loads(request.content));names={t['function']['name'] for t in payload.get('tools',[])}
        if (request.method!='POST' or request.url.scheme!='https' or request.url.host!='api.deepseek.com'
            or not request.url.path.endswith('/chat/completions') or payload.get('model')!=MODEL
            or payload.get('thinking')!={'type':'enabled'} or payload.get('reasoning_effort')!='max'
            or 'temperature' in payload or payload.get('max_tokens',payload.get('max_completion_tokens'))!=49152
            or payload.get('stream') or names not in self.allowed_tool_sets):raise BudgetStopped('Unexpected model or tools')
        self.monitor.scope=self.scope_fn(payload);agent=self.contract.agent(payload,self.monitor.scope)
        self.monitor.check_stop();self.observe_history(payload)
        if self.replayed<len(self.replay):
            path=self.replay[self.replayed]
            if request.content!=(path.parent/'request-body.bin').read_bytes():raise BudgetStopped('Resume request bytes differ from settled request')
            self.replayed+=1;body=read_json(path);self.accept(body,path.parent,agent,replay=True)
            return httpx.Response(200,json=body)
        if self.offline_boundary is not None:
            self.offline_boundary(request,self)
            raise ReplayBoundary('Offline replay reached next provider request')
        for attempt in range(2):
            totals=self.gate();number=totals['attempts']+1;directory=self.out/self.topic/'calls'/f'{number:03d}'
            save(directory/'request.json',payload);(directory/'request-body.bin').write_bytes(request.content)
            ident=self.ledger.reserve(self.run_id,MODEL,2*len(request.content)+2048+49152)
            save(directory/'reservation.json',{'id':ident});emit(request_started=number,topic=self.topic,budget_tokens=totals['budget_tokens'])
            try:response=self.wire.handle_request(request)
            except (httpx.TransportError,OSError) as exc:
                save(directory/'error.json',dict(self.wire.error or {},error_type=type(exc).__name__,error=self.error_text(exc)))
                try:self.ledger.settle(ident,None,None)
                except BudgetStopped:pass
                self.reconcile(ident,estimate(self.out,self.topic,payload,request.content))
                if attempt:raise BudgetStopped('Two consecutive unusable transport responses') from None
                self.gate();emit(retry_wait_seconds=60);self.sleeper(30);emit(retry_wait_seconds=30);self.sleeper(30);continue
            try:body=response.json()
            except (ValueError,UnicodeError):body=None
            if not response.is_success or not isinstance(body,dict) or not isinstance(body.get('usage'),dict):
                save(directory/'error.json',{'http_status':response.status_code,'error_type':'InvalidResponse',
                    'received_bytes':len(response.content),'error_body':self.error_text(response.content.decode('utf-8',errors='replace'))})
                self.ledger.settle(ident,None,response.status_code)
                raise BudgetStopped('HTTP/parse failure; usage unknown, no automatic resend')
            body=self.safe(body);save(directory/'response.json',body);self.ledger.settle(ident,body['usage'],response.status_code)
            totals=accounted(self.out);save(directory/'usage.json',{'usage':body['usage'],'stage_usage':totals})
            self.accept(body,directory,agent);emit(request_settled=number,topic=self.topic,usage=body['usage'],budget_tokens=totals['budget_tokens'])
            return response
