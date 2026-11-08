"""TL5b v0.94.2: authorized empty-response retry and cumulative carry."""
import json
import httpx
from story_agent.tl5b_wire_v941 import AuditedDeepSeek as Previous,accounted,delta
from story_agent.tl5_wire import Contract
from story_agent.pt0_wire_v923 import ReplayBoundary
from story_agent.agent_tl3_d017 import estimate
from story_agent.empty_response_tl5b_v942 import settle_estimate,AUTHORITY
from story_agent.strong_agent_sa1 import read_json,save,emit
from story_agent.budget import BudgetStopped

CARRY=75720


class CompleteReplyContract(Contract):
    """D-016: retain all complete-response checks; empty reasoning only counts."""
    def accept(self,body,agent):
        try:
            choice=body['choices'][0];message=choice['message']
            reasoning=message.get('reasoning_content')
            if not isinstance(reasoning,str):raise BudgetStopped('Malformed reasoning field')
            if choice.get('finish_reason') not in ('stop','tool_calls'):
                raise BudgetStopped('Non-completed response; no feedback or retry')
            content=message.get('content');calls=message.get('tool_calls',[])
            if content is not None and not isinstance(content,str):raise BudgetStopped('Malformed content field')
            if not isinstance(calls,list):raise BudgetStopped('Malformed tool_calls field')
            if not calls and not (content or '').strip():raise BudgetStopped('Reply has neither tools nor content')
            self.responses+=1;empty=not reasoning.strip()
            self.streaks[agent]=self.streaks.get(agent,0)+1 if empty else 0
            return message,dict(agent_id=agent,reasoning_empty=empty,empty_streak=self.streaks[agent])
        except (KeyError,IndexError,TypeError):raise BudgetStopped('Malformed model response') from None


def usable(body):
    try:
        message=body['choices'][0]['message']
        return isinstance(message,dict) and (isinstance(message.get('content'),str) and bool(message['content'].strip())
            or isinstance(message.get('tool_calls'),list) and bool(message['tool_calls']))
    except (KeyError,IndexError,TypeError):return False


class AuditedDeepSeek(Previous):
    def __init__(self,*args,carry=CARRY,**kwargs):
        super().__init__(*args,**kwargs)
        self.run_id='TL5b:v0.94.2:'+self.out.name+':'+self.topic
        self.carry=carry;self.live_contract=False

    def gate(self):
        self.monitor.check_stop();summary=self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts'])-{'settled'}:
            raise BudgetStopped('Shared ledger must be settled')
        new=delta(accounted(self.out),self.baseline)
        with self.ledger.connect() as db:
            rows=db.execute('SELECT status,actual FROM calls WHERE run_id=?',(self.run_id,)).fetchall()
        if len(rows)!=new['attempts'] or sum(r[1] or 0 for r in rows)!=new['accounted_raw_tokens']:
            raise BudgetStopped('New request ledger/evidence mismatch')
        total=self.carry+new['budget_tokens']
        if total>self.cap:raise BudgetStopped('Actual cumulative including75720 carry exceeded cap')
        return dict(new,carry_budget_tokens=self.carry,cumulative_budget_tokens=total)

    def _failure_estimate(self,directory,identifier,payload,request,body,http_status):
        usage=body.get('usage') if isinstance(body,dict) else None
        fields=('prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens')
        if isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0 for k in fields) and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'] and usage['prompt_tokens']==usage['prompt_cache_hit_tokens']+usage['prompt_cache_miss_tokens']:
            value=dict(usage,budget_tokens=usage['prompt_cache_miss_tokens']+usage['completion_tokens'],
                       estimated=False,method='Provider usage measured, response unusable')
        else:value=estimate(self.out,self.topic,payload,request.content)
        value.update(topic=self.topic,call_id=identifier,http_status=http_status,authority=AUTHORITY)
        settled=settle_estimate(self.ledger,identifier,self.run_id,value)
        save(directory/'reconciliation.json',settled)
        save(self.out/'estimates'/f'{identifier}.json',value)
        return value

    def _request(self,request):
        payload=self.safe(json.loads(request.content))
        names={t['function']['name'] for t in payload.get('tools',[])}
        if (request.method!='POST' or request.url.scheme!='https' or request.url.host!='api.deepseek.com'
            or not request.url.path.endswith('/chat/completions') or payload.get('model')!='deepseek-v4-pro'
            or payload.get('thinking')!={'type':'enabled'} or payload.get('reasoning_effort')!='max'
            or 'temperature' in payload or payload.get('max_tokens',payload.get('max_completion_tokens'))!=49152
            or payload.get('stream') or names not in self.allowed_tool_sets):raise BudgetStopped('Unexpected model or tools')
        self.monitor.scope=self.scope_fn(payload);agent=self.contract.agent(payload,self.monitor.scope)
        self.monitor.check_stop();self.observe_history(payload)
        if self.replayed<len(self.replay):
            path=self.replay[self.replayed]
            if request.content!=(path.parent/'request-body.bin').read_bytes():raise BudgetStopped('Replay request bytes differ')
            self.replayed+=1;body=read_json(path);self.accept(body,path.parent,agent,replay=True)
            return httpx.Response(200,json=body)
        if self.offline_boundary is not None:
            self.offline_boundary(request,self);raise ReplayBoundary('Offline replay reached next provider request')
        if self.replayed==100 and accounted(self.out)['attempts']==100:
            if self.first_request is None or request.content!=self.first_request.read_bytes():
                raise BudgetStopped('Live101 differs from offline proof')
        if not self.live_contract:
            contract=CompleteReplyContract();contract.__dict__.update(self.contract.__dict__)
            self.contract=contract;self.live_contract=True
        for attempt in range(2):
            totals=self.gate();number=self.baseline['attempts']+totals['attempts']+1
            directory=self.out/self.topic/'calls'/f'{number:03d}'
            save(directory/'request.json',payload);(directory/'request-body.bin').write_bytes(request.content)
            identifier=self.ledger.reserve(self.run_id,'deepseek-v4-pro',2*len(request.content)+2048+49152)
            save(directory/'reservation.json',{'id':identifier})
            emit(request_started=number,topic=self.topic,budget_tokens=totals['cumulative_budget_tokens'])
            try:response=self.wire.handle_request(request)
            except (httpx.TransportError,OSError) as exc:
                save(directory/'error.json',dict(self.wire.error or {},error_type=type(exc).__name__,error=self.error_text(exc)))
                self._failure_estimate(directory,identifier,payload,request,None,None)
                if attempt:raise BudgetStopped('Two consecutive transport failures') from None
                self.gate();emit(retry_wait_seconds=60);self.sleeper(30);self.sleeper(30);continue
            try:body=response.json()
            except (ValueError,UnicodeError):body=None
            empty=response.status_code==200 and (not usable(body) or not isinstance(body.get('usage') if isinstance(body,dict) else None,dict))
            if empty:
                save(directory/'error.json',dict(http_status=200,error_type='EmptyResponse',received_bytes=len(response.content),
                    error_body=self.error_text(response.content.decode('utf-8',errors='replace')),retry_number=attempt))
                self._failure_estimate(directory,identifier,payload,request,body,200)
                emit(empty_response=number,original_request_retry=not bool(attempt),usage=self.gate())
                if attempt:raise BudgetStopped('Two consecutive unusable HTTP200 responses')
                self.sleeper(30);self.sleeper(30);continue
            if not response.is_success or not isinstance(body,dict) or not isinstance(body.get('usage'),dict):
                save(directory/'error.json',dict(http_status=response.status_code,error_type='InvalidResponse',received_bytes=len(response.content)))
                self.ledger.settle(identifier,None,response.status_code)
                raise BudgetStopped('HTTP/parse failure outside authorized empty200 retry')
            body=self.safe(body);save(directory/'response.json',body)
            self.ledger.settle(identifier,body['usage'],response.status_code)
            save(directory/'usage.json',dict(usage=body['usage'],stage_usage=delta(accounted(self.out),self.baseline)))
            self.accept(body,directory,agent)
            emit(request_settled=number,topic=self.topic,usage=body['usage'],new_budget_tokens=delta(accounted(self.out),self.baseline)['budget_tokens'])
            return response
