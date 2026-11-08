"""PT0 §13: byte-replay62 then continue63, serial T1->N1, total cap1.4M."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import httpx
from langchain_core.tools import StructuredTool
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from scripts import run_pt0_stage2 as old
from story_agent.pt0_wire_v923 import AuditedDeepSeek as Wire,ReplayBoundary,PREFIX,accounted
from story_agent.strong_agent_sa1 import read_json,save,sha,emit
from story_agent.strong_agent_sa2 import credentials,live_model
from story_agent.budget import BudgetStopped
from story_agent.external_pt0_v921 import Wikipedia
from story_agent.texts_tl4b import AgentTools

VERSION='v0.92.4'
PREFIX='PT0:'+VERSION+':'
ROOT,TASK,SPEC,README,GOLDS=old.ROOT,old.TASK,old.SPEC,old.README,old.GOLDS
BASE=TASK/'evidence/pt0-v0.92.4';OUT=BASE/'stage2';OFFLINE=BASE/'stage0'
REFERENCE_DIR=CODE/'data/references/pt0-v0.92.4-stage2'
EXPECTED,CAP,TOPICS=old.EXPECTED,old.CAP,old.TOPICS
s=old.s;scope=old.scope;build=old.build
CARRY_IDS=('PT0:v0.92.2:stage2:T1','PT0:v0.92.3:stage2:T1')


def release():
    spec=SPEC.read_text(encoding='utf-8')
    if '## 13.' not in spec or 'D-20261107-011' not in spec:raise BudgetStopped('§13 authorization required')
    for p,h in GOLDS.values():
        if sha(p)!=h:raise BudgetStopped('Frozen gold SHA256 mismatch (hash-only)')
    return read_json(old.OUT/'protocol.json')['questions']


class Monitor(old.Monitor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.annotation={}
        path=self.directory/'trace-wire.jsonl'
        self.existing={v['call']:v for v in map(json.loads,path.read_text(encoding='utf-8').splitlines())} if path.exists() else {}

    def observe(self,message,call_number,child=False,replay=False):
        self.check_stop()
        if self.scope=='main':self.sequence+=1
        value=dict(call=call_number,scope=self.scope,message=message,**self.annotation);self.records.append(value)
        if call_number in self.existing:assert self.existing[call_number]==value
        else:
            path=self.directory/'trace-wire.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')
            self.existing[call_number]=value


class AuditedDeepSeek(Wire):
    def __init__(self,*args,**kwargs):
        kwargs.setdefault('scope_fn',scope);kwargs.setdefault('monitor_factory',Monitor)
        kwargs.setdefault('carry_run_ids',CARRY_IDS);super().__init__(*args,**kwargs)
        self.run_id=PREFIX+self.out.name+':'+self.topic

    def gate(self):
        self.monitor.check_stop();summary=self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts'])-{'settled'}:raise BudgetStopped('Shared ledger must be settled')
        totals=accounted(self.out)
        ids=(*self.carry_run_ids,*(PREFIX+self.out.name+':'+t for t in TOPICS))
        with self.ledger.connect() as db:
            rows=[r for ident in ids for r in db.execute('SELECT status,actual FROM calls WHERE run_id=?',(ident,)).fetchall()]
        if len(rows)!=totals['attempts'] or sum(r[1] or 0 for r in rows)!=totals['accounted_raw_tokens']:raise BudgetStopped('Continuation ledger/evidence disagree')
        if totals['budget_tokens']>self.cap:raise BudgetStopped('Uncached input plus output cap exceeded')
        return totals

    def _request(self,request):
        if (self.topic=='T1' and self.offline_boundary is None and len(self.replay)==63
            and self.replayed==63 and accounted(self.out)['attempts']==63):
            if request.content!=(OFFLINE/'next064-request-body.bin').read_bytes():raise BudgetStopped('Live064 differs from offline proof')
        return Wire._request(self,request)


def verify():
    p=read_json(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read_json(OFFLINE/'protocol-anchor.json')['sha256']
    assert p['questions']==release()
    for path,digest in p['hashes'].items():
        if sha(path)!=digest:raise BudgetStopped('Frozen file changed: '+path)
    emit(frozen_code_spec_gold_protocol_verified=True,hashes=len(p['hashes']),readme_hash_not_compared=True)


def offline(smoke=False):
    name='tests.test_pt0_v923'+('.ContractTest.test_complete_empty_accepted_and_archived' if smoke else '')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2,failfast=True).run(unittest.defaultTestLoader.loadTestsFromName(name))
    emit(tests=result.testsRun,passed=result.wasSuccessful(),log=stream.getvalue())
    prefix='smoke' if smoke else 'offline-tests';save(OFFLINE/f'{prefix}-{len(list(OFFLINE.glob(prefix+"-*.json")))+1:02d}.json',
        {'passed':result.wasSuccessful(),'tests':result.testsRun,'provider_requests':0,'log':stream.getvalue()})
    if not result.wasSuccessful():raise BudgetStopped('Offline tests failed')


def prepare():
    from tools.codex.pt0_v924_recover import prepare as recover_prepare
    recover_prepare()


def forbidden(_):raise AssertionError('No network in offline replay')


def proof():
    verify()
    if (OUT/'replay63-proof.json').exists():raise BudgetStopped('Replay63 proof already saved')
    from tools.codex.pt0_v924_recover import partial_accesses
    from collections import defaultdict,deque
    originals=AgentTools(ROOT).tools();by_name={t.name:t for t in originals};queues=defaultdict(deque);executed=[]
    def key(t,args):return t.name,json.dumps(t.args_schema.model_validate(args).model_dump(),ensure_ascii=False,sort_keys=True)
    for c in partial_accesses():
        if c['name'] in by_name and not c.get('pending'):queues[key(by_name[c['name']],c['args'])].append(c)
    expected=[json.loads(c['function']['arguments']) for c in read_json(OUT/'T1/calls/063/response.json')['choices'][0]['message']['tool_calls']]
    tools=[]
    for original in originals:
        def make(t):
            def recorded(**kwargs):
                k=key(t,kwargs)
                if queues[k]:return queues[k].popleft()['raw_result']
                assert t.name=='read_text' and any(key(t,kwargs)==key(t,a) for a in expected) and len(executed)<1
                result=t.invoke(kwargs);executed.append({'name':t.name,'args':kwargs,'result':result});return result
            return recorded
        tools.append(StructuredTool.from_function(make(original),name=original.name,description=original.description,args_schema=original.args_schema,infer_schema=False))
    before=accounted(OUT)
    def boundary(request,wire):
        assert wire.replayed==63 and len(executed)==1 and all(not v for v in queues.values())
        assert read_json(OUT/'T1/calls/063/contract.json')['reasoning_empty']
        (OFFLINE/'next064-request-body.bin').write_bytes(request.content)
        save(OFFLINE/'executed063-tools.json',executed)
    wire=AuditedDeepSeek(OUT,'T1','offline-placeholder',CAP,resume=True,inner=httpx.MockTransport(forbidden),offline_boundary=boundary)
    model=live_model('offline-placeholder',wire);wiki=Wikipedia(REFERENCE_DIR,httpx.MockTransport(forbidden))
    try:
        agent,_=build(model,tools,wiki,wire,OUT/'T1');caught=False
        with redirect_stdout(io.StringIO()),patch('socket.socket.connect',side_effect=AssertionError('No network')):
            try:agent.invoke({'messages':[{'role':'user','content':release()['T1']}]},config={'max_concurrency':1})
            except ReplayBoundary:caught=True
        assert caught and wire.replayed==63 and before==accounted(OUT)
        save(OUT/'replay63-proof.json',{'passed':True,'responses':63,'provider_requests':0,'all_request_bodies_byte_equal':True,
            'response63_accepted':True,'reasoning63_file_empty':(OUT/'T1/calls/063/reasoning.txt').read_bytes()==b'',
            'tools63_executed':executed,'next_request_number':64,'next_request_sha256':sha(OFFLINE/'next064-request-body.bin'),
            'usage_unchanged':True,'usage':before})
        emit(replay63_passed=True,provider_requests=0,accepted_empty=63,executed_tools=1,next_request=64)
    finally:model.http_client.close();wiki.close()


def execute(topic,resume=False):
    verify()
    assert any(read_json(p)['passed'] for p in OUT.glob('network-preflight-*.json')),'Network preflight required'
    if (OUT/'HALT.json').exists() or (OUT/'STOP').exists() or any(OUT.glob('*/STOP')):raise BudgetStopped('Stage stopped')
    directory=OUT/topic
    if (directory/'result.json').exists():raise BudgetStopped('Completed topic cannot rerun')
    if topic=='T1':assert read_json(OUT/'replay63-proof.json')['passed']
    if topic=='N1':assert (OUT/'T1/result.json').exists()
    if (directory/'started.json').exists() and not resume:raise BudgetStopped('Explicit audited resume required')
    save(directory/f'readme-record-{len(list(directory.glob("readme-record-*.json")))+1:02d}.json',{'sha256':sha(README),'record_only':True})
    if not (directory/'started.json').exists():save(directory/'started.json',{'topic':topic,'version':VERSION,'carry_budget':259170,'cap':CAP})
    endpoint,key=credentials();wire=AuditedDeepSeek(OUT,topic,key,CAP,resume=topic=='T1' or resume)
    wire.gate();model=live_model(key,wire,endpoint);wiki=Wikipedia(REFERENCE_DIR,filter_log=directory/'reference-filter.jsonl')
    try:
        emit(game_corpus_loading=True,topic=topic);agent,session=build(model,AgentTools(ROOT).tools(),wiki,wire,directory)
        question=read_json(OUT/'inputs'/f'{topic}.json')['question'];seen=set()
        trace=directory/f'trace-{len(list(directory.glob("trace-*.jsonl")))+1:02d}.jsonl'
        for state in agent.stream({'messages':[{'role':'user','content':question}]},config={'max_concurrency':1},stream_mode='values'):
            for m in state.get('messages',[]):
                if m.id in seen:continue
                seen.add(m.id)
                with trace.open('a',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(wire.safe(m.model_dump(mode='json')),ensure_ascii=False)+'\n')
                emit(topic=topic,message=m.type,characters=len(m.text))
        last=state['messages'][-1]
        if last.type!='ai' or last.tool_calls or not last.text.strip() or wire.replayed!=len(wire.replay):raise BudgetStopped('No complete final report')
        totals=wire.gate();answer=wire.safe(last.text);(directory/'report.md').write_text(answer+'\n',encoding='utf-8')
        save(directory/'result.json',{'status':'completed','topic':topic,'question':question,'answer':answer,
            'usage':accounted(OUT,topic),'stage_usage':totals,'approvals':session.approvals,
            'main_responses':wire.monitor.sequence,'request_channel':True,'direct_channel':False})
        emit(finished=topic,usage=totals)
    finally:
        save(directory/f'wiki-network-{len(list(directory.glob("wiki-network-*.json")))+1:02d}.json',wiki.events);wiki.close();model.http_client.close()


def preflight():
    with httpx.Client(trust_env=False,follow_redirects=False,timeout=20) as c:r=c.get('https://api.deepseek.com/models')
    value={'passed':r.status_code==401,'http_status':r.status_code,'credentials_sent':False,'tokens':0}
    save(OUT/f'network-preflight-{len(list(OUT.glob("network-preflight-*.json")))+1:02d}.json',value);emit(preflight=value)
    if not value['passed']:raise BudgetStopped('Preflight failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('smoke','offline','prepare','proof','verify','preflight','run'))
    p.add_argument('--topic',choices=TOPICS);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:
        if a.command in ('smoke','offline'):offline(a.command=='smoke')
        elif a.command=='run':execute(a.topic,a.resume)
        else:globals()[a.command]()
    except Exception as exc:
        if a.command=='run' and not (OUT/'HALT.json').exists():save(OUT/'HALT.json',{'topic':a.topic,'reason':str(exc),'type':type(exc).__name__,'usage':accounted(OUT)})
        emit(stopped=True,reason=str(exc),error_type=type(exc).__name__);sys.exit(2)
