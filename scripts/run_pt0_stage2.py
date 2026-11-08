"""PT0 v0.92.2 / §12: T1->N1, request channel only, actual-only 1.4M cap."""
import argparse
import io
import json
import os
from pathlib import Path
import re
import sys
import unittest

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
for name in ('LANGCHAIN_TRACING_V2','LANGSMITH_TRACING'):os.environ[name]='false'
from scripts import run_pt0_v921 as stage1
from story_agent import agent_pt0 as s
from story_agent.external_pt0 import ExternalSession as OriginalSession
from story_agent.external_pt0_v921 import Wikipedia
from story_agent.pt0_wire import AuditedDeepSeek as OriginalWire,accounted
from story_agent.agent_tl4 import Monitor as OriginalMonitor
from story_agent.budget import BudgetStopped
from story_agent.strong_agent_sa1 import read_json,save,sha,emit
from story_agent.strong_agent_sa2 import credentials,live_model
from story_agent.texts_tl4b import AgentTools

VERSION='v0.92.2'
PREFIX='PT0:'+VERSION+':'
ROOT,TASK,SPEC,README=stage1.ROOT,stage1.TASK,stage1.SPEC,stage1.README
BASE=TASK/'evidence/pt0-v0.92.2'
OUT,OFFLINE=BASE/'stage2',BASE/'stage0'
TOPICS=('T1','N1')
EXPECTED,CAP=700000,1400000
REFERENCE_DIR=CODE/'data/references/pt0-v0.92.2-stage2'
GOLDS={
    'N1':(ROOT/'data/evals/pt0_n1_gold_v1.json','e3c23c399e8e607801408eb5cb608f3fec2f6b90113cb7ddd7c555d8176251b3'),
    'T1':(ROOT/'data/evals/nq4_gold_v1.json','428e3dee3f9cbec87b43f107c4c24cb448d2619b4fbbb02a10a8ca3a459497b3')}


def release():
    spec=SPEC.read_text(encoding='utf-8')
    row=next(v for v in README.read_text(encoding='utf-8').splitlines() if v.startswith('| T-20261107-002 '))
    if '可开始阶段 2' not in row or 'D-20261107-008' not in spec or '## 12.' not in spec:
        raise BudgetStopped('§12 and README release required')
    for path,digest in GOLDS.values():
        if sha(path)!=digest:raise BudgetStopped('Frozen gold SHA256 mismatch (hash-only check)')
    section=spec.split('## 8. 阶段 2：两题',1)[1].split('## 9.',1)[0]
    rows=re.search(r'```text\n(.*?)\n```',section,re.S)[1].splitlines()
    assert len(rows)==2 and all(line.startswith(t+' ') for t,line in zip(TOPICS,rows))
    questions={t:line[len(t)+1:] for t,line in zip(TOPICS,rows)}
    assert questions=={'T1':'帮我重现海坡玻瑞亚的完整故事','N1':'纳西妲的魔神名为什么是布耶尔？'}
    return questions


def scope(payload):
    system=payload['messages'][0]['content']
    if system==s.REVIEW_PROMPT:return 'reviewer'
    if system==s.previous.COMPARE_PROMPT:return 'compare'
    if any(t['function']['name']=='task' for t in payload['tools']):return 'main'
    raise BudgetStopped('Unexpected prompt/scope')


class Monitor(OriginalMonitor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.scope='main';self.records=[]

    def check_stop(self):
        if (self.directory.parent/'STOP').exists():raise BudgetStopped('Stage STOP between requests')
        super().check_stop()

    def observe(self,message,call_number,child=False,replay=False):
        self.check_stop()
        if self.scope=='main':self.sequence+=1
        value={'call':call_number,'scope':self.scope,'message':message}
        self.records.append(value)
        if not replay:
            path=self.directory/'trace-wire.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a',encoding='utf-8',newline='\n') as stream:
                stream.write(json.dumps(value,ensure_ascii=False)+'\n')


class AuditedDeepSeek(OriginalWire):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.run_id=PREFIX+self.out.name+':'+self.topic
        self.monitor=Monitor(self.out/self.topic,self.topic,self.sleeper)

    def gate(self):
        self.monitor.check_stop()
        summary=self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts'])-{'settled'}:
            raise BudgetStopped('Shared ledger must be accounting-only and settled')
        totals=accounted(self.out)
        with self.ledger.connect() as db:
            rows=[r for t in TOPICS for r in db.execute('SELECT status,actual FROM calls WHERE run_id=?',
                (PREFIX+self.out.name+':'+t,)).fetchall()]
        if len(rows)!=totals['attempts'] or sum(r[1] or 0 for r in rows)!=totals['accounted_raw_tokens']:
            raise BudgetStopped('Stage-2 ledger and evidence disagree')
        if totals['budget_tokens']>self.cap:raise BudgetStopped('Uncached input plus output cap exceeded')
        return totals

    def _request(self,request):
        self.monitor.scope=scope(json.loads(request.content))
        return super()._request(request)


class LoggedGuard(s.previous.DelegationGuard):
    def __init__(self,directory):
        super().__init__();self.path=Path(directory)/'delegations.jsonl'
        self.existing={v['tool_call_id']:v for v in map(json.loads,self.path.read_text(encoding='utf-8').splitlines())} if self.path.exists() else {}

    def refusal(self,request):
        before=len(self.events);result=super().refusal(request)
        if len(self.events)>before:
            value=self.events[-1]
            if value['tool_call_id'] in self.existing:assert self.existing[value['tool_call_id']]==value
            else:
                self.path.parent.mkdir(parents=True,exist_ok=True)
                with self.path.open('a',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')
                self.existing[value['tool_call_id']]=value
        return result


class ExternalSession(OriginalSession):
    """Rebuild grants during byte replay without re-fetching prior tool results."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        path=Path(self.event_path) if self.event_path else None
        self.existing=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path and path.exists() else []
        self.event_index=0

    def log(self,event):
        if self.event_index<len(self.existing):
            assert self.existing[self.event_index]==event,'External replay event differs'
            self.events.append(event)
        else:super().log(event)
        self.event_index+=1

    def historical(self,kind,values):
        if self.event_index>=len(self.existing):return None
        event=self.existing[self.event_index]
        assert event['type']==kind and all(event[k]==v for k,v in values.items()),'External replay parameters differ'
        assert self.active is not None
        return event

    def search(self,query,lang='zh'):
        with self.lock:
            event=self.historical('search',{'query':query,'lang':lang})
            if event is None:return super().search(query,lang)
            self.log(event);return event['result']

    def read(self,title,lang='zh',start_line=1,end_line=120):
        with self.lock:
            event=self.historical('read',{'title':title,'lang':lang,'requested_lines':[start_line,end_line]})
            if event is None:return super().read(title,lang,start_line,end_line)
            result=event['result']
            if 'text' in result:assert self.page_allowed((lang,result['pageid']))
            self.log(event);return result


def build(model,tools,wiki,wire,directory):
    """Unchanged prompts; same wire meters parent, compare and review together."""
    reviewer=s.build_reviewer(model,tools);review_number=0
    def review(application):
        nonlocal review_number
        review_number+=1;dest=Path(directory)/'reviews'/f'{review_number:03d}'
        if (dest/'input.json').exists():assert read_json(dest/'input.json')==application
        else:save(dest/'input.json',application)
        begin=len(wire.monitor.records)
        answer=s.review(reviewer,application)
        record={'answer':answer,'calls':[r['call'] for r in wire.monitor.records[begin:] if r['scope']=='reviewer']}
        if (dest/'answer.json').exists():assert read_json(dest/'answer.json')==record
        else:save(dest/'answer.json',record)
        return answer
    session=ExternalSession(wiki,review,Path(directory)/'external-events.jsonl')
    agent=s.build_main(model,tools,session,LoggedGuard(directory),external_enabled=True)
    return agent,session


def offline(smoke=False):
    class Tee(io.StringIO):
        def write(self,value):sys.stdout.write(value);sys.stdout.flush();return super().write(value)
    stream=Tee()
    name='tests.test_pt0_stage2'+('.Stage2Test.test_release_hash_only_and_exact_questions' if smoke else '')
    result=unittest.TextTestRunner(stream=stream,verbosity=2,failfast=True).run(unittest.defaultTestLoader.loadTestsFromName(name))
    prefix='smoke' if smoke else 'offline-tests'
    save(OFFLINE/f'{prefix}-{len(list(OFFLINE.glob(prefix+"-*.json")))+1:02d}.json',
        {'passed':result.wasSuccessful(),'tests':result.testsRun,'provider_requests':0,'log':stream.getvalue()})
    if not result.wasSuccessful() or not result.testsRun:raise BudgetStopped('Offline failure; inspect first')


def prepare():
    questions=release()
    assert sys.flags.no_user_site and (Path(sys.prefix)/'conda-meta').is_dir()
    if (OUT/'protocol.json').exists():raise BudgetStopped('Protocol already frozen')
    assert read_json(sorted(OFFLINE.glob('offline-tests-*.json'))[-1])['passed']
    if REFERENCE_DIR.exists() and any(REFERENCE_DIR.iterdir()):raise BudgetStopped('Stage-2 snapshots must start empty')
    REFERENCE_DIR.mkdir(parents=True,exist_ok=True)
    prior=read_json(CODE/'data/manifests/pt0-v0.92.1-stage1.json')
    import hashlib
    for path,digest in prior['delivery_hashes'].items():
        assert hashlib.sha256((CODE/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,path
    for path,digest in prior['local_evidence_hashes'].items():assert sha(path)==digest,path
    snapshot=OFFLINE/'stage1-report.md'
    with snapshot.open('xb') as stream:stream.write((CODE/'docs/reports/pt0-external-request.md').read_bytes())
    save(OFFLINE/'prior-preservation.json',{'prior_manifest':prior,'report_snapshot':str(snapshot),
        'report_snapshot_sha256':sha(snapshot),'stage1_budget_excluded':122710})
    for t,q in questions.items():save(OUT/'inputs'/f'{t}.json',{'question':q})
    # Capture all three real SDK prompt/schema shapes without any provider call.
    from tests.test_tl4b import fixture,capture
    from scripts.verify_tl3 import reply
    wiki=Wikipedia(OFFLINE/'unused-fixture')
    try:
        session=ExternalSession(wiki,lambda _:'放行\n离线')
        main,_=capture(lambda m,t:s.build_main(m,t,session),fixture().tools(),[reply()])
        review,_=capture(s.build_reviewer,fixture().tools(),[reply()])
        compare,_=capture(s.previous.build_compare,fixture().tools(),[reply()])
        session.active={'offline':True}
        approved,_=capture(lambda m,t:s.build_main(m,t,session),fixture().tools(),[reply()])
    finally:wiki.close()
    contract={'main_system':main[0]['messages'][0]['content'],'main_tools':main[0]['tools'],
        'approved_tools':approved[0]['tools'],'reviewer_system':review[0]['messages'][0]['content'],
        'reviewer_tools':review[0]['tools'],'compare_system':compare[0]['messages'][0]['content'],
        'compare_tools':compare[0]['tools'],'no_prompt_or_schema_tuning':True}
    save(OFFLINE/'wire-contract.json',contract)
    previous=read_json(stage1.OUT/'protocol.json')
    hashes={p:h for p,h in previous['hashes'].items() if p not in (str(SPEC),str(README))}
    paths=[CODE/'scripts/run_pt0_stage2.py',CODE/'tests/test_pt0_stage2.py',SPEC,README,snapshot,
        OFFLINE/'prior-preservation.json',OFFLINE/'wire-contract.json',sorted(OFFLINE.glob('offline-tests-*.json'))[-1]]
    paths+=list((OUT/'inputs').glob('*.json'))
    hashes.update({str(p):sha(p) for p in paths})
    save(OUT/'protocol.json',{'version':VERSION,'hashes':hashes,'questions':questions,'expected':EXPECTED,'cap':CAP,
        'gold_hashes_only':{t:{'path':str(p),'sha256':h} for t,(p,h) in GOLDS.items()},
        'gold_parsed_by_runner':False,'old_reports_or_refs_in_inputs':False,'stage1_budget_excluded':122710,
        'reference_directory':str(REFERENCE_DIR),'request_channel':True,'direct_channel':False,
        'wiki_filter_version':'v0.92.1','wiki_transport':'existing curl','no_manual_interrupt_in_flight':True,
        'model':s.MODEL,'max_tokens':49152,'thinking':{'type':'enabled'},'reasoning_effort':'max',
        'max_retries':0,'request_timeout_seconds':900,'temperature_sent':False,'dependencies':previous['dependencies']})
    verify()


def verify():
    emit(verification_started=True)
    p=read_json(OUT/'protocol.json')
    assert p['questions']==release()
    for path,digest in p['hashes'].items():
        if sha(path)!=digest:raise BudgetStopped('Frozen file changed: '+path)
    emit(frozen_inputs_verified=True,hashes=len(p['hashes']),gold_hashes_checked_not_parsed=True)


def sequence_gate(topic,resume=False):
    if (OUT/'HALT.json').exists() or (OUT/'STOP').exists() or any(OUT.glob('*/STOP')):
        raise BudgetStopped('Stage stopped; no new unit')
    directory=OUT/topic
    if (directory/'result.json').exists():raise BudgetStopped('Completed topic must never rerun')
    if (directory/'started.json').exists() and not resume:raise BudgetStopped('Audited resume required')
    for t in TOPICS[:TOPICS.index(topic)]:
        if not (OUT/t/'result.json').exists():raise BudgetStopped('Prior topic incomplete: '+t)


def execute(topic,resume=False):
    verify();sequence_gate(topic,resume)
    assert any(read_json(p)['passed'] for p in OUT.glob('network-preflight-*.json'))
    directory=OUT/topic
    if not (directory/'started.json').exists():save(directory/'started.json',{'topic':topic,'expected':EXPECTED,'cap':CAP,'request_channel':True,'direct_channel':False})
    endpoint,key=credentials();wire=AuditedDeepSeek(OUT,topic,key,CAP,resume=resume)
    wire.gate();model=live_model(key,wire,endpoint)
    wiki=Wikipedia(REFERENCE_DIR,filter_log=directory/'reference-filter.jsonl')
    try:
        emit(game_corpus_loading=True,topic=topic)
        agent,session=build(model,AgentTools(ROOT).tools(),wiki,wire,directory)
        question=read_json(OUT/'inputs'/f'{topic}.json')['question']
        trace=directory/f'trace-{len(list(directory.glob("trace-*.jsonl")))+1:02d}.jsonl'
        seen=set()
        for state in agent.stream({'messages':[{'role':'user','content':question}]},config={'max_concurrency':1},stream_mode='values'):
            for message in state.get('messages',[]):
                if message.id in seen:continue
                seen.add(message.id)
                with trace.open('a',encoding='utf-8',newline='\n') as stream:
                    stream.write(json.dumps(wire.safe(message.model_dump(mode='json')),ensure_ascii=False)+'\n')
                emit(topic=topic,message=message.type,characters=len(message.text))
        last=state['messages'][-1]
        if last.type!='ai' or last.tool_calls or not last.text.strip() or wire.replayed!=len(wire.replay):
            raise BudgetStopped('No final report or incomplete replay')
        totals=wire.gate();answer=wire.safe(last.text)
        (directory/'report.md').write_text(answer+'\n',encoding='utf-8')
        save(directory/'result.json',{'status':'completed','topic':topic,'question':question,'answer':answer,
            'usage':accounted(OUT,topic),'stage_usage':totals,'main_responses':wire.monitor.sequence,
            'request_channel':True,'direct_channel':False,'approvals':session.approvals})
        emit(finished=topic,usage=totals)
    finally:
        save(directory/f'wiki-network-{len(list(directory.glob("wiki-network-*.json")))+1:02d}.json',wiki.events)
        wiki.close();model.http_client.close()


def preflight():
    import httpx
    try:
        with httpx.Client(trust_env=False,follow_redirects=False,timeout=20) as client:response=client.get('https://api.deepseek.com/models')
        value={'passed':response.status_code==401,'http_status':response.status_code,'credentials_sent':False,'tokens':0}
    except httpx.HTTPError as exc:value={'passed':False,'error_type':type(exc).__name__,'credentials_sent':False,'tokens':0}
    save(OUT/f'network-preflight-{len(list(OUT.glob("network-preflight-*.json")))+1:02d}.json',value);emit(preflight=value)
    if not value['passed']:raise BudgetStopped('Preflight failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('smoke','offline','prepare','verify','preflight','run'))
    p.add_argument('--topic',choices=TOPICS);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:
        if a.command=='smoke':offline(True)
        elif a.command=='offline':offline()
        elif a.command=='run':execute(a.topic,a.resume)
        else:globals()[a.command]()
    except Exception as exc:
        if a.command=='run' and not (OUT/'HALT.json').exists():save(OUT/'HALT.json',{'topic':a.topic,'reason':str(exc),'error_type':type(exc).__name__,'usage':accounted(OUT)})
        emit(stopped=True,reason=str(exc),error_type=type(exc).__name__);sys.exit(2)
