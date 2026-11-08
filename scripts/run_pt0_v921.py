"""PT0 §11 v0.92.1: fresh A1/A2/B only, including old 60,262 estimate."""
import argparse
import importlib.metadata
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
from scripts import run_pt0 as old
from scripts import settle_pt0_d007 as settlement
from story_agent import agent_pt0 as s
from story_agent.external_pt0 import ExternalSession
from story_agent.external_pt0_v921 import Wikipedia,filtered_view
from story_agent.pt0_wire_v921 import AuditedDeepSeek,VERSION,accounted,LEGACY_ID
from story_agent.budget import BudgetStopped
from story_agent.strong_agent_sa1 import read_json,save,sha,emit
from story_agent.strong_agent_sa2 import credentials,live_model
from story_agent.texts_tl4b import AgentTools

ROOT,TASK=old.ROOT,old.TASK
BASE=TASK/'evidence/pt0-v0.92.1'
OUT,OFFLINE=BASE/'stage1',BASE/'stage0'
SPEC=ROOT/'docs/design/codex/108-pt0-external-request.md'
README=ROOT/'docs/design/codex/README.md'
EXPECTED,CAP,TOPICS=80000,160000,('A1','A2','B')
REFERENCE_DIR=CODE/'data/references/pt0-v0.92.1-stage1'


def inputs():
    text=SPEC.read_text(encoding='utf-8')
    assert 'D-20261107-007' in text and '5 份文本快照' in text and '不再重发' in text
    section=text.split('## 7.',1)[1].split('## 8.',1)[0]
    blocks=re.findall(r'```text\n(.*?)\n```',section,re.S)
    result={}
    for topic,block in zip(TOPICS[:2],blocks[:2]):
        parts=re.split(r'；(?=(?:target|evidence|missing|cue|candidates)=)',block.split('：',1)[1])
        app=dict(part.split('=',1) for part in parts)
        app['candidates']=app['candidates'].split('；')
        result[topic]={'raw':block,'application':app}
    result['B']={'question':blocks[2]}
    for t in TOPICS:assert result[t]==read_json(old.OUT/'inputs'/f'{t}.json')
    return result


def offline(smoke=False):
    class Tee(io.StringIO):
        def write(self,value):
            sys.stdout.write(value);sys.stdout.flush();return super().write(value)
    stream=Tee()
    names=['tests.test_pt0_v921.FilterTest.test_buer_line6_minimal_smoke'] if smoke else [
        'tests.test_pt0','tests.test_pt0_v921','tests.test_pt0_v921_wire']
    result=unittest.TextTestRunner(stream=stream,verbosity=2,failfast=True).run(
        unittest.defaultTestLoader.loadTestsFromNames(names))
    prefix='smoke' if smoke else 'offline-tests'
    save(OFFLINE/f'{prefix}-{len(list(OFFLINE.glob(prefix+"-*.json")))+1:02d}.json',
        {'passed':result.wasSuccessful(),'tests':result.testsRun,'provider_calls':0,'log':stream.getvalue()})
    if not result.wasSuccessful() or not result.testsRun:raise BudgetStopped('Offline failure; inspect first')


def prepare():
    if (OUT/'protocol.json').exists():raise BudgetStopped('Protocol already frozen')
    assert sys.flags.no_user_site and (Path(sys.prefix)/'conda-meta').is_dir()
    assert read_json(sorted(OFFLINE.glob('offline-tests-*.json'))[-1])['passed']
    assert (old.OUT/'user-stop.json').exists()
    # Preserve all old files bytewise; never append to the interrupted directory.
    original={str(p):sha(p) for p in old.BASE.rglob('*') if p.is_file()}
    report=CODE/'docs/reports/pt0-external-request.md'
    snapshot=OFFLINE/'v0.92.0-report.md'
    with snapshot.open('xb') as stream:stream.write(report.read_bytes())
    save(OFFLINE/'prior-preservation.json',{'hashes':original,'old_report':str(report),
        'snapshot':str(snapshot),'report_sha256':sha(snapshot),'old_request_will_not_be_resent':True})
    regression=[]
    for txt in sorted((CODE/'docs/reports/pt0/refs').rglob('*.txt')):
        info=read_json(txt.with_suffix('.json'))
        assert sha(txt)==info['sha256']
        view=filtered_view(info['title'],txt.read_text(encoding='utf-8'))
        regression.append({'source':str(txt),'source_sha256':sha(txt),'metadata':info,**view})
    assert len(regression)==5 and len({(r['metadata']['lang'],r['metadata']['pageid']) for r in regression})==4
    buer=next(r for r in regression if r['metadata']['title']=='布耶爾')
    assert 6 in buer['deleted_lines'] and buer['lines'][5]==''
    save(OFFLINE/'five-snapshot-regression.json',{'passed':True,'snapshot_count':5,'unique_pages':4,
        'provider_calls':0,'buer_original_line6_deleted':True,'pages':regression})
    for t,data in inputs().items():save(OUT/'inputs'/f'{t}.json',data)
    from tests.test_tl4b import fixture,capture
    from scripts.verify_tl3 import reply
    wiki=Wikipedia(OFFLINE/'unused-wire-fixture')
    try:
        review,_=capture(s.build_reviewer,fixture().tools(),[reply()])
        reference,_=capture(lambda m,t:s.build_reference(m,ExternalSession(wiki,None)),fixture().tools(),[reply()])
    finally:wiki.close()
    prior_request=read_json(old.OUT/'A1/calls/001/request.json')
    assert review[0]['tools']==prior_request['tools']
    assert review[0]['messages'][0]==prior_request['messages'][0]
    save(OFFLINE/'wire-contract.json',{'review_system':s.REVIEW_PROMPT,'reference_system':s.REFERENCE_PROMPT,
        'review_tools':review[0]['tools'],'reference_tools':reference[0]['tools'],
        'reviewer_prompt_tools_unchanged':True,'only_six_application_parameters':True})
    # Reuse fixed game-file inventory, refresh execution code hashes after old
    # post-interruption fixes. Never load refs, gold, or reports into game tools.
    old_protocol=read_json(old.OUT/'protocol.json')
    hashes={p:h for p,h in old_protocol['hashes'].items() if not Path(p).is_relative_to(CODE)
            and Path(p)!=SPEC}
    paths=list((CODE/'story_agent').glob('*.py'))+[CODE/p for p in (
        'scripts/run_pt0.py','scripts/run_pt0_v921.py','scripts/settle_pt0_d007.py',
        'tests/test_pt0.py','tests/test_pt0_v921.py','tests/test_pt0_v921_wire.py')]
    paths += [SPEC,README,snapshot,OFFLINE/'prior-preservation.json',OFFLINE/'wire-contract.json',
              OFFLINE/'five-snapshot-regression.json',sorted(OFFLINE.glob('offline-tests-*.json'))[-1]]
    paths += list((OUT/'inputs').glob('*.json'))
    hashes.update({str(p):sha(p) for p in paths})
    save(OUT/'recovery-authority.json',{'authority':settlement.AUTHORITY,'old_user_stop_retained':True,
        'old_user_stop_sha256':sha(old.OUT/'user-stop.json'),'no_resend':True,'stage2_authorized':False})
    save(OUT/'protocol.json',{'version':VERSION,'expected_new_tokens':EXPECTED,'cap':CAP,
        'legacy_estimate_included':60262,'new_budget_available_initially':99738,'hashes':hashes,
        'inputs':inputs(),'stage2_entrypoint':False,'new_reference_directory':str(REFERENCE_DIR),
        'refs_or_gold_in_model_inputs':False,'wiki_transport':'existing curl via httpx',
        'filter_version':VERSION,'raw_snapshots_immutable':True,'deleted_sentences_not_sent_to_agent':True,
        'model':s.MODEL,'thinking':{'type':'enabled'},'reasoning_effort':'max','max_tokens':49152,
        'temperature_sent':False,'max_retries':0,'request_timeout_seconds':900,
        'manual_interrupt_in_flight_forbidden':True,'budget_rule':'actual settled cache miss+output plus authorized estimates; stop only >160000',
        'dependencies':{n:importlib.metadata.version(n) for n in ('httpx','deepagents','langchain','langchain-deepseek')}})
    verify()


def verify():
    emit(verification_started=True)
    p=read_json(OUT/'protocol.json')
    for path,digest in p['hashes'].items():
        if sha(path)!=digest:raise BudgetStopped('Frozen file changed: '+path)
    for path,digest in read_json(OFFLINE/'prior-preservation.json')['hashes'].items():
        if sha(path)!=digest:raise BudgetStopped('Old interrupted evidence changed: '+path)
    assert p['inputs']==inputs()
    assert read_json(OUT/'recovery-authority.json')['old_user_stop_sha256']==sha(old.OUT/'user-stop.json')
    emit(frozen_inputs_verified=True,hashes=len(p['hashes']),stage2_entrypoint=False)


def preflight():
    import httpx
    try:
        with httpx.Client(trust_env=False,follow_redirects=False,timeout=20) as client:
            response=client.get('https://api.deepseek.com/models')
        result={'passed':response.status_code==401,'http_status':response.status_code,'credentials_sent':False,'tokens':0}
    except httpx.HTTPError as exc:result={'passed':False,'error_type':type(exc).__name__,'credentials_sent':False,'tokens':0}
    save(OUT/f'network-preflight-{len(list(OUT.glob("network-preflight-*.json")))+1:02d}.json',result)
    emit(preflight=result)
    if not result['passed']:raise BudgetStopped('Preflight failed')


def execute(topic,resume=False):
    if (OUT/'STOP').exists() or (OUT/'HALT.json').exists():raise BudgetStopped('Stage stopped')
    if not (OUT/'recovery-authority.json').exists():raise BudgetStopped('Explicit §11 recovery protocol required')
    verify()
    assert read_json(OUT/'legacy-estimate-settlement.json')['estimate_counted']==60262
    assert any(read_json(p)['passed'] for p in OUT.glob('network-preflight-*.json'))
    for t in TOPICS[:TOPICS.index(topic)]:
        if not (OUT/t/'result.json').exists():raise BudgetStopped('Prior micro unit incomplete')
    directory=OUT/topic
    if (directory/'result.json').exists():raise BudgetStopped('Finished unit must never rerun')
    if (directory/'started.json').exists() and not resume:raise BudgetStopped('Audited resume required')
    if not (directory/'started.json').exists():save(directory/'started.json',{'topic':topic,'expected_new':EXPECTED,'cap':CAP,'carry_in_estimate':60262,'fresh_unit_not_old_request_retry':True})
    endpoint,key=credentials()
    wire=AuditedDeepSeek(OUT,topic,key,CAP,resume=resume)
    wire.gate()
    model=live_model(key,wire,endpoint);wiki=None
    try:
        data=read_json(OUT/'inputs'/f'{topic}.json')
        if topic in ('A1','A2'):
            emit(game_corpus_loading=True,topic=topic)
            agent=s.build_reviewer(model,AgentTools(ROOT).tools())
            content=json.dumps(data['application'],ensure_ascii=False)
        else:
            wiki=Wikipedia(REFERENCE_DIR,filter_log=directory/'reference-filter.jsonl')
            session=ExternalSession(wiki,None,directory/'external-events.jsonl')
            session.grant_direct(read_json(OUT/'inputs/A1.json')['application'])
            agent=s.build_reference(model,session);content=data['question']
        seen=set();answer=''
        trace=directory/f'trace-{len(list(directory.glob("trace-*.jsonl")))+1:02d}.jsonl'
        for state in agent.stream({'messages':[{'role':'user','content':content}]},config={'max_concurrency':1},stream_mode='values'):
            for message in state.get('messages',[]):
                if message.id in seen:continue
                seen.add(message.id)
                with trace.open('a',encoding='utf-8',newline='\n') as stream:
                    stream.write(json.dumps(wire.safe(message.model_dump(mode='json')),ensure_ascii=False)+'\n')
                emit(topic=topic,message=message.type,characters=len(message.text))
            last=state['messages'][-1]
            if last.type=='ai' and not last.tool_calls:answer=last.text
        if not answer.strip() or wire.replayed!=len(wire.replay):raise BudgetStopped('No final answer or replay mismatch')
        totals=wire.gate()
        (directory/'report.md').write_text(wire.safe(answer)+'\n',encoding='utf-8')
        save(directory/'result.json',{'status':'completed','topic':topic,'answer':answer,
            'usage':accounted(OUT,topic),'stage_usage':totals,'stage2_started':False})
        emit(finished=topic,usage=totals,stage2_started=False)
    finally:
        if wiki:save(directory/f'wiki-network-{len(list(directory.glob("wiki-network-*.json")))+1:02d}.json',wiki.events);wiki.close()
        model.http_client.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('smoke','offline','prepare','verify','preflight','run'))
    p.add_argument('--topic',choices=TOPICS);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:
        if a.command=='smoke':offline(True)
        elif a.command=='offline':offline()
        elif a.command=='run':execute(a.topic,a.resume)
        else:globals()[a.command]()
    except Exception as exc:
        if a.command=='run' and not (OUT/'HALT.json').exists():save(OUT/'HALT.json',{'topic':a.topic,'reason':str(exc),'error_type':type(exc).__name__,'usage':accounted(OUT)})
        emit(stopped=True,reason=str(exc),error_type=type(exc).__name__);sys.exit(2)
