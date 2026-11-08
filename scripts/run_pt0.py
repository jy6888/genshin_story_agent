"""PT0 stages 0/1 only. No stage-2 entry point and no direct-channel paid run."""
import argparse
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import sys
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
for name in ('LANGCHAIN_TRACING_V2', 'LANGSMITH_TRACING'):
    os.environ[name] = 'false'
from story_agent import agent_pt0 as s
from story_agent.budget import BudgetStopped
from story_agent.external_pt0 import Wikipedia, ExternalSession, DESCRIPTION
from story_agent.pt0_wire import AuditedDeepSeek, accounted
from story_agent.strong_agent_sa1 import read_json, save, sha, emit
from story_agent.strong_agent_sa2 import credentials, live_model
from story_agent.texts_tl4b import AgentTools

ROOT = Path(__file__).resolve().parents[1]
TASK = CODE/'docs/project/tasks/T-20261107-002'
BASE = TASK/'evidence/pt0-v0.92.0'
OFFLINE, OUT = BASE/'stage0', BASE/'stage1'
SPEC = CODE/'docs/design/codex/108-pt0-external-request.md'
EXPECTED, CAP = 80_000, 160_000
TOPICS = ('A1', 'A2', 'B')


def inputs():
    section = SPEC.read_text(encoding='utf-8').split('## 7.', 1)[1].split('## 8.', 1)[0]
    blocks = re.findall(r'```text\n(.*?)\n```', section, re.S)
    result = {}
    for topic, block in zip(TOPICS[:2], blocks[:2]):
        body = block.split('：', 1)[1]
        parts = re.split(r'；(?=(?:target|evidence|missing|cue|candidates)=)', body)
        application = dict(part.split('=', 1) for part in parts)
        application['candidates'] = application['candidates'].split('；')
        result[topic] = {'raw': block, 'application': application}
    result['B'] = {'question': blocks[2]}
    return result


def offline(smoke=False):
    class Tee(io.StringIO):
        def write(self, value):
            sys.stdout.write(value); sys.stdout.flush()
            return super().write(value)
    stream = Tee()
    name = 'tests.test_pt0'+('.PT0Test.test_program_rejections_and_no_reviewer' if smoke else '')
    result = unittest.TextTestRunner(stream=stream, verbosity=2, failfast=True).run(
        unittest.defaultTestLoader.loadTestsFromName(name))
    prefix = 'smoke' if smoke else 'offline-tests'
    save(OFFLINE/f'{prefix}-{len(list(OFFLINE.glob(prefix+"-*.json")))+1:02d}.json',
         {'passed': result.wasSuccessful(), 'tests': result.testsRun, 'paid_calls': 0, 'log': stream.getvalue()})
    if not result.wasSuccessful() or result.testsRun == 0:
        raise BudgetStopped('Offline failure; stop and inspect')


def prepare():
    if (OUT/'protocol.json').exists():
        raise BudgetStopped('Protocol already frozen')
    if not sys.flags.no_user_site or not (Path(sys.prefix)/'conda-meta').is_dir():
        raise BudgetStopped('Dedicated Conda Python -s required')
    if not read_json(max(OFFLINE.glob('offline-tests*.json'),key=lambda p:p.stat().st_mtime_ns))['passed']:
        raise BudgetStopped('Passing offline tests required')
    spec = SPEC.read_text(encoding='utf-8')
    assert all(x in spec for x in (s.APPENDIX, s.REVIEW_PROMPT, DESCRIPTION))
    for topic, data in inputs().items():
        path = OUT/'inputs'/f'{topic}.json'
        if path.exists():
            assert read_json(path) == data
        else:
            save(path, data)
    from tests.test_tl4b import fixture, capture
    from scripts.verify_tl3 import reply
    from story_agent import agent_tl4b_v912 as old
    wiki = Wikipedia(OFFLINE/'unused-fixture-refs')
    session = ExternalSession(wiki, lambda _: '驳回')
    try:
        before, _ = capture(old.build_main, fixture().tools(), [reply()])
        after, _ = capture(lambda m, t: s.build_main(m, t, session), fixture().tools(), [reply()])
        compare, _ = capture(old.build_compare, fixture().tools(), [reply()])
    finally:
        wiki.close()
    assert after[0]['messages'][0]['content'] == before[0]['messages'][0]['content']+s.APPENDIX
    parity_number=len(list(OFFLINE.glob('actual-wire-parity*.json')))+1
    save(OFFLINE/f'actual-wire-parity-{parity_number:02d}.json', {'passed': True, 'before': before[0],
        'after': after[0], 'compare': compare[0], 'only_main_prompt_delta': s.APPENDIX})
    # Read-only environment and fixed game-corpus inventory, no service startup.
    game = AgentTools(ROOT)
    files = [CODE/p for p in ('story_agent/agent_pt0.py', 'story_agent/external_pt0.py',
        'story_agent/pt0_wire.py', 'scripts/run_pt0.py', 'tests/test_pt0.py',
        'story_agent/agent_tl4b_v912.py', 'story_agent/agent_tl4b.py',
        'story_agent/texts_tl4b.py', 'story_agent/tl4b_wire_v914.py',
        'story_agent/tl4b_delegation_v911.py')]
    hashes = {str(p): sha(p) for p in files+[SPEC, ROOT/'docs/design/codex/108-pt0-external-request.md']}
    hashes.update(game.hashes)
    save(OUT/'protocol.json', {'version': s.VERSION, 'expected': EXPECTED, 'cap': CAP,
        'budget_rule': 'actual cache-miss input + output; stop only after exceeding',
        'hashes': hashes, 'inputs': inputs(), 'stage2_entrypoint': False,
        'main_prompt': s.SYSTEM_PROMPT, 'review_prompt': s.REVIEW_PROMPT,
        'reference_prompt': s.REFERENCE_PROMPT, 'max_retries': 0,
        'model': s.MODEL, 'thinking': {'type':'enabled'}, 'reasoning_effort':'max',
        'max_tokens':49152, 'temperature_sent':False,
        'python': sys.executable, 'no_user_site': sys.flags.no_user_site,
        'dependencies': {n: importlib.metadata.version(n) for n in
                         ('httpx','deepagents','langchain','langchain-deepseek')},
        'new_dependencies': False, 'game_source_files':len(game.hashes),
        'reference_directory':str(CODE/'data/references/pt0-stage1'),
        'n1_draft_or_refs_sent':False})
    verify()


def verify():
    emit(verification_started=True)
    protocol = read_json(OUT/'protocol.json')
    for path, digest in protocol['hashes'].items():
        if sha(path) != digest:
            raise BudgetStopped('Frozen file changed: '+path)
    if protocol['inputs'] != inputs():
        raise BudgetStopped('Authorized inputs changed')
    emit(frozen_inputs_verified=True, hashes=len(protocol['hashes']), stage2_entrypoint=False)


def preflight():
    import httpx
    try:
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=20) as client:
            response = client.get('https://api.deepseek.com/models')
        value = {'passed': response.status_code == 401, 'http_status': response.status_code,
                 'credentials_sent':False, 'tokens':0}
    except httpx.HTTPError as exc:
        value = {'passed':False, 'error_type':type(exc).__name__, 'credentials_sent':False, 'tokens':0}
    number=len(list(OUT.glob('network-preflight*.json')))+1
    save(OUT/f'network-preflight-{number:02d}.json', value)
    emit(preflight=value)
    if not value['passed']:
        raise BudgetStopped('Network preflight failed')


def execute(topic, resume=False):
    if (OUT/'user-stop.json').exists():
        raise BudgetStopped('User stopped paid execution; preserve pending request, no resend')
    verify()
    if not any(read_json(p)['passed'] for p in OUT.glob('network-preflight*.json')):
        raise BudgetStopped('Credential-free preflight required')
    if (OUT/'HALT.json').exists():
        raise BudgetStopped('Stage halted; inspect evidence')
    for prior in TOPICS[:TOPICS.index(topic)]:
        if not (OUT/prior/'result.json').exists():
            raise BudgetStopped('Prior micro unit incomplete')
    directory = OUT/topic
    if (directory/'result.json').exists():
        raise BudgetStopped('Finished unit must never be rerun')
    if (directory/'started.json').exists() and not resume:
        raise BudgetStopped('Prior attempt; audited resume required')
    if not (directory/'started.json').exists():
        save(directory/'started.json', {'topic': topic, 'expected':EXPECTED, 'cap':CAP})
    endpoint, key = credentials()
    wire = AuditedDeepSeek(OUT, topic, key, CAP, resume=resume)
    model = live_model(key, wire, endpoint)
    wiki = None
    try:
        data = read_json(OUT/'inputs'/f'{topic}.json')
        if topic in ('A1', 'A2'):
            emit(game_corpus_loading=True,topic=topic)
            agent = s.build_reviewer(model, AgentTools(ROOT).tools())
            content = json.dumps(data['application'], ensure_ascii=False)
        else:
            wiki = Wikipedia(CODE/'data/references/pt0-stage1')
            session = ExternalSession(wiki, None, directory/'external-events.jsonl')
            session.grant_direct(read_json(OUT/'inputs/A1.json')['application'])
            agent = s.build_reference(model, session)
            content = data['question']
        seen, answer = set(), ''
        trace = directory/f'trace-{len(list(directory.glob("trace-*.jsonl")))+1:02d}.jsonl'
        for state in agent.stream({'messages':[{'role':'user','content':content}]},
                config={'max_concurrency':1}, stream_mode='values'):
            for message in state.get('messages', []):
                if message.id in seen: continue
                seen.add(message.id)
                with trace.open('a', encoding='utf-8', newline='\n') as stream:
                    stream.write(json.dumps(wire.safe(message.model_dump(mode='json')), ensure_ascii=False)+'\n')
                emit(message=message.type, characters=len(message.text), tool_calls=len(getattr(message,'tool_calls',[])))
            last = state['messages'][-1]
            if last.type == 'ai' and not last.tool_calls:
                answer = last.text
        if not answer.strip() or wire.replayed != len(wire.replay):
            raise BudgetStopped('No final answer or replay mismatch')
        totals = wire.gate()
        (directory/'report.md').write_text(wire.safe(answer)+'\n', encoding='utf-8')
        save(directory/'result.json', {'status':'completed', 'answer':answer, 'usage':totals,
            'stage2_started':False, 'topic':topic})
        emit(finished=topic, usage=totals, stage2_started=False)
    finally:
        if wiki:
            save(directory/'wiki-network.json', wiki.events)
            wiki.close()
        model.http_client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke','offline','prepare','verify','preflight','run'))
    parser.add_argument('--topic', choices=TOPICS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        if args.command in ('smoke','offline'):
            if args.command == 'offline' and not any(read_json(p)['passed'] for p in OFFLINE.glob('smoke*.json')):
                raise BudgetStopped('Minimal smoke required')
            offline(args.command == 'smoke')
        elif args.command == 'run':
            if not args.topic: parser.error('--topic required')
            execute(args.topic, args.resume)
        else:
            globals()[args.command]()
    except Exception as exc:
        reason = str(exc) if isinstance(exc, BudgetStopped) else type(exc).__name__
        if args.command == 'run':
            save(OUT/'HALT.json', {'topic':args.topic, 'reason':reason,
                 'error_type':type(exc).__name__, 'usage':accounted(OUT)})
        emit(stopped=True, reason=reason, error_type=type(exc).__name__)
        return 2
    return 0


if __name__ == '__main__': sys.exit(main())
