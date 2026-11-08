"""TL3 stage 1 only: audited direct-compare smoke, never a three-topic rerun."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
for name in ('LANGCHAIN_TRACING_V2', 'LANGSMITH_TRACING'):
    os.environ[name] = 'false'

from story_agent import agent_tl3 as s
from story_agent.budget import BudgetStopped
from story_agent.similar_passages_tl2a_v0862 import AgentTools
from story_agent.strong_agent_sa1 import emit, read_json, save, sha
from story_agent.strong_agent_sa2 import credentials, live_model

ROOT = Path(__file__).resolve().parents[1]
TASK = CODE/'docs/project/tasks/T-20261106-007'
OUT = TASK/'evidence/tl3-v0.88.0/stage1'
SPEC = CODE/'docs/design/codex/104-tl3-review-rerun.md'
SA2 = ROOT/'docs/project/tasks/T-20261106-001/evidence/sa2-v0.82.2'
TL2B = ROOT/'.runtime/worktrees/v49-tl2b-compare-rerun/docs/project/tasks/T-20261106-006/evidence/tl2b-v0.87.0/stage2'
TOPICS = ('NQ1A', 'NQ1B', 'NQ2A')
SMOKES = ('C1', 'C2')
CAP = 20_000


def question(index):
    return read_json(OUT/'inputs'/f'C{index+1}.json')['question']


def prepare():
    if not sys.flags.no_user_site or not (Path(sys.prefix)/'conda-meta').is_dir():
        raise BudgetStopped('Dedicated Conda Python -s required')
    tools = AgentTools(ROOT)
    spec = SPEC.read_text(encoding='utf-8')
    if s.APPENDIX not in spec:
        raise BudgetStopped('Main-system appendix differs from codex/104')
    # Extract only the two expressly authorized question fences, never review notes.
    for i in (1, 2):
        section = spec.split(f'**第 {i} 题**', 1)[1]
        value = section.split('```text\n', 1)[1].split('\n```', 1)[0]
        save(OUT/'inputs'/f'C{i}.json', {'question': value})
    for topic in TOPICS:
        save(OUT/'inputs'/f'{topic}.json', {'question': read_json(SA2/'inputs'/f'{topic}.json')['question']})
    hashes = dict(tools.hashes)
    relatives = ('story_agent/agent_tl3.py', 'scripts/run_tl3.py', 'scripts/verify_tl3.py',
        'scripts/report_tl3.py', 'tests/test_tl3.py', 'tests/test_tl2b_stage2.py',
        'story_agent/agent_tl2b.py', 'story_agent/strong_agent_sa1.py',
        'story_agent/strong_agent_sa2.py', 'story_agent/strong_agent_sa2_v821.py',
        'story_agent/strong_agent_sa2_v822.py', 'story_agent/agent_tools_tl1.py',
        'story_agent/similar_passages_tl2a.py', 'story_agent/similar_passages_tl2a_v0862.py',
        'story_agent/budget.py', 'docs/design/codex/104-tl3-review-rerun.md')
    for relative in relatives:
        hashes[str(CODE/relative)] = sha(CODE/relative)
    for path in (tools.neighbours.database, tools.neighbours.cache/'manifest.json',
                 ROOT/'.runtime/vector-recall/v1/units.json', ROOT/'.runtime/vector-recall/v1/books.json'):
        hashes[str(path)] = sha(path)
    for path in (OUT/'inputs').glob('*.json'):
        hashes[str(path)] = sha(path)
    for topic in TOPICS:
        path = SA2/'inputs'/f'{topic}.json'
        hashes[str(path)] = sha(path)
    save(OUT/'preparation.json', {'version': s.VERSION, 'cap': CAP, 'hashes': hashes,
        'main_prompt': s.SYSTEM_PROMPT, 'compare_prompt': s.COMPARE_PROMPT,
        'description': s.DESCRIPTION, 'python': sys.executable,
        'dependencies': {n: importlib.metadata.version(n) for n in
            ('deepagents', 'langchain', 'langchain-deepseek', 'httpx')},
        'books': len(tools.lines), 'quests': len(tools.quests.lines),
        'stage2_started': False, 'stage2_release': 'await Claude review and explicit README release'})
    emit(prepared=True, output=str(OUT), frozen_hashes=len(hashes))


def verify():
    for path, digest in read_json(OUT/'preparation.json')['hashes'].items():
        if sha(path) != digest:
            raise BudgetStopped('Frozen code/input changed: '+path)
    if not read_json(OUT/'offline-tests.json')['passed']:
        raise BudgetStopped('Passing offline tests required')


def projection():
    done = sum((OUT/t/'result.json').exists() for t in SMOKES)
    totals = s.accounted(OUT)
    return dict(totals, completed=done, cap=CAP,
        projected_budget_tokens=math.ceil(totals['budget_tokens']/done*2) if done else None)


def offline(smoke=False):
    class Tee:
        def __init__(self): self.parts = []
        def write(self, text):
            self.parts.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
        def flush(self): sys.stdout.flush()
    stream = Tee()
    names = ['tests.test_tl3.TL3Test.test_minimal_real_tools'] if smoke else [
        'tests.test_tl3', 'tests.test_tl2b_stage2']
    result = unittest.TextTestRunner(stream=stream, verbosity=2, failfast=True).run(
        unittest.defaultTestLoader.loadTestsFromNames(names))
    save(OUT/('smoke.json' if smoke else 'offline-tests.json'), {
        'passed': result.wasSuccessful(), 'tests': result.testsRun,
        'paid_calls': 0, 'log': ''.join(stream.parts)})
    if not result.wasSuccessful() or not result.testsRun:
        raise BudgetStopped('Offline failure; stop and inspect')


def proof():
    from scripts.verify_tl3 import zero_fee
    save(OUT/'zero-fee-parity.json', zero_fee(AgentTools(ROOT)))
    emit(parity_passed=True, paid_calls=0)


def preflight():
    import httpx
    try:
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=20) as client:
            response = client.get('https://api.deepseek.com/models')
        value = {'passed': response.status_code == 401, 'http_status': response.status_code,
                 'credentials_sent': False, 'tokens': 0}
    except httpx.HTTPError as exc:
        value = {'passed': False, 'error_type': type(exc).__name__, 'credentials_sent': False, 'tokens': 0}
    index = len(list(OUT.glob('network-preflight-*.json')))+1
    save(OUT/f'network-preflight-{index:02d}.json', value)
    emit(preflight=value)
    if not value['passed']:
        raise BudgetStopped('Network preflight failed')


def execute(topic, resume=False):
    verify()
    if not read_json(OUT/'zero-fee-parity.json')['passed']:
        raise BudgetStopped('Actual-wire parity proof required')
    if not any(read_json(p)['passed'] for p in OUT.glob('network-preflight-*.json')):
        raise BudgetStopped('Successful no-credential network preflight required')
    directory = OUT/topic
    if (directory/'result.json').exists():
        raise BudgetStopped('Completed topic must never be rerun')
    if (directory/'started.json').exists() and not resume:
        raise BudgetStopped('Prior attempt; explicit audited resume required')
    if topic == 'C2':
        forecast = projection()
        if forecast['completed'] != 1 or forecast['projected_budget_tokens'] > CAP:
            raise BudgetStopped('First-question projection does not allow second question')
    text = question(SMOKES.index(topic))
    if not (directory/'started.json').exists():
        save(directory/'started.json', {'topic': topic, 'question': text})
    endpoint, key = credentials()
    transport = s.AuditedDeepSeek(OUT, topic, key, CAP, resume=resume)
    model = live_model(key, transport, endpoint)
    agent = s.build_compare(model, AgentTools(ROOT).tools())
    seen, count = set(), 0
    attempt = len(list(directory.glob('trace-*.jsonl')))+1
    trace = directory/f'trace-{attempt:02d}.jsonl'
    try:
        for state in agent.stream({'messages': [{'role': 'user', 'content': text}]},
                config={'max_concurrency': 1}, stream_mode='values'):
            for message in state['messages']:
                if message.id in seen: continue
                seen.add(message.id)
                count += 1
                with trace.open('a', encoding='utf-8', newline='\n') as stream:
                    stream.write(json.dumps(transport.safe(message.model_dump(mode='json')), ensure_ascii=False)+'\n')
                emit(message=count, kind=message.type, content_characters=len(message.text))
        final = state['messages'][-1]
        if final.type != 'ai' or final.tool_calls or not final.text.strip():
            raise BudgetStopped('Zero final report')
        answer = transport.safe(final.text)
        (directory/'report.md').write_text(answer+'\n', encoding='utf-8')
        save(directory/'result.json', {'topic': topic, 'question': text, 'answer': answer,
            'usage': s.accounted(OUT, topic), 'replayed_settled_requests': transport.replayed,
            'tool_calls': sum(len(m.tool_calls) for m in state['messages'] if m.type=='ai')})
        forecast = projection()
        save(directory/'projection.json', forecast)
        emit(finished=topic, projection=forecast)
        if forecast['budget_tokens'] > CAP or forecast['projected_budget_tokens'] > CAP:
            raise BudgetStopped('Actual-based two-question projection exceeds cap; stop')
    finally:
        model.http_client.close()


def main():
    temporary = CODE/'.runtime/tl3-temp'
    temporary.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temporary)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'smoke', 'offline', 'proof', 'preflight', 'run', 'verify', 'projection'))
    parser.add_argument('--topic', choices=SMOKES)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        if args.stage in ('smoke', 'offline'):
            if args.stage == 'offline' and not read_json(OUT/'smoke.json')['passed']:
                raise BudgetStopped('Minimal smoke must pass first')
            offline(args.stage == 'smoke')
        elif args.stage == 'run':
            if not args.topic: parser.error('--topic required')
            execute(args.topic, args.resume)
        elif args.stage == 'projection': emit(projection=projection())
        else: globals()[args.stage]()
    except Exception as exc:
        reason = str(exc) if isinstance(exc, BudgetStopped) else 'runtime_error'
        if args.stage == 'run':
            save(OUT/args.topic/f'HALT-{len(list((OUT/args.topic).glob("HALT-*.json")))+1:02d}.json',
                {'error_type': type(exc).__name__, 'reason': reason, 'usage': s.accounted(OUT)})
        emit(stopped=True, error_type=type(exc).__name__, reason=reason)
        return 2
    return 0


if __name__ == '__main__': sys.exit(main())
