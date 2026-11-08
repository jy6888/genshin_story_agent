"""Run a question through frozen default-v1, with audited resumable output."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ('LANGCHAIN_TRACING_V2', 'LANGSMITH_TRACING'):
    os.environ[name] = 'false'

from story_agent import default_v1 as default
from story_agent.wire_default_v1 import AuditedDeepSeek, TransportStopped
from story_agent.budget import BudgetStopped
from story_agent.strong_agent_sa1 import save, read_json, emit
from story_agent.strong_agent_sa2 import credentials, live_model
from story_agent.external_pt0_v921 import Wikipedia


def prepare(question, directory, cap, resume):
    directory = Path(directory).resolve()
    if (directory/'result.json').exists():
        raise BudgetStopped('Output already completed; refusing overwrite')
    config = dict(default.configuration(), question=question, cap=cap)
    # Code hashes freeze all imported project modules across resume, without
    # reading any evaluation/specification material into the model input.
    config['source_hashes'] = {str(p.relative_to(ROOT)): default.digest(p.read_text(encoding='utf-8'))
        for p in sorted((ROOT/'story_agent').glob('*.py'))}
    config['source_hashes']['scripts/ask.py'] = default.digest(Path(__file__).read_text(encoding='utf-8'))
    if (directory/'config.json').exists():
        if not resume:
            raise BudgetStopped('Output exists; use --resume')
        if read_json(directory/'config.json') != config:
            raise BudgetStopped('Resume config, question, cap or code differs')
    elif resume:
        raise BudgetStopped('No checkpoint to resume')
    else:
        if directory.exists() and any(directory.iterdir()):
            raise BudgetStopped('Output is not empty')
        save(directory/'config.json', config)
    return directory


def execute(question, out, cap=1750000, resume=False, *, key=None, endpoint=None,
            inner=None, sleeper=time.sleep, books=None, wiki=None):
    if books is None:
        from story_agent.public_data import require_local_data
        require_local_data(ROOT)
    directory = prepare(question, out, cap, resume)
    if key is None:
        endpoint, key = credentials()
    wire = AuditedDeepSeek(directory, key, cap, resume=resume, inner=inner, sleeper=sleeper)
    model = live_model(key, wire, endpoint or 'https://api.deepseek.com/v1')
    wiki = wiki if wiki is not None else Wikipedia(directory/'references', filter_log=directory/'reference-filter.jsonl')
    started = time.monotonic()
    try:
        wire.gate()
        emit(corpus_loading=True, configuration=default.CONFIGURATION)
        agent, session, checker, sources = default.build(model, directory, wiki, books)
        config = {'max_concurrency': 1}
        state = agent.invoke({'messages': [{'role': 'user', 'content': question}]}, config=config)

        def invoke(message):
            nonlocal state
            state = agent.invoke({'messages': state['messages']+[{'role':'user','content':message}]}, config=config)
            return state['messages'][-1].text

        def metadata():
            records = wire.monitor.records
            tools = Counter(c['function']['name'] for r in records for c in r['message'].get('tool_calls', []))
            return dict(configuration=default.CONFIGURATION, code_version=default.VERSION,
                model=default.MODEL, thinking={'type':'enabled'}, reasoning_effort='max',
                response_counts=dict(Counter(r['scope'] for r in records)), tool_calls=dict(tools),
                external_requests=sum(e.get('type')=='request' for e in session.events),
                external_approvals=session.approvals, conflict_runs=len(checker.events),
                conflict_submitted=sum(e['validation']['submitted'] for e in checker.events),
                conflict_accepted=sum(len(e['validation']['accepted']) for e in checker.events),
                tokens=wire.cost(), empty_reasoning=wire.cost()['total']['empty_reasoning'],
                elapsed_seconds=round(time.monotonic()-started, 2))

        answer, result = default.finish(state['messages'][-1].text, question, lambda: wire.monitor.records,
                                        invoke, sources, directory, metadata)
        if wire.replayed != len(wire.replay):
            raise BudgetStopped('Checkpoint replay incomplete')
        wire.gate()
        save(directory/'result.json', dict(configuration=default.CONFIGURATION, code_version=default.VERSION,
            status='completed', question=question, answer=answer, metadata=result['metadata'], format_errors=result['errors']))
        default.replace_json(directory/'checkpoint.json', dict(configuration=default.CONFIGURATION, code_version=default.VERSION,
            status='completed', pending_request_sha256=None))
        print((directory/'summary.md').read_text(encoding='utf-8'), flush=True)
        return result
    except Exception as exc:
        save(directory/f'stop-{time.time_ns()}.json', dict(configuration=default.CONFIGURATION,
            code_version=default.VERSION, error_type=type(exc).__name__, reason=wire.error_text(exc)))
        raise
    finally:
        default.replace_json(directory/'cost.json', wire.cost())
        save(directory/f'wiki-network-{time.time_ns()}.json', wiki.events)
        wiki.close()
        model.http_client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cap', type=int, default=1750000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.cap <= 0:
        parser.error('--cap must be positive')
    try:
        execute(args.question, args.out, args.cap, args.resume)
        return 0
    except TransportStopped as exc:
        emit(stopped=True, error_type=type(exc).__name__, reason=str(exc))
        print('续跑命令：python -s scripts/ask.py --question '+json.dumps(args.question, ensure_ascii=False)+
              ' --out '+json.dumps(str(args.out.resolve()), ensure_ascii=False)+' --cap '+str(args.cap)+' --resume', flush=True)
        return 3
    except Exception as exc:
        emit(stopped=True, error_type=type(exc).__name__, reason=str(exc))
        return 2


if __name__ == '__main__':
    sys.exit(main())
