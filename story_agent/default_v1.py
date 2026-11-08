"""Frozen default-v1 composition; the TL5b prompt and tools stay unchanged."""
import hashlib
from pathlib import Path

from story_agent import agent_tl5b, agent_pt0
from story_agent.search_tl5b_v942 import AgentTools
from story_agent.conflict_check_tl5_v931 import ConflictCheck
from story_agent.community_graph_tl5 import CommunityGraph
from story_agent.report_tl5_v934 import Sources
from story_agent import report_default_v952 as report
from scripts.run_pt0_stage2 import ExternalSession, LoggedGuard
from story_agent.strong_agent_sa1 import save, read_json

VERSION = 'v0.95.2'
CONFIGURATION = 'default-v1.1'
MODEL = agent_tl5b.MODEL
SYSTEM_PROMPT = agent_tl5b.SYSTEM_PROMPT
ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / '.runtime/loregraph-8cfd3746b135b87433c27c188b23810a544d0ee9/graphlink_genshin_agent/character_graph.json'


def replace_json(path, value):
    """Atomic mutable summary/checkpoint; call evidence remains append-only."""
    import json
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def same(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise RuntimeError('Checkpoint evidence differs: '+str(path))
    else:
        save(path, value)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def configuration():
    from story_agent.conflict_check_tl5_v931 import PROMPT
    return dict(configuration=CONFIGURATION, code_version=VERSION, model=MODEL,
                thinking={'type': 'enabled'}, reasoning_effort='max', temperature=None,
                max_tokens=49152, max_retries=0, timeout_seconds=900,
                prompt_hashes={name: digest(value) for name, value in {
                    'main': SYSTEM_PROMPT, 'compare': agent_pt0.previous.COMPARE_PROMPT,
                    'reviewer': agent_pt0.REVIEW_PROMPT, 'conflict': PROMPT}.items()},
                transport_retry_seconds=[60, 120, 300, 600, 900],
                failed_requests_limit=10, estimated_usage_counts_toward_cap=False)


def build(model, directory, wiki, books=None):
    directory = Path(directory)
    books = books if books is not None else AgentTools(ROOT)
    tools = books.tools()
    sources = Sources(books)
    reviewer = agent_pt0.build_reviewer(model, tools)
    reviews = []

    def review(application):
        dest = directory / 'reviews' / f'{len(reviews)+1:03d}'
        value = {'input.json': application}
        result = agent_pt0.review(reviewer, application)
        value['answer.json'] = {'answer': result}
        for name, body in value.items():
            path = dest / name
            if path.exists():
                if read_json(path) != body:
                    raise RuntimeError('Review checkpoint differs')
            else:
                save(path, body)
        reviews.append(result)
        return result

    session = ExternalSession(wiki, review, directory / 'external-events.jsonl')
    checker = ConflictCheck(model, tools, CommunityGraph(GRAPH), sources, directory / 'conflicts')
    agent = agent_tl5b.build_main(model, tools, session, checker, LoggedGuard(directory))
    return agent, session, checker, sources


def finish(*args, **kwargs):
    """Reuse the final TL5b checks; only derived version metadata is replaced."""
    answer, question, records, invoke, sources, directory, metadata = args
    directory = Path(directory)
    controller = report.NameCheck(question)
    initial = controller.examine(answer, records())
    same(directory/'name-initial.json', initial)
    message = controller.feedback_once(initial)
    if message:
        same(directory/'name-feedback.json', {'feedback': message})
        answer = invoke(message)
        controller.examine(answer, records())
    same(directory/'name-resubmitted.json', controller.checks[-1])
    errors = report.check(answer, sources)
    same(directory/'format-initial.json', dict(answer=answer, errors=errors))
    count = int(bool(errors))
    if errors:
        message = report.format_feedback(errors)
        same(directory/'format-feedback.json', {'feedback': message})
        answer = invoke(message)
    controller.examine(answer, records())
    result = report.deliver(answer, directory, sources, dict(metadata(), **controller.metadata()), count)
    same(directory/'format-result.json', dict(errors=result['errors'], feedback_count=count, references=result['references']))
    result['metadata'].update(configuration=CONFIGURATION, code_version=VERSION,
                              postprocessor_version=VERSION)
    import json
    path = directory / 'report.md'
    text = path.read_text(encoding='utf-8').rsplit('## 附录 B：运行信息', 1)[0]
    path.write_text(text + '## 附录 B：运行信息\n\n```json\n' +
                    json.dumps(result['metadata'], ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
    return answer, result
