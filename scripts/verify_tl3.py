"""Actual SDK wire parity and real-response replay; no semantic grading."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import httpx

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from scripts import run_tl3 as run
from story_agent import agent_tl2b as old
from story_agent import agent_tl3 as s
from story_agent.budget import TokenLedger
from story_agent.strong_agent_sa1 import read_json, save, sha
from story_agent.strong_agent_sa2 import DB, live_model


def reply(content='离线回答', calls=()):
    message = {'role': 'assistant', 'content': content, 'reasoning_content': '离线思考'}
    if calls:
        message['tool_calls'] = [{'id': f'call_{i}_{name}', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}
            for i, (name, args) in enumerate(calls)]
    return {'id': 'offline', 'object': 'chat.completion', 'created': 1, 'model': old.MODEL,
        'choices': [{'index': 0, 'finish_reason': 'tool_calls' if calls else 'stop', 'message': message}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
            'prompt_cache_hit_tokens': 80, 'prompt_cache_miss_tokens': 20,
            'completion_tokens_details': {'reasoning_tokens': 10}}}


def capture(builder, tools, question='离线核验', nested=False):
    payloads = []
    def handler(request):
        payloads.append(json.loads(request.content))
        n = len(payloads)
        if nested and n == 1:
            value = reply('', [('task', {'subagent_type': 'compare', 'description': question})])
        elif nested and n == 2:
            value = reply('', [('read_book', {'book': '201219', 'start_line': 4, 'end_line': 6}),
                ('read_quest', {'quest': '6026', 'start_line': 492, 'end_line': 501})])
        else:
            value = reply('离线子回答' if nested and n == 3 else '离线主报告')
        return httpx.Response(200, json=value)
    model = live_model('offline-placeholder', httpx.MockTransport(handler))
    try:
        with patch('socket.socket.connect', side_effect=AssertionError('Network prohibited')):
            result = builder(model, tools).invoke({'messages': [{'role': 'user', 'content': question}]},
                config={'max_concurrency': 1})
    finally:
        model.http_client.close()
    return payloads, result


def zero_fee(books):
    tools = books.tools()
    before, _ = capture(old.build_main, tools)
    after, result = capture(s.build_main, tools, nested=True)
    child, _ = capture(old.build_compare, tools)
    assert len(after) == 4
    a, b = after[0], before[0]
    assert a['messages'][0]['content'] == b['messages'][0]['content'].replace(
        old.SYSTEM_PROMPT, s.SYSTEM_PROMPT, 1)
    assert a['messages'][0]['content'].count(s.APPENDIX) == 1
    assert s.SYSTEM_PROMPT == old.SYSTEM_PROMPT+s.APPENDIX
    assert s.DESCRIPTION == old.DESCRIPTION and s.COMPARE_PROMPT == old.COMPARE_PROMPT
    assert s.build_compare is old.build_compare
    assert {t['function']['name'] for t in a['tools']} == old.MAIN_TOOLS
    assert len(a['tools']) == 8  # Seven research tools plus task(compare).
    assert {k: v for k, v in a.items() if k != 'messages'} == {k: v for k, v in b.items() if k != 'messages'}
    assert {k: v for k, v in after[1].items() if k != 'messages'} == {
        k: v for k, v in child[0].items() if k != 'messages'}
    assert after[1]['messages'][0]['content'] == child[0]['messages'][0]['content'] == s.COMPARE_PROMPT
    assert after[2]['messages'][2]['reasoning_content'] == '离线思考'
    assert 'L6:' in after[2]['messages'][3]['content']
    assert 'L492:' in after[2]['messages'][4]['content']
    assert after[3]['messages'][2]['reasoning_content'] == '离线思考'
    assert after[3]['messages'][3]['content'] == '离线子回答'
    assert result['messages'][-1].text == '离线主报告'
    # Cross-check against actually sent frozen TL2B requests, not only builders.
    frozen = sorted((run.TL2B/'NQ1A/calls').glob('*/request.json'))
    assert frozen, 'Frozen TL2B stage-2 wire evidence required'
    previous = next(read_json(p) for p in frozen if len(read_json(p)['tools']) == 8)
    assert previous['messages'][0]['content'] == b['messages'][0]['content']
    assert {k: v for k, v in previous.items() if k != 'messages'} == {
        k: v for k, v in b.items() if k != 'messages'}
    for topic in run.TOPICS:
        assert read_json(run.OUT/'inputs'/f'{topic}.json')['question'] == read_json(
            run.SA2/'inputs'/f'{topic}.json')['question']
    spec = run.SPEC.read_text(encoding='utf-8')
    for i in (1, 2):
        expected = spec.split(f'**第 {i} 题**', 1)[1].split('```text\n', 1)[1].split('\n```', 1)[0]
        assert run.question(i-1) == expected
    return {'passed': True, 'paid_calls': 0, 'requests': after,
        'tl2b_actual_request': str(frozen[0]), 'baseline_request': b,
        'child_baseline_request': child[0], 'main_only_authorized_appendix': True,
        'compare_builder_reused': True, 'compare_description_prompt_tools_model_equal': True,
        'seven_tool_schemas_equal': True, 'nested_reasoning_roundtrip': True,
        'questions_equal_sa2': list(run.TOPICS), 'smoke_questions_equal_spec': True}


def integrity_and_replay():
    run.verify()
    assert read_json(run.OUT/'zero-fee-parity.json')['passed']
    tools = run.AgentTools(run.ROOT).tools()
    rows = []
    def forbidden(request): raise AssertionError('Settled request resent')
    for index, topic in enumerate(run.SMOKES):
        root = run.OUT/topic
        result = read_json(root/'result.json')
        assert result['question'] == run.question(index)
        assert (root/'report.md').read_text(encoding='utf-8') == result['answer']+'\n'
        trace = [json.loads(line) for path in sorted(root.glob('trace-*.jsonl'))
                 for line in path.read_text(encoding='utf-8').splitlines()]
        assert trace[0]['content'] == result['question'] and trace[-1]['content'] == result['answer']
        calls = sorted((root/'calls').glob('*/response.json'))
        assert len(calls) == result['usage']['requests']
        for path in calls:
            directory = path.parent
            request = read_json(directory/'request.json')
            assert json.loads((directory/'request-body.bin').read_bytes()) == request
            assert request['messages'][0]['content'] == s.COMPARE_PROMPT
            assert request['messages'][1]['content'] == result['question']
            assert {t['function']['name'] for t in request['tools']} == old.CHILD_TOOLS
            message = read_json(path)['choices'][0]['message']
            assert message['reasoning_content'] == (directory/'reasoning.txt').read_text(encoding='utf-8')
            assert (message.get('content') or '') == (directory/'content.txt').read_text(encoding='utf-8')
            assert any(m['type']=='ai' and m['additional_kwargs'].get('reasoning_content') ==
                       message['reasoning_content'] for m in trace)
        tool_rows = [m for m in trace if m['type']=='tool']
        assert len(tool_rows) == result['tool_calls']
        assert any('text' in json.loads(m['content']) for m in tool_rows)
        transport = s.AuditedDeepSeek(run.OUT, topic, 'offline-replay-placeholder', resume=True,
                                    inner=httpx.MockTransport(forbidden))
        model = live_model('offline-replay-placeholder', transport)
        try:
            with patch('socket.socket.connect', side_effect=AssertionError('Network disabled')):
                state = s.build_compare(model, tools).invoke({'messages': [
                    {'role': 'user', 'content': result['question']}]}, config={'max_concurrency': 1})
        finally: model.http_client.close()
        assert state['messages'][-1].text == result['answer']
        assert transport.replayed == len(calls)
        rows.append({'topic': topic, 'responses_replayed': len(calls), 'request_bytes_equal': True,
            'answer_equal': True, 'tool_calls': len(tool_rows), 'normal_invalid_arguments':
                sum('error' in json.loads(m['content']) for m in tool_rows),
            'usage': s.accounted(run.OUT, topic), 'answer_sha256': sha(root/'report.md')})
    totals = s.accounted(run.OUT)
    assert totals['budget_tokens'] <= run.CAP
    assert read_json(run.OUT/'C1/projection.json')['projected_budget_tokens'] <= run.CAP
    ledger = TokenLedger(DB)
    with ledger.connect() as db:
        settled = db.execute('SELECT status,actual FROM calls WHERE run_id IN (?,?)',
                             (s.PREFIX+'C1', s.PREFIX+'C2')).fetchall()
    assert len(settled) == totals['attempts'] and all(r[0]=='settled' for r in settled)
    assert sum(r[1] for r in settled) == totals['accounted_raw_tokens']
    save(run.OUT/'stage1-integrity-replay.json', {'passed': True, 'paid_calls': 0,
        'new_provider_requests': 0, 'topics': rows, 'usage': totals, 'ledger_reconciled': True,
        'frozen_hashes_verified': len(read_json(run.OUT/'preparation.json')['hashes']),
        'stage2_started': False, 'semantic_grading': False})
    print(json.dumps({'integrity_replay_passed': True, 'topics': rows}, ensure_ascii=False))


if __name__ == '__main__': integrity_and_replay()
