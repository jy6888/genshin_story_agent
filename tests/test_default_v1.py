"""Meaningful offline contracts for the default CLI and D-019 transport."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from story_agent import default_v1 as default, agent_tl5b
from story_agent.wire_default_v1 import AuditedDeepSeek, TransportStopped, DELAYS
from story_agent.budget import BudgetStopped
from story_agent.external_pt0_v921 import Wikipedia
from story_agent.strong_agent_sa1 import read_json
from scripts import ask
from tests.test_tl4b import fixture, capture
from tests.test_tl5 import answer, graph
from scripts.verify_tl3 import reply

TEMP = default.ROOT/'.runtime/default-v1-tests'
TEMP.mkdir(parents=True, exist_ok=True)


def payload(content='测试'):
    return dict(model='deepseek-v4-pro', thinking={'type':'enabled'}, reasoning_effort='max',
                max_tokens=49152, stream=False, tools=[],
                messages=[dict(role='system',content='离线测试'),dict(role='user',content=content)])


def response(text='完整回答'):
    return dict(id='offline',object='chat.completion',model='deepseek-v4-pro',created=0,
        choices=[dict(index=0, finish_reason='stop',message=dict(role='assistant',content=text,reasoning_content=''))],
        usage=dict(prompt_tokens=20,completion_tokens=10,total_tokens=30,
                   prompt_cache_hit_tokens=12,prompt_cache_miss_tokens=8,
                   completion_tokens_details={'reasoning_tokens':0}))


def request(value=None):
    return httpx.Request('POST','https://api.deepseek.com/chat/completions',
                        content=json.dumps(value or payload(),ensure_ascii=False).encode('utf-8'))


class WireTest(unittest.TestCase):
    def test_disambiguation_is_recoverable_but_corrupt_tool_result_stops(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            wire=AuditedDeepSeek(Path(temp)/'Q','offline-placeholder',100000)
            value=payload()
            value['tools']=[{'function':{'name':'read_text','parameters':{'required':['file']}}}]
            value['messages'] += [dict(role='assistant',content=None,tool_calls=[dict(id='tool1',
                function=dict(name='read_text',arguments='{"file":"10000128"}'))]),
                dict(role='tool',tool_call_id='tool1',content=json.dumps({'error':'请用完整文件名','candidates':['agd_voiceline/10000128_法尔伽.txt','agd_character_story/10000128_法尔伽.txt']},ensure_ascii=False))]
            wire.observe_history(value)
            value['messages'][-1]['content']='damaged non-JSON tool result'
            with self.assertRaisesRegex(BudgetStopped,'Unparseable tool result'):wire.observe_history(value)

    def exercise(self, failures, expected=True, cap=10000000):
        temp = tempfile.TemporaryDirectory(dir=TEMP)
        self.addCleanup(temp.cleanup)
        directory=Path(temp.name)/'Q'
        sent=[]; pauses=[]

        def handler(req):
            sent.append(req.content)
            failure=failures[len(sent)-1] if len(sent)<=len(failures) else None
            if isinstance(failure,Exception):raise failure
            if isinstance(failure,int):return httpx.Response(failure,json={})
            if isinstance(failure,bytes):return httpx.Response(200,content=failure)
            return httpx.Response(200,json=response())

        wire=AuditedDeepSeek(directory,'offline-placeholder',cap,inner=httpx.MockTransport(handler),sleeper=pauses.append)
        if expected:wire.handle_request(request())
        else:
            with self.assertRaises(BudgetStopped):wire.handle_request(request())
        self.assertTrue(all(b==sent[0] for b in sent))
        self.assertEqual(wire.ledger.summary()['status_counts'],{'settled':len(sent)})
        return wire,sent,pauses

    def test_byte_identical_mixed_failures_backoff_and_estimates(self):
        wire,sent,pauses=self.exercise([b'null',httpx.ConnectError('offline'),429,503,b''])
        self.assertEqual(len(sent),6)
        self.assertEqual(sum(pauses),sum(DELAYS))
        self.assertEqual([read_json(p)['wait_seconds'] for p in sorted((wire.directory/'calls').glob('*/error.json'))],list(DELAYS))
        cost=wire.gate()
        self.assertEqual(cost['total']['budget_tokens'],18)
        self.assertGreater(cost['estimated_missing_usage_budget_tokens'],18)
        self.assertEqual(cost['total']['empty_reasoning'],1)

    def test_sixth_failure_stops_and_other_4xx_do_not_retry(self):
        wire,sent,pauses=self.exercise([503]*6,False)
        self.assertEqual((len(sent),sum(pauses)),(6,sum(DELAYS)))
        for status in (400,401,404):
            _,sent,pauses=self.exercise([status],False)
            self.assertEqual((len(sent),pauses),(1,[]))

    def test_retry_counter_resets_and_eleventh_failure_stops(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            calls=0;pauses=[]
            def handler(req):
                nonlocal calls
                calls+=1
                return httpx.Response(200,json=response()) if calls%2==0 else httpx.Response(503,json={})
            wire=AuditedDeepSeek(Path(temp)/'Q','offline-placeholder',10000000,
                                 inner=httpx.MockTransport(handler),sleeper=pauses.append)
            for _ in range(10):wire.handle_request(request())
            with self.assertRaises(TransportStopped):wire.handle_request(request())
            self.assertEqual((calls,sum(pauses),wire.failures),(21,600,11))
            self.assertEqual(wire.cost()['retry_count'],10)

    def test_structural_empty_reply_not_transport_and_measured_cap(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            wire=AuditedDeepSeek(Path(temp)/'Q','offline-placeholder',100,
                inner=httpx.MockTransport(lambda _:httpx.Response(200,json=response(''))))
            with self.assertRaisesRegex(BudgetStopped,'neither tools nor content'):wire.handle_request(request())
            self.assertEqual(wire.cost()['transport_failures'],{})
        wire,_,_=self.exercise([],True,17)
        with self.assertRaisesRegex(BudgetStopped,'budget exceeded'):wire.gate()

    def test_success_replay_and_pending_resume_bytes(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            directory=Path(temp)/'Q';sent=[]
            def handler(req):
                sent.append(req.content)
                return httpx.Response(200,json=response()) if len(sent)==1 else httpx.Response(503,json={})
            wire=AuditedDeepSeek(directory,'offline-placeholder',10000000,
                inner=httpx.MockTransport(handler),sleeper=lambda _:None)
            wire.handle_request(request())
            with self.assertRaises(TransportStopped):wire.handle_request(request(payload('第二问')))
            resumed=[]
            replay=AuditedDeepSeek(directory,'offline-placeholder',10000000,resume=True,
                inner=httpx.MockTransport(lambda req:(resumed.append(req.content) or httpx.Response(200,json=response()))))
            replay.handle_request(request())
            self.assertEqual(resumed,[])
            replay.handle_request(request(payload('第二问')))
            self.assertEqual(resumed,[sent[-1]])


class EntryTest(unittest.TestCase):
    def setUp(self):
        graph_patch = patch.object(default, 'CommunityGraph', return_value=graph())
        graph_patch.start()
        self.addCleanup(graph_patch.stop)

    def test_entry_resume_after_successful_tool_and_transport_stop(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            directory=Path(temp)/'Q';sent=[]
            first=response(None)
            first['choices'][0]['finish_reason']='tool_calls'
            first['choices'][0]['message']['tool_calls']=[dict(id='offline-tool',type='function',
                function=dict(name='read_book',arguments=json.dumps(dict(book='100_书甲',start_line=1,end_line=3),ensure_ascii=False)))]
            def handler(req):
                sent.append(req.content)
                return httpx.Response(200,json=first) if len(sent)==1 else httpx.Response(503,json={})
            wiki=lambda:Wikipedia(Path(temp)/'wiki',httpx.MockTransport(lambda _:self.fail('No wiki network')))
            with self.assertRaises(TransportStopped):
                ask.execute('测试题',directory,10000000,key='offline-placeholder',books=fixture(),wiki=wiki(),
                            inner=httpx.MockTransport(handler),sleeper=lambda _:None)
            final=[]
            ask.execute('测试题',directory,10000000,True,key='offline-placeholder',books=fixture(),wiki=wiki(),
                inner=httpx.MockTransport(lambda req:(final.append(req.content) or httpx.Response(200,json=response(answer())))))
            self.assertEqual(final,[sent[-1]])
            self.assertEqual(read_json(directory/'result.json')['status'],'completed')

    def test_cli_fake_model_outputs_overwrite_and_resume(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            directory=Path(temp)/'Q'
            inner=httpx.MockTransport(lambda _:httpx.Response(200,json=response(answer())))
            wiki=Wikipedia(Path(temp)/'wiki',httpx.MockTransport(lambda _:self.fail('No wiki network')))
            with patch('socket.socket.connect',side_effect=AssertionError('No network')):
                result=ask.execute('测试题',directory,100000, key='offline-placeholder',inner=inner,books=fixture(),wiki=wiki)
            self.assertEqual(result['errors'],[])
            for name in ('summary.md','report.md','cost.json','trace.jsonl','checkpoint.json','config.json','result.json'):
                self.assertTrue((directory/name).is_file(),name)
            self.assertEqual(read_json(directory/'result.json')['configuration'],default.CONFIGURATION)
            self.assertEqual(read_json(directory/'config.json')['prompt_hashes']['main'],default.digest(agent_tl5b.SYSTEM_PROMPT))
            with self.assertRaisesRegex(BudgetStopped,'completed'):
                ask.prepare('测试题',directory,100000,False)
            with self.assertRaisesRegex(BudgetStopped,'completed'):
                ask.prepare('测试题',directory,100000,True)
        with tempfile.TemporaryDirectory(dir=TEMP) as temp:
            directory=Path(temp)/'Q'
            ask.prepare('测试题',directory,100000,False)
            with self.assertRaisesRegex(BudgetStopped,'use --resume'):ask.prepare('测试题',directory,100000,False)
            self.assertEqual(ask.prepare('测试题',directory,100000,True),directory.resolve())

    def test_transport_exit_code_and_resume_command(self):
        with patch.object(ask,'execute',side_effect=TransportStopped('offline')):
            self.assertEqual(ask.main(['--question','测试题','--out',str(TEMP/'stopped')]),3)

    def test_prompt_tools_and_graph_isolation(self):
        books=fixture();session=SimpleNamespace(active=None,tools=lambda:[]);checker=SimpleNamespace(run=lambda _: '')
        rows,_=capture(lambda m,t:agent_tl5b.build_main(m,t,session,checker),books.tools(),[reply()])
        self.assertEqual(rows[0]['messages'][0]['content'],default.SYSTEM_PROMPT)
        main={t['function']['name'] for t in rows[0]['tools']}
        child,_=capture(default.agent_pt0.previous.build_compare,books.tools(),[reply()])
        self.assertNotIn('read_community_graph',main)
        self.assertNotIn('read_community_graph',{t['function']['name'] for t in child[0]['tools']})
        for name in ('blind-test-v1.md','data/evals/','scoring_','观察点（核查者用）'):
            self.assertNotIn(name,json.dumps(rows+child,ensure_ascii=False))


if __name__=='__main__':unittest.main()
