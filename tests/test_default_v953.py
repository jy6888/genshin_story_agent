"""D-022 real-response regressions and measured empty-reply replay contracts."""
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from story_agent import report_default_v953 as fixed, report_default_v952 as old
from story_agent.default_v953 import ROOT
from story_agent.wire_default_v953 import AuditedDeepSeek, complete_empty
from story_agent.budget import BudgetStopped
from story_agent.strong_agent_sa1 import read_json
from tests.test_default_v1 import payload, response, request, TEMP
from tests.test_tl4b import fixture
from tests.test_tl5 import answer
from story_agent import agent_pt0
from story_agent.conflict_check_tl5_v931 import PROMPT as CONFLICT

BASE=ROOT/'docs/project/tasks/T-20261107-005/evidence'
REAL_FIXTURES = all((BASE/name).is_file() for name in (
    'blind-v0.95.2/B1/result.json', 'blind-v0.95.1/B4/result.json', 'blind-v0.95.2/B2/result.json'))
REAL_FIXTURES = REAL_FIXTURES and (ROOT/'data/manifests/book-map-v1.json').is_file()
REAL_FIXTURES = REAL_FIXTURES and (ROOT/'.runtime/istaroth-checkpoint/text/agd_book').is_dir()


class FormatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not REAL_FIXTURES:
            return
        from story_agent.default_v953 import AgentTools
        cls.sources=fixed.Sources(AgentTools(ROOT))

    @unittest.skipUnless(REAL_FIXTURES, 'Historical response fixtures and private corpus excluded from source distribution')
    def test_real_b1_two_guesses_and_two_abutting_sources(self):
        text=read_json(BASE/'blind-v0.95.2/B1/result.json')['answer']
        errors=fixed.check(text,self.sources)
        self.assertEqual(len(errors),2,errors)
        self.assertTrue(all(e.startswith('猜测节：') for e in errors))
        refs=self.sources.references(text)
        self.assertTrue(all(r['resolved'] is not None for r in refs))
        for fragment,name,start,end in [
            ('L435与200403_似乎是什么人手写的笔记.txt L17–18','200403_似乎是什么人手写的笔记.txt',17,18),
            ('已在71064_千年千岩.txt L195–199读到','71064_千年千岩.txt',195,199)]:
            found=[r['resolved'] for r in self.sources.references(fragment) if r['resolved']]
            self.assertTrue(any(r['file']==name and r['start']==start and r['end']==end for r in found),found)

    @unittest.skipUnless(REAL_FIXTURES, 'Historical response fixtures and private corpus excluded from source distribution')
    def test_b4_and_b2_real_reports_stay_valid(self):
        for path in ('blind-v0.95.1/B4/result.json','blind-v0.95.2/B2/result.json'):
            self.assertEqual(fixed.check(read_json(BASE/path)['answer'],self.sources),[])

    def test_guess_content_is_not_a_subfield(self):
        sources=fixed.Sources(fixture())
        value=answer().replace('## 猜测\n\n无','## 猜测\n\n1. 猜测内容：待查。\n抓手：原文。\n为什么：未确定。\n如何检验：新文本。')
        self.assertEqual(len(fixed.check(value,sources)),1)
        self.assertEqual(old.check(value,sources),[])
        self.assertEqual(fixed.check(value.replace('1. 猜测内容：','【猜测】猜测内容：'),sources),[])


class EmptyReplyTest(unittest.TestCase):
    def exercise(self,bodies,system=None):
        temporary=tempfile.TemporaryDirectory(dir=TEMP);self.addCleanup(temporary.cleanup)
        directory=Path(temporary.name)/'Q';sent=[];pauses=[]
        def handler(req):
            sent.append(req.content)
            return httpx.Response(200,json=bodies[min(len(sent)-1,len(bodies)-1)])
        wire=AuditedDeepSeek(directory,'offline-placeholder',1000000,
            inner=httpx.MockTransport(handler),sleeper=pauses.append)
        value=payload()
        if system is not None:value['messages'][0]['content']=system
        return wire,sent,pauses,request(value)

    def test_resend_succeeds_any_role_and_is_immediate(self):
        for system in ('离线主Agent',agent_pt0.previous.COMPARE_PROMPT,CONFLICT,agent_pt0.REVIEW_PROMPT):
            wire,sent,pauses,req=self.exercise([response('  '),response('完成')],system)
            wire.handle_request(req)
            self.assertEqual(sent,[req.content,req.content])
            self.assertEqual(pauses,[])
            cost=wire.cost()
            self.assertEqual(cost['total']['requests'],2)
            self.assertEqual(cost['total']['budget_tokens'],36)
            self.assertEqual(cost['complete_empty_reply_count'],1)
            self.assertEqual(cost['complete_empty_resend_requests'],1)
            self.assertEqual(cost['transport_failures'],{})
            self.assertEqual(cost['configuration'],'default-v1.2')

    def test_second_empty_stops_and_both_are_billed(self):
        wire,sent,pauses,req=self.exercise([response(''),response(None)])
        with self.assertRaisesRegex(BudgetStopped,'Two measured empty'):wire.handle_request(req)
        self.assertEqual(len(sent),2);self.assertEqual(pauses,[])
        self.assertEqual(wire.cost()['total']['budget_tokens'],36)
        self.assertEqual(wire.cost()['total']['successful_responses'],0)
        self.assertEqual(wire.ledger.summary()['status_counts'],{'settled':2})

    def test_reasoning_and_tools_without_content_is_not_empty(self):
        body=response(None)
        body['choices'][0]['finish_reason']='tool_calls'
        body['choices'][0]['message']['reasoning_content']='思考'
        body['choices'][0]['message']['tool_calls']=[dict(id='t',type='function',function=dict(name='search_books',arguments='{"query":"测试"}'))]
        self.assertFalse(complete_empty(body))
        wire,sent,_,req=self.exercise([body]);wire.handle_request(req)
        self.assertEqual(len(sent),1)

    def test_replay_after_first_reply_was_empty_keeps_invocation_id(self):
        wire,sent,_,req=self.exercise([response(''),response('完成')])
        wire.handle_request(req)
        replay_sent=[]
        replay=AuditedDeepSeek(wire.directory,'offline-placeholder',1000000,resume=True,
            inner=httpx.MockTransport(lambda r:(replay_sent.append(r.content) or httpx.Response(200,json=response('下一步')))))
        replay.handle_request(req)
        self.assertEqual(replay_sent,[])
        self.assertEqual(replay.replayed,1)


if __name__=='__main__':unittest.main()
