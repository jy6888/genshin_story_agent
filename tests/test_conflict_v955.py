"""Meaningful return-boundary checks, including a real SDK format retry."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx

from story_agent import conflict_check_default_v955 as s
from story_agent.report_default_v953 import Sources
from story_agent.community_graph_tl5 import GraphRead
from story_agent.strong_agent_sa2 import live_model
from scripts.verify_tl3 import reply
from tests.test_tl4b import fixture
from tests.test_tl5 import graph


class ReturnBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.books=fixture();self.sources=Sources(self.books);self.reader=GraphRead(graph())
        self.reader.read('人物0');self.description='人物0在书中；第二条结论'
        self.item=dict(conclusion='人物0在书中',source='100_书甲.txt L2',quote='书籍命中',note='原文不支持该说法。')

    def check(self,item):
        return s.validate(json.dumps({'conflicts':[item]},ensure_ascii=False),self.description,self.sources,self.reader,[])

    def test_multi_locations_and_compacted_merged_quote(self):
        for separator in ('/','、','，',';'):
            with self.subTest(separator=separator):
                item=dict(self.item,source='100_书甲.txt L2'+separator+'L4–5',quote='书籍命中书文书文')
                result=self.check(item)
                self.assertEqual(result['accepted'],[item])
                self.assertEqual(result['source_resolutions'][0]['ranges'],[[2,2],[4,5]])

    def test_every_location_exists_and_no_cross_file_or_trailing_junk(self):
        for source in ('100_书甲.txt L2/L999','100_书甲.txt L2/200_任务甲.txt L2',
                       '100_书甲.txt L2、200_任务甲.txt L2','100_书甲.txt L2 extra',
                       '100_书甲.txt L0/L2','100_书甲.txt L5–4'):
            with self.subTest(source=source):
                self.assertIn('source_invalid',self.check(dict(self.item,source=source))['items'][0]['reasons'])

    def test_each_format_reason_has_its_own_actionable_instruction(self):
        cases=[('fields',{'source':'x'}),('conclusion_length',dict(self.item,conclusion='长'*41)),
            ('quote_length',dict(self.item,quote='长'*61)),('note_length',dict(self.item,note='长'*81)),
            ('conclusion_not_substring',dict(self.item,conclusion='新添的结论')),
            ('source_invalid',dict(self.item,source='100_书甲.txt L999')),
            ('quote_not_in_lines',dict(self.item,quote='不存在的摘引'))]
        for reason,item in cases:
            with self.subTest(reason=reason):
                result=self.check(item);message=s.feedback(result,self.description,self.sources)
                self.assertIn(reason,message)
                if reason in ('source_invalid','quote_not_in_lines'):self.assertIn('100_书甲.txt',message)
        result=s.validate(json.dumps({'conflicts':[self.item]*6}),self.description,self.sources,self.reader,[])
        self.assertIn('最多5条',s.feedback(result,self.description,self.sources))

    def test_retry_once_success_and_still_invalid(self):
        bad=json.dumps({'conflicts':[dict(self.item,quote='错误')]},ensure_ascii=False)
        good=json.dumps({'conflicts':[self.item]},ensure_ascii=False)
        for resubmitted,accepted in ((good,1),(bad,0)):
            with self.subTest(accepted=accepted):
                calls=[]
                result=s.review_once(bad,self.description,self.sources,self.reader,[],[],
                    lambda message:(calls.append(message) or resubmitted))
                self.assertEqual(len(calls),1);self.assertEqual(len(result['accepted']),accepted)
                self.assertEqual(result['format_feedback_count'],1)

    def test_direct_privacy_rejections_never_receive_feedback_even_if_malformed(self):
        items=[dict(self.item,note='社区告诉我'),dict(self.item,note=self.reader.notes()[0]),
               dict(conclusion='人物0在书中',note='图谱中的说法')]
        for item in items:
            with self.subTest(item=item):
                def unexpected(_):raise AssertionError('Privacy rejection must not request feedback')
                result=s.review_once(json.dumps({'conflicts':[item]},ensure_ascii=False),self.description,
                    self.sources,self.reader,[],[],unexpected)
                self.assertEqual(result['accepted'],[]);self.assertEqual(result['format_feedback_count'],0)

    def test_malformed_resubmission_is_discarded_instead_of_raising(self):
        raw=json.dumps({'conflicts':[dict(self.item,quote='错误')]},ensure_ascii=False)
        for item in (dict(self.item,conclusion=123),dict(self.item,conclusion=None),7):
            with self.subTest(item=item):
                result=s.review_once(raw,self.description,self.sources,self.reader,[],[],
                    lambda _:json.dumps({'conflicts':[item]},ensure_ascii=False))
                self.assertEqual(result['accepted'],[])
                self.assertIn('fields',result['items'][0]['reasons'])
                self.assertEqual(result['format_feedback_count'],1)

    def test_unrelated_retry_cannot_revive_directly_deleted_item(self):
        private=dict(self.item,note='社区告诉我')
        other=dict(self.item,conclusion='第二条结论',quote='错误')
        sanitized=dict(self.item,note='原文不同。')
        corrected=dict(other,quote='书籍命中')
        messages=[]
        result=s.review_once(json.dumps({'conflicts':[private,other]},ensure_ascii=False),self.description,
            self.sources,self.reader,[],[],lambda message:(messages.append(message) or
                json.dumps({'conflicts':[sanitized,corrected]},ensure_ascii=False)))
        self.assertNotIn('社区告诉我',messages[0]);self.assertNotIn('第1条',messages[0])
        self.assertEqual(result['accepted'],[corrected])

    def test_sdk_checker_keeps_same_prompt_and_sends_only_one_feedback(self):
        parent=Path(__file__).resolve().parents[1]/'.runtime/v955-test-temp';parent.mkdir(parents=True,exist_ok=True)
        sent=[]
        bad=json.dumps({'conflicts':[dict(self.item,source='100_书甲.txt L999')]},ensure_ascii=False)
        good=json.dumps({'conflicts':[dict(self.item,source='100_书甲.txt L2/L4')]},ensure_ascii=False)
        def handler(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200,json=reply(bad if len(sent)==1 else good))
        model=live_model('offline-placeholder',httpx.MockTransport(handler))
        try:
            with tempfile.TemporaryDirectory(dir=parent) as temp,patch('socket.socket.connect',side_effect=AssertionError('No network')):
                checker=s.ConflictCheck(model,self.books.tools(),graph(),self.sources,temp)
                addition=checker.run(self.description)
                self.assertIn('原文冲突核对',addition)
                self.assertEqual(len(sent),2)
                self.assertEqual(sent[0]['messages'][0]['content'],s.PROMPT)
                self.assertEqual(sent[1]['messages'][0]['content'],s.PROMPT)
                self.assertEqual(checker.events[0]['validation']['format_feedback_count'],1)
                self.assertIn('100_书甲.txt',sent[1]['messages'][-1]['content'])
        finally:model.http_client.close()


if __name__=='__main__':unittest.main()
