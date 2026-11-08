"""TL5 private graph boundary and format delivery, no provider calls."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from story_agent import agent_pt0, agent_tl5
from story_agent.community_graph_tl5 import CommunityGraph, GraphRead, overlaps
from story_agent.conflict_check_tl5 import ConflictCheck, validate, addition
from story_agent.report_tl5 import Sources, HEADINGS, check, deliver, feedback, APPENDIX
from tests.test_tl4b import fixture, capture
from scripts.verify_tl3 import reply
from story_agent.strong_agent_sa2 import live_model
from story_agent.tl5_wire import AuditedDeepSeek,accounted
from story_agent.budget import TokenLedger

def graph():
    return CommunityGraph(nodes=[dict(id=str(i),primary_name='人物'+str(i),aliases=['别名'+str(i)],
        notes=['这是十二个连续文字用于防止泄漏的秘密内容'],outgoing=[],incoming=[]) for i in range(7)])

def answer():
    bodies=['题','答','无','### 小问题\n\n【明说】书籍命中（100_书甲 L2）。“书籍命中”。','无','无','无']
    return '总结\n=====报告=====\n'+'\n\n'.join('## '+h+'\n\n'+b for h,b in zip(HEADINGS,bodies))

class TL5Test(unittest.TestCase):
    def setUp(self):
        self.temp=Path(__file__).resolve().parents[1]/'.runtime/tl5-test-temp'
        self.temp.mkdir(parents=True,exist_ok=True)
        self.books=fixture(); self.sources=Sources(self.books); self.reader=GraphRead(graph())
        self.reader.read('人物0')
        self.item={'conclusion':'人物0在书中','source':'100_书甲 L2','quote':'书籍命中','note':'原文只写书籍命中。'}

    def validate(self,item,game=()):
        return validate(json.dumps({'conflicts':[item]},ensure_ascii=False),'人物0在书中',self.sources,self.reader,game)

    def test_names_candidates_limits_and_bound(self):
        g=graph(); self.assertEqual(g.trigger('别名0和人物1'),['人物0','人物1'])
        g.nodes[1]['aliases'].append('别名0')
        r=GraphRead(g); self.assertEqual(len(r.read('别名0')['candidates']),2)
        for i in range(5): self.assertIn('text',r.read('人物'+str(i)))
        self.assertIn('error',r.read('人物5')); self.assertIn('text',r.read('人物0'))
        g.nodes[0]['notes']=['长'*9000]; self.assertLessEqual(len(r.read('人物0')['text']),6000)
        self.assertIn('未列完',r.read('人物0')['text'])

    def test_each_rejection_and_valid_append(self):
        good=self.validate(self.item); self.assertEqual(len(good['accepted']),1)
        self.assertIn('结论“人物0在书中”与 100_书甲 L2 原文“书籍命中”不符',addition(good['accepted']))
        cases=[('conclusion','新添结论','conclusion_not_substring'),('source','100_书甲 L999','source_invalid'),
               ('quote','错误引文','quote_not_in_lines'),('note','社区的判断','forbidden_word'),
               ('note','这是十二个连续文字用于防止泄漏的秘密内容','notes_leak')]
        for key,value,reason in cases:
            with self.subTest(reason=reason): self.assertIn(reason,self.validate(dict(self.item,**{key:value}))['items'][0]['reasons'])
        retained=self.validate(dict(self.item,note='这是十二个连续文字用于防止泄漏的秘密内容'),['这是十二个连续文字用于防止泄漏的秘密内容'])
        self.assertEqual(len(retained['accepted']),1)
        self.assertTrue(validate('坏JSON','',self.sources,self.reader,[])['structure_failure'])
        mixed=validate(json.dumps({'conflicts':[self.item,dict(self.item,quote='错误')]}),'人物0在书中',self.sources,self.reader,[])
        self.assertEqual(len(mixed['accepted']),1)

    def test_no_name_no_cost_and_structure_fail_no_addition(self):
        with tempfile.TemporaryDirectory(dir=self.temp) as d:
            checker=ConflictCheck(None,[],graph(),self.sources,d)
            self.assertEqual(checker.run('无名结论'),'');self.assertEqual(checker.events,[])
        self.assertEqual(addition([]),'')

    def test_format_repair_and_appendices(self):
        self.assertEqual(check(answer(),self.sources),[])
        self.assertTrue(check('错误回答',self.sources));self.assertIn('允许写法',feedback(check('错误回答',self.sources)))
        with tempfile.TemporaryDirectory(dir=self.temp) as d:
            r=deliver(answer(),d,self.sources,{'model':'offline'},1)
            self.assertEqual(r['errors'],[])
            text=(Path(d)/'report.md').read_text(encoding='utf-8')
            self.assertIn('附录 A',text);self.assertIn('> L2: 书籍命中',text);self.assertIn('附录 B',text)
            deliver('仍然不合格',d,self.sources,{},1)
            text=(Path(d)/'report.md').read_text(encoding='utf-8')
            for h in HEADINGS:self.assertIn('## '+h,text)
            self.assertIn('仍然不合格',text)
        refs=self.sources.references('《书甲》L2、agd_character_story/300_角色甲.txt L1–2')
        self.assertEqual(len(refs),2);self.assertTrue(all(r['resolved'] for r in refs))

    def test_wire_prompt_parity_and_private_return(self):
        session=SimpleNamespace(active=None,tools=lambda:[])
        checker=SimpleNamespace(run=lambda _: '\n\n已核实的冲突')
        before,_=capture(lambda m,t:agent_pt0.build_main(m,t,session),self.books.tools(),[reply()])
        after,_=capture(lambda m,t:agent_tl5.build_main(m,t,session,checker),self.books.tools(),[reply()])
        self.assertEqual(after[0]['messages'][0]['content'],before[0]['messages'][0]['content']+APPENDIX)
        self.assertEqual(after[0]['tools'],before[0]['tools'])
        self.assertNotIn('图谱',after[0]['messages'][0]['content'])
        payloads,state=capture(lambda m,t:agent_tl5.build_main(m,t,session,checker),self.books.tools(),[
            reply('',[('task',{'subagent_type':'compare','description':'核对人物0在书中（100_书甲 L2）'})]),reply('对照回答'),reply('主回答')])
        self.assertEqual(payloads[1]['messages'][0]['content'],agent_pt0.previous.COMPARE_PROMPT)
        self.assertEqual({t['function']['name'] for t in payloads[1]['tools']},agent_pt0.REVIEW_TOOLS)
        self.assertIn('已核实的冲突',payloads[2]['messages'][-1]['content'])
        self.assertEqual(state['messages'][-1].text,'主回答')

    def test_audited_wire_no_tools_feedback_and_replay(self):
        with tempfile.TemporaryDirectory(dir=self.temp) as d:
            root=Path(d); payloads=[]
            def provider(request):
                payloads.append(json.loads(request.content))
                return httpx.Response(200,json=reply('格式前' if len(payloads)==1 else '格式后'))
            ledger=TokenLedger(root/'ledger.sqlite')
            def run(wire):
                model=live_model('offline-placeholder',wire)
                try:
                    agent=create_agent(model,tools=[],system_prompt=agent_tl5.SYSTEM_PROMPT)
                    state=agent.invoke({'messages':[{'role':'user','content':'格式样例'}]})
                    state=agent.invoke({'messages':state['messages']+[{'role':'user','content':'请改写'}]})
                    return state['messages'][-1].text
                finally:model.http_client.close()
            out=root/'stage1'
            wire=AuditedDeepSeek(out,'B','offline-placeholder',1000,ledger=ledger,inner=httpx.MockTransport(provider))
            self.assertEqual(run(wire),'格式后');self.assertEqual(accounted(out)['requests'],2)
            replay=AuditedDeepSeek(out,'B','offline-placeholder',1000,ledger=ledger,resume=True,
                inner=httpx.MockTransport(lambda _:self.fail('No provider on replay')))
            self.assertEqual(run(replay),'格式后');self.assertEqual(replay.replayed,2)

    def test_private_checker_reads_and_invalid_json(self):
        with tempfile.TemporaryDirectory(dir=self.temp) as d:
            for index,raw in enumerate([json.dumps({'conflicts':[self.item]},ensure_ascii=False),'invalid-json']):
                responses=iter([reply('',[('read_community_graph',{'name':'别名0'}),
                                         ('read_book',{'book':'100','start_line':2,'end_line':2})]),reply(raw)])
                model=live_model('offline-placeholder',httpx.MockTransport(lambda _:httpx.Response(200,json=next(responses))))
                try:
                    checker=ConflictCheck(model,self.books.tools(),graph(),self.sources,Path(d)/str(index))
                    result=checker.run('人物0在书中')
                    if index==0:self.assertIn('原文冲突核对',result)
                    else:self.assertEqual(result,'')
                    self.assertEqual(len(checker.events[0]['nodes']),1)
                finally:model.http_client.close()

if __name__=='__main__':unittest.main()
