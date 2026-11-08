"""TL4b actual SDK routing, wire prompt parity, accounting and D-017 tests."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from langchain_core.utils.function_calling import convert_to_openai_tool
from scripts.verify_tl3 import reply
from story_agent import agent_tl4 as frozen
from story_agent import agent_tl4b as s
from story_agent.agent_tools_tl1 import Quests
from story_agent.budget import BudgetStopped, TokenLedger
from story_agent.similar_passages_tl2a_v0862 import BookLocator
from story_agent.strong_agent_sa2 import live_model
from story_agent.texts_tl4 import Texts
from story_agent.texts_tl4b import AgentTools
from story_agent.tl4b_delegation_v911 import delegation_error, LINE, SRC
from story_agent.tl4b_delegation import REJECTION

TEMP = Path(__file__).resolve().parents[1]/'.runtime/tl4b-temp'


def fixture():
    """Tiny in-memory corpus; no real game corpus or historical reports in tests."""
    books = AgentTools.__new__(AgentTools)
    books.lines = {'100_书甲.txt': ['# 书甲', '书籍命中'] + ['书文']*130,
                   '101_同名.txt': ['# 同名', '书籍甲'], '102_同名.txt': ['# 同名', '书籍乙'],
                   '103_跨库.txt': ['# 跨库']}
    books.aliases = {'书甲': ['100_书甲.txt'], '同名': ['101_同名.txt', '102_同名.txt'],
                     '跨库': ['103_跨库.txt']}
    books.locator = BookLocator(books.lines)
    books.maps, books.catalogue, books.hashes = {}, [], {}
    books.neighbours = SimpleNamespace(lookup=lambda *a: {'books': []})
    books.quests = Quests.__new__(Quests)
    books.quests.lines = {'200_任务甲.txt': ['# 任务甲', '人物: 任务命中'] + ['任务文']*130,
                         '201_跨库.txt': ['# 跨库']}
    books.quests.aliases = {'任务甲': ['200_任务甲.txt'], '200': ['200_任务甲.txt'],
                           '跨库': ['201_跨库.txt']}
    books.texts = Texts.__new__(Texts)
    books.texts.lines = {'agd_character_story/300_角色甲.txt': ['# 角色甲', '文本命中']}
    books.texts.kinds = {'agd_character_story/300_角色甲.txt': 'character_story'}
    books.texts.aliases = {'角色甲': ['agd_character_story/300_角色甲.txt']}
    return books


def capture(builder, tools, responses, guard=None):
    payloads = []
    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=responses[len(payloads)-1])
    model = live_model('offline-placeholder', httpx.MockTransport(handler))
    try:
        with patch('socket.socket.connect', side_effect=AssertionError('Network prohibited')):
            agent = builder(model, tools, guard) if guard is not None else builder(model, tools)
            state = agent.invoke({'messages': [{'role': 'user', 'content': '离线工具核验'}]},
                                 config={'max_concurrency': 1})
    finally:
        model.http_client.close()
    return payloads, state


class TL4bTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEMP.mkdir(parents=True, exist_ok=True)
        cls.books = fixture()

    def test_revised_reference_syntax(self):
        refs = ['L 27', '第27行', '行27', '行号约43-55', '行 号 大约 43',
                '《书甲》', '1068_终北祷歌集', '样本.txt']
        for dash in '-–—~～至到':
            refs += [f'L27{dash}30', f'第27{dash}30行', f'43{dash}55行']
        for ref in refs:
            with self.subTest(ref=ref):
                self.assertIsNone(delegation_error(ref))
        for text in ['', '核对散兵考据结论', '核对结论（附结论与出处）']:
            self.assertEqual(delegation_error(text), REJECTION)

    def test_unified_reads_and_ambiguities(self):
        for name, direct, args, notice in [
            ('100_书甲.txt', self.books.read_book, ('100_书甲.txt', 1, 130), '这是书籍文本，已按 read_book 读取。'),
            ('任务甲', self.books.quests.read_quest, ('任务甲', 1, 130), '这是任务文本，已按 read_quest 读取。')]:
            expected = direct(*args)
            actual = self.books.read_text(name, 1, 130)
            expected['notice'] = notice+expected['notice']
            self.assertEqual(actual, expected)
            self.assertEqual(actual['end_line'], 120)
            self.assertTrue(actual['truncated'])
        self.assertEqual(self.books.read_text('角色甲'), self.books.texts.read_text('角色甲'))
        self.assertEqual(self.books.read_text('100', 900, 901), self.books.read_book('100', 900, 901))
        self.assertEqual(len(self.books.read_text('同名')['candidates']), 2)
        self.assertEqual(len(self.books.read_text('跨库')['candidates']), 2)
        self.assertEqual(self.books.read_text('../../docs/reports/private.txt')['total_candidates'], 0)
        self.assertIn('error', self.books.read_text('不存在'))

    def test_guard_before_child_model_and_next_delegation(self):
        guard = s.DelegationGuard()
        responses = [reply('', [('task', {'subagent_type': 'compare', 'description': '核对散兵考据结论'})]),
                     reply('', [('task', {'subagent_type': 'compare', 'description': '核对《书甲》的说法'})]),
                     reply('子回答'), reply('主回答')]
        payloads, state = capture(s.build_main, self.books.tools(), responses, guard)
        self.assertEqual(len(payloads), 4)
        self.assertEqual([len(p['tools']) for p in payloads], [10, 10, 6, 10])
        self.assertEqual([e['blocked'] for e in guard.events], [True, False])
        refusals = [m for m in payloads[1]['messages'] if m['role'] == 'tool']
        self.assertEqual(refusals[-1]['content'], REJECTION)
        self.assertEqual(payloads[2]['messages'][1]['content'], '核对《书甲》的说法')
        self.assertEqual(state['messages'][-1].text, '主回答')
        self.assertEqual(len([m for m in state['messages'] if m.type == 'tool']), 2)
        self.assertIn('reasoning_content', payloads[-1]['messages'][-2])

    def test_compare_six_tools_roundtrip_and_reasoning(self):
        calls = [('search_books', {'query': '命中'}), ('search_quests', {'query': '命中'}),
                 ('search_texts', {'query': '命中'}), ('read_text', {'file': '书甲'}),
                 ('read_text', {'file': '任务甲'}), ('read_book', {'book': '100'}),
                 ('read_quest', {'quest': '200'})]
        payloads, state = capture(s.build_compare, self.books.tools(), [reply('', calls), reply('完成')])
        self.assertEqual(state['messages'][-1].text, '完成')
        self.assertEqual(len(payloads), 2)
        returns = [json.loads(m['content']) for m in payloads[1]['messages'] if m['role']=='tool']
        self.assertEqual(len(returns), 7)
        self.assertTrue(all(v.get('total_matches', 1)>0 and 'error' not in v for v in returns))
        self.assertEqual(payloads[1]['messages'][2]['reasoning_content'], '离线思考')
        main_defs = {t.name: convert_to_openai_tool(t) for t in self.books.tools()}
        self.assertEqual({t['function']['name'] for t in payloads[0]['tools']}, s.CHILD_TOOLS)
        for t in payloads[0]['tools']:
            self.assertEqual(t, main_defs[t['function']['name']])

    def test_actual_wire_prompts_only_authorized_edits(self):
        tools = self.books.tools()
        old_main, _ = capture(frozen.build_main, tools, [reply()])
        new_main, _ = capture(s.build_main, tools, [reply()])
        old_child, _ = capture(frozen.build_compare, tools, [reply()])
        new_child, _ = capture(s.build_compare, tools, [reply()])
        self.assertEqual(new_main[0]['messages'][0]['content'],
                         old_main[0]['messages'][0]['content'].replace(s.OLD, s.NEW, 1))
        self.assertEqual(new_child[0]['messages'][0]['content'],
                         old_child[0]['messages'][0]['content']+s.APPENDIX)
        self.assertEqual(s.DESCRIPTION, frozen.DESCRIPTION)
        self.assertEqual(new_main[0]['tools'], old_main[0]['tools'])

    def test_async_guard_rejects_without_handler(self):
        request = SimpleNamespace(tool_call={'name': 'task', 'id': 'x',
            'args': {'subagent_type': 'compare', 'description': '核对散兵考据结论'}})
        async def forbidden(_):
            self.fail('Child must not start')
        # Rejection must complete before its first await; no OS event loop or
        # socket pair is needed for this zero-child-execution assertion.
        coroutine = s.DelegationGuard().awrap_tool_call(request, forbidden)
        with self.assertRaises(StopIteration) as completed:
            coroutine.send(None)
        self.assertEqual(completed.exception.value.content, REJECTION)

    def test_audited_resume_budget_and_reasoning(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            out, sent = Path(directory), []
            ledger = TokenLedger(out/'ledger.sqlite')
            def handler(request):
                sent.append(request.content)
                return httpx.Response(200, json=reply('结果'))
            for resume in [False, True]:
                wire = s.AuditedDeepSeek(out, 'C1', 'offline-placeholder', cap=40, ledger=ledger,
                    resume=resume, inner=httpx.MockTransport((lambda _: self.fail('No resend')) if resume else handler))
                model = live_model('offline-placeholder', wire)
                try:
                    result = s.build_compare(model, self.books.tools()).invoke(
                        {'messages': [{'role': 'user', 'content': '离线'}]})
                    self.assertEqual(result['messages'][-1].text, '结果')
                    self.assertEqual(wire.gate()['budget_tokens'], 40)
                    wire.cap = 39
                    with self.assertRaisesRegex(BudgetStopped, 'cap exceeded'):
                        wire.gate()
                finally:
                    model.http_client.close()
            self.assertEqual(len(sent), 1)
            self.assertEqual(wire.replayed, 1)
            self.assertEqual((out/'C1/calls/001/reasoning.txt').read_text(encoding='utf-8'), '离线思考')

    def test_d017_same_bytes_retry_and_two_failures(self):
        for failures in [1, 2]:
            with self.subTest(failures=failures), tempfile.TemporaryDirectory(dir=TEMP) as directory:
                out, sent, waits = Path(directory), [], []
                def handler(request):
                    sent.append(request.content)
                    if len(sent) <= failures:
                        raise httpx.ReadError('离线传输失败')
                    return httpx.Response(200, json=reply())
                wire = s.AuditedDeepSeek(out, 'C1', 'offline-placeholder', cap=1_000_000,
                    ledger=TokenLedger(out/'ledger.sqlite'), inner=httpx.MockTransport(handler), sleeper=waits.append)
                model = live_model('offline-placeholder', wire)
                try:
                    if failures == 1:
                        result = s.build_compare(model, self.books.tools()).invoke(
                            {'messages': [{'role': 'user', 'content': '离线'}]})
                        self.assertEqual(result['messages'][-1].text, '离线回答')
                    else:
                        with self.assertRaisesRegex(BudgetStopped, 'Two consecutive'):
                            s.build_compare(model, self.books.tools()).invoke(
                                {'messages': [{'role': 'user', 'content': '离线'}]})
                    self.assertEqual(sent[0], sent[1])
                    self.assertEqual(waits, [30, 30])
                    self.assertEqual(s.accounted(out)['estimated_requests'], failures)
                    self.assertEqual(wire.ledger.summary()['status_counts'], {'settled': 2})
                finally:
                    model.http_client.close()


if __name__ == '__main__':
    unittest.main(verbosity=2, failfast=True)
