"""TL4b unified reads over the same immutable, allowlisted game corpora."""
from langchain_core.tools import tool
from story_agent.texts_tl4 import AgentTools as FrozenTools


class AgentTools(FrozenTools):
    def read_text(self, file, start_line=1, end_line=120):
        # Ambiguous or invalid ranges within the sixteen text kinds retain TL4's
        # result exactly. Only a genuinely absent identifier falls back.
        if file in self.texts.lines or self.texts.aliases.get(file):
            return self.texts.read_text(file, start_line, end_line)
        book, book_candidates = self.locator.locate(file)
        quests = [file] if file in self.quests.lines else self.quests.aliases.get(file, [])
        if book and not quests:
            value = self.read_book(book, start_line, end_line)
            noun, reader = '书籍', 'read_book'
        elif len(quests) == 1 and not book:
            value = self.quests.read_quest(quests[0], start_line, end_line)
            noun, reader = '任务', 'read_quest'
        else:
            candidates = list(dict.fromkeys(([book] if book else book_candidates) + list(quests)))
            return {'error': '未找到唯一文本；请使用候选的完整文件名。',
                    'candidates': candidates[:10], 'total_candidates': len(candidates)}
        if 'error' not in value:
            value['notice'] = f'这是{noun}文本，已按 {reader} 读取。' + value.get('notice', '')
        return value

    def tools(self):
        original = super().tools()

        @tool(description=next(t.description for t in original if t.name == 'read_text'))
        def read_text(file: str, start_line: int = 1, end_line: int = 120) -> dict:
            return self.read_text(file, start_line, end_line)

        # Keep every public schema/description identical to the frozen tools.
        return [read_text if t.name == 'read_text' else t for t in original]
