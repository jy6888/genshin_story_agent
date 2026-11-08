"""TL2A v0.86.0: static, read-only passage neighbours; no encoder at runtime."""
import hashlib
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from langchain_core.tools import tool
from story_agent.agent_tools_tl1 import AgentTools as TL1Tools

VERSION = 'v0.86.0'
NOTICE = ('只是意象或说法相近的候选；是否真有联系，要读原文判断，或交给对照读者。'
          '两个通道相似度不可比，各通道按每本书最高相似度排序，再从书图开始交替取书；'
          '重复书跳过，最多10本，每本最多3行。短行的相似度来自前后非空行窗口。')


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Neighbours:
    """An immutable SQLite table, opened read-only separately for each call."""
    def __init__(self, cache, books):
        self.cache = Path(cache).resolve()
        self.database = self.cache / 'neighbors.sqlite'
        self.manifest = json.loads((self.cache / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['version'] != VERSION or file_sha(self.database) != self.manifest['database_sha256']:
            raise ValueError('Similar-passages cache version/hash mismatch')
        for name, expected in self.manifest['corpus'].items():
            text = '\n'.join(books.lines[name])
            if hashlib.sha256(text.encode('utf-8')).hexdigest() != expected:
                raise ValueError('Similar-passages corpus mismatch: ' + name)
        self.books = books
        with self.connect() as db:
            self.units = {i: (channel, name, json.loads(lines)) for i, channel, name, lines
                          in db.execute('SELECT id,channel,book,lines FROM units')}

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database.as_uri() + '?mode=ro&immutable=1', uri=True)
        try:
            yield db
        finally:
            db.close()

    def lookup(self, book, line):
        name = self.books.resolve(book)
        if name is None:
            return {'error': '未找到唯一书名；请用完整文件名或唯一书名。'}
        if type(line) is not int or not 1 <= line <= len(self.books.lines[name]):
            return {'error': '物理行号无效。', 'total_lines': len(self.books.lines[name])}
        by_channel = {'书图': {}, '逐行': {}}
        with self.connect() as db:
            queries = db.execute('SELECT unit FROM queries WHERE book=? AND line=?', (name, line)).fetchall()
            for (query,) in queries:
                found = db.execute('SELECT items FROM neighbors WHERE unit=?', (query,)).fetchone()
                if found is None:
                    raise ValueError('Incomplete neighbour table')
                for target, score in json.loads(found[0]):
                    channel, other, numbers = self.units[target]
                    if other == name:
                        raise ValueError('Same-book neighbour in cache')
                    rows = by_channel[channel].setdefault(other, {})
                    for number in numbers:
                        if number not in rows or score > rows[number]:
                            rows[number] = score
        ranked = {channel: sorted(candidates, key=lambda book: (-max(candidates[book].values()), book))
                  for channel, candidates in by_channel.items()}
        selected, seen = [], set()
        positions = {channel: 0 for channel in ranked}
        while len(selected) < 10:
            advanced = False
            for channel in ('书图', '逐行'):
                names = ranked[channel]
                while positions[channel] < len(names) and names[positions[channel]] in seen:
                    positions[channel] += 1
                if positions[channel] < len(names):
                    other = names[positions[channel]]
                    positions[channel] += 1
                    seen.add(other)
                    selected.append((channel, other))
                    advanced = True
                    if len(selected) == 10:
                        break
            if not advanced:
                break
        results = []
        for first, other in selected:
            rows, used = [], set()
            # Interleave per-channel lines too: never compare scores across channels.
            queues = {ch: sorted(by_channel[ch].get(other, {}).items(), key=lambda x: (-x[1], x[0]))
                      for ch in (first, '逐行' if first == '书图' else '书图')}
            while len(rows) < 3 and any(queues.values()):
                for ch, queue in queues.items():
                    while queue and queue[0][0] in used:
                        queue.pop(0)
                    if queue:
                        number, score = queue.pop(0)
                        used.add(number)
                        rows.append({'book': other, 'line': number,
                                     'text': self.books.lines[other][number-1],
                                     'similarity': score, 'channel': ch})
                        if len(rows) == 3:
                            break
            results.append({'book': other, 'passages': rows})
        return {'book': name, 'line': line, 'notice': NOTICE, 'books': results}


class AgentTools(TL1Tools):
    def __init__(self, root, cache=None):
        super().__init__(root)
        self.neighbours = Neighbours(cache or Path(root)/'.runtime/similar-passages/v1', self)

    def similar_passages(self, book, line):
        return self.neighbours.lookup(book, line)

    def tools(self):
        @tool
        def similar_passages(book: str, line: int) -> dict:
            """给定一本书的完整文件名或唯一书名与物理行号，列出其他书里意象或说法相近的段落，只是候选；是否真有联系，要读原文判断，或交给对照读者。不接受任意文本查询。书图、逐行通道分别排序后交替取，分数不可跨通道比较；最多10本书，每本最多3行。"""
            return self.similar_passages(book, line)
        return [*super().tools(), similar_passages]
