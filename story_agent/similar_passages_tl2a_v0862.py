"""TL2A v0.86.2: tolerant book identity and presentation-only neighbour cleanup.

The v0.86.0 module, SQLite ranking, embeddings and source metadata stay frozen.
"""
from collections import defaultdict
import json
from pathlib import Path
import re
import unicodedata

from langchain_core.tools import tool
from story_agent.agent_tools_tl1 import AgentTools as TL1Tools
from story_agent.similar_passages_tl2a import Neighbours as FrozenNeighbours, file_sha

VERSION = 'v0.86.2'
NOTICE = ('只是意象或说法相近的候选；是否真有联系，要读原文判断，或交给对照读者。'
          '两个通道相似度不可比，仍各按书最高相似度排序，从书图开始交替取书。'
          '最多10个代表书项，每项最多3段；书图显示条目概括及行号，少于4字的原文行显示窗口。'
          '同文合并并注明“同文另见”，腾出的书项按原顺序补足。')


def normalized_title(value):
    value = value.strip()
    value = re.sub(r'\.txt$', '', value, flags=re.IGNORECASE)
    value = value.replace('《', '').replace('》', '')
    # Bracket annotations are stripped only at the edges, not inside a title.
    edge = r'[（(][^()（）]*[）)]'
    previous = None
    while previous != value:
        previous = value
        value = re.sub(rf'^\s*{edge}\s*|\s*{edge}\s*$', '', value)
    value = re.sub(r'\.txt$', '', value.strip(), flags=re.IGNORECASE)
    return ''.join(c for c in value if not c.isspace() and c != '·')


def content_length(text):
    return sum(not c.isspace() and not unicodedata.category(c).startswith('P') for c in text)


class BookLocator:
    def __init__(self, filenames):
        self.filenames = sorted(filenames)
        self.numbers, self.titles = defaultdict(list), defaultdict(list)
        for name in self.filenames:
            number, title = Path(name).stem.split('_', 1)
            self.numbers[number].append(name)
            self.titles[normalized_title(title)].append(name)

    def locate(self, value):
        if not isinstance(value, str) or not value or '/' in value or '\\' in value:
            return None, []
        if value in self.filenames:
            return value, []
        numbers = set(re.findall(r'(?<!\d)\d+(?!\d)', value))
        matches = sorted({name for n in numbers for name in self.numbers.get(n, [])})
        if len(matches) == 1:
            return matches[0], []
        if matches:
            return None, matches
        title = normalized_title(value)
        exact = self.titles.get(title, [])
        if len(exact) == 1:
            return exact[0], []
        if exact:
            return None, exact
        candidates = {n for t, names in self.titles.items() if title and (title in t or t in title)
                      for n in names}
        for number in numbers:
            # A typo/prefix suggests candidates but never silently resolves them.
            for size in range(len(number), 2, -1):
                near = {name for n, names in self.numbers.items() if n.startswith(number[:size])
                        for name in names}
                if near:
                    candidates.update(near)
                    break
        return None, sorted(candidates)


class Neighbours(FrozenNeighbours):
    def __init__(self, cache, books, root):
        super().__init__(cache, books)
        vr = Path(root)/'.runtime/vector-recall/v1'
        for filename in ('units.json', 'books.json'):
            path = vr/filename
            expected = self.manifest['sources'].get(path.relative_to(Path(root)).as_posix(),
                self.manifest['sources'].get(str(path)))
            if expected is None:
                raise ValueError('Cache manifest missing repository-relative metadata hash: ' + path.name)
            if file_sha(path) != expected:
                raise ValueError('Frozen book-map metadata changed')
        metadata = json.loads((vr/'units.json').read_text(encoding='utf-8'))
        metadata = [r for r in metadata if r['target'] in ('T1', 'T2', 'T3')]
        names = {r['document']:r['filename'] for r in json.loads((vr/'books.json').read_text(encoding='utf-8'))}
        ids = [i for i in sorted(self.units) if self.units[i][0] == '书图']
        if len(ids) != len(metadata):
            raise ValueError('Book-map metadata alignment differs')
        self.map_text = {}
        for i, row in zip(ids, metadata):
            if self.units[i][1:] != (names[row['book']], row['lines']):
                raise ValueError('Book-map metadata/unit mismatch')
            self.map_text[i] = row['text']
        self.nonempty = {name:[i for i, text in enumerate(lines, 1) if text.strip()]
                         for name, lines in self.books.lines.items()}

    def passage(self, unit, score):
        channel, name, numbers = self.units[unit]
        if channel == '书图':
            return {'book':name, 'lines':numbers, 'text':self.map_text[unit],
                    'similarity':score, 'channel':channel, 'kind':'书图条目'}
        number = numbers[0]
        text = self.books.lines[name][number-1]
        result = {'book':name, 'line':number, 'text':text, 'similarity':score,
                  'channel':channel, 'kind':'原文'}
        if content_length(text) < 4:
            nonempty = self.nonempty[name]
            pos = nonempty.index(number)
            window = nonempty[max(0,pos-1):pos+2]
            result.update(lines=window, kind='窗口',
                          text='\n'.join(self.books.lines[name][n-1] for n in window))
        return result

    def display_key(self, row):
        # Equal summaries alone do not establish equal original text.
        numbers = row.get('lines',[row.get('line')])
        original = tuple(self.books.lines[row['book']][n-1] for n in numbers)
        return row['text'], original

    def lookup(self, book, line):
        name = self.books.resolve(book)
        if name is None:
            _, candidates = self.books.locator.locate(book)
            return {'error':'未找到唯一书名；请使用候选完整文件名。',
                    'candidates':candidates[:10], 'total_candidates':len(candidates)}
        if type(line) is not int or not 1 <= line <= len(self.books.lines[name]):
            return {'error':'物理行号无效。', 'total_lines':len(self.books.lines[name])}
        candidates = {'书图':defaultdict(dict), '逐行':defaultdict(dict)}
        with self.connect() as db:
            for (query,) in db.execute('SELECT unit FROM queries WHERE book=? AND line=?',(name,line)).fetchall():
                found = db.execute('SELECT items FROM neighbors WHERE unit=?',(query,)).fetchone()
                if found is None:
                    raise ValueError('Incomplete neighbour table')
                for unit,score in json.loads(found[0]):
                    channel,other,_ = self.units[unit]
                    if other == name:
                        raise ValueError('Same-book neighbour in cache')
                    rows = candidates[channel][other]
                    rows[unit] = max(rows.get(unit,-2),score)
        ranked = {ch:sorted(rows,key=lambda n:(-max(rows[n].values()),n)) for ch,rows in candidates.items()}
        positions = dict.fromkeys(ranked,0)
        seen, displayed, result = set(), {}, []
        # This is the same alternating book stream as v0.86.0, extended on duplicates.
        while len(result) < 10:
            advanced = False
            for first in ('书图','逐行'):
                names = ranked[first]
                while positions[first] < len(names) and names[positions[first]] in seen:
                    positions[first] += 1
                if positions[first] == len(names):
                    continue
                other = names[positions[first]]
                positions[first] += 1
                seen.add(other)
                advanced = True
                channels = (first, '逐行' if first == '书图' else '书图')
                queues = {ch:sorted(candidates[ch].get(other,{}).items(),
                    key=lambda item:(-item[1], min(self.units[item[0]][2]), item[0])) for ch in channels}
                passages = []
                while len(passages) < 3 and any(queues.values()):
                    for ch,queue in queues.items():
                        if not queue:
                            continue
                        unit,score = queue.pop(0)
                        row = self.passage(unit,score)
                        key = self.display_key(row)
                        if key in displayed:
                            representative = displayed[key]
                            ref = {k:row[k] for k in ('book','channel','similarity')}
                            ref['lines'] = row.get('lines',[row.get('line')])
                            refs = representative.setdefault('same_text_elsewhere',[])
                            if ref not in refs:
                                refs.append(ref)
                            representative['notice'] = '同文另见：' + '、'.join(dict.fromkeys(r['book'] for r in refs))
                        else:
                            displayed[key] = row
                            passages.append(row)
                        if len(passages) == 3:
                            break
                if passages:
                    result.append({'book':other,'passages':passages})
                    if len(result) == 10:
                        break
            if not advanced:
                break
        return {'book':name, 'line':line, 'notice':NOTICE, 'books':result}


class AgentTools(TL1Tools):
    def __init__(self, root, cache=None):
        super().__init__(root)
        self.locator = BookLocator(self.lines)
        self.neighbours = Neighbours(cache or Path(root)/'.runtime/similar-passages/v1',self,root)

    def resolve(self, book):
        return self.locator.locate(book)[0]

    def read_book(self, book, start_line=1, end_line=120):
        name,candidates = self.locator.locate(book)
        if name is None:
            return {'error':'未找到唯一书名；请使用候选完整文件名。',
                    'candidates':candidates[:10], 'total_candidates':len(candidates)}
        return super().read_book(name,start_line,end_line)

    def similar_passages(self, book, line):
        return self.neighbours.lookup(book,line)

    def tools(self):
        @tool
        def read_book(book: str, start_line: int = 1, end_line: int = 120) -> dict:
            """按完整文件名、编号或唯一书名读原文；可忽略间隔号、空格、书名号和.txt。歧义或未找到时列候选，请用完整文件名重试。物理行号从1起含空行；默认1–120行，未读完会提示。"""
            return self.read_book(book,start_line,end_line)

        @tool
        def similar_passages(book: str, line: int) -> dict:
            """给定书名/编号与物理行号，列出其他书里意象或说法相近的段落，只是候选；是否真有联系，要读原文判断，或交给对照读者。两个通道分别排序交替取书，分数不可跨通道比较；书图显示条目概括，极短行显示窗口，同文合并注明另见，最多10个代表书项、每项最多3段。"""
            return self.similar_passages(book,line)
        return [read_book if t.name=='read_book' else t for t in super().tools()] + [similar_passages]
