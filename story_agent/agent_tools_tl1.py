"""TL1B v0.85.0: read-only quest and catalogue tools; no model or prompt changes."""
from collections import Counter
import hashlib
from pathlib import Path
import re

from langchain_core.tools import tool

from story_agent.strong_agent_sa1 import Books

VERSION = 'v0.85.0'


def speaker_for_line(line):
    """Report a textual label, never infer whether a character's claim is true."""
    if (line.startswith('#') or re.match(r'^Objective\b', line)
            or re.match(r'^\(Quest is part of (?:chapter|group):', line)):
        return '结构'
    if ((line.startswith('(') and line.endswith(')'))
            or (line.startswith('（') and line.endswith('）'))):
        return '场景'
    match = re.match(r'^([^:：]+): ', line)
    return match[1] if match else ''


def unread_notice(total, end, noun):
    return f'本{noun}共 {total} 行，本次读到第 {end} 行，未读完。' if end < total else ''


def chinese_number(text):
    digits = dict(zip('零一二三四五六七八九', range(10)))
    if '十' in text:
        left, right = text.split('十', 1)
        return (digits[left] if left else 1) * 10 + (digits[right] if right else 0)
    return digits[text]


def series_key(title):
    """Strip explicit ordinal endings, keeping the ordinal for natural ordering."""
    match = re.search(r'(?:[·・\s]*其(?:之)?|[·・])([一二三四五六七八九十]{1,3})$', title)
    if match:
        try:
            return title[:match.start()].rstrip('·・ '), chinese_number(match[1])
        except (KeyError, ValueError):
            pass
    match = re.search(r'[·・\s]*([上中下])$', title)
    if match:
        return title[:match.start()].rstrip('·・ '), {'上': 1, '中': 2, '下': 3}[match[1]]
    return title, 0


class Quests:
    """Load UTF-8 once from the fixed corpus; later calls cannot open paths."""
    def __init__(self, root):
        root = Path(root).resolve()
        corpus = root / '.runtime/istaroth-checkpoint/text/agd_quest'
        for path in (corpus, *corpus.parents):
            if path == root:
                break
            if path.is_symlink() or path.is_junction():
                raise ValueError('Quest corpus directory link rejected')
        if not corpus.is_dir():
            raise ValueError('Fixed quest corpus missing')
        self.lines, self.aliases, self.hashes = {}, {}, {}
        for path in sorted(corpus.glob('*.txt')):
            if (path.is_symlink() or path.is_junction() or not path.is_file()
                    or path.resolve().parent != corpus):
                raise ValueError('Quest outside fixed corpus')
            raw = path.read_bytes()
            lines = raw.decode('utf-8').splitlines()
            self.lines[path.name] = lines
            self.hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            aliases = {path.stem, re.sub(r'^\d+_', '', path.stem)}
            number = re.match(r'^\d+(?=_)', path.name)
            if number:
                aliases.add(number[0])
            heading = next((s[2:] for s in lines if s.startswith('# ')), None)
            if heading:
                aliases.add(heading)
            for alias in aliases:
                self.aliases.setdefault(alias, []).append(path.name)
        if not self.lines:
            raise ValueError('Fixed quest corpus empty')

    def search_quests(self, query):
        if not query:
            return {'error': '请给出非空字面搜索词。'}
        matches, counts = [], Counter()
        for name, lines in self.lines.items():
            for number, line in enumerate(lines, 1):
                if query in line:
                    counts[name] += 1
                    if len(matches) < 80:
                        matches.append({'file': name, 'line': number, 'text': line,
                                        'speaker': speaker_for_line(line)})
        total = sum(counts.values())
        result = {'query': query, 'matches': matches, 'total_matches': total,
                  'truncated': total > 80}
        if total > 80:
            result['top_files'] = [{'file': name, 'matches': count}
                                   for name, count in counts.most_common(20)]
            result['notice'] = '最多返回80条；结果已截断，请换更具体的词。'
        return result

    def read_quest(self, quest, start_line=1, end_line=120):
        choices = [quest] if quest in self.lines else self.aliases.get(quest, [])
        if len(choices) != 1:
            return {'error': '未找到唯一任务；请使用候选的完整文件名。',
                    'candidates': choices[:10], 'total_candidates': len(choices)}
        name = choices[0]
        lines = self.lines[name]
        if start_line < 1 or end_line < start_line or start_line > len(lines):
            return {'error': '行号范围无效。', 'total_lines': len(lines)}
        end = min(end_line, start_line + 119, len(lines))
        truncated = end < min(end_line, len(lines))
        notices = ['单次最多120行，请继续分段读取。'] if truncated else []
        if notice := unread_notice(len(lines), end, '任务'):
            notices.append(notice)
        return {'file': name, 'total_lines': len(lines), 'start_line': start_line,
                'end_line': end, 'truncated': truncated, 'notice': ''.join(notices),
                'text': '\n'.join(f'L{i}: {lines[i-1]}' for i in range(start_line, end+1))}


class AgentTools(Books):
    """Reuse the frozen SA1 corpus/maps and methods without changing that module."""
    def __init__(self, root):
        super().__init__(root)
        self.quests = Quests(root)
        self.hashes.update(self.quests.hashes)
        self.catalogue = []
        for name, lines in self.lines.items():
            title = re.sub(r'^\d+_', '', Path(name).stem)
            series, ordinal = series_key(title)
            number = re.match(r'^\d+(?=_)', name)
            self.catalogue.append((series, ordinal, int(number[0]) if number else -1,
                                   {'file': name, 'title': title,
                                    'total_lines': len(lines), 'series': series}))
        self.catalogue.sort(key=lambda row: row[:3] + (row[3]['file'],))

    def list_books(self, keyword='', id_from=None, id_to=None):
        if id_from is not None and id_to is not None and id_from > id_to:
            return {'error': '编号范围无效。'}
        rows = [row for _, _, number, row in self.catalogue
                if keyword in row['title']
                and (id_from is None or number >= id_from)
                and (id_to is None or number <= id_to)]
        return {'books': rows[:200], 'total_matches': len(rows), 'truncated': len(rows) > 200,
                'notice': '最多返回200本；请缩小书名或编号范围。' if len(rows) > 200 else ''}

    def read_book(self, book, start_line=1, end_line=120):
        result = super().read_book(book, start_line, end_line)
        if 'error' not in result:
            result['notice'] += unread_notice(result['total_lines'], result['end_line'], '书')
        return result

    def tools(self):
        # SA1's book_map and search_books closures bind this instance, unchanged.
        original = {item.name: item for item in super().tools()}

        @tool
        def read_book(book: str, start_line: int = 1, end_line: int = 120) -> dict:
            """按精确书名或文件名读原文，行号从1开始并含空行。不给行号即从第1行读起，一次最多120行；短书可一次读完。未到书末会提示未读完。"""
            return self.read_book(book, start_line, end_line)

        @tool
        def search_quests(query: str) -> dict:
            """字面搜索任务与主线的对话和叙述原文，最多80条。场景说明、系统说明与人物台词不同；人物台词只表示该人物这样说，所指对象要看上下文。截断时附命中最多的文件，请换更具体的词。"""
            return self.quests.search_quests(query)

        @tool
        def read_quest(quest: str, start_line: int = 1, end_line: int = 120) -> dict:
            """按完整文件名、唯一标题或唯一编号读取任务与主线原文。行号从1开始并含空行，默认1至120行，一次最多120行；保留说话人、场景说明和Objective。人物台词只表示该人物这样说，所指对象要看上下文。"""
            return self.quests.read_quest(quest, start_line, end_line)

        @tool
        def list_books(keyword: str = '', id_from: int | None = None,
                       id_to: int | None = None) -> dict:
            """浏览全部书籍目录，按书名关键词、编号闭区间或两者筛选；列文件名、书名、总行数与系列，同系列按序号连续排列，最多200本。书名不确定时先用本工具查，不要猜文件名。"""
            return self.list_books(keyword, id_from, id_to)

        return [original['book_map'], original['search_books'], read_book,
                search_quests, read_quest, list_books]
