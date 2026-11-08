"""PT0 v0.92.1: §11 filtered views, immutable raw snapshots, curl unchanged."""
import hashlib
import json
from pathlib import Path
import re

from story_agent.external_pt0 import Wikipedia as OriginalWikipedia, HEADER
from story_agent.agent_tools_tl1 import unread_notice

VERSION='v0.92.1'
KEYWORD=re.compile(r'原神|Genshin|米哈游|miHoYo|HoYoverse',re.I)
CULTURE=re.compile(r'流行文化|大众文化|大眾文化|衍生作品|in\s+popular\s+culture|popular\s+culture|in\s+media',re.I)
HEADING=re.compile(r'^\s*(={2,6})\s*(.*?)\s*\1\s*$')
GAME_THEME=re.compile(r'游戏|遊戲|角色|character|video\s*game|role.playing|playable',re.I)
SENTENCE=re.compile(r'.+?(?:[。！？!?]+[”’」』\"\']*|\.(?=\s|$)|$)',re.S)
GAME_REJECTION='该条目以游戏为主题，不在可读范围内'


def game_topic(title,text):
    paragraphs=re.split(r'\n\s*\n',text.strip(),maxsplit=1)
    first=paragraphs[0] if paragraphs else ''
    return bool(KEYWORD.search(title) or (KEYWORD.search(first) and GAME_THEME.search(first)))


def filtered_view(title,text):
    """Keep physical line numbers; deleted text becomes empty, never a marker."""
    original=text.splitlines()
    kept=list(original);deletions=[];section=None;i=0
    while i<len(original):
        line=original[i];heading=HEADING.match(line)
        if heading and section is not None and len(heading[1])<=section:
            section=None
        if heading and CULTURE.search(heading[2]) and section is None:
            section=len(heading[1])
        if section is not None:
            kept[i]=''
            deletions.append({'source_lines':[i+1],'original_sentence':line,
                              'reason':'popular_culture_section'})
            i+=1;continue
        if not line.strip() or heading:
            if KEYWORD.search(line):
                kept[i]=''
                deletions.append({'source_lines':[i+1],'original_sentence':line,'reason':'keyword'})
            i+=1;continue
        end=i+1
        while end<len(original) and original[end].strip() and not HEADING.match(original[end]):
            end+=1
        paragraph='\n'.join(original[i:end]);chars=list(paragraph)
        for match in SENTENCE.finditer(paragraph):
            if not KEYWORD.search(match[0]):continue
            start_line=i+1+paragraph[:match.start()].count('\n')
            stop_line=i+1+paragraph[:match.end()].count('\n')
            deletions.append({'source_lines':list(range(start_line,stop_line+1)),
                              'original_sentence':match[0],'reason':'keyword'})
            for n in range(match.start(),match.end()):
                if chars[n]!='\n':chars[n]=''
        kept[i:end]=''.join(chars).split('\n')
        i=end
    assert len(kept)==len(original)
    return {'version':VERSION,'rejected':game_topic(title,text),'lines':kept,
            'deleted_lines':sorted({n for d in deletions for n in d['source_lines']}),
            'deletions':deletions,'source_line_numbers_preserved':True}


class Wikipedia(OriginalWikipedia):
    def __init__(self,*args,filter_log=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.filter_events=[]
        self.filter_log=Path(filter_log) if filter_log else None

    def record_filter(self,value):
        self.filter_events.append(value)
        if self.filter_log:
            self.filter_log.parent.mkdir(parents=True,exist_ok=True)
            with self.filter_log.open('a',encoding='utf-8',newline='\n') as stream:
                stream.write(json.dumps(value,ensure_ascii=False)+'\n')

    def view(self,snapshot):
        info=snapshot['metadata']
        raw_path,_=self.paths(info['title'],info['lang'])
        raw=raw_path.read_bytes()
        view=filtered_view(info['title'],raw.decode('utf-8'))
        data='\n'.join(view['lines']).encode('utf-8')
        txt=raw_path.with_suffix('.filtered.txt');meta=raw_path.with_suffix('.filtered.json')
        record={'title':info['title'],'lang':info['lang'],'pageid':info['pageid'],
                'revision_id':info['revision_id'],'raw_sha256':info['sha256'],
                'filtered_sha256':hashlib.sha256(data).hexdigest(),**view}
        if txt.exists() or meta.exists():
            if not txt.exists() or not meta.exists():raise ValueError('Incomplete filtered snapshot')
            if txt.read_bytes()!=data or json.loads(meta.read_text(encoding='utf-8'))!=record:
                raise ValueError('Filtered snapshot differs; preserve evidence')
        else:
            txt.write_bytes(data)
            with meta.open('x',encoding='utf-8',newline='\n') as stream:
                stream.write(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        # Audit records are deliberately never returned by the Agent tools.
        self.record_filter({'type':'page_filter',**record})
        return view,record['filtered_sha256']

    def read(self,title,lang='zh',start_line=1,end_line=120,allowed_page=None):
        if start_line<1 or end_line<start_line:return {'error':'行号范围无效。'}
        snap=self.snapshot(title,lang)
        if 'error' in snap:return snap
        view,digest=self.view(snap)
        if view['rejected']:return {'error':GAME_REJECTION}
        info,lines=snap['metadata'],view['lines']
        if start_line>len(lines):return {'error':'行号范围无效。','total_lines':len(lines)}
        if allowed_page is not None and not allowed_page((lang,info['pageid'])):
            return {'error':'本次放行的阅读页数已用完'}
        end=min(end_line,start_line+119,len(lines))
        return {'header':HEADER,**info,'snapshot_hit':snap['snapshot_hit'],
                'filter_version':VERSION,'filtered_sha256':digest,
                'total_lines':len(lines),'start_line':start_line,'end_line':end,
                'notice':unread_notice(len(lines),end,'资料'),
                'text':'\n'.join(f'L{i}: {lines[i-1]}' for i in range(start_line,end+1))}

    def search(self,query,lang='zh'):
        result=super().search(query,lang)
        if 'error' in result:return result
        values=[]
        for item in result['results']:
            view=filtered_view(item['title'],item['summary'])
            self.record_filter({'type':'search_filter','query':query,'lang':lang,
                'title':item['title'],'pageid':item['pageid'],**view})
            if not view['rejected']:
                values.append(dict(item,summary='\n'.join(view['lines']).strip()))
        return dict(result,results=values)
