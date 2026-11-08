"""D-008 self-repair: resolve literal corpus aliases and give usable source candidates."""
import re
from pathlib import Path
from story_agent.report_tl5 import Sources as Previous,LINE

class Sources(Previous):
    def __init__(self,books):
        super().__init__(books)
        self.character_names={}
        for name in books.texts.lines:
            short=re.sub(r'^\d+_','',Path(name).stem)
            if name.startswith('agd_character_story/'):
                self.character_names.setdefault(short,{})['story']=name
                for label in ('角色故事','角色故事1','角色故事2','角色故事3','角色故事4','角色故事5','角色故事6','角色故事7','角色故事8'):
                    self.aliases.setdefault(short+label,[]).append(name)
            elif name.startswith('agd_voiceline/'):
                self.character_names.setdefault(short,{})['voice']=name
                for label in ('语音','角色语音'):
                    self.aliases.setdefault(short+label,[]).append(name)
        self.sorted_aliases=sorted(self.aliases,key=len,reverse=True)

    def references(self,text):
        output=[];last=None
        for match in re.finditer(LINE,text):
            start,end=int(match[1]),int(match[2] or match[1])
            prefix=text[max(0,match.start()-160):match.start()]
            clean=prefix.rstrip(' `*（(:：')
            # Exact suffix matching against the immutable corpus aliases; no semantic inference.
            found=None
            for alias in self.sorted_aliases:
                plain=clean.endswith(alias)
                before=clean[:-len(alias)] if plain and alias else ''
                boundary=not before or not re.match(r'[\w\u3400-\u9fff]',before[-1])
                if (plain and boundary) or clean.endswith('《'+alias+'》'):
                    found=alias;break
            resolved=self.resolve(found,start,end) if found else None
            raw=(found+' '+match[0]) if found else match[0]
            if found is None and last is not None:
                between=text[last['end']:match.start()]
                if re.fullmatch(r'[\s,，、;；]*',between) and last['resolved']:
                    resolved=self.resolve(last['resolved']['file'],start,end)
                    raw=last['resolved']['file']+' '+match[0]
            if found is None and resolved is None:
                token=re.search(r'([\w\u3400-\u9fff《》·]+)\s*$',clean)
                raw=(token[1]+' ' if token else '')+match[0]
            if resolved is None:
                candidates=[]
                if found:candidates=list(dict.fromkeys(self.aliases.get(found,[])))
                if not candidates and ('角色故事' in raw or '语音' in raw):
                    category='story' if '角色故事' in raw else 'voice'
                    candidates=[v[category] for n,v in self.character_names.items() if n in text and category in v]
                if candidates:
                    raw+='（可用完整标识：'+'；'.join(n+' '+match[0] for n in candidates[:6])+'）'
            output.append({'raw':raw,'resolved':resolved})
            last={'end':match.end(),'resolved':resolved}
        return output
