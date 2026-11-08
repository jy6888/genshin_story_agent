"""TL5b v0.94.2 feedback sentence; unchanged format diagnostics."""
import json
from pathlib import Path
from story_agent import report_tl5b as previous
from story_agent.name_check_tl5b import NameCheck as PreviousCheck,feedback as previous_feedback
from story_agent.report_tl5 import feedback as format_feedback
from story_agent.strong_agent_sa1 import save

EXTRA='补进报告的新结论，重新交稿前先交对照读者核对。'
check=previous.check


def feedback(value):return previous_feedback(value)+EXTRA


class NameCheck(PreviousCheck):
    def feedback_once(self,result):
        message=super().feedback_once(result)
        return message+EXTRA if message else None


def deliver(answer,directory,sources,metadata,feedback_count):
    result=previous.deliver(answer,directory,sources,metadata,feedback_count)
    result['metadata'].update(configuration='v0.94.2',postprocessor_version='v0.94.2')
    path=Path(directory)/'report.md'
    text=path.read_text(encoding='utf-8').rsplit('## 附录 B：运行信息',1)[0]
    path.write_text(text+'## 附录 B：运行信息\n\n```json\n'+json.dumps(result['metadata'],ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    return result


def finish(answer,question,records,invoke,sources,directory,metadata):
    controller=NameCheck(question)
    initial=controller.examine(answer,records());save(Path(directory)/'name-initial.json',initial)
    message=controller.feedback_once(initial)
    if message:
        save(Path(directory)/'name-feedback.json',{'feedback':message})
        answer=invoke(message);controller.examine(answer,records())
    save(Path(directory)/'name-resubmitted.json',controller.checks[-1])
    errors=check(answer,sources);save(Path(directory)/'format-initial.json',dict(answer=answer,errors=errors))
    count=int(bool(errors))
    if errors:
        message=format_feedback(errors);save(Path(directory)/'format-feedback.json',{'feedback':message})
        answer=invoke(message)
    controller.examine(answer,records())
    result=deliver(answer,directory,sources,dict(metadata(),**controller.metadata()),count)
    save(Path(directory)/'format-result.json',dict(errors=result['errors'],feedback_count=count,references=result['references']))
    return answer,result
