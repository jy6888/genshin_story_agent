"""TL5b finalization: name check before one format feedback, derived appendices."""
import json
import re
from pathlib import Path
from story_agent import report_tl5 as base
from story_agent import report_tl5_v934 as frozen
from story_agent.name_check_tl5b import NameCheck
from story_agent.strong_agent_sa1 import save

SUMMARY_RULE = base.APPENDIX.split('第一部分是', 1)[1].split('第二部分是', 1)[0]


def check(answer, sources):
    errors = frozen.check(answer, sources)
    summary, _ = base.split(answer)
    if not summary.strip():
        errors.append('位置：分隔行之前；问题：缺对话总结；允许写法：第一部分是'+SUMMARY_RULE)
    return errors


def deliver(answer, directory, sources, metadata, feedback_count):
    result = frozen.deliver(answer, directory, sources, metadata, feedback_count)
    result['errors'] = check(answer, sources)
    summary, _ = base.split(answer)
    result['metadata'].update(configuration='v0.94.0', postprocessor_version='v0.94.0',
        final_invalid_count=len(result['errors']), format_errors=result['errors'],
        summary_missing=not summary.strip(), summary_status='总结缺失' if not summary.strip() else '已生成')
    directory = Path(directory)
    if not summary.strip():
        (directory/'summary.md').write_text('（本次未生成对话总结）\n完整报告：'+str((directory/'report.md').resolve())+'\n', encoding='utf-8')
    path = directory/'report.md'
    text = path.read_text(encoding='utf-8').rsplit('## 附录 B：运行信息', 1)[0]
    path.write_text(text+'## 附录 B：运行信息\n\n```json\n'+json.dumps(result['metadata'],ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    return result


def finish(answer, question, records, invoke, sources, directory, metadata):
    """invoke keeps all normal tools enabled for both authorized feedback turns."""
    controller = NameCheck(question)
    initial = controller.examine(answer, records())
    save(Path(directory)/'name-initial.json', initial)
    message = controller.feedback_once(initial)
    if message:
        save(Path(directory)/'name-feedback.json', {'feedback': message})
        answer = invoke(message)
        controller.examine(answer, records())
    save(Path(directory)/'name-resubmitted.json', controller.checks[-1])
    errors = check(answer, sources)
    save(Path(directory)/'format-initial.json', {'answer':answer,'errors':errors})
    count = int(bool(errors))
    if errors:
        message = base.feedback(errors)
        save(Path(directory)/'format-feedback.json', {'feedback':message})
        answer = invoke(message)
    controller.examine(answer, records())
    meta = dict(metadata(), **controller.metadata())
    result = deliver(answer, directory, sources, meta, count)
    save(Path(directory)/'format-result.json', {'errors':result['errors'],'feedback_count':count,'references':result['references']})
    return answer, result
