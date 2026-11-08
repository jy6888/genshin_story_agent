"""TL4b §2 literal line-reference rule; pending historical replay acceptance."""
import re

VERSION = 'v0.91.0'
LINE_REFERENCE = re.compile(r'L[0-9]+(?:[-–—][0-9]+)?|第[0-9]+行|[0-9]+[-–—][0-9]+行')
REJECTION = '没有启动对照读者：问题里没有找到要比较或核对的具体内容。请写明每条要核对的结论（或要比较的两处材料），以及依据的书名、任务名或文本名与行号，再调用。'


def delegation_error(description):
    """Return the exact zero-model-call refusal, or None when a reference exists."""
    return None if LINE_REFERENCE.search(description) else REJECTION
