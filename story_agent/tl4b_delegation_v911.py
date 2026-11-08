"""TL4b D-20261107-002: concrete source reference gate (codex/107 §9)."""
import re
from story_agent.tl4b_delegation import REJECTION

VERSION = 'v0.91.1'
LINE = re.compile(r'(L\s*\d+|第\s*\d+\s*(?:[-–—~～至到]\s*\d+\s*)?行|行\s*(?:号)?\s*(?:约|大约)?\s*\d+|\d+\s*(?:[-–—~～至到]\s*\d+\s*)?行)')
SRC = re.compile(r'(《[^》]{1,40}》|\d{3,}_[^\s，。、（）()]+|\.txt\b)')


def source_reference(description):
    return LINE.search(description) or SRC.search(description)


def delegation_error(description):
    return None if source_reference(description) else REJECTION
