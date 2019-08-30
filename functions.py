import html
import unicodedata
import re

def extract_id(s):
    regex = re.compile(r'\d*$')
    return regex.findall(s)[0]

def strip_element(s):
    regex = re.compile(r'<[^>]+>')
    s = html.unescape(s)
    s = regex.sub('', s)
    s = s.replace('\\r\\n', '')
    s = s.replace('\t', '')
    s = html.unescape(s)
    s = unicodedata.normalize("NFKD", s)
    return s

def strip_response(r):
    s = ''
    for d in r:
        for k, v in d.items():
            s += strip_element(v)
    return s