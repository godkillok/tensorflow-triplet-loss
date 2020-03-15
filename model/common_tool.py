
#coding=utf-8
import re
import json
punctuation = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~。，"""
def clean_str(text):
    text = text.strip()
    text = re.sub(r'[{}]+'.format(punctuation), ' ', str(text))
    text=' '.join(text.split())
    text=text.lower()
    return text

def per_line(line):
    li = line.split("__label__")
    labels=[]
    for l in li[1:]:
        punctuation = r"""0123456789"""
        text = re.sub(r'[{}]+'.format(punctuation), ' ', str(l))
        text = ' '.join(text.split())
        text = text.lower()
        text = ' '.join(text.split("\x01_"))
        text = ' '.join(text.split())
        labels.append(text)

    punctuation = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~。，"""
    text = line.split("__label__")[0].strip()
    text = re.sub(r'[{}]+'.format(punctuation), ' ', str(text))
    text = ' '.join(text.split())
    text = re.sub(
        '[.com\u200b\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+',
        '', text)
    #         print(str)
    tokens = text.lower()
    # print(tokens)
    tokens = tokens.split()
    tokens = [w.strip() for w in tokens if len(w.strip()) > 0 and not w.isdigit()]
    return tokens,labels
#1