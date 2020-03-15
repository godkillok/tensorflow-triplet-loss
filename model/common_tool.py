
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

def parse_line_dict(tokens,labels,vocab_dict,label_dict,OOV):

    print(tokens)
    print(labels)
    text = [vocab_dict.get(r,OOV) for r in tokens]
    # labels=labels[:12]
    if len(labels) >= 12:
        labels = labels[0: 12]
    else:
        labels += ['-111'] * (12 - len(labels))
    tags=[]
    for lab in labels:
        tag=[]
        for la in lab.split(' '):
            # if la not in vocab_dict:
            #     print("'{}' not exist".format(la))
            tag.append(vocab_dict.get(la,0))
        tags.append(tag)
    labels=[label_dict.get(lab,-1) for lab in labels]
    print([text,labels,tags])
    return [text,labels,tags]


def ini(path_vocab,path_label,pad_word,OOV):
    with open(path_vocab, 'r', encoding='utf8') as f:
        lines = f.readlines()
        vocab_dict = {l.strip(): (i) for i, l in enumerate(lines)}
        pad_word=vocab_dict.get(pad_word)
        OOV=vocab_dict.get(OOV)
        print("pad_word {},OOV {} --vocab_dict {}".format(pad_word,OOV,len(vocab_dict)))

    with open(path_label, 'r', encoding='utf8') as f:
        lines = f.readlines()
        label_dict = {l.strip().split("\x01\t")[0]: i for i, l in enumerate(lines)}

    return vocab_dict,label_dict
