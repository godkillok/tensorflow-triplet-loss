import re
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

with open("/data/tanggp/tmp/Starspace/python/test/tag_space2","r",encoding="utf8") as f:
    lines=f.readlines()

results=[]
i=0
for li in lines:
    tokens, labels=per_line(li.strip())
    results.append(" ".join(tokens))
    results+=labels
    i+=1
    if i==2:
        print(results)

with open("/data/tanggp/tmp/Starspace/python/test/tag_space_token","w",encoding="utf8") as f:
    for res in results:
        f.writelines(res+'\n')


