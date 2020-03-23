import codecs

chars = []

def readchar(file):
    chars = []
    with codecs.open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            for ch in line:
                if ch == ' ' or ch == '\r' or ch == '\n':
                    continue
                else:
                    chars.append(ch)
    return chars

chars.extend(readchar("ctrip_hotel_neg.utf8"))
chars.extend(readchar("ctrip_hotel_pos.utf8"))

chars = list(set(chars))
chars.sort()
print(len(chars))
with codecs.open('../ctrip_dic/ctrip_dict.utf8', 'w', encoding='utf8') as f:
    for c in chars:
        f.write(c)

print("FIN")
