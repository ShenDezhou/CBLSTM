import codecs

import os


posindex = 1
with codecs.open("ctrip_hotel_pos_raw.utf8", 'w', encoding='utf-8') as fw:
    with codecs.open("ctrip_hotel_pos_group_raw.utf8", 'w', encoding='utf-8') as fg:
        for filename in os.listdir(u"../hotelcomment/正面"):
            if filename.endswith(".txt"):
                with codecs.open("../hotelcomment/正面/" + filename, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    lines = [line.strip() for line in lines if len(line.strip())>0]
                    for g in [posindex] * len(lines):
                        fg.write(str(g)+'\t')
                    posindex += 1
                    for line in lines:
                        fw.write(line+"\n")

negindex = -1
with codecs.open("ctrip_hotel_neg_raw.utf8", 'w', encoding='utf-8') as fw:
    with codecs.open("ctrip_hotel_neg_group_raw.utf8", 'w', encoding='utf-8') as fg:
        for filename in os.listdir(u"../hotelcomment/负面"):
            if filename.endswith(".txt"):
                with codecs.open(u"../hotelcomment/负面/" + filename, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    lines = [line.strip() for line in lines if len(line.strip()) > 0]
                    for g in [negindex] * len(lines):
                        fg.write(str(g) + '\t')
                    negindex -= 1
                    for line in lines:
                        fw.write(line + "\n")
print("FIN")