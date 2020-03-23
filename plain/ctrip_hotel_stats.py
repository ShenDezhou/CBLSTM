import codecs

import os

with codecs.open('ctrip_hotel_pos.utf8', 'r', encoding='utf8') as fp:
    with codecs.open('ctrip_hotel_neg.utf8', 'r', encoding='utf8') as fn:
        x = fp.readlines()
        y = fn.readlines()
        x = [len(i) for i in x]
        y = [len(i) for i in y]
        print(max(x), max(y))
print("FIN")