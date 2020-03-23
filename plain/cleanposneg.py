import codecs

with codecs.open("ctrip_hotel_neg_raw.utf8", "r", encoding='utf-8') as fn:
    with codecs.open("ctrip_hotel_neg_group_raw.utf8", "r", encoding='utf-8') as fng:
        with codecs.open("ctrip_hotel_neg.utf8", "w", encoding='utf-8') as fnf:
            with codecs.open("ctrip_hotel_neg_group.utf8", "w", encoding='utf-8') as fngf:

                lines = fn.readlines()
                glines = fng.readlines()[0].strip().split("\t")
                assert len(lines) == len(glines)
                lset = set()
                filteredlines = []
                fllteredg = []

                for i in range(len(lines)):
                    if '免费注册 网站导航 宾馆索引 服务说明 关于携程 诚聘英才 代理合作 广告业务 联系我们' in lines[i] or 'Copyright' in lines[i]:
                        continue

                    if lines[i] not in lset:
                        filteredlines.append(lines[i])
                        fllteredg.append(glines[i])
                        lset.add(lines[i])

                for line in filteredlines:
                    fnf.write(line)
                for s in fllteredg:
                    fngf.write(s)
                    fngf.write('\t')

with codecs.open("ctrip_hotel_pos_raw.utf8", "r", encoding='utf-8') as fp:
     with codecs.open("ctrip_hotel_pos_group_raw.utf8", "r", encoding='utf-8') as fpg:
         with codecs.open("ctrip_hotel_pos.utf8", "w", encoding='utf-8') as fnf:
             with codecs.open("ctrip_hotel_pos_group.utf8", "w", encoding='utf-8') as fngf:

                 lines = fp.readlines()
                 glines = fpg.readlines()[0].strip().split("\t")
                 assert len(lines) == len(glines)
                 lset = set()
                 filteredlines = []
                 fllteredg = []

                 for i in range(len(lines)):
                     if '免费注册 网站导航 宾馆索引 服务说明 关于携程 诚聘英才 代理合作 广告业务 联系我们' in lines[i] or 'Copyright' in lines[i]:
                         continue

                     if lines[i] not in lset:
                         filteredlines.append(lines[i])
                         fllteredg.append(glines[i])
                         lset.add(lines[i])

                 for line in filteredlines:
                     fnf.write(line)
                 for s in fllteredg:
                     fngf.write(s)
                     fngf.write('\t')
print("FIN")