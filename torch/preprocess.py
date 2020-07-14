import os
import pandas

input='2000/'
df = pandas.DataFrame()
for dir in os.listdir(input):
    subdir = os.path.join(input, dir)
    if dir=='pos':
        label = 1
    if dir=='neg':
        label = 0
    for file in os.listdir(subdir):
        with open(os.path.join(subdir,file),'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if len(line.strip()) > 0]
            content = "。".join(lines)
            row = {'content': content, 'label': str(label)}
            df = df.append(row, ignore_index=True)
            print('.')
df.to_csv("valid.csv", index=False)
print('FIN')