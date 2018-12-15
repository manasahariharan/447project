import re
import io
import pandas as pd
chaps = []
chapcount = -1
start = False
chaplabels = []
chaplabels1 = []
p1 = re.compile('    \[\d*\]')
p2 = re.compile('\[\d*\]')
with io.open("dangerous.txt", mode="r", encoding="utf-8") as f:
    for line in f:
        line = line.strip('\n')
        if line == "LETTRE PREMIÈRE":
            start = True
        if start == True and line.startswith("LETTRE") and not(line.endswith(']')):
            if p1.match(line):
                print(line)
                line = next(f).strip('\n')
            
            chapcount += 1
            chaps.append([])
            next(f)
            chaplabels.append(next(f).strip())
            next(f)
            next(f)
        if start:
            if line.startswith("    [57]"):
                break
            if p1.match(line):
                line = next(f).strip('\n')
                while line.startswith('    '):
                    line = next(f).strip('\n')
            
            if line != "" and not(line.startswith("LETTRE")):
                #print(line)
                line = ''.join([i for i in line if not i.isdigit() and i != '[' and i != ']'])
                line = line.strip('.')
                chaps[chapcount].extend(line.split(' '))
            
            
from collections import Counter

chaplabels = [re.split(r'à|au', lab.lower()) for lab in chaplabels]
chapsen = [a[0] for a in chaplabels] 
chaprec = [a[1] for a in chaplabels]
mc = [a for a in set(chapsen) if chapsen.count(a) > 2]
dellist = []
for i in range(len(chaps)):
    if (chapsen[i] not in mc):
        dellist.append(i)
print(dellist)
chapsen = [item for i,item in enumerate(chapsen) if i not in dellist]
chaps = [item for i,item in enumerate(chaps) if i not in dellist]
chapsen = [item.strip(' ') for item in chapsen]
##create dataset given chapter and author
data = pd.DataFrame({'character':chapsen,
                     'chapter': chaps})
data["Character"] = data['character'].astype('category')
cat_map = {'_cécile volanges': 'CV', '_la marquise de merteuil': 'MM', '_le vicomte de valmont': 'VV',
           '_la présidente de tourvel':'PT','_madame de volanges':'MV','_le chevalier danceny':'CD','_madame de rosemonde':'MR'}
data['Character'].cat.categories = [cat_map[g] for g in data['Character'].cat.categories]
data = data.drop(['character'], axis=1)
data.to_csv('proc_data.csv')
print(data.loc[0])