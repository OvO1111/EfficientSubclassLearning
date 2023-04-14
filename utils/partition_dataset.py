import os, json, shutil
import random
from os.path import *

path = '/data/dailinrui/dataset/refuge2020'
new_path = '/data/dailinrui/dataset/refuge2020_trainExpand'

with open(join(path, 'train.list'), 'r') as fp:
    all_list = [x for x in fp.readlines() if x.startswith('n')]
    
random.shuffle(all_list)
train_list = all_list[:round(0.7 * len(all_list))]
val_list = all_list[round(0.7 * len(all_list)):round(0.8 * len(all_list))]
test_list = all_list[round(0.9 * len(all_list)):]

os.makedirs(join(new_path, 'data'), exist_ok=True)

with open(join(new_path, 'train.list'), 'w') as fp:
    fp.writelines(train_list)
with open(join(new_path, 'val.list'), 'w') as fp:
    fp.writelines(val_list)
with open(join(new_path, 'test.list'), 'w') as fp:
    fp.writelines(test_list)
mapping = {1: [1, 2]}
with open(join(path, 'mapping.json'), 'w') as fp:
    json.dump(mapping, fp)
    
for h5 in os.listdir(join(path, 'data')):
    if not h5.startswith('n'):
        continue
    shutil.copy(join(path, 'data', h5), join(new_path, 'data', h5))
    print(join(path, 'data', h5), end='\r')