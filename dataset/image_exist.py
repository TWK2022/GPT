import os
import json

# -------------------------------------------------------------------------------------------------------------------- #
# 去除图片不存在的数据
# 图片传输的过程中可能会丢失
# -------------------------------------------------------------------------------------------------------------------- #
label_path = 'data_demo.json'
with open(label_path, 'r', encoding='utf-8') as f:
    load = json.load(f)
new_list = []
for dict_ in load:
    if dict_['image_path'] is not None and os.path.exists(dict_['image_path']):
        new_list.append(dict_)
    else:
        print(dict_['image_path'])
with open(label_path, 'w', encoding='utf-8') as f:
    json.dump(new_list, f, ensure_ascii=False)
