import json


def data_get(args):
    data_dict = data_prepare(args).load()
    return data_dict


class data_prepare(object):
    def __init__(self, args):
        self.divide = args.divide
        self.data_path = args.data_path

    def load(self):
        # 读取数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        data_len = len(data_list)  # 输入数据的数量
        boundary = int(data_len * self.divide[0] / (self.divide[0] + self.divide[1]))  # 数据划分
        train = data_list[:boundary]
        val = data_list[boundary:]
        data_dict = {'train': train, 'val': val}
        return data_dict
