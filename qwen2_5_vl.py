import os
import cv2
import peft
import fitz
import torch
import flask
import pickle
import argparse
import requests
import threading
import numpy as np
import transformers
from concurrent.futures import Future

# -------------------------------------------------------------------------------------------------------------------- #
# 注意批量预测时，图片的形状不一致(有填充)会影响效果
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|qwen2.5_vl|')
parser.add_argument('--input_size', default=960, type=int, help='|输入图片高度|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
parser.add_argument('--flask_start', default=False, type=bool)
parser.add_argument('--port', default=9980, type=int)
parser.add_argument('--model_dir', default='', type=str)
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
class predict_class:
    def __init__(self, model_dir=None, peft_dir='', config=None, args=args):
        model_dir = args.model_dir if model_dir is None or args.model_dir else model_dir
        config = {} if config is None else config
        self.input_size = args.input_size if config.get('input_size') is None else config['input_size']
        self.device = args.device if config.get('device') is None else config['device']
        # 模型
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.float16,
                                                                              device_map=args.device).eval()
        if os.path.exists(peft_dir):  # 加载peft模型
            self.model = peft.PeftModel.from_pretrained(self.model, peft_dir, is_trainable=False)
        self.processor = transformers.AutoProcessor.from_pretrained(model_dir, use_fast=True, padding_side='left')
        # 提示词模板
        self.template = ('<|im_start|>system\n{system}<|im_end|>\n'
                         '<|im_start|>user\n{image_template}{text_input}<|im_end|>\n'
                         '<|im_start|>assistant\n')
        self.system_template = ('You are a helpful assistant.')
        self.image_template = '<|vision_start|><|image_pad|><|vision_end|>'
        self.generation_config = {'max_new_tokens': 128}
        # flask
        if args.flask_start:
            import logging
            werkzeug_log = logging.getLogger('werkzeug')
            werkzeug_log.setLevel(logging.ERROR)
            self.lock = threading.Lock()  # 线程锁
            self.request_queue = []  # 请求队列
            self.state = True  # 模型是否空闲
            self.batch = 12
            app = flask.Flask(__name__)
            app.add_url_rule('/', methods=['POST'], view_func=self._flask_request)
            app.run(host='127.0.0.1', port=args.port, threaded=True)

    def __call__(self, text, image=None, system=None, generation_config=None):
        # 输入处理
        if not isinstance(text, list):
            text = [text]
        if not isinstance(image, list):
            image = [image]
        if not isinstance(system, list):
            system = [system] if system is not None else ['']
        system = [self.system_template + _ for _ in system]
        input_prompt = []
        input_image = []
        for index in range(len(image)):  # 注意批量预测时，图片的形状不一致(有填充)会影响效果
            if image[index] is None:  # 只传入文本
                input_prompt.append(self.template.format(system=system[index], image_template='',
                                                         text_input=text[index]))
            else:
                input_image.append(self._image_process(image[index]))
                input_prompt.append(self.template.format(system=system[index], image_template=self.image_template,
                                                         text_input=text[index]))
        input_image = None if len(input_image) == 0 else input_image
        input_dict = self.processor(text=input_prompt, images=input_image, padding=True,
                                    return_tensors='pt', add_special_tokens=False).to(self.device)
        # 模型预测
        if generation_config is None:
            generation_config = self.generation_config
        input_dict.update(generation_config)
        pred = self.model.generate(**input_dict)
        # 输出处理
        pred = [_[len(__):] for _, __ in zip(pred, input_dict['input_ids'])]
        output = self.processor.batch_decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output

    @staticmethod
    def flask_call(text, image=None, system='', port=9980):  # 发送请求
        request = {'text': text, 'image': image, 'system': system}
        binary_data = pickle.dumps(request)
        response = requests.post(f'http://127.0.0.1:{port}/', data=binary_data)
        result = response.text
        return result

    def _flask_request(self):  # 接收请求并执行功能
        binary_data = flask.request.get_data()
        input_dict = pickle.loads(binary_data)
        future = Future()  # 异步返回结果接口
        with self.lock:  # 让所有请求排队
            self.request_queue.append({'input_dict': input_dict, 'future': future})
            if self.state:  # 模型空闲
                if len(self.request_queue) >= self.batch:  # 队列太多时立即处理
                    self.state = False
                    self._flask_batch()
                elif len(self.request_queue) > 0:  # 延迟启动
                    self.state = False
                    threading.Timer(2, self._flask_batch).start()
        return future.result()

    def _flask_batch(self):  # 接收请求并执行功能
        with self.lock:
            input_batch = self.request_queue[:self.batch].copy()
            self.request_queue = self.request_queue[self.batch:]
            text = [_['input_dict'].get('text', '') for _ in input_batch]
            image = [_['input_dict'].get('image', None) for _ in input_batch]
            system = [_['input_dict'].get('system', '') for _ in input_batch]
            result = self.__call__(text=text, image=image, system=system)
            for input_batch_, result_ in zip(input_batch, result):
                input_batch_['future'].set_result(result_)
            self.state = True  # 模型空闲

    def _image_process(self, image, gray=False):  # 输入处理
        if isinstance(image, str):  # 路径
            if os.path.splitext(image)[1].lower() == '.pdf':  # pdf
                document = fitz.open(image)
                page = document.load_page(0)
                max_size = max(page.rect.width, page.rect.height)
                scale = min(4096, 3 * max_size) / max_size
                image = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                image = np.frombuffer(image.samples, dtype=np.uint8).reshape(image.height, image.width, image.n)
            else:  # 图片
                image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        h, w, _ = image.shape
        # 尺寸变形
        if max(w, h) != self.input_size:
            scale = w / h
            max_size = max(min(self.input_size, int(max(w, h) * 1.2 // 32) * 32), 160)
            if scale >= 1:
                resize_w = max_size
                resize_h = min(int(resize_w / scale // 32) * 32, max_size)
            else:
                resize_h = max_size
                resize_w = min(int(resize_h * scale // 32) * 32, max_size)
            image = cv2.resize(image, (resize_w, resize_h))
        # 图片处理
        if gray:
            image = np.min(image, axis=2)
            image = np.stack([image, image, image], axis=2)
        return image


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    text = f'根据图片找到物体的零件名称、材料、表面处理，没有找到为无。'
    system = ('''你是一个机加工工程师，需要你识别图纸中的信息。\n
    提示：零件名称、材料、表面处理、热处理、料号、项目编号、类别号不要混淆；没有找到为无。\n
    你要严格按照格式回答。\n回答格式：{"零件名称":零件名称,"材料":材料,"表面处理":表面处理,"热处理":热处理}''')
    model_dir = r'Qwen2.5-VL-3B-Instruct'
    peft_dir = r'peft_last'
    model = predict_class(model_dir, peft_dir)
    result = model([text, text], ['a.jpg', 'b.jpg'], [system, system])
    print(result)
