import torch
import argparse
import transformers

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|模型预测|')
parser.add_argument('--model_path', default='chinese-alpaca-2-1.3b', type=str, help='|tokenizer和模型文件夹位置|')
parser.add_argument('--model', default='llama2', type=str, help='|模型类型|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
class predict_class:
    def __init__(self, args):
        self.device = args.device
        self.generation_config = transformers.GenerationConfig(max_new_tokens=1024, do_sample=True, temperature=0.5)
        self.record = 0
        self.record_list = []
        if args.model == 'llama2':
            self.tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path)
            self.model = transformers.LlamaForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True,
                                                                       torch_dtype=torch.float16).eval()
            self.model = self.model.float() if args.device.lower() == 'cpu' else self.model.half()
            self.model = self.model.to(self.device)
            self.system = 'You are a helpful assistant. 你是一个乐于助人的助手。'  # 默认系统提示
            self.template = ('<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{input} [/INST]')  # 单轮对话提示模版
            self.template_add = ' {output_add}</s><s>[INST] {input_add} [/INST]'  # 多轮对话追加的提示模版
        elif args.model == 'baichuan2':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,
                                                                           torch_dtype=torch.float16).eval()
            self.model = self.model.float() if args.device.lower() == 'cpu' else self.model.half()
            self.model = self.model.to(self.device)
            self.system = ''  # 默认系统提示
            self.template = '{system}<reserved_106>{input}<reserved_107>'  # 单轮对话提示模版
            self.template_add = '{output_add}<reserved_106>{input_add}<reserved_107>'  # 多轮对话追加的提示模版
        elif args.model == 'qwen':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,
                                                                           torch_dtype=torch.float16).eval()
            self.model = self.model.float() if args.device.lower() == 'cpu' else self.model.half()
            self.model = self.model.to(self.device)
            self.system = 'You are a helpful assistant.'  # 默认系统提示
            self.template = ('<|im_start|>{system}<|im_end|>\n<|im_start|>user{input}<|im_end|>\n'
                             '<|im_start|>assistant')  # 单轮对话提示模版
            self.template_add = ('{output_add}<|im_end|>\n<|im_start|>user{input}<|im_end|>\n'
                                 '<|im_start|>assistant')  # 多轮对话追加的提示模版

    def predict(self, input_, generation_config=None):  # 输入字符串
        generation_config = generation_config if generation_config else self.generation_config
        with torch.no_grad():
            prompt = self.template.format(system=self.system, input=input_)
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            pred = self.model.generate(input_ids=input_ids, generation_config=generation_config)
            result = self.tokenizer.decode(pred[0], skip_special_tokens=True)
        return result


if __name__ == '__main__':
    text = '你好呀'
    model = predict_class(args)
    result = model.predict(text)
    print(result)
