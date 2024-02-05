import torch
import argparse
import transformers

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|llama模型预测|')
parser.add_argument('--model_path', default='chinese-alpaca-2-1.3b', type=str, help='|tokenizer和模型文件夹位置|')
parser.add_argument('--model', default='llama2', type=str, help='|模型类型|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
class predict_class:
    def __init__(self, args):
        self.device = args.device
        # 模型
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path)
        self.model = transformers.LlamaForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True).eval()
        self.model = self.model.float() if args.device.lower() == 'cpu' else self.model.half()
        self.model = self.model.to(self.device)
        self.generation_config = transformers.GenerationConfig(do_sample=True, temperature=0.5)
        # 验证
        assert len(self.tokenizer) == self.model.vocab_size  # 检查模型和文本编码是否匹配
        # 输入提示词模版
        if args.model == 'llama2':
            self.system = 'You are a helpful assistant. 你是一个乐于助人的助手。'  # 默认系统提示
            self.template = ('<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{input} [/INST]')  # 单轮对话提示模版
            self.template_add = ' {output_add}</s><s>[INST] {input_add} [/INST]'  # 多轮对话追加的提示模版
        elif args.model == 'baichuan2':
            self.system = ''  # 默认系统提示
            self.template = '{system}<reserved_106>{input}'  # 单轮对话提示模版
            self.template_add = '<reserved_107>{output_add}<reserved_106>{input_add}'  # 多轮对话追加的提示模版
        self.record = 0
        self.record_list = []

    def predict(self, input_, generation_config=None):  # 输入字符串
        generation_config = generation_config if generation_config else self.generation_config
        with torch.no_grad():
            prompt = self.template.format(system=self.system, input=input_)
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            pred = self.model.generate(input_ids=input_ids, generation_config=generation_config)
            result = self.tokenizer.decode(pred[0], skip_special_tokens=True)
            result = result.split("[/INST]")[-1].strip()
        return result


if __name__ == '__main__':
    text = '你好呀'
    model = predict_class(args)
    result = model.predict(text)
    print(result)
