import torch
import argparse
import transformers

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|llama模型预测|')
parser.add_argument('--model_path', default='chinese-alpaca-2-1.3b', type=str, help='|tokenizer和模型文件夹位置|')
parser.add_argument('--device', default='cpu', type=str, help='|设备|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
class llama_class:
    def __init__(self, args):
        self.device = args.device
        # 模型
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path)
        self.model = transformers.LlamaForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True).eval()
        self.model = self.model.float() if args.device.lower() == 'cpu' else self.model.half()
        self.model = self.model.to(self.device)
        self.generation_config = transformers.GenerationConfig(max_new_tokens=400, do_sample=True, num_beams=1,
                                                               temperature=0.2, top_k=40, top_p=0.9,
                                                               repetition_penalty=1.1)
        # 验证
        assert len(self.tokenizer) == self.model.vocab_size  # 检查模型和文本编码是否匹配
        # 输入提示词模版
        self.template = '[INST] <<SYS>>\n{prompt}\n<</SYS>>\n\n{instruction} [/INST]'  # 对话模型的输入需要经过特殊的处理
        self.prompt = 'You are a helpful assistant. 你是一个乐于助人的助手。'  # 提示词
        self.template_add = '{answer}</s><s>[INST] {instruction} [/INST]'
        self.record = 0
        self.record_list = []

    def predict(self, text, generation_config=None):  # 输入字符串
        generation_config = generation_config if generation_config else self.generation_config
        with torch.no_grad():
            text_merge = self.template.format(prompt=self.prompt, instruction=text)
            text_dict = self.tokenizer(text_merge, return_tensors='pt')
            input_ids = text_dict["input_ids"].to(self.device)
            attention_mask = text_dict['attention_mask'].to(self.device)
            pred = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       generation_config=generation_config)
            result = self.tokenizer.decode(pred[0], skip_special_tokens=True)
            result = result.split("[/INST]")[-1].strip()
        return result


if __name__ == '__main__':
    text = '你好呀'
    model = llama_class(args)
    result = model.predict(text)
    print(result)
