import os
import peft
import torch
import argparse
import threading
import transformers

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser('|模型预测|')
parser.add_argument('--model_path', default='qwen3_0.6b', type=str, help='|tokenizer和模型文件夹位置|')
parser.add_argument('--peft_model_path', default='', type=str, help='|peft模型文件夹位置(空则不使用)|')
parser.add_argument('--model', default='qwen3', type=str, help='|模型类型|')
parser.add_argument('--system', default='', type=str, help='|追加的系统提示词|')
parser.add_argument('--think', default=False, type=bool, help='|思维链|')
parser.add_argument('--temperature', default=0.2, type=float, help='|值越小回答越稳定，0.2-0.8|')
parser.add_argument('--max_new_tokens', default=768, type=int, help='|模型最大输出长度限制|')
parser.add_argument('--repetition_penalty', default=1.1, type=float, help='|防止模型输出重复的惩罚权重，1为不惩罚|')
parser.add_argument('--stream', default=False, type=bool, help='|流式输出|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()
args.device = 'cpu' if not torch.cuda.is_available() else args.device


# -------------------------------------------------------------------------------------------------------------------- #
class predict_class:
    def __init__(self, args=args):
        self.think = args.think
        self.model_type = args.model
        if args.model == 'qwen3':
            self.system = ''  # 默认系统提示
            self.template = ('<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n'
                             '<|im_start|>assistant\n')  # 单轮对话提示模版
            self.template_think = '<think>\n\n<think>\n\n'  # 思维链
            self.template_history = ('{output_add}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n'
                                     '<|im_start|>assistant\n')  # 多轮对话追加的提示模版
            self.bos_token_id = 151644  # <|im_start|>
            self.eos_token_id = 151645  # <|im_end|>
            self.pad_token_id = 151643  # <|endoftext|>
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,
                                                                       torch_dtype=torch.float16).eval()
        if os.path.exists(args.peft_model_path):
            self.model = peft.PeftModel.from_pretrained(self.model, args.peft_model_path)
        self.device = args.device
        self.model = self.model.float() if args.device.lower() == 'cpu' else self.model.half()
        self.model = self.model.to(args.device)
        self.stream = transformers.TextIteratorStreamer(self.tokenizer)
        self.generation_config = transformers.GenerationConfig(eos_token_id=self.eos_token_id,
                                                               pad_token_id=self.pad_token_id,
                                                               temperature=args.temperature,
                                                               max_new_tokens=args.max_new_tokens,
                                                               repetition_penalty=args.repetition_penalty,
                                                               do_sample=True)

    def predict(self, input_, system='', config_dict=None):
        if config_dict:
            self.generation_config.max_new_tokens = config_dict['max_new_tokens']
            self.generation_config.temperature = config_dict['temperature']
            self.generation_config.repetition_penalty = config_dict['repetition_penalty']
        with torch.no_grad():
            prompt = self.template.format(system=self.system + system, input=input_)
            prompt += self.template_think if not self.think else ''
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            pred = self.model.generate(input_ids=input_ids, generation_config=self.generation_config)
            result = self.tokenizer.decode(pred[0][len(input_ids[0]):], skip_special_tokens=True)
        return result

    def predict_stream(self, input_, system='', config_dict=None):
        if config_dict:
            self.generation_config.max_new_tokens = config_dict['max_new_tokens']
            self.generation_config.temperature = config_dict['temperature']
            self.generation_config.repetition_penalty = config_dict['repetition_penalty']
        with torch.no_grad():
            prompt = self.template.format(system=self.system + system, input=input_)
            prompt += self.template_think if not self.think else ''
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            kwargs = {'input_ids': input_ids, 'generation_config': self.generation_config, 'streamer': self.stream}
            thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
            thread.start()
            for str_ in self.stream:
                yield str_


if __name__ == '__main__':
    model = predict_class()
    for str_ in model.predict_stream('你是谁，精简回答'):
        print(str_, end='')