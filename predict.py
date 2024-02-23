import peft
import torch
import threading
import transformers


class predict_class:
    def __init__(self, args):
        self.model_type = args.model
        if args.model == 'llama2':
            self.system = 'You are a helpful assistant. 你是一个乐于助人的助手。'  # 默认系统提示
            self.template = ('<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{input} [/INST]')  # 单轮对话提示模版
            self.template_add = ' {output_add}</s><s>[INST] {input_add} [/INST]'  # 多轮对话追加的提示模版
            self.split = '[/INST]'
            self.tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path)
            self.eos_token_id = 2  # </s>
            self.pad_token_id = 32000  # <pad>
            self.model = transformers.LlamaForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True,
                                                                       torch_dtype=torch.float16).eval()
        elif args.model == 'baichuan2':
            self.system = ''  # 默认系统提示
            self.template = '{system}<reserved_106>{input}<reserved_107>'  # 单轮对话提示模版
            self.template_add = '{output_add}<reserved_106>{input_add}<reserved_107>'  # 多轮对话追加的提示模版
            self.split = '<reserved_107>'
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            self.eos_token_id = 2  # </s>
            self.pad_token_id = 0  # <unk>
            self.model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,
                                                                           torch_dtype=torch.float16).eval()
        elif args.model == 'qwen':
            self.system = 'You are a helpful assistant.\n'  # 默认系统提示
            self.template = ('<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n'
                             '<|im_start|>assistant\n')  # 单轮对话提示模版
            self.template_add = ('{output_add}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n'
                                 '<|im_start|>assistant\n')  # 多轮对话追加的提示模版
            self.split = '<|im_start|>assistant\n'
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            self.eos_token_id = 151645  # <|im_end|>
            self.pad_token_id = 151643  # <|endoftext|>
            self.model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,
                                                                           torch_dtype=torch.float16).eval()
        if args.peft_model_path:
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

    def test(self, system, input_, config_dict=None):
        if config_dict:
            self.generation_config.max_new_tokens = config_dict['max_new_tokens']
            self.generation_config.temperature = config_dict['temperature']
            self.generation_config.repetition_penalty = config_dict['repetition_penalty']
        with torch.no_grad():
            prompt = self.template.format(system=self.system + system, input=input_)
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            pred = self.model.generate(input_ids=input_ids, generation_config=self.generation_config)
            result = self.tokenizer.decode(pred[0], skip_special_tokens=True)
            result = result.split(self.split)[-1]
        return prompt, result

    def predict(self, system, input_, config_dict=None):
        if config_dict:
            self.generation_config.max_new_tokens = config_dict['max_new_tokens']
            self.generation_config.temperature = config_dict['temperature']
            self.generation_config.repetition_penalty = config_dict['repetition_penalty']
        with torch.no_grad():
            prompt = self.template.format(system=self.system + system, input=input_)
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            pred = self.model.generate(input_ids=input_ids, generation_config=self.generation_config)
            result = self.tokenizer.decode(pred[0], skip_special_tokens=True)
            result = result.split(self.split)[-1]
        return result

    def predict_stream(self, system, input_, config_dict=None):
        if config_dict:
            self.generation_config.max_new_tokens = config_dict['max_new_tokens']
            self.generation_config.temperature = config_dict['temperature']
            self.generation_config.repetition_penalty = config_dict['repetition_penalty']
        with torch.no_grad():
            prompt = self.template.format(system=self.system + system, input=input_)
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(self.device)
            kwargs = {'input_ids': input_ids, 'generation_config': self.generation_config, 'streamer': self.stream}
            thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
            thread.start()
            return eval(f'self._stream_{self.model_type}()')

    def _stream_llama2(self):
        for index, str_ in enumerate(self.stream):
            if index == 0:
                continue
            elif index == 1:
                str_ = str_.replace('[/INST] ', '')
            elif str_[-4:] == '</s>':
                str_ = str_[1:-4]
            yield str_

    def _stream_baichuan2(self):
        for index, str_ in enumerate(self.stream):
            if index == 0:
                continue
            elif index == 1:
                str_ = str_.split('<reserved_107>')[-1]
            elif str_[-4:] == '</s>':
                str_ = str_[1:-4]
            yield str_

    def _stream_qwen(self):
        for index, str_ in enumerate(self.stream):
            if index == 0:
                continue
            elif str_[-10:] == '<|im_end|>':
                str_ = str_.replace('<|im_end|>', '')
            yield str_


if __name__ == '__main__':
    import argparse

    # ---------------------------------------------------------------------------------------------------------------- #
    parser = argparse.ArgumentParser('|模型预测|')
    parser.add_argument('--model_path', default='Qwen-1_8B-Chat', type=str, help='|tokenizer和模型文件夹位置|')
    parser.add_argument('--peft_model_path', default='', type=str, help='|peft模型文件夹位置(空则不使用)|')
    parser.add_argument('--model', default='qwen', type=str, help='|模型类型|')
    parser.add_argument('--system', default='', type=str, help='|追加的系统提示词|')
    parser.add_argument('--temperature', default=0.2, type=float, help='|值越小回答稳定概率，0.2-0.8|')
    parser.add_argument('--max_new_tokens', default=768, type=int, help='|模型最大输出长度限制|')
    parser.add_argument('--repetition_penalty', default=1.1, type=float, help='|防止模型输出重复的惩罚权重，1为不惩罚|')
    parser.add_argument('--stream', default=False, type=bool, help='|流式输出，需要特殊处理|')
    parser.add_argument('--device', default='cuda', type=str, help='|设备|')
    args = parser.parse_args()
    # ---------------------------------------------------------------------------------------------------------------- #
    model = predict_class(args)
    if args.stream:
        while True:
            system = args.system
            input_ = input('用户输入：').strip()
            stream = model.predict_stream(system, input_)
            print(f'----------stream----------')
            for str_ in stream:
                print(str_, end='')
            print('\n')
    else:
        while True:
            system = args.system
            input_ = input('用户输入：').strip()
            prompt, result = model.test(system, input_)
            print(f'----------prompt----------\n{prompt}')
            print(f'----------result----------\n{result}')
