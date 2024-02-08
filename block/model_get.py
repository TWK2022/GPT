import os
import peft
import torch
import transformers


def model_get(args):
    if os.path.exists(args.weight):  # 继续训练peft模型
        model_dict = torch.load(args.weight, map_location='cpu')
    else:  # 重新训练peft模型
        choice_dict = {'llama2': 'model_prepare(args).llama2()',
                       'baichuan2': 'model_prepare(args).baichuan2()'}
        tokenizer, model = eval(choice_dict[args.model])
        model_dict = {}
        model_dict['model'] = model
        model_dict['tokenizer'] = tokenizer
        model_dict['epoch_finished'] = 0  # 已训练的轮次
        model_dict['optimizer_state_dict'] = None  # 学习率参数
        model_dict['ema_updates'] = 0  # ema参数
        model_dict['standard'] = 1  # 评价指标
    model_dict['model'].print_trainable_parameters()  # 显示模型的可训练参数和总参数
    return model_dict


class model_prepare:
    def __init__(self, args):
        self.args = args

    def llama2(self):
        tokenizer = transformers.LlamaTokenizer.from_pretrained(self.args.model_path)
        model = transformers.LlamaForCausalLM.from_pretrained(self.args.model_path, low_cpu_mem_usage=True)
        peft_config = peft.LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, inference_mode=False,
                                      task_type=peft.TaskType.CAUSAL_LM,
                                      target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj',
                                                      'up_proj'])
        model = peft.get_peft_model(model, peft_config)
        return tokenizer, model

    def baichuan2(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True,
                                                               use_fast=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(self.args.model_path, trust_remote_code=True)
        peft_config = peft.LoraConfig(r=1, lora_alpha=32, lora_dropout=0.1, inference_mode=False,
                                      task_type=peft.TaskType.CAUSAL_LM, target_modules=['W_pack'])
        model = peft.get_peft_model(model, peft_config)
        return tokenizer, model

    def qwen(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True,
                                                               use_fast=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(self.args.model_path, trust_remote_code=True)
        peft_config = peft.LoraConfig(r=64, lora_alpha=16, lora_dropout=0.05, inference_mode=False,
                                      task_type=peft.TaskType.CAUSAL_LM,
                                      target_modules=['c_attn', 'c_proj', 'w1', 'w2'])
        model = peft.get_peft_model(model, peft_config)
        return tokenizer, model
