import os
import peft
import torch
import transformers


def model_get(args):
    if os.path.exists(args.weight):  # 继续训练peft模型
        model_dict = torch.load(args.weight, map_location='cpu')
    else:  # 重新训练peft模型
        tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path)
        model = transformers.LlamaForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)
        peft_config = peft.LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, inference_mode=False,
                                      task_type=peft.TaskType.CAUSAL_LM,
                                      target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj',
                                                      'up_proj'])  # peft模型配置，可根据情况修改
        model = peft.get_peft_model(model, peft_config)  # 在原模型上创建peft模型
        model_dict = {}
        model_dict['model'] = model
        model_dict['tokenizer'] = tokenizer
        model_dict['epoch'] = 0  # 已训练的轮次
        model_dict['optimizer_state_dict'] = None  # 学习率参数
        model_dict['lr_adjust_index'] = 0  # 学习率调整参数
        model_dict['ema_updates'] = 0  # ema参数
    model_dict['model'].print_trainable_parameters()  # 显示模型的可训练参数和总参数
    return model_dict
