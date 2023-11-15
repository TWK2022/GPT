import peft
import transformers


def model_get(args):
    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.weight)
    model = transformers.LlamaForCausalLM.from_pretrained(args.weight, low_cpu_mem_usage=True)
    peft_config = peft.LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, inference_mode=False,
                                  task_type=peft.TaskType.CAUSAL_LM,
                                  target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj',
                                                  'up_proj'])
    model = peft.get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model_dict = {}
    model_dict['tokenizer'] = tokenizer
    model_dict['model'] = model
    model_dict['ema_updates'] = 0  # ema参数
    return model_dict
