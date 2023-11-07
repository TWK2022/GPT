import transformers


def model_get(args):
    model = transformers.BertForSequenceClassification.from_pretrained(args.model)
    model_dict = {'model': model}
    model_dict['ema_updates'] = 0  # ema参数
    return model_dict
