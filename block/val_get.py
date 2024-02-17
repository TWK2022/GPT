import tqdm
import torch


def val_get(args, val_dataloader, model, data_dict, ema):
    with torch.no_grad():
        model = ema.ema if args.ema else model.eval()
        val_loss = 0  # 记录损失
        tqdm_len = len(data_dict['val']) // args.batch * args.device_number
        tqdm_show = tqdm.tqdm(total=tqdm_len, mininterval=0.2)
        for index, (input_ids_batch, attention_mask_batch, label_batch) in enumerate(val_dataloader):
            input_ids_batch = input_ids_batch.to(args.device, non_blocking=args.latch)
            attention_mask_batch = attention_mask_batch.to(args.device, non_blocking=args.latch)
            label_batch = label_batch.to(args.device, non_blocking=args.latch)
            pred_batch = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=label_batch)
            loss_batch = pred_batch.loss  # 当传入labels时模型内部会自动计算损失
            # 调整参数，ema.updates会自动+1
            ema.update(model) if args.ema else None
            # 记录损失
            val_loss += loss_batch.item()
            # tqdm
            tqdm_show.set_postfix({'val_loss': loss_batch.item()})  # 添加loss显示
            tqdm_show.update(args.device_number)  # 更新进度条
        # tqdm
        tqdm_show.close()
        # 计算平均损失
        val_loss = val_loss / (index + 1)
        print(f'\n| 验证 | val_loss:{val_loss:.4f} |\n')
    return val_loss
