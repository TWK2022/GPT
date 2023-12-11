import tqdm
import torch
from block.ModelEMA import ModelEMA
from block.lr_get import adam, lr_adjust


def train_get(args, data_dict, model_dict):
    # 加载模型
    model = model_dict['model'].to(args.device, non_blocking=args.latch)
    # 学习率
    optimizer = adam(args.regularization, args.r_value, model.parameters(), lr=args.lr_start, betas=(0.937, 0.999))
    optimizer.load_state_dict(model_dict['optimizer_state_dict']) if model_dict['optimizer_state_dict'] else None
    optimizer_adjust = lr_adjust(args, model_dict['lr_adjust_item'])  # 学习率调整函数
    optimizer = optimizer_adjust(optimizer, model_dict['epoch'] + 1, 0)  # 初始化学习率
    # 使用平均指数移动(EMA)调整参数(不能将ema放到args中，否则会导致模型保存出错)
    ema = ModelEMA(model) if args.ema else None
    if args.ema:
        ema.updates = model_dict['ema_updates']
    # 数据集
    train_dataset = torch_dataset(args, data_dict['train_input'], model_dict['tokenizer'])
    train_shuffle = False if args.distributed else True  # 分布式设置sampler后shuffle要为False
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=train_shuffle,
                                                   drop_last=True, pin_memory=args.latch, num_workers=args.num_worker,
                                                   sampler=train_sampler, collate_fn=train_dataset.collate_fn)
    # 分布式初始化
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank) if args.distributed else model
    epoch_base = model_dict['epoch'] + 1  # 新的一轮要+1
    for epoch in range(epoch_base, epoch_base + args.epoch):
        # 训练
        print(f'\n-----------------------第{epoch}轮-----------------------') if args.local_rank == 0 else None
        model.train()
        train_loss = 0  # 记录训练损失
        tqdm_show = tqdm.tqdm(
            total=len(data_dict['train_input']) // args.batch // args.device_number * args.device_number, postfix=dict,
            mininterval=0.2) if args.local_rank == 0 else None  # tqdm
        for index, (input_ids_batch, attention_mask_batch, label_batch) in enumerate(train_dataloader):
            input_ids_batch = input_ids_batch.to(args.device, non_blocking=args.latch)
            attention_mask_batch = attention_mask_batch.to(args.device, non_blocking=args.latch)
            label_batch = label_batch.to(args.device, non_blocking=args.latch)
            if args.amp:
                with torch.cuda.amp.autocast():
                    pred_batch = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch,
                                       labels=label_batch)
                    loss_batch = pred_batch.loss  # 当传入labels时模型内部会自动计算损失
                args.amp.scale(loss_batch).backward()
                args.amp.step(optimizer)
                args.amp.update()
                optimizer.zero_grad()
            else:
                pred_batch = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=label_batch)
                loss_batch = pred_batch.loss  # 当传入labels时模型内部会自动计算损失
                loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad()
            # 调整参数，ema.updates会自动+1
            ema.update(model) if args.ema else None
            # 记录损失
            train_loss += loss_batch.item()
            # tqdm
            if args.local_rank == 0:
                tqdm_show.set_postfix({'当前loss': loss_batch.item()})  # 添加loss显示
                tqdm_show.update(args.device_number)  # 更新进度条
        # tqdm
        tqdm_show.close() if args.local_rank == 0 else None
        # 计算平均损失
        train_loss = train_loss / (index + 1)
        print('\n| train_loss:{:.4f} | lr:{:.6f} |\n'.format(train_loss, optimizer.param_groups[0]['lr']))
        # 调整学习率
        optimizer = optimizer_adjust(optimizer, epoch + 1, train_loss)
        # 清理显存空间
        del input_ids_batch, attention_mask_batch, label_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 保存
        if args.local_rank == 0:  # 分布式时只保存一次
            model_dict['model'] = model.module if args.distributed else model
            model_dict['epoch'] = epoch
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
            model_dict['lr_adjust_item'] = optimizer_adjust.lr_adjust_item
            model_dict['ema_updates'] = ema.updates if args.ema else model_dict['ema_updates']
            if epoch % 1 == 0:  # 保存peft模型
                save_name = f'save_peft_{epoch}'
                model_dict['model'].save_pretrained(save_name)
                model_dict['tokenizer'].save_pretrained(save_name)
            if args.save_pt != 0 and epoch % args.save_pt == 0:  # 保存完整模型
                torch.save(model_dict, 'last.pt')
            # wandb
            if args.wandb:
                args.wandb_run.log({'metric/train_loss': train_loss})
        torch.distributed.barrier() if args.distributed else None  # 分布式时每轮训练后让所有GPU进行同步，快的GPU会在此等待


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, input_data, tokenizer):
        self.input_data = input_data
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.pad_token_id
        self.prompt = 'You are a helpful assistant. 你是一个乐于助人的助手。'  # 提示词
        self.template = ('[INST] <<SYS>>\n'
                         '{prompt}\n'
                         '<</SYS>>\n\n'
                         '{instruction} [/INST]')  # 对话模型的输入需要经过特殊的处理

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        sft_dict = self.input_data[index]
        instruction = sft_dict['instruction'] + '\n' + sft_dict['input'] if sft_dict['input'] \
            else sft_dict['instruction']
        output = sft_dict['output'] + str(self.eos_token)
        text_merge = self.template.format(prompt=self.prompt, instruction=instruction)  # 对话时的完整输入
        input_encode = self.tokenizer.encode(text_merge, add_special_tokens=True, return_tensors='pt').squeeze(0)
        output_encode = self.tokenizer.encode(output, add_special_tokens=False, return_tensors='pt').squeeze(0)
        input_ids = torch.concat([input_encode, output_encode], dim=0)  # 训练时的完整输入input_ids
        attention_mask = torch.full_like(input_ids, 1)  # attention_mask
        label = torch.full_like(input_ids, -100)  # 不需要的地方变为-100
        label[len(input_encode):] = output_encode  # 训练时的完整标签label
        return input_ids, attention_mask, label

    def collate_fn(self, getitem_list):  # 自定义__getitem__的合并方式，填充数据然后合并为批量
        input_ids_list = [_[0] for _ in getitem_list]
        attention_mask_list = [_[1] for _ in getitem_list]
        label_list = [_[2] for _ in getitem_list]
        input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True,
                                                          padding_value=self.pad_token_id)
        attention_mask_batch = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        label_batch = torch.nn.utils.rnn.pad_sequence(label_list, batch_first=True, padding_value=-100)
        return input_ids_batch, attention_mask_batch, label_batch
