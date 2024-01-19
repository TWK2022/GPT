import tqdm
import torch
from block.val_get import val_get
from block.ModelEMA import ModelEMA
from block.lr_get import adam, lr_adjust


def train_get(args, data_dict, model_dict):
    # 加载模型
    model = model_dict['model'].to(args.device, non_blocking=args.latch)
    # 学习率
    optimizer = adam(args.regularization, args.r_value, model.parameters(), lr=args.lr_start, betas=(0.937, 0.999))
    optimizer.load_state_dict(model_dict['optimizer_state_dict']) if model_dict['optimizer_state_dict'] else None
    optimizer_adjust = lr_adjust(args, model_dict['lr_adjust_index'])  # 学习率调整函数
    optimizer = optimizer_adjust(optimizer, model_dict['epoch'] + 1, 0)  # 初始化学习率
    # 使用平均指数移动(EMA)调整参数(不能将ema放到args中，否则会导致模型保存出错)
    ema = ModelEMA(model) if args.ema else None
    if args.ema:
        ema.updates = model_dict['ema_updates']
    # 数据集
    train_dataset = torch_dataset(args, data_dict['train'], model_dict['tokenizer'])
    train_shuffle = False if args.distributed else True  # 分布式设置sampler后shuffle要为False
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=train_shuffle,
                                                   drop_last=True, pin_memory=args.latch, num_workers=args.num_worker,
                                                   sampler=train_sampler, collate_fn=train_dataset.collate_fn)
    val_dataset = torch_dataset(args, data_dict['val'], model_dict['tokenizer'])
    val_sampler = None  # 分布式时数据合在主GPU上进行验证
    val_batch = args.batch // args.device_number  # 分布式验证时batch要减少为一个GPU的量
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch, shuffle=False,
                                                 drop_last=False, pin_memory=args.latch, num_workers=args.num_worker,
                                                 sampler=val_sampler)
    # 分布式初始化
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank) if args.distributed else model
    epoch_base = model_dict['epoch'] + 1  # 新的一轮要+1
    for epoch in range(epoch_base, epoch_base + args.epoch):  # 训练
        print(f'\n-----------------------第{epoch}轮-----------------------') if args.local_rank == 0 else None
        model.train()
        train_loss = 0  # 记录损失
        if args.local_rank == 0:  # tqdm
            tqdm_len = len(data_dict['train']) // args.batch // args.device_number * args.device_number
            tqdm_show = tqdm.tqdm(total=tqdm_len, mininterval=0.2)
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
                tqdm_show.set_postfix({'train_loss': loss_batch.item()})  # 添加loss显示
                tqdm_show.update(args.device_number)  # 更新进度条
        # tqdm
        if args.local_rank == 0:
            tqdm_show.close()
        # 计算平均损失
        train_loss = train_loss / (index + 1)
        if args.local_rank == 0:
            print(f'\n| train_loss:{train_loss:.4f} | lr:{optimizer.param_groups[0]["lr"]:.6f} |\n')
        # 调整学习率
        optimizer = optimizer_adjust(optimizer, epoch + 1, train_loss)
        # 清理显存空间
        del input_ids_batch, attention_mask_batch, label_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        if args.local_rank == 0:  # 分布式时只验证一次
            val_loss = val_get(args, val_dataloader, model, data_dict, ema)
        # 保存
        if args.local_rank == 0:  # 分布式时只保存一次
            model_dict['model'] = model.module if args.distributed else model
            model_dict['epoch'] = epoch
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
            model_dict['lr_adjust_index'] = optimizer_adjust.lr_adjust_index
            model_dict['ema_updates'] = ema.updates if args.ema else model_dict['ema_updates']
            model_dict['train_loss'] = train_loss
            model_dict['val_loss'] = val_loss
            if val_loss < model_dict['standard']:  # 保存最佳peft模型
                model_dict['standard'] = val_loss
                save_name = f'peft_{epoch}_{val_loss:.2f}'
                model_dict['model'].save_pretrained(save_name)
                model_dict['tokenizer'].save_pretrained(save_name)
                print(f'\n| 保存最佳模型:{save_name} | val_loss:{val_loss:.4f} |\n')
            if args.save_pt > 0 and epoch % args.save_pt == 0:  # 保存完整模型以便中断后继续训练
                torch.save(model_dict, 'last.pt')
            # wandb
            if args.wandb:
                args.wandb_run.log({'metric/train_loss': train_loss,
                                    'metric/val_loss': val_loss})
        torch.distributed.barrier() if args.distributed else None  # 分布式时每轮训练后让所有GPU进行同步，快的GPU会在此等待


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, input_data, tokenizer):
        self.input_data = input_data
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.pad_token_id
        if args.model == 'llama2':
            self.system = 'You are a helpful assistant. 你是一个乐于助人的助手。'  # 默认系统提示
            self.template = ('[INST] <<SYS>>\n'
                             '{system}\n'
                             '<</SYS>>\n\n'
                             '{input} [/INST]')  # 单轮对话提示模版
            self.template_add = ' {output}</s><s>[INST] {input2} [/INST]'  # 多轮对话追加的提示模版

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        data_dict = self.input_data[index]
        system = data_dict['system'] if 'system' in data_dict.keys() else self.system  # 系统提示
        input_ = data_dict['input']  # 问题
        output = data_dict['output'] + str(self.eos_token)  # 标签
        prompt = self.template.format(system=system, input=input_)  # 完整输入
        if 'input2' in data_dict.keys():  # 多轮对话
            input2 = data_dict['input2']
            prompt = prompt + self.template_add.format(output=output, input2=input2)
            output = data_dict['output2']
        input_encode = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt').squeeze(0)
        output_encode = self.tokenizer.encode(output, add_special_tokens=False, return_tensors='pt').squeeze(0)
        input_ids = torch.concat([input_encode, output_encode], dim=0)  # 训练时的完整输入
        attention_mask = torch.full_like(input_ids, 1)
        label = torch.full_like(input_ids, -100)  # 不需要的地方变为-100
        label[len(input_encode):] = output_encode  # 训练时的完整标签
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
