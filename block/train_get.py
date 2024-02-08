import tqdm
import torch
from block.val_get import val_get
from block.model_ema import model_ema
from block.lr_get import adam, lr_adjust


def train_get(args, data_dict, model_dict):
    # 加载模型
    model = model_dict['model'].to(args.device, non_blocking=args.latch)
    # 学习率
    optimizer = adam(args.regularization, args.r_value, model.parameters(), lr=args.lr_start, betas=(0.937, 0.999))
    optimizer.load_state_dict(model_dict['optimizer_state_dict']) if model_dict['optimizer_state_dict'] else None
    step_epoch = len(data_dict['train']) // args.batch // args.device_number * args.device_number  # 每轮的步数
    optimizer_adjust = lr_adjust(args, step_epoch, model_dict['epoch_finished'])  # 学习率调整函数
    optimizer = optimizer_adjust(optimizer)  # 学习率初始化
    # 使用平均指数移动(EMA)调整参数(不能将ema放到args中，否则会导致模型保存出错)
    ema = model_ema(model) if args.ema else None
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
    epoch_base = model_dict['epoch_finished'] + 1  # 新的一轮要+1
    for epoch in range(epoch_base, args.epoch + 1):  # 训练
        print(f'\n-----------------------第{epoch}轮-----------------------') if args.local_rank == 0 else None
        model.train()
        train_loss = 0  # 记录损失
        if args.local_rank == 0:  # tqdm
            tqdm_show = tqdm.tqdm(total=step_epoch, mininterval=0.2)
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
            # 调整学习率
            optimizer = optimizer_adjust(optimizer)
            # tqdm
            if args.local_rank == 0:
                tqdm_show.set_postfix({'train_loss': loss_batch.item(),
                                       'lr': optimizer.param_groups[0]['lr']})  # 添加显示
                tqdm_show.update(args.device_number)  # 更新进度条
        # tqdm
        if args.local_rank == 0:
            tqdm_show.close()
        # 计算平均损失
        train_loss = train_loss / (index + 1)
        if args.local_rank == 0:
            print(f'\n| 训练 | train_loss:{train_loss:.4f} | lr:{optimizer.param_groups[0]["lr"]:.6f} |\n')
        # 清理显存空间
        del input_ids_batch, attention_mask_batch, label_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        if args.local_rank == 0:  # 分布式时只验证一次
            val_loss = val_get(args, val_dataloader, model, data_dict, ema)
        # 保存
        if args.local_rank == 0:  # 分布式时只保存一次
            model_dict['model'] = model.module if args.distributed else model
            model_dict['epoch_finished'] = epoch
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
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
        self.model = args.model
        self.input_data = input_data
        self.max_length = args.max_length
        self.tokenizer = tokenizer
        if args.model == 'llama2':
            self.system = 'You are a helpful assistant. 你是一个乐于助人的助手。'  # 默认系统提示
            self.template = ('<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{input} [/INST]')  # 单轮对话提示模版
            self.template_add = ' {output_add}</s><s>[INST] {input_add} [/INST]'  # 多轮对话追加的提示模版
            self.ignore_index = -100
        elif args.model == 'baichuan2':
            self.system = ''  # 默认系统提示
            self.template = '{system}<reserved_106>{input}<reserved_107>'  # 单轮对话提示模版
            self.template_add = '{output_add}<reserved_106>{input_add}<reserved_107>'  # 多轮对话追加的提示模版
            self.ignore_index = -100
            self.reserved_106 = tokenizer.encode('<reserved_106>', add_special_tokens=False)  # 对应<reserved_106>
        elif args.model == 'qwen':
            self.system = 'You are a helpful assistant.\n'  # 默认系统提示
            self.template = ('<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n'
                             '<|im_start|>assistant\n')  # 单轮对话提示模版
            self.template_add = ('{output_add}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n'
                                 '<|im_start|>assistant\n')  # 多轮对话追加的提示模版
            self.ignore_index = -100
            self.im_start_id = tokenizer.im_start_id  # 对应<|im_start|>
            self.im_end_id = tokenizer.im_end_id  # 对应<|im_end|>
            self.n = tokenizer.encode('\n', add_special_tokens=False)  # 对应<|im_end|>后的\n

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        data_dict = self.input_data[index]
        system = self.system + data_dict['system'] if 'system' in data_dict.keys() else self.system  # 系统提示
        input_ = data_dict['input']  # 问题
        output = data_dict['output']  # 回答
        input_ids, attention_mask, label = eval(f'self._{self.model}')(system, input_, output)
        return input_ids, attention_mask, label

    def collate_fn(self, getitem_list):  # 自定义__getitem__的合并方式，填充数据然后合并为批量
        input_ids_list = [_[0] for _ in getitem_list]
        attention_mask_list = [_[1] for _ in getitem_list]
        label_list = [_[2] for _ in getitem_list]
        input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True,
                                                          padding_value=self.tokenizer.pad_token_id)
        attention_mask_batch = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        label_batch = torch.nn.utils.rnn.pad_sequence(label_list, batch_first=True, padding_value=self.ignore_index)
        return input_ids_batch, attention_mask_batch, label_batch

    def _llama2(self, system, input_, output):
        prompt = self.template.format(system=system, input=input_)
        prompt_encode = self.tokenizer.encode(prompt, add_special_tokens=False)
        output_encode = self.tokenizer.encode(output, add_special_tokens=False)
        input_ids = torch.tensor(prompt_encode + output_encode + [self.tokenizer.eos_token_id], dtype=torch.int64)
        attention_mask = torch.full_like(input_ids, 1)
        label = torch.full_like(input_ids, self.ignore_index)
        label[-len(output_encode):] = torch.tensor(output_encode, dtype=torch.int64)
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        label = label[:self.max_length]
        return input_ids, attention_mask, label

    def _baichuan2(self, system, input_, output):
        prompt = self.template.format(system=system, input=input_)
        prompt_encode = self.tokenizer.encode(prompt, add_special_tokens=False)
        output_encode = self.tokenizer.encode(output, add_special_tokens=False)
        input_ids = torch.tensor(prompt_encode + output_encode + [self.tokenizer.eos_token_id], dtype=torch.int64)
        attention_mask = torch.full_like(input_ids, 1)
        label = torch.full_like(input_ids, self.ignore_index)
        label[-len(output_encode):] = torch.tensor(output_encode, dtype=torch.int64)
        index = torch.nonzero(input_ids == self.reserved_106).squeeze(1)  # <reserved_106>对应的地方
        label[index] = self.tokenizer.eos_token_id  # 变为eos_token_id
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        label = label[:self.max_length]
        return input_ids, attention_mask, label

    def _qwen(self, system, input_, output):
        prompt = self.template.format(system=system, input=input_)
        prompt_encode = self.tokenizer.encode(prompt, add_special_tokens=False)
        output_encode = self.tokenizer.encode(output, add_special_tokens=False)
        input_ids = torch.tensor(prompt_encode + output_encode + [self.im_end_id] + self.n, dtype=torch.int64)
        attention_mask = torch.full_like(input_ids, 1)
        label = torch.full_like(input_ids, self.ignore_index)
        label[-len(output_encode):] = torch.tensor(output_encode, dtype=torch.int64)
        index = torch.nonzero(input_ids == self.im_start_id).squeeze(1)  # <|im_start|>对应的地方
        label[index] = self.im_start_id  # 变为im_start_id
        index = torch.nonzero(input_ids == self.im_end_id).squeeze(1)  # <|im_end|>对应的地方
        label[index] = self.im_end_id  # 变为im_end_id
        index += 1  # <|im_end|>后的\n对应的地方
        label[index] = self.n  # 变为198
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        label = label[:self.max_length]
        return input_ids, attention_mask, label
