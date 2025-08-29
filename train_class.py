import os
import cv2
import json
import math
import peft
import torch
import logging
import numpy as np
import transformers


class train_class:
    '''
        model_load: 加载模型
        data_load: 加载数据
        dataloader_load: 加载数据处理器
        optimizer_load: 加载学习率
        loss_load: 训练损失
        train: 训练模型
        validation: 训练时的模型验证
    '''

    def __init__(self, args):
        self.args = args
        self.model_dict = self.model_load()  # 模型
        self.model_dict['model'] = self.model_dict['model'].to(args.device, non_blocking=args.latch)  # 设备
        self.data_dict = self.data_load()  # 数据
        self.train_dataloader, self.val_dataloader, self.train_dataset = self.dataloader_load()  # 数据处理器
        self.optimizer, self.optimizer_adjust = self.optimizer_load()  # 学习率、学习率调整
        if args.distributed:  # 分布式初始化
            self.model_dict['model'] = torch.nn.parallel.DistributedDataParallel(self.model_dict['model'],
                                                                                 device_ids=[args.local_rank],
                                                                                 output_device=args.local_rank)
        if args.local_rank == 0 and args.log:  # 日志
            log_path = os.path.dirname(__file__) + '/log.log'
            logging.basicConfig(filename=log_path, level=logging.INFO,
                                format='%(asctime)s | %(levelname)s | %(message)s')
            logging.info('-------------------- log --------------------')

    def model_load(self):
        args = self.args
        if args.model == 'qwen2.5_vl':
            model = transformers.AutoModelForImageTextToText.from_pretrained(args.model_path,
                                                                             torch_dtype=torch.float16,
                                                                             device_map=args.device).eval()
            processor = transformers.AutoProcessor.from_pretrained(args.model_path, use_fast=True,
                                                                   padding_side='left')
            tokenizer = processor.tokenizer
        else:  # qwen3
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
            processor = None
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                                                   use_fast=False)
        if os.path.exists(args.weight_path):  # 加载已有peft
            model = peft.PeftModel.from_pretrained(model, args.weight_path, is_trainable=True)
            model_dict = torch.load(f'{args.weight_path}/last.pt', map_location='cpu', weights_only=False)
            if args.weight_again:
                model_dict.update({'epoch_finished': 0, 'optimizer_state_dict': None, 'standard': 1})
        else:  # 创建新peft
            peft_config = peft.LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, inference_mode=False,
                                          task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                                                                 'gate_proj', 'up_proj', 'down_proj'])
            model = peft.get_peft_model(model, peft_config)
            model_dict = {'epoch_finished': 0, 'optimizer_state_dict': None, 'standard': 1}
        model.print_trainable_parameters()  # 显示模型的可训练参数和总参数
        model_dict.update({'model': model, 'tokenizer': tokenizer, 'processor': processor})
        return model_dict

    def data_load(self):
        args = self.args
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        data_len = len(data_list)  # 输入数据的数量
        boundary = int(data_len * args.divide[0] / (args.divide[0] + args.divide[1]))  # 数据划分
        train = data_list[:boundary]
        val = data_list[boundary:]
        data_dict = {'train': train, 'val': val}
        return data_dict

    def dataloader_load(self):
        args = self.args
        # 数据集
        train_dataset = torch_dataset(args, 'train', self.data_dict['train'], self.model_dict['tokenizer'],
                                      self.model_dict['processor'])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        train_shuffle = False if args.distributed else True  # 分布式设置sampler后shuffle要为False
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=train_shuffle,
                                                       drop_last=True, pin_memory=args.latch,
                                                       num_workers=args.num_worker,
                                                       sampler=train_sampler, collate_fn=train_dataset.collate_fn)
        val_dataset = torch_dataset(args, 'val', self.data_dict['val'], self.model_dict['tokenizer'],
                                    self.model_dict['processor'])
        val_sampler = None  # 分布式时数据合在主GPU上进行验证
        val_batch = args.batch // args.device_number  # 分布式验证时batch要减少为一个GPU的量
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch, shuffle=False,
                                                     drop_last=False, pin_memory=args.latch,
                                                     num_workers=args.num_worker,
                                                     sampler=val_sampler, collate_fn=val_dataset.collate_fn)
        return train_dataloader, val_dataloader, train_dataset

    def optimizer_load(self):
        args = self.args
        if args.regularization == 'L2':
            optimizer = torch.optim.Adam(self.model_dict['model'].parameters(),
                                         lr=args.lr_start, betas=(0.937, 0.999), weight_decay=args.r_value)
        else:
            optimizer = torch.optim.Adam(self.model_dict['model'].parameters(),
                                         lr=args.lr_start, betas=(0.937, 0.999))
        if self.model_dict['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(self.model_dict['optimizer_state_dict'])
        step_epoch = len(self.data_dict['train']) // args.batch // args.device_number * args.device_number  # 每轮步数
        optimizer_adjust = lr_adjust(args, step_epoch, self.model_dict['epoch_finished'])  # 学习率调整函数
        optimizer = optimizer_adjust(optimizer)  # 学习率初始化
        return optimizer, optimizer_adjust

    def train(self):
        args = self.args
        model = self.model_dict['model']
        epoch_base = self.model_dict['epoch_finished'] + 1  # 新的一轮要+1
        for epoch in range(epoch_base, args.epoch + 1):
            if args.local_rank == 0 and args.print_info:
                info = f'-----------------------epoch:{epoch}-----------------------'
                print(info)
            model.train()
            train_loss = 0  # 记录损失
            for index, input_dict in enumerate(self.train_dataloader):
                for key in input_dict.keys():
                    input_dict[key] = input_dict[key].to(args.device, non_blocking=args.latch)
                if args.amp:
                    with torch.cuda.amp.autocast():
                        pred_batch = model(**input_dict)
                        loss_batch = pred_batch.loss  # 当传入labels时模型内部会自动计算损失
                    args.amp.scale(loss_batch).backward()
                    args.amp.step(self.optimizer)
                    args.amp.update()
                    self.optimizer.zero_grad()
                else:
                    pred_batch = model(**input_dict)
                    loss_batch = pred_batch.loss  # 当传入labels时模型内部会自动计算损失
                    loss_batch.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                train_loss += loss_batch.item()  # 记录损失
                self.optimizer = self.optimizer_adjust(self.optimizer)  # 调整学习率
            # 计算平均损失
            train_loss /= index + 1
            # 日志
            if args.local_rank == 0 and args.print_info:
                info = f'| train | train_loss:{train_loss:.4f} | lr:{self.optimizer.param_groups[0]["lr"]:.6f} |'
                print(info)
            # 清理显存空间
            del input_dict
            torch.cuda.empty_cache()
            # 验证
            if args.local_rank == 0:  # 分布式时只验证一次
                val_loss = self.validation()
            # 保存
            if args.local_rank == 0:  # 分布式时只保存一次
                self.model_dict['model'] = model.module if args.distributed else model
                self.model_dict['epoch_finished'] = epoch
                self.model_dict['optimizer_state_dict'] = self.optimizer.state_dict()
                self.model_dict['train_loss'] = train_loss
                self.model_dict['val_loss'] = val_loss
                if epoch % args.save_epoch == 0 or epoch == args.epoch:
                    self.model_dict['model'].save_pretrained(args.save_path)  # 保存peft模型
                    torch.save({'epoch_finished': epoch, 'optimizer_state_dict': self.optimizer.state_dict(),
                                'val_loss': val_loss, 'standard': val_loss}, f'{args.save_path}/last.pt')
                if val_loss <= self.model_dict['standard'] and val_loss <= 1:
                    save_best = f'peft_{epoch}_{val_loss:.2f}'
                    self.model_dict['standard'] = val_loss
                    self.model_dict['model'].save_pretrained(save_best)  # 保存peft模型
                    torch.save({'epoch_finished': epoch, 'optimizer_state_dict': self.optimizer.state_dict(),
                                'val_loss': val_loss, 'standard': val_loss}, f'{save_best}/last.pt')
                    if args.local_rank == 0:  # 日志
                        info = (f'| best_model | val_loss:{val_loss:.4f} |')
                        print(info) if args.print_info else None
                        logging.info(info) if args.log else None
                # wandb
                if args.wandb:
                    wandb_log = {}
                    if epoch == 0:
                        wandb_log.update({f'image/train_image': self.wandb_image_list})
                    wandb_log.update({'metric/train_loss': train_loss,
                                      'metric/val_loss': val_loss})
                    args.wandb_run.log(wandb_log)
            torch.distributed.barrier() if args.distributed else None  # 分布式时每轮训练后让所有GPU进行同步，快的GPU会在此等待

    def validation(self):
        args = self.args
        with torch.no_grad():
            model = self.model_dict['model'].eval()
            val_loss = 0
            for index, input_dict in enumerate(self.val_dataloader):
                for key in input_dict.keys():
                    input_dict[key] = input_dict[key].to(args.device, non_blocking=args.latch)
                pred_batch = model(**input_dict)
                loss_batch = pred_batch.loss  # 当传入labels时模型内部会自动计算损失
                val_loss += loss_batch.item()
            # 计算指标
            val_loss /= (index + 1)
            # 日志
            info = (f'| val | val_loss:{val_loss:.4f} |')
            print(info) if args.print_info else None
        return val_loss

    def _bn_prune(self, model):  # 通过bn层裁剪模型
        args = self.args
        weight = []  # 权重
        weight_layer = []  # 每个权重所在的层
        layer = 0  # 层数记录
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                weight.append(module.weight.data.clone())
                weight_layer.append(np.full((len(module.weight.data),), layer))
                layer += 1
        weight_abs = torch.concatenate(weight, dim=0).abs()
        weight_index = np.concatenate(weight_layer, axis=0)
        # 剪枝
        boundary = int(len(weight_abs) * args.prune_ratio)
        weight_index_keep = weight_index[np.argsort(weight_abs)[-boundary:]]  # 保留的参数所在的层数
        config = []  # 裁剪结果
        for layer, weight_one in enumerate(weight):
            layer_number = max(np.sum(weight_index_keep == layer).item(), 1)  # 剪枝后该层的参数个数，至少1个
            config.append(layer_number)
        return config


class lr_adjust:
    def __init__(self, args, step_epoch, epoch_finished):
        self.lr_start = args.lr_start  # 初始学习率
        self.lr_end = args.lr_end_ratio * args.lr_start  # 最终学习率
        self.lr_end_epoch = args.lr_end_epoch  # 最终学习率达到的轮数
        self.step_all = self.lr_end_epoch * step_epoch  # 总调整步数
        self.step_finished = epoch_finished * step_epoch  # 已调整步数
        self.warmup_step = max(5, int(args.warmup_ratio * self.step_all))  # 预热训练步数

    def __call__(self, optimizer):
        self.step_finished += 1
        step_now = self.step_finished
        decay = step_now / self.step_all
        lr = self.lr_end + (self.lr_start - self.lr_end) * math.cos(math.pi / 2 * decay)
        if step_now <= self.warmup_step:
            lr = lr * (0.1 + 0.9 * step_now / self.warmup_step)
        lr = max(lr, 0.000001)
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
        return optimizer


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, tag, data, tokenizer, processor):
        self.tag = tag
        self.data = data
        self.model = args.model
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = args.max_length
        if self.model == 'qwen3':
            self.template = ('<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n'
                             '<|im_start|>assistant\n')  # 单轮对话提示模版
            self.template_think = '<think>\n\n<think>\n\n'  # 思维链
            self.template_output = ('{output}<|im_end|>\n')  # 多轮对话追加的提示模版
            self.bos_token_id = 151644  # <|im_start|>
            self.eos_token_id = 151645  # <|im_end|>
            self.pad_token_id = 151643  # <|endoftext|>
            self.ignore_index = -100
        elif self.model == 'qwen2.5_vl':
            self.template = ('<|im_start|>system\n{system}<|im_end|>\n'
                             '<|im_start|>user\n{image_template}{text_input}<|im_end|>\n'
                             '<|im_start|>assistant\n')  # 单轮对话提示模版
            self.template_output = ('{output}<|im_end|>\n')  # 多轮对话追加的提示模版
            self.image_template = '<|vision_start|><|image_pad|><|vision_end|>'
            self.bos_token_id = 151644  # <|im_start|>
            self.eos_token_id = 151645  # <|im_end|>
            self.pad_token_id = 151643  # <|endoftext|>
            self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_dict = self.data[index]
        system = data_dict.get(data_dict['system'], '')  # 系统提示
        text_input = data_dict['input']  # 问题
        text_output = data_dict['output']  # 回答
        image_input = data_dict.get('image_path', '')  # 图片
        if not image_input:  # 文本预测
            prompt_input = self.template.format(system=system, input=text_input) + self.template_think
            prompt_output = self.template_output.format(output=text_output)
            encode_input = self.tokenizer.encode(prompt_input, add_special_tokens=False)
            encode_output = self.tokenizer.encode(prompt_output, add_special_tokens=False)
            input_id = torch.tensor(encode_input + encode_output, dtype=torch.int64)
            attention_mask = torch.full_like(input_id, 1)  # 有内容对应1，填充对应0
            label = torch.full_like(input_id, self.ignore_index)
            label[len(encode_input):] = input_id[len(encode_input):]  # 非回答部分为-100
            return input_id, attention_mask, label
        else:  # 文本加图片预测
            image = cv2.imdecode(np.fromfile(image_input, dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
            prompt_input = self.template.format(system=system, image_template=self.image_template,
                                                text_input=text_input)
            prompt_output = self.template_output.format(output=text_output)
            encode_dict = self.processor(text=prompt_input, images=image, add_special_tokens=False)
            encode_input = encode_dict['input_ids'][0]
            encode_output = self.tokenizer.encode(prompt_output, add_special_tokens=False)
            input_id = torch.tensor(encode_input + encode_output, dtype=torch.int64)
            attention_mask = torch.full_like(input_id, 1)
            label = torch.full_like(input_id, self.ignore_index)
            label[len(encode_input):] = input_id[len(encode_input):]  # 非回答部分为-100
            pixel_value = encode_dict['pixel_values']
            image_grid_thw = encode_dict['image_grid_thw']
            return input_id, attention_mask, label, pixel_value, image_grid_thw

    def collate_fn(self, getitem_list):  # 自定义__getitem__的合并方式，填充数据然后合并为批量
        input_id_batch = torch.nn.utils.rnn.pad_sequence([_[0] for _ in getitem_list], batch_first=True,
                                                         padding_value=self.pad_token_id)[:self.max_length]
        attention_mask_batch = torch.nn.utils.rnn.pad_sequence([_[1] for _ in getitem_list], batch_first=True,
                                                               padding_value=0)[:self.max_length]
        label_batch = torch.nn.utils.rnn.pad_sequence([_[2] for _ in getitem_list], batch_first=True,
                                                      padding_value=self.ignore_index)[:self.max_length]
        input_dict = {'input_ids': input_id_batch, 'attention_mask': attention_mask_batch, 'labels': label_batch}
        if len(getitem_list[0]) > 3:  # 图文识别
            input_dict['pixel_values'] = torch.concatenate([_[3] for _ in getitem_list], dim=0)
            input_dict['image_grid_thw'] = torch.concatenate([_[4] for _ in getitem_list], dim=0)
        return input_dict
