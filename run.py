import os
import wandb
import torch
import argparse
from block.data_get import data_get
from block.model_get import model_get
from block.train_get import train_get

# -------------------------------------------------------------------------------------------------------------------- #
# 数据格式：sft格式
# [{'instruction':'A','input':'B','output':'C'},...]
# 多轮对话数据：
# prompt='You are a helpful assistant. 你是一个乐于助人的助手。'
# template='[INST] <<SYS>>\n{prompt}\n<</SYS>>\n\n{instruction} [/INST]'
# text=template.format(prompt=prompt,instruction=instruction)
# template_add=' {answer}</s><s>[INST] {instruction} [/INST]'
# text=text+template_add.format(answer=answer,instruction=instruction)
# -------------------------------------------------------------------------------------------------------------------- #
# 分布式训练：
# python -m torch.distributed.launch --master_port 9999 --nproc_per_node n run.py --distributed True
# master_port为GPU之间的通讯端口，空闲的即可
# n为GPU数量
# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|llama类大模型微调:peft模型训练|')
parser.add_argument('--data_path', default=r'data_demo.json', type=str, help='|sft(.json)数据路径|')
parser.add_argument('--divide', default='9,1', type=str, help='|训练集和验证集划分比例|')
parser.add_argument('--model_path', default='chinese-alpaca-2-1.3b', type=str, help='|原模型位置|')
parser.add_argument('--weight', default='last.pt', type=str, help='|已有模型的位置，没有则新建peft再训练|')
parser.add_argument('--save_pt', default=1, type=int, help='|每几轮保存一次last.pt模型以便中断后继续训练，0为不保存|')
parser.add_argument('--wandb', default=False, type=bool, help='|是否使用wandb可视化|')
parser.add_argument('--wandb_project', default='GPT', type=str, help='|wandb项目名称|')
parser.add_argument('--wandb_name', default='train', type=str, help='|wandb项目中的训练名称|')
parser.add_argument('--epoch', default=15, type=int, help='|训练轮数|')
parser.add_argument('--batch', default=1, type=int, help='|训练批量大小|')
parser.add_argument('--lr_start', default=0.00002, type=float, help='|初始学习率，adam算法，3轮预热训练，基准为0.00002|')
parser.add_argument('--lr_end_ratio', default=0.2, type=float, help='|最终学习率=lr_end_ratio*lr_start，基准为0.2|')
parser.add_argument('--lr_adjust_num', default=10, type=int, help='|学习率下降调整次数，余玄下降法，要小于总轮次|')
parser.add_argument('--lr_adjust_threshold', default=0.9, type=float, help='|损失下降比较快时不调整学习率，基准为0.9|')
parser.add_argument('--regularization', default='L2', type=str, help='|正则化，有L2、None|')
parser.add_argument('--r_value', default=0.0005, type=float, help='|正则化的权重系数|')
parser.add_argument('--device', default='cuda', type=str, help='|训练设备|')
parser.add_argument('--latch', default=True, type=bool, help='|模型和数据是否为锁存，True为锁存|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU处理数据的进程数，0表示只有一个主进程，一般为0、2、4、8|')
parser.add_argument('--ema', default=True, type=bool, help='|使用平均指数移动(EMA)调整参数|')
parser.add_argument('--amp', default=True, type=bool, help='|混合float16精度训练，CPU时不可用|')
parser.add_argument('--distributed', default=False, type=bool, help='|单机多卡分布式训练，分布式训练时batch为总batch|')
parser.add_argument('--local_rank', default=0, type=int, help='|分布式训练使用命令后会自动传入的参数|')
args = parser.parse_args()
args.divide = list(map(int, args.divide.split(',')))
args.device_number = max(torch.cuda.device_count(), 1)  # 使用的GPU数，可能为CPU
print(f'| args:{args} |')
# 为CPU设置随机种子
torch.manual_seed(999)
# 为所有GPU设置随机种子
torch.cuda.manual_seed_all(999)
# 固定每次返回的卷积算法
torch.backends.cudnn.deterministic = True
# cuDNN使用非确定性算法
torch.backends.cudnn.enabled = True
# 训练前cuDNN会先搜寻每个卷积层最适合实现它的卷积算法，加速运行；但对于复杂变化的输入数据，可能会有过长的搜寻时间，对于训练比较快的网络建议设为False
torch.backends.cudnn.benchmark = False
# wandb可视化:https://wandb.ai
if args.wandb and args.local_rank == 0:  # 分布式时只记录一次wandb
    args.wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
# 混合float16精度训练
if args.amp:
    args.amp = torch.cuda.amp.GradScaler()
# 分布式训练
if args.distributed:
    torch.distributed.init_process_group(backend="nccl")
    args.device = torch.device("cuda", args.local_rank)
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.data_path), f'! data_path不存在:{args.data_path} !'
if os.path.exists(args.weight):
    print(f'| 加载模型继续训练:{args.weight} |')
else:
    print(f'| 创建peft模型以微调模型:{args.model_path} |')
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    # 数据
    data_dict = data_get(args)
    # 模型
    model_dict = model_get(args)
    # 训练
    train_get(args, data_dict, model_dict)
