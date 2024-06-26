import os
import wandb
import torch
import argparse
from block.data_get import data_get
from block.model_get import model_get
from block.train_get import train_get

# -------------------------------------------------------------------------------------------------------------------- #
# 单轮对话数据格式(json)：没有system时用默认的
# [{'input':'B','output':'C'},...]
# [{'system':'A','input':'B','output':'C'},...]
# -------------------------------------------------------------------------------------------------------------------- #
# 分布式数据并行训练：
# python -m torch.distributed.launch --master_port 9999 --nproc_per_node n run.py --distributed True
# master_port为GPU之间的通讯端口，空闲的即可
# n为GPU数量
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|大模型微调:peft模型训练|')
parser.add_argument('--wandb', default=False, type=bool, help='|是否使用wandb可视化|')
parser.add_argument('--wandb_project', default='GPT', type=str, help='|wandb项目名称|')
parser.add_argument('--wandb_name', default='train', type=str, help='|wandb项目中的训练名称|')
parser.add_argument('--data_path', default=r'data_demo.json', type=str, help='|json数据路径|')
parser.add_argument('--divide', default='99,1', type=str, help='|训练集和验证集划分比例|')
parser.add_argument('--max_length', default=768, type=int, help='|模型输入的最大长度|')
parser.add_argument('--weight', default='last.pt', type=str, help='|已有模型的位置，没有则新建peft再训练|')
parser.add_argument('--model', default='qwen', type=str, help='|模型选择|')
parser.add_argument('--model_path', default='Qwen-1_8B-Chat', type=str, help='|原模型位置|')
parser.add_argument('--save_pt', default=0, type=int, help='|每几轮保存一次last.pt模型以便中断后继续训练，0为不保存|')
parser.add_argument('--epoch', default=30, type=int, help='|训练总轮数(包含之前已训练轮数)|')
parser.add_argument('--batch', default=2, type=int, help='|训练批量大小，分布式时为总批量|')
parser.add_argument('--warmup_ratio', default=0.01, type=float, help='|预热训练步数占总步数比例，最少5步，基准为0.01|')
parser.add_argument('--lr_start', default=0.0001, type=float, help='|初始学习率，adam算法，批量大时要增大，基准为0.0001|')
parser.add_argument('--lr_end_ratio', default=0.01, type=float, help='|最终学习率=lr_end_ratio*lr_start，基准为0.01|')
parser.add_argument('--lr_end_epoch', default=30, type=int, help='|最终学习率达到的轮数，每一步都调整，余玄下降法|')
parser.add_argument('--regularization', default='L2', type=str, help='|正则化，有L2、None|')
parser.add_argument('--r_value', default=0.0002, type=float, help='|正则化权重系数，可从0.0001开始逐渐增加，直到最佳值|')
parser.add_argument('--device', default='cuda', type=str, help='|训练设备|')
parser.add_argument('--latch', default=False, type=bool, help='|模型和数据是否为锁存，True为锁存|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU处理数据的进程数，0只有一个主进程，一般为0、2、4、8|')
parser.add_argument('--ema', default=False, type=bool, help='|使用平均指数移动(EMA)调整参数|')
parser.add_argument('--amp', default=True, type=bool, help='|混合float16精度训练，CPU时不可用|')
parser.add_argument('--distributed', default=False, type=bool, help='|单机多卡分布式训练，分布式训练时batch为总batch|')
parser.add_argument('--local_rank', default=0, type=int, help='|分布式训练使用命令后会自动传入的参数|')
args = parser.parse_args()
args.divide = list(map(int, args.divide.split(',')))
args.device_number = max(torch.cuda.device_count(), 1)  # 使用的GPU数，可能为CPU
# 为CPU设置随机种子
torch.manual_seed(999)
# 为所有GPU设置随机种子
torch.cuda.manual_seed_all(999)
# 固定每次返回的卷积算法
torch.backends.cudnn.deterministic = True
# cuDNN使用非确定性算法
torch.backends.cudnn.enabled = True
# 训练前cuDNN会先搜寻每个卷积层最适合实现它的卷积算法，加速运行；但对于复杂变化的输入数据，可能会有过长的搜寻时间，对于训练比较快的网络建议设为False
torch.backends.cudnn.benchmark = True
# wandb可视化:https://wandb.ai
if args.wandb and args.local_rank == 0:  # 分布式时只记录一次wandb
    args.wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
# 混合float16精度训练
if args.amp:
    args.amp = torch.cuda.amp.GradScaler()
# 分布式训练
if args.distributed:
    torch.distributed.init_process_group(backend='nccl')
    args.device = torch.device("cuda", args.local_rank)
# -------------------------------------------------------------------------------------------------------------------- #
assert os.path.exists(args.data_path), f'! data_path不存在:{args.data_path} !'
if os.path.exists(args.weight):
    print(f'| 加载模型继续训练:{args.weight} |')
else:
    print(f'| 创建peft模型以微调模型:{args.model_path} |')
# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    # 摘要
    print(f'| args:{args} |') if args.local_rank == 0 else None
    # 数据
    data_dict = data_get(args)
    # 模型
    model_dict = model_get(args)
    # 训练
    train_get(args, data_dict, model_dict)
