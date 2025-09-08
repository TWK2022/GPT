import wandb
import torch
import argparse
from train_class import train_class

# -------------------------------------------------------------------------------------------------------------------- #
# 单轮对话数据格式(json)：没有system时用默认的
# [{'input':'B','output':'C'},...]
# [{'system':'A','input':'B','output':'C'},...]
# -------------------------------------------------------------------------------------------------------------------- #
# 分布式数据并行训练:
# python -m torch.distributed.launch --master_port 9999 --nproc_per_node n run.py --distributed True
# master_port为gpu之间的通讯端口，空闲的即可。n为gpu数量
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|大模型微调:peft模型训练|')
parser.add_argument('--log', default=True, type=bool, help='|日志|')
parser.add_argument('--print_info', default=True, type=bool, help='|打印信息|')
parser.add_argument('--wandb', default=False, type=bool, help='|wandb可视化|')
parser.add_argument('--data_path', default='dataset/data_demo.json', type=str, help='|数据位置|')
parser.add_argument('--divide', default=(19, 1), type=tuple, help='|训练集和验证集划分比例|')
parser.add_argument('--weight_path', default='peft_last', type=str, help='|加载模型，优先级:加载模型>剪枝训练>创建新模型|')
parser.add_argument('--weight_again', default=True, type=bool, help='|重置学习率等状态，在weight_path上重新训练|')
parser.add_argument('--model', default='qwen2.5_vl', type=str, help='|模型选择|')
parser.add_argument('--model_path', default='Qwen2.5-VL-3B-Instruct', type=str, help='|原模型位置|')
parser.add_argument('--save_epoch', default=1, type=int, help='|每x轮和最后一轮保存peft模型|')
parser.add_argument('--save_path', default='peft_last', type=str, help='|保存模型|')
parser.add_argument('--max_length', default=1500, type=int, help='|模型输入+输出最大长度|')
parser.add_argument('--epoch', default=5, type=int, help='|训练总轮数(包含之前已训练轮数)|')
parser.add_argument('--batch', default=1, type=int, help='|训练批量大小，分布式时为总批量|')
parser.add_argument('--warmup_ratio', default=0.01, type=float, help='|预热训练步数占总步数比例，最少5步，基准为0.01|')
parser.add_argument('--lr_start', default=2e-5, type=float, help='|初始学习率，adam算法，批量小时要减小，基准为2e-5|')
parser.add_argument('--lr_end_ratio', default=0.1, type=float, help='|最终学习率=lr_end_ratio*lr_start，基准为0.1|')
parser.add_argument('--lr_end_epoch', default=10, type=int, help='|最终学习率达到的轮数，每一步都调整，余玄下降法|')
parser.add_argument('--regularization', default='L2', type=str, help='|正则化，有L2、None|')
parser.add_argument('--r_value', default=0.0005, type=float, help='|正则化权重系数，基准为0.0005|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
parser.add_argument('--latch', default=True, type=bool, help='|模型和数据是否为锁存|')
parser.add_argument('--num_worker', default=0, type=int, help='|cpu处理数据进程数，0为一个主进程，一般为0、2、4、8|')
parser.add_argument('--amp', default=True, type=bool, help='|混合float16精度训练，cpu时不可用，出现nan可能与gpu有关|')
parser.add_argument('--distributed', default=False, type=bool, help='|单机多卡分布式训练，分布式训练时batch为总batch|')
parser.add_argument('--local_rank', default=0, type=int, help='|分布式训练使用命令后会自动传入的参数|')
args = parser.parse_args()
if not torch.cuda.is_available():  # 没有gpu
    args.device = 'cpu'
    args.amp = False
args.device_number = max(torch.cuda.device_count(), 1)  # 使用的gpu数，可能为cpu
# wandb可视化:https://wandb.ai
if args.wandb and args.local_rank == 0:  # 分布式时只记录一次wandb
    args.wandb_run = wandb.init(project='GPT', name='train', config=args)
# 混合float16精度训练
if args.amp:
    args.amp = torch.cuda.amp.GradScaler()
# 分布式训练
if args.distributed:
    torch.distributed.init_process_group(backend='nccl')  # 分布式训练初始化
    args.device = torch.device('cuda', args.local_rank)
# 设置
torch.manual_seed(999)  # 为cpu设置随机种子
torch.cuda.manual_seed_all(999)  # 为所有gpu设置随机种子
torch.backends.cudnn.deterministic = True  # 固定每次返回的卷积算法
torch.backends.cudnn.enabled = True  # cuDNN使用非确定性算法
torch.backends.cudnn.benchmark = False  # 训练前cuDNN会先搜寻每个卷积层最适合实现它的卷积算法，加速运行；但对于复杂变化的输入数据，可能会有过长的搜寻时间，对于训练比较快的网络建议设为False
# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    train = train_class(args)
    train.train()
