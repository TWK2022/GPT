import torch
import argparse
from block.data_get import data_get
from block.model_get import model_get
from block.loss_get import loss_get
from block.train_get import train_get
# -------------------------------------------------------------------------------------------------------------------- #
# 数据data.csv格式：values768长度的特征向量，column为特征向量对应的中文文本
# -------------------------------------------------------------------------------------------------------------------- #
# 分布式训练:
# python -m torch.distributed.launch --master_port 9999 --nproc_per_node n run.py --distributed True
# master_port为GPU之间的通讯端口，空闲的即可
# n为GPU数量
# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|clip中文文本模型微调|')
parser.add_argument('--data_path', default=r'data.csv', type=str, help='|数据路径|')
parser.add_argument('--model', default='chinese_model', type=str, help='|模型|')
parser.add_argument('--epoch', default=26, type=int, help='|训练轮数|')
parser.add_argument('--batch', default=10, type=int, help='|训练批量大小|')
parser.add_argument('--loss', default='mse', type=str, help='|损失函数|')
parser.add_argument('--lr_start', default=0.000002, type=float, help='|初始学习率|')
parser.add_argument('--regularization', default='L2', type=str, help='|正则化，有L2、None|')
parser.add_argument('--r_value', default=0.0005, type=float, help='|正则化的权重系数|')
parser.add_argument('--device', default='cpu', type=str, help='|训练设备|')
parser.add_argument('--latch', default=True, type=bool, help='|模型和数据是否为锁存，True为锁存|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU在处理数据时使用的进程数，0表示只有一个主进程，一般为0、2、4、8|')
parser.add_argument('--ema', default=True, type=bool, help='|使用平均指数移动(EMA)调整参数|')
parser.add_argument('--amp', default=False, type=bool, help='|混合float16精度训练，CPU时不可用|')
parser.add_argument('--distributed', default=False, type=bool, help='|单机多卡分布式训练，分布式训练时batch为总batch|')
parser.add_argument('--local_rank', default=0, type=int, help='|分布式训练使用命令后会自动传入的参数|')
args = parser.parse_args()
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
# 混合float16精度训练
if args.amp:
    args.amp = torch.cuda.amp.GradScaler()
# 分布式训练
if args.distributed:
    torch.distributed.init_process_group(backend="nccl")
    args.device = torch.device("cuda", args.local_rank)
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    # 数据
    data_dict = data_get(args)
    # 模型
    model_dict = model_get(args)
    # 损失
    loss = loss_get(args)
    # 训练
    train_get(args, data_dict, model_dict, loss)
