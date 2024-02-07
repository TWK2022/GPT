# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用gradio将程序包装成一个可视化的界面，可以在网页可视化的展示
import gradio
import argparse
from predict import predict_class

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动gradio服务|')
parser.add_argument('--model_path', default='Qwen-1_8B-Chat', type=str, help='|tokenizer和模型文件夹位置|')
parser.add_argument('--model', default='qwen', type=str, help='|模型类型|')
parser.add_argument('--temperature', default=0.2, type=float, help='|回答稳定概率，0.2-0.8，越小越稳定|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def function(input_):
    result = model.predict(system='', input_=input_)
    return result


if __name__ == '__main__':
    print('| 使用gradio启动服务 |')
    model = predict_class(args)
    gradio_app = gradio.Interface(fn=function, inputs=['text'], outputs=['text'], examples=[['你好呀']])
    gradio_app.launch(share=False)
