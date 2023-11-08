# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用gradio将程序包装成一个可视化的界面，可以在网页可视化的展示
import gradio
import argparse
from predict import llama_class

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动gradio服务|')
parser.add_argument('--model_path', default='chinese-alpaca-2-1.3b', type=str, help='|模型文件夹位置(合并后的模型)|')
parser.add_argument('--device', default='cpu', type=str, help='|设备|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def function(text):
    result = model.predict(text)
    return result


if __name__ == '__main__':
    print('| 使用gradio启动服务 |')
    model = llama_class(args)
    gradio_app = gradio.Interface(fn=function, inputs=['text'], outputs=['text'],
                                  examples=[['你好呀'], ['我刚刚说了什么']])
    gradio_app.launch(share=False)
