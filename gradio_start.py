# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用gradio将程序包装成一个可视化的界面，可以在网页可视化的展示
import gradio
import argparse
from predict import predict_class

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动gradio服务|')
parser.add_argument('--model_path', default='chinese-alpaca-2-1.3b', type=str, help='|tokenizer和模型文件夹位置|')
parser.add_argument('--model', default='llama2', type=str, help='|模型类型|')
parser.add_argument('--temperature', default=0.2, type=float, help='|回答稳定概率，0.2-0.8，越小越稳定|')
parser.add_argument('--device', default='cpu', type=str, help='|设备|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def function(input_, history):
    result = model.predict(system='', input_=input_)
    history.append((input_, result))
    output = history
    return history, output


if __name__ == '__main__':
    print('| 使用gradio启动服务 |')
    model = predict_class(args)
    input_text = gradio.Textbox(placeholder='', label='输入', lines=1)
    state = gradio.State(value=[])
    chatbot = gradio.Chatbot()
    gradio_app = gradio.Interface(fn=function, inputs=[input_text, state], outputs=[state, chatbot],
                                  examples=[['你好呀']])
    gradio_app.launch(share=False)
