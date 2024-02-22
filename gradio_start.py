# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用gradio将程序包装成一个可视化的页面，可以在网页可视化的展示
import gradio
import argparse
from predict import predict_class

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser('|在服务器上启动gradio服务|')
parser.add_argument('--model_path', default='Qwen-1_8B-Chat', type=str, help='|tokenizer和模型文件夹位置|')
parser.add_argument('--peft_model_path', default='', type=str, help='|peft模型文件夹位置(空则不使用)|')
parser.add_argument('--model', default='qwen', type=str, help='|模型类型|')
parser.add_argument('--system', default='', type=str, help='|追加的系统提示词|')
parser.add_argument('--temperature', default=0.2, type=float, help='|回答稳定概率，0.2-0.8，越小越稳定|')
parser.add_argument('--repetition_penalty', default=1, type=float, help='|防止模型输出重复的惩罚权重，1为不惩罚|')
parser.add_argument('--stream', default=True, type=bool, help='|流式输出，需要特殊处理|')
parser.add_argument('--device', default='cpu', type=str, help='|设备|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
def function(input_, history, temperature, repetition_penalty):
    temperature = args.temperature if temperature == 'default' else temperature
    repetition_penalty = args.repetition_penalty if repetition_penalty == 'default' else repetition_penalty
    if args.stream:
        stream = model.predict_stream(system='', input_=input_, temperature=temperature)
        history.append([input_, ''])
        for index, str_ in enumerate(stream):
            if index > 0:
                history[-1][-1] += str_
                yield history
    else:
        result = model.predict(system='', input_=input_, temperature=temperature, repetition_penalty=repetition_penalty)
        history.append([input_, result])
        yield history


def gradio_start():
    # 输入
    input_ = gradio.Textbox(placeholder='在此输入内容', label='用户输入', lines=1, scale=9)
    temperature = gradio.components.Radio(choices=['default', 0.2, 0.5, 0.8, 1.0], value='default',
                                          label='temperature')
    repetition_penalty = gradio.components.Radio(choices=['default', 1.0, 1.1, 1.3, 1.5], value='default',
                                                 label='repetition_penalty')
    # 输出
    chatbot = gradio.Chatbot(value=[], label=args.model_path, height=450, bubble_full_width=False)
    # 按钮
    button = gradio.Button(value='确定🚀', scale=1)
    # 渲染
    with gradio.Blocks(theme='Default') as gradio_app:
        gradio.Markdown('## 对话模型')
        with gradio.Row():  # 水平排列
            with gradio.Column(scale=9):  # 垂直排列
                chatbot.render()
                with gradio.Row():  # 水平排列
                    input_.render()
                    button.render()
                    gradio.ClearButton(components=[input_, chatbot], value='清除历史', scale=1)
                gradio.Examples(examples=['你好呀', '你是谁'], inputs=input_, label='示例')
            with gradio.Column(scale=1):  # 垂直排列
                gradio.Markdown('### 设置')
                temperature.render()
                repetition_penalty.render()
        # 事件
        button.click(fn=function, inputs=[input_, chatbot, temperature, repetition_penalty], outputs=[chatbot])
    # 启动
    gradio_app.queue()
    gradio_app.launch(share=False)


if __name__ == '__main__':
    print('| 使用gradio启动服务 |')
    model = predict_class(args)
    gradio_start()
