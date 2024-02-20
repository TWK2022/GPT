# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用gradio将程序包装成一个可视化的页面，可以在网页可视化的展示
import gradio
import argparse
import transformers
from predict import predict_class

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动gradio服务|')
parser.add_argument('--model_path', default='chinese-alpaca-2-1.3b', type=str, help='|tokenizer和模型文件夹位置|')
parser.add_argument('--peft_model_path', default='', type=str, help='|peft模型文件夹位置(空则不使用)|')
parser.add_argument('--model', default='llama2', type=str, help='|模型类型|')
parser.add_argument('--system', default='', type=str, help='|追加的系统提示词|')
parser.add_argument('--temperature', default=0.2, type=float, help='|回答稳定概率，0.2-0.8，越小越稳定|')
parser.add_argument('--device', default='cpu', type=str, help='|设备|')
parser.add_argument('--stream', default=False, type=bool, help='|流式输出|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def function(input_, temperature, history):
    temperature = args.temperature if temperature == 'default' else temperature
    generation_config = transformers.GenerationConfig(max_new_tokens=1024, temperature=temperature)
    if args.stream:
        stream = model.predict_stream(system='', input_=input_, generation_config=generation_config)
        history.append([input_, ''])
        for index, str_ in enumerate(stream):
            if index > 0:
                history[-1][-1] += str_
                yield history
    else:
        result = model.predict(system='', input_=input_, generation_config=generation_config)
        history.append([input_, result])
        yield history


if __name__ == '__main__':
    print('| 使用gradio启动服务 |')
    model = predict_class(args)
    # 输入
    input_ = gradio.Textbox(placeholder='在此输入内容', label='用户输入', lines=1, scale=9)
    temperature = gradio.components.Radio(choices=['default', 0.2, 0.4, 0.6, 0.8], value='default',
                                          label='temperature')
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
        # 事件
        button.click(fn=function, inputs=[input_, temperature, chatbot], outputs=[chatbot])
    # 启动
    gradio_app.queue()
    gradio_app.launch(share=False)
