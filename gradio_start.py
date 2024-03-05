# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用gradio将程序包装成一个可视化的页面，可以在网页可视化的展示
import gradio
import argparse
from predict import predict_class
from feature_search import search_class

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser('|在服务器上启动gradio服务|')
parser.add_argument('--model_path', default='Qwen-1_8B-Chat', type=str, help='|tokenizer和模型文件夹位置|')
parser.add_argument('--peft_model_path', default='', type=str, help='|peft模型文件夹位置(空则不使用)|')
parser.add_argument('--model', default='qwen', type=str, help='|模型类型|')
parser.add_argument('--temperature', default=0.2, type=float, help='|值越小回答稳定概率，0.2-0.8|')
parser.add_argument('--max_new_tokens', default=768, type=int, help='|模型最大输出长度限制|')
parser.add_argument('--repetition_penalty', default=1.1, type=float, help='|防止模型输出重复的惩罚权重，1为不惩罚|')
parser.add_argument('--device', default='cpu', type=str, help='|设备|')
parser.add_argument('--embed_model_path', default='text2vec_base_chinese', type=str, help='|编码模型位置|')
parser.add_argument('--database_path', default='feature_database', type=str, help='|特征数据库|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
def function(input_, history, system, search, search_score, stream, max_new_tokens, temperature, repetition_penalty):
    config_dict = {'max_new_tokens': max_new_tokens, 'temperature': temperature,
                   'repetition_penalty': repetition_penalty}
    if search:
        score, text = model_search.search(input_)
        print(f'| score:{score} |')
        print(f'| text:{text[0:50]}... |')
        if score > search_score:
            system += f'\n你的回答要参考以下资料：\n{text}'
    if stream:
        stream = model.predict_stream(system=system, input_=input_, config_dict=config_dict)
        history.append([input_, ''])
        for str_ in stream:
            history[-1][-1] += str_
            yield history
    else:
        result = model.predict(system=system, input_=input_, config_dict=config_dict)
        history.append([input_, result])
        yield history


def gradio_start():
    # 输入
    input_ = gradio.Textbox(placeholder='在此输入内容', label='用户输入', lines=1, scale=9)
    system = gradio.Textbox(placeholder='在此输入系统提示词(可不填)', label='系统提示词')
    search = gradio.components.Radio(choices=[False, True], value=False, label='知识库搜索')
    search_score = gradio.components.Radio(choices=[0.6, 0.7, 0.8], value=0.6, label='知识库搜索阈值')
    stream = gradio.components.Radio(choices=[False, True], value=True, label='流式输出')
    temperature = gradio.components.Radio(choices=[0.2, 0.5, 0.8], value=0.2, label='temperature')
    max_new_tokens = gradio.components.Radio(choices=[512, 768, 1024], value=768, label='max_new_tokens')
    repetition_penalty = gradio.components.Radio(choices=[1.1, 1.3, 1.5], value=1.1, label='repetition_penalty')
    # 输出
    chatbot = gradio.Chatbot(value=[], label=args.model_path, height=450, bubble_full_width=False)
    # 按钮
    button = gradio.Button(value='确定🚀', scale=1)
    # 渲染
    theme = gradio.themes.Base(primary_hue='pink', secondary_hue='rose', neutral_hue='pink')
    with gradio.Blocks(theme=theme, title=args.model_path) as gradio_app:
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
                system.render()
                search.render()
                search_score.render()
                stream.render()
                temperature.render()
                max_new_tokens.render()
                repetition_penalty.render()
        # 事件
        button.click(fn=function,
                     inputs=[input_, chatbot, system, search, search_score, stream, max_new_tokens,
                             temperature, repetition_penalty],
                     outputs=[chatbot])
    # 启动
    gradio_app.queue()
    gradio_app.launch(share=False)


if __name__ == '__main__':
    print('| 使用gradio启动服务 |')
    model = predict_class(args)
    model_search = search_class(args)
    gradio_start()
