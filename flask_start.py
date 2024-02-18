# pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用flask将程序包装成一个服务，并在服务器上启动
import json
import flask
import argparse
from predict import predict_class

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser('|在服务器上启动flask服务|')
parser.add_argument('--model_path', default='Qwen-1_8B-Chat', type=str, help='|tokenizer和模型文件夹位置|')
parser.add_argument('--peft_model_path', default='', type=str, help='|peft模型文件夹位置(空则不使用)|')
parser.add_argument('--model', default='qwen', type=str, help='|模型类型|')
parser.add_argument('--system', default='', type=str, help='|追加的系统提示词|')
parser.add_argument('--temperature', default=0.2, type=float, help='|回答稳定概率，0.2-0.8，越小越稳定|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()
app = flask.Flask(__name__)  # 创建一个服务框架


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
@app.route('/test/', methods=['POST'])  # 每当调用服务时会执行一次flask_app函数
def flask_app():
    request_json = flask.request.get_data()
    request_dict = json.loads(request_json)
    system = request_dict['system']
    input_ = request_dict['input']
    result = model.predict(system=system, input_=input_)
    return result


if __name__ == '__main__':
    print('| 使用flask启动服务 |')
    model = predict_class(args)
    app.run(host='0.0.0.0', port=9999, debug=False)  # 启动服务
