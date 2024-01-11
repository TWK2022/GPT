import openai
import argparse

parser = argparse.ArgumentParser(description='|openai_API|')
parser.add_argument('--api_key', default='', type=str, help='|密钥|')
parser.add_argument('--model', default='gpt-3.5-turbo-0613', type=str, help='|模型选择，如gpt-3.5-turbo-0613|')
parser.add_argument('--temperature', default=0.8, type=float, help='|值越小回答越简短稳定，默认1|')
args = parser.parse_args()


class openai_API:
    def __init__(self, args):
        self.API = openai.OpenAI(api_key=args.api_key)
        self.model = args.model
        self.temperature = args.temperature

    def predict(self, message):
        response = self.API.chat.completions.create(model=self.model, temperature=self.temperature, messages=message)
        result = response.choices[0].message.content
        return result


if __name__ == '__main__':
    API = openai_API(args)
    message = [{'role': 'system', 'content': '乐于助人的知识专家'},
               {'role': 'user', 'content': '大模型训练时要注意什么'}]
    result = API.predict(message)
    print(result)
