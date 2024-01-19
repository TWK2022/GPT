## GPT
>基于pytorch实现的语言生成模型指令微调框架：peft模型训练
### 1，环境
>torch：https://pytorch.org/get-started/previous-versions/
>```
>pip install transformers peft -i https://pypi.tuna.tsinghua.edu.cn/simple
>```
### *，模型下载
#### Baichuan2  
>Baichuan2官方项目：https://github.com/baichuan-inc/Baichuan2  
>Baichuan2-7B-Chat(15G)：https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat  
>Baichuan2-13B-Chat(28G)：https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat  
#### Chinese-LLaMA-Alpaca-2  
>LLaMA2官方项目：https://github.com/ymcui/Chinese-LLaMA-Alpaca-2  
>chinese-alpaca-2-1.3b(2.4G)(模型很小，可以在本地用cpu运行以便调试代码)：https://huggingface.co/hfl/chinese-alpaca-2-1.3b  
>chinese-alpaca-2-7b(13G)：https://huggingface.co/hfl/chinese-alpaca-2-7b  
>chinese-alpaca-2-13b(25G)：https://huggingface.co/hfl/chinese-alpaca-2-13b
#### 下载方法
>```
>sudo apt-get install git-lfs：linux安装git-lfs。windows安装git时自带
>git lfs install：启用lfs。不使用lfs无法下载大文件
>git clone https://huggingface.co/hfl/chinese-alpaca-2-1.3b：下载模型chinese-alpaca-2-1.3b
>```
### 2，predict.py
>使用模型
### 3，gradio_start.py
>用gradio将程序包装成一个可视化的界面，可以在网页可视化的展示
### 4，flask_start.py
>用flask将程序包装成一个服务，并在服务器上启动
### 5，flask_request.py
>以post请求传输数据调用服务
### 6，gunicorn_config.py
>用gunicorn多进程启动flask服务：gunicorn -c gunicorn_config.py flask_start:app
### 7，run.py
>微调模型：训练peft模型
### 8，openai_API.py
>使用密钥用API调用openai的GPT模型
***
![image](README_IMAGE/001.jpg)
