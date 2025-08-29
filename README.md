## GPT
>基于pytorch实现的语言生成模型指令微调框架：peft模型训练
### 1，环境
>torch：https://pytorch.org/get-started/previous-versions/
>```
>pip install transformers peft==0.6 transformers_stream_generator -i https://pypi.tuna.tsinghua.edu.cn/simple
>```
### 模型下载
>hf-mirror: https://hf-mirror.com/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5  
### 知识库检索编码模型
>shibing624/text2vec-bge-large-chinese(1.2G)：https://huggingface.co/shibing624/text2vec-bge-large-chinese
### 下载方法
>```
>sudo apt-get install git-lfs：linux安装git-lfs。windows安装git时自带
>git lfs install：启用lfs。不使用lfs无法下载大文件
>git clone https://huggingface.co/Qwen/Qwen-1_8B-Chat：下载模型Qwen-1_8B-Chat
>```
### 2，predict.py
>使用模型
### 3，feature_make.py
>制作特征数据库
### 4，feature_search.py
>使用数据库
### 5，gradio_start.py
>用gradio将程序包装成一个可视化的页面，可以在网页可视化的展示
### 6，flask_start.py
>用flask将程序包装成一个服务，并在服务器上启动
### 7，flask_request.py
>以post请求传输数据调用服务
### 8，gunicorn_config.py
>用gunicorn多进程启动flask服务：gunicorn -c gunicorn_config.py flask_start:app
### 9，openai_API.py
>使用密钥用API调用openai的GPT模型来获取数据
### 10，run.py
>微调模型：训练peft模型
### 其他
>学习笔记：https://github.com/TWK2022/notebook
***
![image](README_IMAGE/001.jpg)
