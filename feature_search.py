from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage


class search_class:
    def __init__(self, args):
        self.model = HuggingFaceEmbedding(model_name=args.embed_model_path)
        storage = StorageContext.from_defaults(persist_dir=args.database_path)
        self.feature_database = load_index_from_storage(storage, embed_model=self.model)
        self.retriever = self.feature_database.as_retriever()

    def search(self, input_text):
        result_list = self.retriever.retrieve(input_text)
        return result_list[0].score, result_list[0].text

    def add_data(self):
        pass


if __name__ == '__main__':
    import argparse

    # ---------------------------------------------------------------------------------------------------------------- #
    parser = argparse.ArgumentParser('|模型预测|')
    parser.add_argument('--embed_model_path', default='text2vec_base_chinese', type=str, help='|编码模型位置|')
    parser.add_argument('--database_path', default='feature_database', type=str, help='|特征数据库|')
    parser.add_argument('--device', default='cpu', type=str, help='|设备|')
    args = parser.parse_args()
    # ---------------------------------------------------------------------------------------------------------------- #
    input_text = '知识库检索怎么做'
    model = search_class(args)
    score, text = model.search(input_text)
    print(f'| 相似度:{score} |')
    print(f'| 文本:{text} |')
