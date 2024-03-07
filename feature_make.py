import os
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# -------------------------------------------------------------------------------------------------------------------- #
# 数据格式：
# 1级目录(args.data_path)
# ------|2级目录(逻辑上加的)
# -------------|3级目录(最小操作单位document)
# --------------------|4级目录(划分好的.txt，最小存储单位node)
# -------------------------------------------------------------------------------------------------------------------- #
def database_make(args):
    document_list = []
    for path_2 in os.listdir(args.data_path):
        path_2_abs = f'{args.data_path}/{path_2}'
        for path_3 in os.listdir(path_2_abs):
            path_3_abs = f'{path_2_abs}/{path_3}'
            for path_4 in os.listdir(path_3_abs):
                path_4_abs = f'{path_3_abs}/{path_4}'
                with open(path_4_abs, 'r', encoding='utf-8') as f:
                    text = f.read()
                document = Document(text=text, id_=path_3_abs)
                document_list.append(document)
    model = HuggingFaceEmbedding(model_name=args.embed_model_path)
    feature_database = VectorStoreIndex.from_documents(document_list, embed_model=model)
    feature_database.storage_context.persist(persist_dir=args.save_path)
    print(f'| document集合:{len(feature_database.ref_doc_info)} | node总数:{len(feature_database.docstore.docs)} '
          f'| 保存位置:{args.save_path} |')


if __name__ == '__main__':
    import argparse

    # ---------------------------------------------------------------------------------------------------------------- #
    parser = argparse.ArgumentParser('|模型预测|')
    parser.add_argument('--embed_model_path', default='text2vec-bge-large-chinese', type=str, help='|编码模型位置|')
    parser.add_argument('--data_path', default='data_knowledge', type=str, help='|数据库文件夹(含.txt文件)|')
    parser.add_argument('--save_path', default='feature_database', type=str, help='|保存的特征数据库位置|')
    parser.add_argument('--device', default='cpu', type=str, help='|设备|')
    args = parser.parse_args()
    # ---------------------------------------------------------------------------------------------------------------- #
    database_make(args)
