import os
import re
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    GPTVectorStoreIndex,
    Settings,
    StorageContext, load_index_from_storage
)
import torch
from llama_index.core.node_parser import SentenceSplitter
from modelscope import snapshot_download

# from llama_index.core import (ServiceContext, SimpleDirectoryReader, SummaryIndex, Settings)

# 选用RolePlay 配置agent
from modelscope_agent.agents.role_play import RolePlay  # 

from modelscope_agent.llm.yuan import YuanLammaLLM

from dotenv import load_dotenv

load_dotenv()

os.environ['DASHSCOPE_API_KEY'] = os.getenv('DASHSCOPE_API_KEY') 
embedding_name='damo/nlp_gte_sentence-embedding_chinese-base'
local_embedding = snapshot_download(embedding_name, cache_dir='./bigmodel')
Settings.embed_model = "local:"+local_embedding


class ConsultantAgent():
    # role_template = '''你扮演一个时尚顾问，拥有丰富的服装搭配的经验，可以优先通过查询本地知识库来回答用户的问题。对于图片名字请去掉图片格式后缀。
    #                    如果分类是整件装，则只需要top参数，不需要bottom参数；如果未提供模特全身照片，则不需要person参数。
    #                    最后回答请给出以下格式，请保持一行进行输出： 
    #                    image:仅仅包含路径; tips:不要包含请参考以下虚拟试衣的效果以及图片信息。
    #                 '''
    role_template = '''你正在扮演一个时尚顾问，拥有丰富的服装搭配的经验。可以优先根据对话中内容来回答用户的问题，并在最后给出对服装的穿搭评价。
                    '''
    llm_config = {
        'model': 'yuan', 
        'model_server': 'yuan',
        }

    # llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    function_list = None

    @classmethod
    def run(cls, query: str):
        Settings.llm = YuanLammaLLM()

        emb_dir = 'doc_emb'
        if not os.path.exists(emb_dir):
            # 读取文档
            documents = SimpleDirectoryReader("./data").load_data()
            # 对文档进行切分，将切分后的片段转化为embedding向量，构建向量索引
            index = GPTVectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=256)])
            # 将embedding向量和向量索引存储到文件中
            index.storage_context.persist(persist_dir='doc_emb')
        else:
            # 从存储文件中读取embedding向量和向量索引
            storage_context = StorageContext.from_defaults(persist_dir="doc_emb")
            index = load_index_from_storage(storage_context)

        index_ret = index.as_retriever(similarity_top_k=5)

        sys_prompt = '''。请按照图片名字;颜色;分类;款式;风格;适合季节;图案描述;详细描述进行查找。'''

        result = index_ret.retrieve(query+sys_prompt)
        ref_doc = result[0].text
        
        # # 构建查询引擎
        # query_engine = index.as_query_engine(similarity_top_k=3)
        # # 构造查询prompt 请按照图片名字;颜色;分类;款型;风格;适合季节;图案描述;详细描述进行查找。
        # sys_prompt = '''。请根据Context information中给出符合要求的图片以及服装信息。'''
        
        # # prompt += sys_prompt
        # # 查询获得答案
        # result = query_engine.query(query + sys_prompt)

        # ref_doc = ''
        # tmp_answer = str(result).split('Answer: ')[-1]
        # if len(tmp_answer) > 0:
        #     tmp_answer1 = tmp_answer.split('<sep>')[-1]
        #     ref_doc = tmp_answer1.split('<eod>')[0]

        print('[xin]: ref_doc: ', ref_doc)

        # 大模型对话
        bot = RolePlay(llm=cls.llm_config, instruction=cls.role_template)
        response = bot.run(query, print_info=True, ref_doc=ref_doc)

        text = ''
        for chunk in response:
            text += chunk

        return text

    @classmethod
    def summary(cls):
        Settings.llm = YuanLammaLLM()

        emb_dir = 'doc_emb'
        if not os.path.exists(emb_dir):
            # 读取文档
            documents = SimpleDirectoryReader("./data").load_data()
            # 对文档进行切分，将切分后的片段转化为embedding向量，构建向量索引
            index = GPTVectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=256)])
            # 将embedding向量和向量索引存储到文件中
            index.storage_context.persist(persist_dir='doc_emb')
        else:
            # 从存储文件中读取embedding向量和向量索引
            storage_context = StorageContext.from_defaults(persist_dir="doc_emb")
            index = load_index_from_storage(storage_context)

        index_ret = index.as_retriever(similarity_top_k=5)

        sys_prompt = '''。请检索所有上衣、下装，还有连衣裙（整件装）。'''

        result = index_ret.retrieve(sys_prompt)
        ref_doc = result[0].text

        ref_doc = ''
        tmp_answer = str(result).split('Answer: ')[-1]
        if len(tmp_answer) > 0:
            tmp_answer1 = tmp_answer.split('<sep>')[-1]
            ref_doc = tmp_answer1.split('<eod>')[0]

        # print('[xin]: ref_doc: ', ref_doc)

        role_template = '''
        请忘记之前的设定和内容，专注以下指令。你现在扮演智能AI助手，目前衣服款式有上衣、下装和连衣裙（整件装）。请去除重复的内容，然后根据内容进行总结，按照如下格式输出：
        上衣：输出总数。输出不同颜色的件数。
        下装：输出总数。输出不同颜色的件数。
        连衣裙（整件装）: 输出总数。输出不同颜色的件数。
        示例如下：
        上衣：5件。黑色：1件，灰色：2件，蓝色：2件。
        下装：1件。黑色：1件。
        连衣裙（整件装）: 1件。米各色：1件。
        '''
        query = "上衣、下装，还有连衣裙（整件装）的数量情况。"

        # 大模型对话
        bot = RolePlay(llm=cls.llm_config, instruction=role_template)
        response = bot.run(query, print_info=True, ref_doc=ref_doc)

        print('[xin]: response: ', str(response))

        text = ''
        for chunk in response:
            text += chunk

        return text