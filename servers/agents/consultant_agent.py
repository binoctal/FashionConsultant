import os
import re
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)
from modelscope import snapshot_download

# 选用RolePlay 配置agent
from modelscope_agent.agents.role_play import RolePlay  # 

from dotenv import load_dotenv

load_dotenv()

os.environ['DASHSCOPE_API_KEY'] = os.getenv('DASHSCOPE_API_KEY') 
embedding_name='damo/nlp_gte_sentence-embedding_chinese-base'
local_embedding = snapshot_download(embedding_name)
Settings.embed_model = "local:"+local_embedding

class ConsultantAgent():
    role_template = '''你扮演一个时尚顾问，拥有丰富的服装搭配的经验，可以优先通过查询本地知识库来回答用户的问题。对于图片名字请去掉图片格式后缀。
                       如果分类是整件装，则只需要top参数，不需要bottom参数；如果未提供模特全身照片，则不需要person参数。
                       最后回答请给出以下格式，请保持一行进行输出： 
                       image:仅仅包含路径; tips:不要包含请参考以下虚拟试衣的效果以及图片信息;。
                    '''
    llm_config = {
        'model': 'qwen-max', 
        'model_server': 'dashscope',
        }
    function_list = ['image2image']

    @classmethod
    def run(cls, query: str):
        # load documents
        documents = SimpleDirectoryReader("data/").load_data()
        index = VectorStoreIndex.from_documents(documents)

        bot = RolePlay(function_list=cls.function_list,llm=cls.llm_config, instruction=cls.role_template)

        index_ret = index.as_retriever(similarity_top_k=3)
        result = index_ret.retrieve(query)
        ref_doc = result[0].text
        response = bot.run(query, remote=False, print_info=True, ref_doc=ref_doc)
        text = ''
        for chunk in response:
            text += chunk
        print(text)

        #处理数据
        pattern = r'Answer:(.*)'
        result = re.findall(pattern, text)
        print(result)

        if len(result) > 0:
            return result[0]
        else:
            return text

