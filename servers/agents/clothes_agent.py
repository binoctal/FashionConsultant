import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)
from modelscope import snapshot_download
from modelscope_agent.agents import RolePlay

from dotenv import load_dotenv

load_dotenv()

os.environ['DASHSCOPE_API_KEY'] = os.getenv('DASHSCOPE_API_KEY') 

# load documents
documents = SimpleDirectoryReader("data/").load_data()

embedding_name='damo/nlp_gte_sentence-embedding_chinese-base'
local_embedding = snapshot_download(embedding_name)
Settings.embed_model = "local:"+local_embedding

index = VectorStoreIndex.from_documents(documents)


role_template = '服装知识库查询小助手，可以优先通过查询本地知识库来回答用户的问题'
llm_config = {
    'model': 'qwen-max', 
    'model_server': 'dashscope',
    }
function_list = []

bot = RolePlay(function_list=function_list,llm=llm_config, instruction=role_template)

index_ret = index.as_retriever(similarity_top_k=3)
query = "我的深蓝色的衣服的款型是什么？图片名字是什么？"
result = index_ret.retrieve(query)
ref_doc = result[0].text
response = bot.run(query, remote=False, print_info=True, ref_doc=ref_doc)
text = ''
for chunk in response:
    text += chunk
print(text)

