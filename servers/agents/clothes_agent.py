# 配置环境变量；如果您已经提前将api-key提前配置到您的运行环境中，可以省略这个步骤
import os
import re
from dotenv import load_dotenv

load_dotenv()

os.environ['DASHSCOPE_API_KEY'] = os.getenv('DASHSCOPE_API_KEY') 
os.environ['AMAP_TOKEN'] = os.getenv('AMAP_TOKEN') 

# 选用RolePlay 配置agent
from modelscope_agent.agents.role_play import RolePlay  # 

class ClothesAgent():
    role_template = '''你扮演一个时尚设计师，能够通过图片准确的给出衣服或者裤子的颜色、衣服分类（上衣，下装和整件装）、款型、风格、适合季节以及图案描述等信息。最终结果请在一行给出以下格式：
                       图片名字： ***; 颜色： ***； 分类： 限定为上衣，下装或者整件装之一； 款型： ***, 风格： ***； 适合季节： ***； 图案描述： ***； 详细描述： ***。
                    '''
    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}
    
    # input tool name
    # function_list = ['qwen_vl']

    @classmethod
    def run(cls, image_path: str):
        print('xin]: image_path: ', image_path)
        if len(image_path) == 0:
            return
        
        bot = RolePlay(llm=cls.llm_config, instruction=cls.role_template)

        response = bot.run(f'[上传文件{image_path}],描述这张照片')

        print('[xin]: response: ', response)

        text = ''
        for chunk in response:
            text += chunk

        print('[xin]: text: ', text)

        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        #处理数据
        # pattern = r'Answer:([^Answer:]*)'
        # result = re.findall(pattern, text)
        result = text
        print('[xin]: result: ', result)

        if len(result) > 0:
            with open(data_dir + '/clothes_data.txt',"a", encoding='utf-8') as f: 
                f.write(result + '\n')     