import os
import re
from typing import Dict, Optional

import json
import time
import math
import pandas as pd
import requests
from modelscope_agent.constants import ApiNames
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.utils.utils import get_api_key
from utils.press_db import get_clothes
from datetime import datetime

@register_tool('image2image')
class Image2image(BaseTool):
    description = '虚拟试衣'
    name = 'image2image'
    parameters: list = [{
        'name': 'top',
        'type': 'string',
        'description': '上半身服饰图片URL',
        'required': True
    },
    {
        'name': 'bottom',
        'type': 'string',
        'description': '下半身服饰图片URL',
        'required': False
    },
    {
        'name': 'person',
        'type': 'string',
        'description': '模特人物图片URL',
        'required': False
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)

        # remote call
        self.request_url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis/'
        self.query_url = 'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}'
        self.default_person = 'https://dashscope-swap.oss-cn-beijing.aliyuncs.com/aa-test/sample-person.png'

        self.api_key = self.cfg.get(
            ApiNames.dashscope_api_key.name,
            os.environ.get(ApiNames.dashscope_api_key.value, ''))

    def parse_image_format(self, url: str):
        if '.jpg' in url:
            return 'jpg'
        elif '.png' in url:
            return 'png'
        elif '.jpeg' in url:
            return 'jpeg'
        elif '.bmp' in url:
            return 'bmp'
        else:
            return 'jpg'

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        try:
            # print(os.environ['DASHSCOPE_API_KEY'] )
            self.api_key = get_api_key(ApiNames.dashscope_api_key, self.api_key, **kwargs)
        except AssertionError:
            raise ValueError('Please set valid DASHSCOPE_API_KEY!')

        top_garment_url = ''
        top_name = params['top']
        if top_name is None:
            return None
        else:
            top_garment_url = get_clothes(top_name)

        bottom_garment_url = ''
        if 'bottom' in params:
            bottom_name = params['bottom']
            if bottom_name is None:
                bottom_garment_url = ''
            elif bottom_name.isdigit():
                bottom_garment_url = get_clothes(bottom_name)
            else:
                bottom_garment_url = ''
        else:
            bottom_garment_url = ''

        person_image_url = ''
        if 'person' in params:
            person_name = params['person']
            if person_name is None:
                person_image_url = self.default_person
            elif person_name.isdigit():
                person_image_url = get_clothes(person_name)
            else:
                person_image_url = self.default_person
        else:
            person_image_url = self.default_person


         # 参考api详情，确定headers参数
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'X-DashScope-Async': 'enable'
        }

        input_data = {
            "model": "aitryon",
            "input": {
                "top_garment_url": top_garment_url,
                "bottom_garment_url": bottom_garment_url,
                "person_image_url": person_image_url
            },
            "parameters": {
                "resolution": -1,
                "restore_face": True
            }
        }

                # requests请求
        response = requests.post(url=self.request_url, headers=headers, data=json.dumps(input_data))        
        data = response.json()
        task_id = data['output']['task_id']

        # 查询任务
        headers = { 'Authorization': f'Bearer {self.api_key}' }

        retry_times = 35
        delay_cnt = 0
        limit_num = 3.5
        step_num = 0.25
        epsilon = 1e-9 
        while retry_times:
            delay_num = limit_num - step_num * delay_cnt
            if delay_num > 0 and not math.isclose(delay_num, 1.0, abs_tol=epsilon):
                time.sleep(delay_num)
            else:
                time.sleep(1)

            retry_times -= 1
            delay_cnt += 1

            response = requests.get(self.query_url.format(task_id=task_id), headers=headers)
            data = response.json()

            if response.status_code == 400:
                raise RuntimeError(data)
            elif response.status_code == 200:
                # print('task_status: ', data['output']['task_status'])
                if data['output']['task_status'] == 'SUCCEEDED':
                    final_image_url = data['output']['image_url']

                    image_format = self.parse_image_format(final_image_url)
                    now_str = datetime.now().strftime("%Y%m%d%H%M%S") 
                    img_path = 'images/final/' + now_str + '.' + image_format
                                 
                    response = requests.get(final_image_url, headers=headers)
                    with open(img_path, 'wb') as f:
                        f.write(response.content)

                    return f'您穿搭的结果是{img_path}。'    
        
        if retry_times == -1:
            return f'无法进行获取穿搭结果。'  