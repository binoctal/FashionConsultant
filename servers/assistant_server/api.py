import json
import re
from fastapi import FastAPI, HTTPException
from assistant_server.models import FileRequest, ChatRequest
from assistant_server.service_utils import create_resp_msg
from utils.press_oss import PressOss
from utils.press_db import create_db_and_tables, add_clothes, Clothes
from agents.clothes_agent import ClothesAgent
from agents.consultant_agent import ConsultantAgent

create_db_and_tables()
press_oss = PressOss()
app = FastAPI()


@app.post("/upload")
async def upload(request: FileRequest):
  
  try:
    if len(request.file_path) > 0:
      result = press_oss.upload_image(request.file_path)

      if len(result) == 0:
        return create_resp_msg(f'{request.file_path} upload failed!');
      else:
        #AI识别图片
        ClothesAgent.run(request.file_path)

        #更新数据库
        pattern = r'initial/(.*?)\.' 
        match_ret = re.findall(pattern, request.file_path)
        add_clothes(Clothes(name=match_ret[0], url=result))
        return create_resp_msg(f'{request.file_path} upload successful!')

  except Exception as e:
        import traceback
        print(
            f'The error is {e}, and the traceback is {traceback.format_exc()}')
        return create_resp_msg(
            status_code=400,
            message=f"Failed to execute tool '{app.tool_name}' with error {e}")
  

@app.post("/chat")
async def chat(request: ChatRequest):
  
  try:
    if len(request.text) > 0:
      #AI对话
      resp = ConsultantAgent.run(request.text)
      return create_resp_msg(resp)
  except Exception as e:
        import traceback
        print(
            f'The error is {e}, and the traceback is {traceback.format_exc()}')
        return create_resp_msg(
            status_code=400,
            message=f"Failed to execute chat")  
        
@app.get("/summary")
async def summary():  
  try:
    print('this is summary')
    #AI对话
    resp = ConsultantAgent.summary()
    return create_resp_msg(resp)      
  except Exception as e:
        import traceback
        print(
            f'The error is {e}, and the traceback is {traceback.format_exc()}')
        return create_resp_msg(
            status_code=400,
            message=f"Failed to execute summary")  