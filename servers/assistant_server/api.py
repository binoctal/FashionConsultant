import json
from fastapi import FastAPI, HTTPException
from assistant_server.press_oss import PressOss
from assistant_server.models import FileRequest
from assistant_server.service_utils import create_resp_msg

press_oss = PressOss()
app = FastAPI()

@app.post("/upload")
async def upload(request: FileRequest):
  
  try:
    if request.file_path:
      result = press_oss.upload_image(request.file_path)
      return create_resp_msg(f'{request.file_path} upload failed!' if result == None else f'{request.file_path} upload successful!');
  except Exception as e:
        import traceback
        print(
            f'The error is {e}, and the traceback is {traceback.format_exc()}')
        return create_resp_msg(
            status_code=400,
            message=f"Failed to execute tool '{app.tool_name}' with error {e}")