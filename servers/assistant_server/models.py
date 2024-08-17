from pydantic import BaseModel


class FileRequest(BaseModel):
    file_path: str

class ChatRequest(BaseModel):
    text: str