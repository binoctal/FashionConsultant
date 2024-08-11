from pydantic import BaseModel


class FileRequest(BaseModel):
    file_path: str

