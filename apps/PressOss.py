# -*- coding: utf-8 -*-
import os
import oss2
from dotenv import load_dotenv

class PressOss():
  __instance = None
  __is_init = False

  def __new__(cls):
      if not cls.__instance:
          cls.__instance = super(PressOss, cls).__new__(cls)
      return cls.__instance

  def __init__(self):
    if not self.__is_init:
      load_dotenv()
      endpoint = os.getenv('OSS_TEST_ENDPOINT')
      access_key_id = os.getenv('OSS_TEST_ACCESS_KEY_ID')
      access_key_secret = os.getenv('OSS_TEST_ACCESS_KEY_SECRET')
      buck = os.getenv('OSS_TEST_BUCKET') 

      auth = oss2.Auth(access_key_id, access_key_secret)
      self.bucket = oss2.Bucket(auth, endpoint, buck)
      
      __is_init = True
    
  def upload_image(self, image_path): 
    tmp_token = image_path.split('/')   
    image_name = tmp_token[len(tmp_token) - 1]

    print("image_name: " + image_name)

    # # Upload
    self.bucket.put_object_from_file('press/' + image_name, image_path)

if __name__ == "__main__":
  press_oss = PressOss()
  press_oss.upload_image('./images/top5.png')