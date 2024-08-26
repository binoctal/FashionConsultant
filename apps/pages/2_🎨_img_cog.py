import base64
import os
import json
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

add_selectbox = st.sidebar.selectbox(
            "图片来源",
            ("本地上传",)
        )

if add_selectbox == '本地上传':
    uploaded_file = st.sidebar.file_uploader(label='上传图片')
else:
    img_url = st.sidebar.text_input('图片url')

uploaded_file = None
img_url = None

# 请求结果
img_base64 = None
if uploaded_file:
    st.image(uploaded_file, caption='本地图片')

    base64_data = base64.b64encode(uploaded_file.getvalue())
    img_base64 = base64_data.decode()

    #缓存到本地
    image_dir = 'images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        os.makedirs(image_dir + '/initial')
        os.makedirs(image_dir + '/final')
        os.makedirs('db')

    now = datetime.now()
    now_str = now.strftime("%Y%m%d%H%M%S") 
    file_name = image_dir +'/initial/' + now_str + '.' + uploaded_file.type.split('/')[1]

    with open(file_name,"wb") as f: 
        f.write(uploaded_file.getbuffer())

    #保存到oss
    response = requests.post(url="http://localhost:8000/upload", json={"file_path": file_name}, headers={'content-type': 'application/json'})
    print(response.json())
    st.write(response.json()['message'])

if img_url:
    st.image(img_url, caption='网络图片')