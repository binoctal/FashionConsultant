
import requests
import streamlit as st

# 初次运行时，session_state中没有"messages"，需要创建一个空列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 每次对话时，都需要遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 将用户的输入添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    print('[xin]: user prompt: ', prompt)

    # 调用模型
    response = requests.post(url="http://localhost:8000/chat", json={"text": prompt}, headers={'content-type': 'application/json'})
    print(response.json())

    message_text = response.json()['message']

    # 将模型的输出添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "assistant", "content": message_text})

    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(message_text)