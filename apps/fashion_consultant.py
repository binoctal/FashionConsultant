import requests
import streamlit as st

st.set_page_config(
    page_title="智能时尚顾问",
)


# 假设我们有一些树形结构的数据
tree_data = {
    "Category 1": ["Item 1.1", "Item 1.2", {"Subcategory 1.1": ["Item 1.1.1", "Item 1.1.2"]}],
    "Category 2": ["Item 2.1", "Item 2.2"],
    "Category 3": ["Item 3.1", {"Subcategory 3.1": ["Item 3.1.1", "Item 3.1.2"]}]
}

st.title = "衣橱信息："

def generate_html_tree(data, level=0):
    indent = '  ' * (level * 2)  # 每一层增加两个空格的缩进
    html = f"{indent}<ul style='list-style-type: none; padding-left: {20 * level}px;'>"
    for key, value in data.items():
        html += f"<li>{key}"
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    html += generate_html_tree(item, level + 1)
                else:
                    html += f"<li>{item}</li>"
        elif isinstance(value, dict):
            html += generate_html_tree({key: value}, level + 1)
        html += "</li>"
    html += f"{indent}</ul>"
    return html

# 生成Markdown格式的树形数据
markdown_tree = generate_html_tree(tree_data)

# 创建第一行，并设置背景颜色为黄色
with st.container():
    st.markdown(
        """
        <style>
        .yellow-background {
            background-color: #262730; /* 浅灰色背景 */
            padding: 10px;
            border-radius: 5px; /* 圆角效果 */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 在Streamlit中展示Markdown格式的树形数据
    st.markdown(
        '<div class="yellow-background">'
        f'<p style="text-align:center;">{markdown_tree}</p>'
        '</div>',
        unsafe_allow_html=True
    )
    

# 创建第二行，默认背景颜色
with st.container():
    st.write("-"*50)


# 调用模型
response = requests.get(url="http://localhost:8000/summary")
print(response.json())