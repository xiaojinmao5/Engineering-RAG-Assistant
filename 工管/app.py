import os
import re
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="工程规范助手", page_icon="🏗️")
st.title("🏗️ 工程规范与流程智能助手")

@st.cache_data
def load_data():
    # 获取当前 app.py 所在的文件夹路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "规范.txt")
    
    # 如果找不到文件，直接返回空列表，防止网页崩溃
    if not os.path.exists(file_path):
        st.error(f"❌ 找不到文件！系统尝试读取的路径是：{file_path}")
        return []
        
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = re.split(r'(?=第[一二三四五六七八九十百\d]+条)', text)
    return [c.strip() for c in chunks if c.strip()]

clean_chunks = load_data()

# 在网页上显示是否加载成功
if len(clean_chunks) > 0:
    st.success(f"✅ 知识库加载成功！共识别出 {len(clean_chunks)} 个条款片段。")
else:
    st.warning("⚠️ 知识库为空，请检查 data 文件夹下的规范.txt")

def search(query, top_k=2):
    query_chars = set(query) 
    scores = []
    for chunk in clean_chunks:
        score = sum(1 for char in query_chars if char in chunk)
        scores.append((score, chunk))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scores[:top_k]]

user_question = st.chat_input("请输入你的工程问题...")

if user_question:
    st.chat_message("user").write(user_question)
    
    with st.chat_message("assistant"):
        with st.spinner("正在检索规范并思考..."):
            retrieved_chunks = search(user_question, top_k=2)
            context = "\n".join(retrieved_chunks)
            
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
            
            prompt = f"""你是一个专业的工程管理助手。
请只根据以下【参考规范】回答用户的问题。如果规范中没有答案，请回答“未在知识库中找到”。
回答完毕后，请务必在末尾加上引用来源。

【参考规范】
{context}

【用户问题】
{user_question}
"""
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"模型调用失败: {e}")
            
            with st.expander("📚 查看引用的规范原文"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.info(f"**参考片段 {i+1}：**\n{chunk}")
