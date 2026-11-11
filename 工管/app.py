import os
import re
import streamlit as st
from openai import OpenAI

# ================= 1. 网页基础设置 =================
st.set_page_config(page_title="工程规范助手", page_icon="🏗️")
st.title("🏗️ 工程规范与流程智能助手")
st.caption("基于本地规范知识库的 RAG 问答系统")

# ================= 2. 读取并切分文档 =================
# 使用缓存，避免每次刷新网页都重新切分
@st.cache_data
def load_data():
    with open("data/规范.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = re.split(r'(?=第[一二三四五六七八九十百\d]+条)', text)
    return [c.strip() for c in chunks if c.strip()]

clean_chunks = load_data()

# ================= 3. 检索函数 =================
def search(query, top_k=2):
    query_chars = set(query) 
    scores = []
    for chunk in clean_chunks:
        score = sum(1 for char in query_chars if char in chunk)
        scores.append((score, chunk))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scores[:top_k]]

# ================= 4. 页面交互 =================
user_question = st.chat_input("请输入你的工程问题，例如：混凝土浇筑后要浇水吗？")

if user_question:
    # 显示用户的提问
    st.chat_message("user").write(user_question)
    
    # 检索并生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在检索规范并思考..."):
            # 检索
            retrieved_chunks = search(user_question, top_k=2)
            context = "\n".join(retrieved_chunks)
            
            # 调大模型
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
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
            
            # 在网页上输出回答
            st.write(answer)
            
            # 把检索到的原文放进一个可折叠的框里（体现“引用溯源”）
            with st.expander("📚 查看引用的规范原文"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.info(f"**参考片段 {i+1}：**\n{chunk}")
