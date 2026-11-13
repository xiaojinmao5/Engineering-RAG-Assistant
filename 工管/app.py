import os
import re
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader # 引入解析 PDF 的工具

st.set_page_config(page_title="工程规范助手", page_icon="🏗️")
st.title("🏗️ 工程规范与流程智能助手")

@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    all_text = ""
    
    # 如果连 data 文件夹都没有，直接返回
    if not os.path.exists(data_dir):
        st.error(f"❌ 找不到 data 文件夹！路径：{data_dir}")
        return []
        
    # 遍历 data 文件夹里的所有文件
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        # 如果是 txt 文件
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n"
                
        # 如果是 pdf 文件
        elif filename.endswith(".pdf"):
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
            except Exception as e:
                st.warning(f"⚠️ 读取 PDF 文件 {filename} 失败: {e}")
                
    if not all_text.strip():
        return []
        
    # 把所有的文字合并后，统一进行正则切分
    chunks = re.split(r'(?=第[一二三四五六七八九十百\d]+条)', all_text)
    return [c.strip() for c in chunks if c.strip()]

clean_chunks = load_data()

# 在网页上显示加载状态
if len(clean_chunks) > 0:
    st.success(f"✅ 知识库加载成功！共识别出 {len(clean_chunks)} 个条款片段。")
else:
    st.warning("⚠️ 知识库为空，请检查 data 文件夹下是否有 txt 或 pdf 文件。")

def search(query, top_k=3): # 稍微增加返回的数量
    query_chars = set(query) 
    scores = []
    for chunk in clean_chunks:
        score = sum(1 for char in query_chars if char in chunk)
        scores.append((score, chunk))
    scores.sort(key=lambda x: x[0], reverse=True)
    # 过滤掉得分为 0 的完全无关片段（如果有的话）
    return [item[1] for item in scores[:top_k] if item[0] > 0]

user_question = st.chat_input("请输入你的工程问题...")

if user_question:
    st.chat_message("user").write(user_question)
    
    with st.chat_message("assistant"):
        with st.spinner("正在检索规范并思考..."):
            retrieved_chunks = search(user_question, top_k=3)
            
            if not retrieved_chunks:
                st.write("未在知识库中找到相关规范，请尝试换个问法。")
            else:
                context = "\n\n".join(retrieved_chunks)
                
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
