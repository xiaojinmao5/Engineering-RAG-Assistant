import os
import re
import streamlit as st
from openai import OpenAI
import chromadb
from zhipuai import ZhipuAI

st.set_page_config(page_title="工程规范助手", page_icon="🏗️")
st.title("🏗️ 工程规范与流程智能助手 (语义检索版)")

@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    all_text = ""
    if not os.path.exists(data_dir):
        return []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename == "guifan.pdf":
            continue
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n"
        elif filename.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text: all_text += text + "\n"
            except Exception:
                pass
    if not all_text.strip():
        return []
    chunks = re.split(r'(?=第[一二三四五六七八九十百\d\.]+条)', all_text)
    return [c.strip() for c in chunks if len(c.strip()) > 10]

clean_chunks = load_data()

class ZhipuEmbeddingFunction:
    def __init__(self):
        self.client = ZhipuAI(api_key=os.environ.get("ZHIPU_API_KEY"))
        self.model_name = "embedding-3"
    def __call__(self, input: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=input)
        return [d.embedding for d in resp.data]
    def name(self) -> str:
        return f"zhipuai/{self.model_name}"
    def get_config(self) -> dict:
        return {"model": self.model_name}

@st.cache_resource
def get_chroma_collection(chunks):
    if not chunks:
        return None
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection(name="engineering_rules")
    except:
        pass
    collection = client.create_collection(
        name="engineering_rules", 
        embedding_function=ZhipuEmbeddingFunction() 
    )
    ids = [str(i) for i in range(len(chunks))]
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=chunks[i : i + batch_size],
            ids=ids[i : i + batch_size]
        )
    return collection

collection = get_chroma_collection(clean_chunks)

if collection and len(clean_chunks) > 0:
    st.success(f"✅ 语义知识库加载成功！共建立 {len(clean_chunks)} 个向量索引。")
else:
    st.warning("⚠️ 知识库为空，请检查 data 文件夹。")

user_question = st.chat_input("请输入你的工程问题...")

if user_question:
    st.chat_message("user").write(user_question)
    with st.chat_message("assistant"):
        with st.spinner("正在理解语义并检索规范..."):
            if collection:
                # 【核心修复】手动计算查询向量
                embedding_func = ZhipuEmbeddingFunction()
                query_vector = embedding_func([user_question])
                results = collection.query(
                    query_embeddings=query_vector,
                    n_results=3
                )
                try:
                    retrieved_chunks = results['documents'][0] if results['documents'] else []
                except (TypeError, KeyError):
                    retrieved_chunks = results.documents[0] if results.documents else []
            else:
                retrieved_chunks = []
                
            if not retrieved_chunks:
                st.write("未在知识库中找到相关规范。")
            else:
                context = "\n\n".join(retrieved_chunks)
                client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
                prompt = f"""你是一个专业的工程管理助手。请只根据以下【参考规范】回答用户的问题。如果规范中没有答案，请回答“未在知识库中找到”。回答完毕后，务必在末尾加上引用来源（具体到第X条）。
【参考规范】\n{context}\n【用户问题】\n{user_question}"""
                try:
                    response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}])
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"模型调用失败: {e}")
                
                with st.expander("📚 查看语义检索到的规范原文"):
                    for i, chunk in enumerate(retrieved_chunks):
                        st.info(f"**参考片段 {i+1}：**\n{chunk}")
