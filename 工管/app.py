import os
import re
import streamlit as st
from openai import OpenAI
import chromadb
from zhipuai import ZhipuAI
import jieba
import math

# ================= 0. 页面与侧边栏 =================
st.set_page_config(page_title="工程规范智能助手", page_icon="🏗️", layout="wide")
st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>""", unsafe_allow_html=True)
st.title("🏗️ 工程规范与流程智能助手 (混合检索+多轮对话版)")

# ================= 1. 读取与切分 =================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    all_text = ""
    if not os.path.exists(data_dir):
        return []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename == "guifan.pdf": continue
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
            except Exception: pass
    chunks = re.split(r'(?=第[一二三四五六七八九十百\d\.]+[条章])', all_text)
    return [c.strip() for c in chunks if len(c.strip()) > 10]

clean_chunks = load_data()

# ================= 2. 自定义智谱 Embedding =================
class ZhipuEmbeddingFunction:
    def __init__(self):
        self.client = ZhipuAI(api_key=os.environ.get("ZHIPU_API_KEY"))
        self.model_name = "embedding-3"
    def __call__(self, input: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=input)
        return [d.embedding for d in resp.data]
    def name(self) -> str: return f"zhipuai/{self.model_name}"
    def get_config(self) -> dict: return {"model": self.model_name}

# ================= 3. 初始化向量库 =================
@st.cache_resource
def get_chroma_collection(chunks):
    if not chunks: return None
    client = chromadb.EphemeralClient()
    try: client.delete_collection(name="engineering_rules")
    except: pass
    collection = client.create_collection(name="engineering_rules", embedding_function=ZhipuEmbeddingFunction())
    ids = [str(i) for i in range(len(chunks))]
    for i in range(0, len(chunks), 64):
        collection.add(documents=chunks[i : i + 64], ids=ids[i : i + 64])
    return collection

collection = get_chroma_collection(clean_chunks)

# ================= 4. 混合检索算法 (核心升级点) =================
# 原理：结合“向量相似度”和“关键词重合度”共同打分
def hybrid_search(query, k=3):
    if not collection or not clean_chunks: return []
    
    # 1. 关键词打分 (BM25 简化版：统计词频)
    jieba.load_userdict(['渗漏', '防渗漏', '保修期', '强制监理']) # 补充工程词典
    query_words = list(jieba.cut(query))
    kw_scores = {}
    for idx, chunk in enumerate(clean_chunks):
        score = sum(1 for word in query_words if word in chunk)
        kw_scores[str(idx)] = score
        
    # 2. 向量打分
    query_vector = ZhipuEmbeddingFunction()([query])
    vector_results = collection.query(query_embeddings=query_vector, n_results=10) # 先召回前10个
    vector_ids = vector_results['ids'][0]
    
    # 3. 综合打分排序
    final_scores = []
    for vid in vector_ids:
        # 向量得分权重 0.7，关键词得分权重 0.3
        v_score = 1.0 / (vector_ids.index(vid) + 1) # 排名越靠前分越高
        k_score = kw_scores.get(vid, 0) / max(max(kw_scores.values()), 1)
        combined_score = 0.7 * v_score + 0.3 * k_score
        final_scores.append((combined_score, clean_chunks[int(vid)]))
        
    final_scores.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in final_scores[:k]]

# ================= 5. 多轮对话记忆管理 =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统控制台")
    st.success(f"✅ 知识库就绪：{len(clean_chunks)} 个片段")
    if st.button("🗑️ 清空对话记忆"):
        st.session_state.messages = []
        st.rerun()

# 渲染历史对话
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ================= 6. 主界面交互 =================
if user_question := st.chat_input("例如：施工单位偷工减料会面临什么处罚？"):
    # 1. 记录用户提问
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.chat_message("user").write(user_question)
    
    with st.chat_message("assistant"):
        with st.spinner("正在双路检索并思考..."):
            # 2. 混合检索
            retrieved_chunks = hybrid_search(user_question)
            context = "\n\n".join(retrieved_chunks)
            
            # 3. 构建多轮 Prompt (取最近 3 轮对话作为记忆)
            client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
            
            # 构建历史对话文本
            history_text = ""
            for msg in st.session_state.messages[-5:-1]: # 不包含当前这句
                history_text += f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}\n"

            prompt = f"""你是一个专业的工程管理助手。
请结合【历史对话】和【参考规范】回答用户的问题。如果规范中没有答案，请回答“未在知识库中找到”。
回答完毕后，务必在末尾加上引用来源。

【历史对话】
{history_text}

【参考规范】
{context}

【用户问题】
{user_question}
"""
            try:
                response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}])
                answer = response.choices[0].message.content
                st.write(answer)
                
                # 4. 记录 AI 回答
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"模型调用失败: {e}")
                
            with st.expander("📚 查看混合检索到的原文依据"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.info(f"**参考片段 {i+1}：**\n{chunk}")
