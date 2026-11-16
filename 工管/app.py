import os
import re
import streamlit as st
from openai import OpenAI
import chromadb
from zhipuai import ZhipuAI
import jieba
import requests

# ================= 0. 页面与全局 CSS =================
st.set_page_config(page_title="工程规范智能助手", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 隐藏主页面和侧边栏的滚动条 */
    ::-webkit-scrollbar { width: 0px; background: transparent; }
    
    /* 强制固定侧边栏宽度 */
    [data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 320px !important;
    }
    [data-testid="stSidebarCollapseButton"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    
    /* 极限压缩侧边栏边距 */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    [data-testid="stSidebar"] .element-container { margin-bottom: 0.2rem; }
    [data-testid="stSidebar"] hr { margin-top: 0.4rem; margin-bottom: 0.4rem; }
    [data-testid="stSidebar"] [data-testid="stExpander"] { margin-bottom: 0.4rem; }
    
    /* 【修改：去掉蓝色，改成深灰色】 */
    .tiny-config { font-size: 11px; color: #6B7280; line-height: 1.6; margin-top: 5px; }
    .tiny-config span { color: #374151; font-weight: 600; }
    
    /* 历史记录样式 */
    .history-item { margin-bottom: 8px; padding: 6px 0 6px 8px; font-size: 12px; line-height: 1.4; }
    .history-user { border-left: 2px solid #9CA3AF; color: #374151; }
    .history-assistant { border-left: 2px solid #3B82F6; color: #374151; }
    .history-role { font-weight: 600; font-size: 11px; margin-bottom: 2px; display: block; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ 工程规范与流程智能助手")

USER_AVATAR = "https://img.icons8.com/ios/50/6B7280/user--v1.png"
AI_AVATAR = "https://img.icons8.com/ios/50/3B82F6/bot.png"

# ================= 1. 读取与切分 =================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    all_text = ""
    file_list = []
    if not os.path.exists(data_dir): return [], []
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename == "guifan.pdf": continue
        file_list.append(filename)
        
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
    return [c.strip() for c in chunks if len(c.strip()) > 10], file_list

clean_chunks, loaded_files = load_data()

# ================= 2. 智谱 Embedding 与 向量库 =================
class ZhipuEmbeddingFunction:
    def __init__(self):
        self.client = ZhipuAI(api_key=os.environ.get("ZHIPU_API_KEY"))
        self.model_name = "embedding-3"
    def __call__(self, input: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=input)
        return [d.embedding for d in resp.data]
    def name(self) -> str: return f"zhipuai/{self.model_name}"
    def get_config(self) -> dict: return {"model": self.model_name}

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

# ================= 3. 混合检索 (海选) =================
def hybrid_search(query, k=10): 
    if not collection or not clean_chunks: return []
    jieba.add_word('防渗漏')
    jieba.add_word('保修期')
    jieba.add_word('强制监理')
    query_words = list(jieba.cut(query))
    kw_scores = {}
    for idx, chunk in enumerate(clean_chunks):
        score = sum(1 for word in query_words if word in chunk)
        kw_scores[str(idx)] = score
    query_vector = ZhipuEmbeddingFunction()([query])
    vector_results = collection.query(query_embeddings=query_vector, n_results=k)
    vector_ids = vector_results['ids'][0]
    final_scores = []
    for vid in vector_ids:
        v_score = 1.0 / (vector_ids.index(vid) + 1) 
        k_score = kw_scores.get(vid, 0) / max(max(kw_scores.values()), 1)
        combined_score = 0.7 * v_score + 0.3 * k_score
        final_scores.append((combined_score, clean_chunks[int(vid)]))
    final_scores.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in final_scores]

# ================= 4. Reranker 重排序 (复试) =================
def rerank_chunks(query, chunks, top_n=3):
    if not chunks: return []
    url = "https://api.siliconflow.cn/v1/rerank"
    payload = {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": query,
        "documents": chunks,
        "top_n": top_n,
        "return_documents": False
    }
    headers = {
        "Authorization": f"Bearer {os.environ.get('SILICONFLOW_API_KEY')}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json()
        reranked = [chunks[item['index']] for item in results['results']]
        return reranked
    except Exception as e:
        st.warning(f"Reranker 降级处理：{e}")
        return chunks[:top_n] 

# ================= 5. 侧边栏与交互 =================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

with st.sidebar:
    st.caption("⚙️ **系统控制台**")
    st.success(f"✅ 知识库：{len(clean_chunks)} 个片段")
    
    # 1. 模型参数调节
    st.caption("🎛️ **模型参数**")
    temperature = st.slider("发散度 (Temperature)", min_value=0.0, max_value=1.0, value=0.2, step=0.1, help="越低越严谨（推荐工程场景），越高越有创造力。")

    # 2. 知识库文件
    with st.expander("📁 已加载知识库", expanded=False):
        if loaded_files:
            for file in loaded_files:
                st.caption(f"📄 {file}")
        else:
            st.caption("暂无文件")

    # 3. 快捷提问
    st.caption("💡 **快捷提问**")
    if st.button("外墙渗水保修几年？", use_container_width=True):
        st.session_state.quick_prompt = "外墙渗水保修几年？"
    if st.button("偷工减料会受什么处罚？", use_container_width=True):
        st.session_state.quick_prompt = "施工单位偷工减料会面临什么处罚？"

    if st.button("🗑️ 清空对话记忆", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    
    # 4. 历史记忆
    with st.expander("📜 历史记忆回顾", expanded=False):
        if st.session_state.messages:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="history-item history-user"><span class="history-role">用户</span>{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="history-item history-assistant"><span class="history-role">AI 助手</span>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.caption("暂无历史记录~")
            
    st.divider()
    
    # 5. 底层架构 (深灰色)
    st.markdown("""
    <div class="tiny-config">
    💡 <b>底层架构</b><br>
    • LLM: <span>DeepSeek-chat</span><br>
    • 向量: <span>智谱 embedding-3</span><br>
    • 召回: <span>混合检索 (Top-10)</span><br>
    • 精排: <span>BGE-Reranker (Top-3)</span>
    </div>
    """, unsafe_allow_html=True)

# --- 欢迎语 ---
if len(st.session_state.messages) == 0:
    st.info("👋 你好！我是工程规范智能助手。你可以问我关于《建设工程质量管理条例》的任何问题。")

# 渲染主界面历史对话
for msg in st.session_state.messages:
    avatar_img = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
    st.chat_message(msg["role"], avatar=avatar_img).write(msg["content"])

# ================= 6. 主界面逻辑 =================
# 处理快捷按钮的输入
user_question = st.chat_input("例如：施工单位偷工减料会面临什么处罚？")
if st.session_state.quick_prompt:
    user_question = st.session_state.quick_prompt
    st.session_state.quick_prompt = None

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.chat_message("user", avatar=USER_AVATAR).write(user_question)
    
    with st.chat_message("assistant", avatar=AI_AVATAR):
        with st.spinner("正在检索、重排并思考..."):
            client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
            
            # 查询重写
            search_query = user_question
            if len(st.session_state.messages) > 1:
                history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:-1]])
                rewrite_prompt = f"根据历史对话，将最新追问改写为独立搜索句，直接输出问题。\n历史对话：\n{history}\n最新追问：{user_question}"
                try:
                    rw_res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": rewrite_prompt}])
                    search_query = rw_res.choices[0].message.content.strip()
                    st.caption(f"💡 结合上下文，系统正在检索：**{search_query}**")
                except Exception: pass
            
            # 海选 + 精排
            raw_chunks = hybrid_search(search_query, k=10) 
            retrieved_chunks = rerank_chunks(search_query, raw_chunks, top_n=3) 
            
            context = "\n\n".join(retrieved_chunks)
            
            history_text = ""
            for msg in st.session_state.messages[-5:-1]:
                history_text += f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}\n"

            prompt = f"""你是一个专业的工程管理助手。
请结合【历史对话】和【参考规范】回答用户的问题。如果规范中没有答案，请回答“未在知识库中找到”。
回答完毕后，务必在末尾加上引用来源。

【历史对话】
{history_text}

【参考规范】
{context}

【当前用户问题】
{search_query}
"""
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat", 
                    messages=[{"role": "user", "content": prompt}], 
                    stream=True,
                    temperature=temperature # 新增：接入了滑块的参数
                )
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                
                answer = st.write_stream(stream_generator())
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"模型调用失败: {e}")
                
            with st.expander("📚 查看重排后精选的原文依据"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.info(f"**参考片段 {i+1}：**\n{chunk}")
