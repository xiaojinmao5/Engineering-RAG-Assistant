import os
import re
import streamlit as st
from openai import OpenAI
import chromadb
from zhipuai import ZhipuAI
import jieba
import requests
import io

# ================= 0. 页面与全局 CSS =================
st.set_page_config(page_title="工程规范智能助手", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    ::-webkit-scrollbar { display: none !important; }
    [data-testid="stSidebar"] { width: 320px !important; min-width: 320px !important; background-color: #FAFAFA !important; }
    [data-testid="stSidebarCollapseButton"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    [data-testid="stSidebarUserContent"] { padding: 1rem 0.5rem !important; }
    [data-testid="stSidebar"] .element-container { margin-bottom: 0.4rem !important; }
    .history-item { margin-bottom: 8px; padding: 6px 8px; font-size: 12px; line-height: 1.4; border-radius: 4px;}
    .history-user { border-left: 3px solid #9CA3AF; color: #374151; background: #fff;}
    .history-assistant { border-left: 3px solid #3B82F6; color: #374151; background: #fff;}
    .history-role { font-weight: 600; font-size: 11px; margin-bottom: 2px; display: block; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ 工程规范与流程智能助手")

st.markdown("""
<div style="font-size: 13px; color: #6B7280; margin-bottom: 25px; background-color: #F9FAFB; padding: 10px; border-radius: 8px; border-left: 4px solid #3B82F6;">
    💡 <b>底层架构:</b> DeepSeek-chat &nbsp;|&nbsp; 
    <b>向量:</b> 智谱 embedding-3 &nbsp;|&nbsp; 
    <b>召回:</b> 混合检索 (Top-10) &nbsp;|&nbsp; 
    <b>精排:</b> BGE-Reranker (Top-3)
</div>
""", unsafe_allow_html=True)

USER_AVATAR = "https://img.icons8.com/ios/50/6B7280/user--v1.png"
AI_AVATAR = "https://img.icons8.com/ios/50/3B82F6/bot.png"

# ================= 1. 静态数据加载 =================
@st.cache_data
def load_static_data():
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

static_chunks, loaded_files = load_static_data()

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

# 注意：这里用 st.session_state 来存储动态扩充的文档和集合，防止页面刷新丢失
if "dynamic_chunks" not in st.session_state:
    st.session_state.dynamic_chunks = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

@st.cache_resource
def get_chroma_collection():
    client = chromadb.EphemeralClient()
    try: client.delete_collection(name="engineering_rules")
    except: pass
    collection = client.create_collection(name="engineering_rules", embedding_function=ZhipuEmbeddingFunction())
    return collection

collection = get_chroma_collection()

def add_to_vector_db(chunks):
    """将切分好的文本加入向量库"""
    if not chunks: return
    ids = [str(hash(c)) for c in chunks] # 用 hash 作为 id 防止重复
    # 分批处理
    for i in range(0, len(chunks), 64):
        batch_chunks = chunks[i : i + 64]
        batch_ids = ids[i : i + 64]
        collection.add(documents=batch_chunks, ids=batch_ids)

# 初始化时把静态数据放进去
all_chunks = static_chunks + st.session_state.dynamic_chunks
if all_chunks and collection.count() == 0:
    add_to_vector_db(all_chunks)

# ================= 3. 混合检索与 Reranker =================
def hybrid_search(query, k=10): 
    if not all_chunks: return []
    jieba.add_word('防渗漏'); jieba.add_word('保修期'); jieba.add_word('强制监理')
    query_words = list(jieba.cut(query))
    kw_scores = {}
    for idx, chunk in enumerate(all_chunks):
        score = sum(1 for word in query_words if word in chunk)
        kw_scores[str(hash(chunk))] = score
        
    query_vector = ZhipuEmbeddingFunction()([query])
    vector_results = collection.query(query_embeddings=query_vector, n_results=k)
    vector_ids = vector_results['ids'][0]
    
    final_scores = []
    for vid in vector_ids:
        v_score = 1.0 / (vector_ids.index(vid) + 1) 
        k_score = kw_scores.get(vid, 0) / max(max(kw_scores.values()), 1)
        combined_score = 0.7 * v_score + 0.3 * k_score
        # 找到对应的文本
        doc_text = vector_results['documents'][0][vector_ids.index(vid)]
        final_scores.append((combined_score, doc_text))
        
    final_scores.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in final_scores]

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
    headers = {"Authorization": f"Bearer {os.environ.get('SILICONFLOW_API_KEY')}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json()
        reranked = [chunks[item['index']] for item in results['results']]
        return reranked
    except Exception as e:
        return chunks[:top_n] 

# ================= 4. 侧边栏交互 =================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

with st.sidebar:
    st.caption("⚙️ **系统控制台**")
    st.success(f"✅ 知识库：{len(all_chunks)} 个片段")
    
    # 【核心新增：文件上传组件】
    with st.expander("📤 动态扩充知识库 (支持PDF/TXT)", expanded=False):
        uploaded_file = st.file_uploader("拖拽上传新规范", type=['txt', 'pdf'])
        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner("正在解析并向量化..."):
                    text = ""
                    if uploaded_file.name.endswith(".txt"):
                        text = uploaded_file.read().decode("utf-8")
                    elif uploaded_file.name.endswith(".pdf"):
                        try:
                            from pypdf import PdfReader
                            reader = PdfReader(io.BytesIO(uploaded_file.read()))
                            for page in reader.pages:
                                t = page.extract_text()
                                if t: text += t + "\n"
                        except Exception as e:
                            st.error(f"PDF解析失败: {e}")
                    
                    if text:
                        new_chunks = re.split(r'(?=第[一二三四五六七八九十百\d\.]+[条章])', text)
                        new_chunks = [c.strip() for c in new_chunks if len(c.strip()) > 10]
                        if new_chunks:
                            add_to_vector_db(new_chunks)
                            st.session_state.dynamic_chunks.extend(new_chunks)
                            st.session_state.uploaded_files.append(uploaded_file.name)
                            st.success(f"上传成功！新增 {len(new_chunks)} 个片段")
                            st.rerun() # 刷新页面更新数量

    st.caption("🎛️ **模型参数**")
    temperature = st.slider("发散度 (Temperature)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    with st.expander("📁 知识库 / 💡 快捷提问 / 📜 历史", expanded=False):
        st.write("**已加载文件:**")
        for f in loaded_files: st.caption(f"📄 {f}")
        for f in st.session_state.uploaded_files: st.caption(f"📄 {f} (动态)")
        
        st.divider()
        if st.button("外墙渗水保修几年？", use_container_width=True):
            st.session_state.quick_prompt = "外墙渗水保修几年？"
        if st.button("偷工减料会受什么处罚？", use_container_width=True):
            st.session_state.quick_prompt = "施工单位偷工减料会面临什么处罚？"
            
        st.divider()
        if st.session_state.messages:
            for msg in st.session_state.messages[-4:]: # 只显示最近4条防卡顿
                role_name = "用户" if msg["role"] == "user" else "AI 助手"
                css_class = "history-user" if msg["role"] == "user" else "history-assistant"
                st.markdown(f'<div class="history-item {css_class}"><span class="history-role">{role_name}</span>{msg["content"][:50]}...</div>', unsafe_allow_html=True)
        else:
            st.caption("暂无记录~")

    if st.button("🗑️ 清空对话记忆", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ================= 5. 主界面交互 =================
if len(st.session_state.messages) == 0:
    st.info("👋 你好！我是工程规范智能助手。你可以直接提问，或者在左侧上传新的规范文件。")

for msg in st.session_state.messages:
    avatar_img = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
    st.chat_message(msg["role"], avatar=avatar_img).write(msg["content"])

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
            
            search_query = user_question
            if len(st.session_state.messages) > 1:
                history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:-1]])
                rewrite_prompt = f"根据历史对话，将最新追问改写为独立搜索句，直接输出问题。\n历史对话：\n{history}\n最新追问：{user_question}"
                try:
                    rw_res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": rewrite_prompt}])
                    search_query = rw_res.choices[0].message.content.strip()
                    st.caption(f"💡 结合上下文，系统正在检索：**{search_query}**")
                except Exception: pass
            
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
                    model="deepseek-chat", messages=[{"role": "user", "content": prompt}], stream=True, temperature=temperature
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
