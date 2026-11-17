import os
import re
import streamlit as st
from openai import OpenAI
import chromadb
from zhipuai import ZhipuAI
import jieba
import requests
import io
import json

# ================= 0. 页面与全局 CSS =================
st.set_page_config(page_title="工程规范智能助手", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    ::-webkit-scrollbar { display: none !important; }
    
    [data-testid="stSidebar"] {
        background-color: #FAFAFA !important;
    }
    [data-testid="stSidebarCollapseButton"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    [data-testid="stSidebarUserContent"] { padding: 1rem 0.8rem !important; }
    [data-testid="stSidebar"] .element-container { margin-bottom: 0.4rem !important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] { margin-bottom: 0.4rem !important; }
</style>
""", unsafe_allow_html=True)

# 【修改：主标题变小，去掉 Agent 版】
st.markdown("<h1 style='font-size: 30px; margin-bottom: 20px;'>🏗️ 工程规范与流程智能助手</h1>", unsafe_allow_html=True)

USER_AVATAR = "https://img.icons8.com/ios/50/6B7280/user--v1.png"
AI_AVATAR = "https://img.icons8.com/ios/50/3B82F6/bot.png"

# ================= 1. Agent 工具定义 =================
def calculate_penalty(contract_amount: float, penalty_rate: float, delay_days: int):
    """计算工程延期违约金"""
    penalty = contract_amount * penalty_rate * delay_days
    return f"计算结果：违约金总额为 {penalty:.2f} 万元。"

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_penalty",
            "description": "计算工程合同延期违约金。当用户询问涉及金额计算、违约金、赔偿金的问题时，必须调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_amount": {"type": "number", "description": "合同总金额，单位万元"},
                    "penalty_rate": {"type": "number", "description": "每日违约金比例，小数形式，如 0.1% 应传 0.001"},
                    "delay_days": {"type": "integer", "description": "延误天数"}
                },
                "required": ["contract_amount", "penalty_rate", "delay_days"]
            }
        }
    }
]

# ================= 2. 智谱 Embedding 函数 =================
class ZhipuEmbeddingFunction:
    def __init__(self):
        self.client = ZhipuAI(api_key=os.environ.get("ZHIPU_API_KEY"))
        self.model_name = "embedding-3"
    def __call__(self, input: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=input)
        return [d.embedding for d in resp.data]
    def name(self) -> str: return f"zhipuai/{self.model_name}"
    def get_config(self) -> dict: return {"model": self.model_name}

# ================= 3. 初始化持久化向量库 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
db_dir = os.path.join(current_dir, "chroma_db") 

if not os.path.exists(data_dir): os.makedirs(data_dir)

@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=db_dir)
    try:
        collection = client.get_collection(name="engineering_rules", embedding_function=ZhipuEmbeddingFunction())
    except:
        collection = client.create_collection(name="engineering_rules", embedding_function=ZhipuEmbeddingFunction())
    return collection

collection = get_chroma_collection()

def add_to_vector_db(chunks):
    if not chunks: return
    ids = [str(hash(c)) for c in chunks]
    for i in range(0, len(chunks), 64):
        collection.add(documents=chunks[i:i+64], ids=ids[i:i+64])

# ================= 4. 读取文件 =================
@st.cache_data
def load_static_data():
    all_text = ""
    file_list = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n"
            file_list.append(filename)
        elif filename.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text: all_text += text + "\n"
                file_list.append(filename)
            except Exception: pass
    chunks = re.split(r'(?=第[一二三四五六七八九十百\d\.]+[条章])', all_text)
    return [c.strip() for c in chunks if len(c.strip()) > 10], file_list

static_chunks, loaded_files = load_static_data()

if collection.count() == 0 and static_chunks:
    with st.spinner("首次启动，正在将文件向量化并存入硬盘..."):
        add_to_vector_db(static_chunks)

@st.cache_data
def get_all_chunks():
    return collection.get()["documents"]

all_chunks = get_all_chunks()

# ================= 5. 混合检索与 Reranker =================
def hybrid_search(query, k=10): 
    if not all_chunks: return []
    jieba.add_word('防渗漏'); jieba.add_word('保修期')
    query_words = list(jieba.cut(query))
    kw_scores = {str(hash(chunk)): sum(1 for word in query_words if word in chunk) for chunk in all_chunks}
    query_vector = ZhipuEmbeddingFunction()([query])
    vector_results = collection.query(query_embeddings=query_vector, n_results=k)
    vector_ids = vector_results['ids'][0]
    final_scores = []
    for vid in vector_ids:
        v_score = 1.0 / (vector_ids.index(vid) + 1) 
        k_score = kw_scores.get(vid, 0) / max(max(kw_scores.values()), 1)
        combined_score = 0.7 * v_score + 0.3 * k_score
        doc_text = vector_results['documents'][0][vector_ids.index(vid)]
        final_scores.append((combined_score, doc_text))
    final_scores.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in final_scores]

def rerank_chunks(query, chunks, top_n=3):
    if not chunks: return []
    url = "https://api.siliconflow.cn/v1/rerank"
    payload = {"model": "BAAI/bge-reranker-v2-m3", "query": query, "documents": chunks, "top_n": top_n, "return_documents": False}
    headers = {"Authorization": f"Bearer {os.environ.get('SILICONFLOW_API_KEY')}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        results = response.json()
        return [chunks[item['index']] for item in results['results']]
    except Exception:
        return chunks[:top_n] 

# ================= 6. 侧边栏 =================
if "messages" not in st.session_state: st.session_state.messages = []
if "quick_prompt" not in st.session_state: st.session_state.quick_prompt = None

with st.sidebar:
    # 【修改：移到最顶部，并加上 Agent 版标记】
    st.markdown("⚙️ **系统控制台 | Agent 版**")
    
    st.markdown("""
    <div style="font-size: 12px; color: #6B7280; background-color: #F3F4F6; padding: 8px 10px; border-radius: 6px; border-left: 3px solid #3B82F6; line-height: 1.5; margin-bottom: 10px;">
        💡 <b>技术架构</b><br>
        • <b>大模型:</b> DeepSeek (Function Calling)<br>
        • <b>数据库:</b> Chroma (持久化)<br>
        • <b>检索:</b> 混合检索 + Reranker
    </div>
    """, unsafe_allow_html=True)
    
    st.success(f"✅ 硬盘知识库：{len(all_chunks)} 个片段")
    
    with st.expander("📤 动态扩充知识库 (永久生效)", expanded=False):
        uploaded_file = st.file_uploader("拖拽上传新规范", type=['txt', 'pdf'])
        if uploaded_file:
            if uploaded_file.name not in loaded_files:
                with st.spinner("解析并写入硬盘..."):
                    text = ""
                    save_path = os.path.join(data_dir, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    if uploaded_file.name.endswith(".txt"):
                        text = uploaded_file.read().decode("utf-8")
                    elif uploaded_file.name.endswith(".pdf"):
                        try:
                            from pypdf import PdfReader
                            reader = PdfReader(io.BytesIO(uploaded_file.read()))
                            for page in reader.pages:
                                if page.extract_text(): text += page.extract_text() + "\n"
                        except: pass
                        
                    if text:
                        new_chunks = [c.strip() for c in re.split(r'(?=第[一二三四五六七八九十百\d\.]+[条章])', text) if len(c.strip()) > 10]
                        add_to_vector_db(new_chunks)
                        st.success(f"成功！永久新增 {len(new_chunks)} 个片段")
                        st.rerun()

    temperature = st.slider("发散度 (Temperature)", 0.0, 1.0, 0.2, 0.1)

    with st.expander("💡 快捷提问", expanded=False):
        if st.button("外墙渗水保修几年？", use_container_width=True): st.session_state.quick_prompt = "外墙渗水保修几年？"
        if st.button("偷工减料会受什么处罚？", use_container_width=True): st.session_state.quick_prompt = "施工单位偷工减料会面临什么处罚？"
        if st.button("帮我算算违约金", use_container_width=True): st.session_state.quick_prompt = "合同价500万，延误了45天，每日违约金比例是0.1%，需要赔多少钱？"

    st.caption("📜 **历史记忆回顾**")
    if st.session_state.messages:
        recent_messages = st.session_state.messages[-4:] 
        pairs = []
        temp_q = None
        for msg in recent_messages:
            if msg["role"] == "user":
                temp_q = msg["content"]
            elif msg["role"] == "assistant" and temp_q:
                pairs.append((temp_q, msg["content"]))
                temp_q = None
                
        for q, a in pairs:
            short_q = q[:12] + "..." if len(q) > 12 else q
            with st.expander(f"🙋‍♂️ {short_q}", expanded=False):
                st.write(a)
    else:
        st.caption("暂无历史记录~")

    st.divider()
    if st.button("🗑️ 清空对话记忆", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ================= 7. 主界面与 Agent 交互 =================
if len(st.session_state.messages) == 0:
    st.info("👋 你好！我是工程规范智能助手。你可以问我规范问题，也可以让我帮你算违约金。")

for msg in st.session_state.messages:
    avatar_img = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
    st.chat_message(msg["role"], avatar=avatar_img).write(msg["content"])

user_question = st.chat_input("例如：合同500万，延误45天，每日违约金0.1%，要赔多少？")
if st.session_state.quick_prompt:
    user_question = st.session_state.quick_prompt
    st.session_state.quick_prompt = None

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.chat_message("user", avatar=USER_AVATAR).write(user_question)
    
    with st.chat_message("assistant", avatar=AI_AVATAR):
        with st.spinner("Agent 正在思考..."):
            client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
            
            # 1. 先跑常规 RAG 检索
            search_query = user_question
            raw_chunks = hybrid_search(search_query, k=10) 
            retrieved_chunks = rerank_chunks(search_query, raw_chunks, top_n=3) 
            context = "\n\n".join(retrieved_chunks)
            
            # 2. 构建包含工具定义的对话
            messages = [
                {"role": "system", "content": f"你是工程助手。请参考规范回答问题：\n{context}\n如果涉及金额计算，必须调用 calculate_penalty 工具，不要自己瞎算。"},
                {"role": "user", "content": user_question}
            ]

            try:
                # 3. 第一轮请求：让模型判断是否调用工具
                response = client.chat.completions.create(
                    model="deepseek-chat", 
                    messages=messages, 
                    tools=tools,
                    tool_choice="auto", 
                    temperature=temperature
                )
                
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                # 4. 如果模型决定调用工具
                if tool_calls:
                    st.caption("🛠️ **Agent 发现需要精准计算，正在调用【违约金计算器】...**")
                    messages.append(response_message) 
                    
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        if function_name == "calculate_penalty":
                            function_response = calculate_penalty(
                                contract_amount=function_args.get("contract_amount"),
                                penalty_rate=function_args.get("penalty_rate"),
                                delay_days=function_args.get("delay_days")
                            )
                            
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            })
                    
                    # 5. 第二轮请求：模型拿到计算结果，生成最终回答
                    second_response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        stream=True
                    )
                    def tool_stream():
                        for chunk in second_response:
                            if chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content
                    answer = st.write_stream(tool_stream())
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                else:
                    # 如果没调用工具，走 RAG 流式回答
                    messages[0]["content"] += f"\n\n【参考规范】\n{context}"
                    stream_res = client.chat.completions.create(
                        model="deepseek-chat", messages=messages, stream=True, temperature=temperature
                    )
                    def normal_stream():
                        for chunk in stream_res:
                            if chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content
                    answer = st.write_stream(normal_stream())
                    st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Agent 执行失败: {e}")
                
            with st.expander("📚 查看参考的规范原文"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.info(f"**参考片段 {i+1}：**\n{chunk}")
