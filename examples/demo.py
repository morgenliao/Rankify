import streamlit as st
import os
import base64
import logging
from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.retrievers.retriever import Retriever

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入重排序相关模块，如果不可用则提供备选方案
try:
    from rankify.models.reranking import Reranking
    from rankify.utils.pre_defind_models import HF_PRE_DEFIND_MODELS
    reranking_available = True
    logger.info("重排序功能已成功加载")
except ImportError as e:
    logger.warning(f"重排序功能加载失败: {str(e)}")
    reranking_available = False
    # 创建空的替代品
    HF_PRE_DEFIND_MODELS = {"基础重排序": {"default": "default"}}

# 设置页面配置
st.set_page_config(
    page_title="Rankify Demo",
    page_icon="🔍",  # 使用表情符号替代图片，避免路径问题
    layout="centered"
)

# CSS样式保持不变
st.markdown("""
<style>
    /* Logo + title layout */
    .logo-container {
        text-align: center;
        margin-bottom: 10px;
    }

    .logo-img {
        width: 150px;
        border-radius: 15px;
    }

    h1 {
        text-align: center;
        color: #1e90ff;
        font-weight: bold;
    }

    .stTextArea, .stSelectbox, .stSlider {
        background-color: #f0f8ff !important;
        border-radius: 10px;
        padding: 5px;
    }

    .stButton>button {
        width: 100%;
        background-color: #1e90ff;
        color: white;
        border-radius: 10px;
        font-size: 1.1rem;
        padding: 10px;
        margin-top: 20px;
    }

    .stButton>button:hover {
        background-color: #63b3ed;
    }

    .stExpander {
        border: 1px solid #cce7ff !important;
        border-radius: 10px;
    }

    .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 尝试加载标志，但在失败时提供备选方案
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(os.path.dirname(script_dir), "images", "rankify-crop.png")
    
    with open(logo_path, "rb") as img_file:
        b64_logo = base64.b64encode(img_file.read()).decode()
        
    st.markdown(f"""
    <div class="logo-container">
        <img class="logo-img" src="data:image/png;base64,{b64_logo}" />
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    logger.warning(f"无法加载标志: {str(e)}")

st.title("Rankify 检索演示")

# 用户输入和配置
query = st.text_area("输入问题:")

# 配置部分
st.markdown("### 设置")
retriever_method = st.selectbox("选择检索器:", ["dpr", "bm25", "contriever", "ance", "colbert"])
index_type = st.selectbox("选择索引类型:", ["wiki", "msmarco"])
top_k = st.slider("召回文档数量:", 1, 20, 5)

# 重排序选项
if reranking_available:
    apply_reranking = st.selectbox("应用重排:", ["No Reranking"] + list(HF_PRE_DEFIND_MODELS.keys()))
    reranking_model = None
    if apply_reranking != "No Reranking":
        reranking_model = st.selectbox("选择重排模型:", list(HF_PRE_DEFIND_MODELS[apply_reranking].keys()))
else:
    st.warning("⚠️ 重排序功能不可用。要启用完整功能，请安装 vLLM 或 rankify[reranking] 依赖。")
    apply_reranking = "No Reranking"
    reranking_model = None

# 检索按钮
if st.button("🚀 检索文档"):
    if not query.strip():
        st.warning("⚠️ 请输入一个问题")
    else:
        try:
            with st.spinner("🔍 正在检索文档..."):
                documents = [Document(question=Question(query), answers=Answer([]), contexts=[])]
                retriever = Retriever(method=retriever_method.lower(), n_docs=top_k, index_type=index_type.lower())
                retrieved_documents = retriever.retrieve(documents)

                # 如果重排序可用且被选择
                reranked_documents = None
                if reranking_available and apply_reranking != "No Reranking" and reranking_model:
                    try:
                        reranker = Reranking(method=apply_reranking, model_name=reranking_model)
                        with st.spinner("✨ 正在进行重排序..."):
                            reranked_documents = reranker.rank(retrieved_documents)
                    except Exception as e:
                        st.error(f"重排序过程出错: {str(e)}")
                        logger.error(f"重排序错误: {str(e)}")

            # 显示检索结果
            st.subheader("📄 检索的文档")
            if retrieved_documents and hasattr(retrieved_documents[0], 'contexts'):
                for doc in retrieved_documents:
                    for i, context in enumerate(doc.contexts[:top_k]):
                        with st.expander(f"📘 文档 {i+1}: {context.title}"):
                            st.write(context.text)
            else:
                st.info("未找到相关文档")

            # 显示重排序结果
            if reranking_available and apply_reranking != "No Reranking" and reranking_model and reranked_documents:
                st.subheader("🔁 重排序后的文档")
                if hasattr(reranked_documents[0], 'reorder_contexts'):
                    for doc in reranked_documents:
                        for i, context in enumerate(doc.reorder_contexts[:top_k]):
                            with st.expander(f"📗 文档 {i+1}: {context.title}"):
                                st.write(context.text)
                else:
                    st.info("重排序未返回结果")
                
        except Exception as e:
            st.error(f"处理过程中出错: {str(e)}")
            logger.error(f"检索错误: {str(e)}")

# 添加说明信息
with st.expander("ℹ️ 关于 Rankify"):
    st.markdown("""
    **Rankify** 是一个模块化且高效的检索、重排序和 RAG 框架，专为最新的检索、排序和 RAG 任务设计。
    
    - 支持多种检索方法
    - 提供先进的重排序模型
    - 集成了检索增强生成功能
    
    [查看 GitHub 项目](https://github.com/DataScienceUIBK/rankify)
    """)