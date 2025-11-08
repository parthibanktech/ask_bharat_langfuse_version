"""
🤖 AskBharat RAG Chatbot
-------------------------
Simple educational chatbot to compare different RAG approaches.

Features:
- Choose from 5 different RAG variants
- Ask questions about Indian government documents
- See retrieved sources and generated answers
- Compare performance of different approaches

Usage:
    streamlit run chatbot.py
"""

import streamlit as st
import logging
from vector_db_manager import VARIANT_CONFIG
from rag.pipeline import run_rag
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Setup logging - Set to INFO to see Langfuse initialization logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AskBharat RAG Chatbot",
    page_icon="",
    layout="wide"
)

# Title
st.title(" AskBharat RAG Chatbot")
st.markdown("Ask questions about Indian government policies and get AI-powered answers!")

# Sidebar - Model selection
st.sidebar.header("⚙️ Configuration")

variant_name = st.sidebar.selectbox(
    "Choose RAG Model",
    options=list(VARIANT_CONFIG.keys()),
    help="Select which RAG variant to use for retrieval"
)

# Show variant info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Model Info")
config = VARIANT_CONFIG[variant_name]
st.sidebar.markdown(f"**Chunking:** {config['splitter_type']}")
st.sidebar.markdown(f"**Chunk Size:** {config['chunk_size']}")
st.sidebar.markdown(f"**Overlap:** {config['overlap']}")
st.sidebar.markdown(f"**Embeddings:** {config['embedding']}")

if config['embedding'] == 'openai':
    st.sidebar.markdown("💰 *Uses OpenAI API (costs apply)*")
else:
    st.sidebar.markdown("🆓 *Uses local models (FREE!)*")

# Model descriptions
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About Variants")
st.sidebar.markdown("""
**Fixed Chunking**: Simple fixed-size chunks
**Sentence Overlap**: Sentence-aware with overlap
**Hybrid Retrieval**: TF-IDF + semantic search
**Reranking**: Adds LLM-based reranking
**Cross-Encoder**: Local cross-encoder reranking
""")

# Main area - Query input
st.markdown("---")

# Initialize session state for query if not exists
if "query" not in st.session_state:
    st.session_state.query = ""

# Sample questions - handle button clicks
sample_questions = {
    "sample1": "What is the contribution of MSME to GDP?",
    "sample2": "What are the benefits of GST?",
    "sample3": "Why was GST introduced?",
    "sample4": "How does GST impact small businesses?"
}

with st.expander("💡 Try these sample questions"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What is the contribution of MSME to GDP?", key="sample1"):
            st.session_state.query = sample_questions["sample1"]
            st.rerun()
        if st.button("What are the benefits of GST?", key="sample2"):
            st.session_state.query = sample_questions["sample2"]
            st.rerun()
    with col2:
        if st.button("Why was GST introduced?", key="sample3"):
            st.session_state.query = sample_questions["sample3"]
            st.rerun()
        if st.button("How does GST impact small businesses?", key="sample4"):
            st.session_state.query = sample_questions["sample4"]
            st.rerun()

query = st.text_input(
    "💬 Ask your question:",
    value=st.session_state.query,
    placeholder="e.g., What is the contribution of MSME to GDP?",
    help="Ask questions about GST, MSME, or other Indian policies",
    key="query_input"
)


# All RAG logic has been moved to rag/ folder
# This file now only contains UI code


# Ask button and processing
if st.button("🔍 Ask", type="primary"):
    if not query:
        st.warning("⚠️ Please enter a question first!")
    else:
        with st.spinner(f"Processing with {variant_name}..."):
            try:
                # Run RAG pipeline - all logic is in rag/pipeline.py
                result = run_rag(variant_name, query, top_k=5)
                
                # Check if we got any documents
                if not result["documents"] or not result["sources"]:
                    st.warning("⚠️ No documents retrieved. Please try a different query or check your vector database.")
                else:
                    # Display results
                    st.markdown("---")
                    st.markdown("### 🤖 Answer")
                    st.markdown(result["answer"])
                    
                    st.markdown("---")
                    st.markdown("### 📄 Retrieved Documents")
                    
                    # Show top 3 documents
                    for i, (doc, source) in enumerate(zip(result["documents"][:3], result["sources"][:3])):
                        with st.expander(f"📄 Document {i+1} - {source}"):
                            st.markdown(doc)
                    
                    st.markdown("---")
                    st.markdown("### 🔍 Sources")
                    unique_sources = list(set(result["sources"][:3]))
                    for source in unique_sources:
                        st.markdown(f"- {source}")
                    
                    st.success("✓ Query completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error processing query: {e}", exc_info=True)

# Footer
st.markdown("---")

