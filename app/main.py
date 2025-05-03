"""
Main application file for the Enhanced AI Creation Studio
"""
import streamlit as st
import logging
from vector_store import load_vector_store, create_or_load_vector_store, add_to_vector_store
from memory import load_memory
from llm import init_llm
from ui import render_creation_tab, render_memory_browser, render_advanced_tab
from core.stub import Stub
from config import configurations

# Set page config first - MUST BE CALLED BEFORE ANY OTHER STREAMLIT COMMAND
st.set_page_config(
    page_title="Enhanced AI Creation Studio",
    page_icon="🚀",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Initialize session state
def init_session_state():
    if 'llm' not in st.session_state:
        st.session_state.llm = init_llm()
        
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if 'memory_db' not in st.session_state:
        st.session_state.memory_db = []
    
    if 'conversation_memory' not in st.session_state:
        from langchain.memory import ConversationBufferMemory
        st.session_state.conversation_memory = ConversationBufferMemory(memory_key="chat_history")
    
    if 'vector_store' not in st.session_state:
        # Create or load vector store
        try:
            st.session_state.vector_store = load_vector_store()
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            st.session_state.vector_store = None

def main():
    """Main application function"""
    st.title("🚀 Enhanced AI Creation Studio")
    st.subheader("Turn your ideas into stunning images and 3D models with intelligent memory")
    
    # Initialize session state
    init_session_state()
    
    # Load existing memories if not already loaded
    if len(st.session_state.memory_db) == 0:
        memories = load_memory()
        if memories:
            st.session_state.memory_db = memories
            # Add memories to vector store if needed
            if st.session_state.vector_store is None:
                st.session_state.vector_store = create_or_load_vector_store()
                for memory in memories:
                    add_to_vector_store(memory)
    
    # Get app IDs to pass into the creation tab
    user_config = configurations.get('super-user')
    app_ids = user_config.app_ids if user_config else []
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["Create", "Memory Browser", "Advanced"])
    
    with tab1:
        render_creation_tab(app_ids)
    
    with tab2:
        render_memory_browser()
    
    with tab3:
        render_advanced_tab()

if __name__ == "__main__":
    main()