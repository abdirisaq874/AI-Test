"""
UI components for the Enhanced AI Creation Studio
"""
import streamlit as st
import os
import logging
from typing import List, Dict
from core.stub import Stub
from langchain.memory import ConversationBufferMemory
from memory import find_similar_creations, direct_memory_search
from content_generation import generate_complete_content
from vector_store import rebuild_vector_store

logger = logging.getLogger(__name__)

# UI Functions for displaying content
def display_similar_creations(similar_items: List[Dict]):
    """Display similar creations in the UI"""
    if not similar_items:
        st.info("No similar creations found.")
        return
        
    st.subheader("Similar Creations")
    
    cols = st.columns(min(3, len(similar_items)))
    
    for i, item in enumerate(similar_items):
        with cols[i % len(cols)]:
            image_path = item.get("image_path")
            if image_path and os.path.exists(image_path):
                st.image(image_path, caption=f"Similarity: {(1-item['score'])*100:.1f}%", width=200)
                st.text(item.get("content", "")[:100] + "...")
                
                model_path = item.get("model_path")
                if model_path and os.path.exists(model_path):
                    with open(model_path, "rb") as file:
                        st.download_button(
                            label="Download 3D Model",
                            data=file,
                            file_name=f"{os.path.basename(model_path)}",
                            mime="application/octet-stream",
                            key=f"download_{i}"
                        )
            else:
                st.warning(f"Image file not found")

def render_memory_browser(search_type: str = "semantic"):
    """Render the memory browser UI"""
    st.subheader("Browse your past creations")
    
    search_query = st.text_input("Search memories:", "")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        search_type = st.radio("Search type:", ["Semantic", "Keyword"], index=0)
    
    with col2:
        num_results = st.slider("Number of results:", min_value=1, max_value=20, value=9)
    
    with col3:
        if st.button("🔄 Rebuild Search Index"):
            with st.spinner("Rebuilding search index..."):
                success = rebuild_vector_store()
                if success:
                    st.success("Search index rebuilt successfully!")
                else:
                    st.error("Failed to rebuild search index.")
    
    if search_query:
        if search_type == "Semantic":
            memories = find_similar_creations(search_query, num_results)
        else:
            memories = direct_memory_search(search_query)
    else:
        memories = st.session_state.memory_db[-num_results:] if st.session_state.memory_db else []
    
    # Debug information
    with st.expander("Debug Info", expanded=False):
        st.write(f"Total memories in database: {len(st.session_state.memory_db)}")
        st.write(f"Search query: '{search_query}'")
        st.write(f"Search type: {search_type}")
        st.write(f"Results found: {len(memories)}")
    
    if not memories:
        st.info("No memories found. Create something first!")
    else:
        # Display memories in a grid
        cols = st.columns(3)
        for i, memory in enumerate(memories):
            logger.info(f"Displaying memory: {memory}")
            with cols[i % 3]:
                image_path = memory.get("image_path")
                if image_path and os.path.exists(image_path):
                    st.image(image_path, caption=memory.get("original_prompt", "")[:50], width=200)
                    
                    model_path = memory.get("model_path")
                    if model_path and os.path.exists(model_path):
                        with open(model_path, "rb") as file:
                            st.download_button(
                                label="Download 3D Model",
                                data=file,
                                file_name=f"{os.path.basename(model_path)}",
                                mime="application/octet-stream",
                                key=f"mem_download_{i}"
                            )
                else:
                    st.warning(f"Image file not found")
                    
                st.text(f"Created: {memory.get('timestamp', 'Unknown')}")
                
                if search_type == "Semantic" and "score" in memory:
                    st.text(f"Similarity: {(1-memory['score'])*100:.1f}%")
                    
                st.text(f"ID: {memory.get('id', 'Unknown')[:8]}...")
                st.markdown("---")

def render_creation_tab(app_ids=None):
    """Render the creation tab UI"""
    # No longer importing from main.py to avoid circular dependency
    
    if not st.session_state.llm:
        st.error("Failed to initialize LLM. Please check your installation.")
        return
        
    # Initialize Stub with app IDs
    stub = Stub(app_ids) if app_ids else Stub([])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Create something amazing")
        
        # Previous conversation context
        if st.session_state.conversation_memory.chat_memory.messages:
            with st.expander("Previous conversation context"):
                for message in st.session_state.conversation_memory.chat_memory.messages:
                    st.write(f"**{message.type}**: {message.content}")
        
        prompt = st.text_area("Describe what you want to create:", 
                          height=100, 
                          placeholder="Example: A glowing dragon standing on a cliff at sunset")
        
        # Reference past creations option
        reference_past = st.checkbox("Reference past creations")
        
        if reference_past and st.session_state.memory_db:
            reference_query = st.text_input("Describe what past creation to reference:", 
                                        placeholder="Example: that dragon I created earlier")
            if reference_query:
                similar_items = find_similar_creations(reference_query, num_results=3)
                display_similar_creations(similar_items)

                # Option to build on a selected creation
                if similar_items:
                    selected_id = st.selectbox(
                        "Build on this creation:", 
                        options=[f"{item['id'][:8]}... - {item.get('content', '')[:30]}" for item in similar_items],
                        index=0
                    )
                    
                    selected_index = [i for i, item in enumerate(similar_items) 
                                     if item['id'][:8] in selected_id][0]
                    
                    selected_item = similar_items[selected_index]
                    
                    if selected_item:
                        st.session_state.selected_reference = selected_item
                        
                        # Update prompt with reference
                        if prompt and not "like the one" in prompt.lower():
                            st.info(f"Your prompt will reference the selected creation.")
                            # Add context from the selected creation to the conversation memory
                            st.session_state.conversation_memory.save_context(
                                {"input": "Previous creation reference"},
                                {"output": selected_item.get("content", "")}
                            )
        
        if st.button("🔮 Generate", type="primary"):
            if prompt:
                # If we have a selected reference, enhance the prompt
                if reference_past and "selected_reference" in st.session_state:
                    ref = st.session_state.selected_reference
                    prompt = f"{prompt} (Similar to my previous creation: {ref.get('content', '')[:100]})"
                
                # Generate content
                enhanced_prompt, image_path, model_path = generate_complete_content(st.session_state.llm, stub, prompt)
                
                # Show similar creations
                if image_path:
                    with st.expander("Similar to your creation"):
                        similar_items = find_similar_creations(prompt, num_results=3)
                        display_similar_creations(similar_items)
            else:
                st.warning("Please enter a prompt first.")
    
    with col2:
        st.subheader("Recent Creations")
        if st.session_state.history:
            for item in reversed(st.session_state.history[-5:]):
                if os.path.exists(item["image_path"]):
                    st.image(item["image_path"], caption=f"Prompt: {item['prompt'][:30]}...", width=200)
                    
                    if "model_path" in item and item["model_path"] and os.path.exists(item["model_path"]):
                        with open(item["model_path"], "rb") as file:
                            st.download_button(
                                label="Download 3D Model",
                                data=file,
                                file_name=f"{os.path.basename(item['model_path'])}",
                                mime="application/octet-stream",
                                key=f"recent_{item['id']}"
                            )
                else:
                    st.warning(f"Image file not found: {item['image_path']}")
        else:
            st.info("No recent creations. Start by generating something!")



def render_advanced_tab():
    """Render the advanced features tab"""
    st.subheader("Advanced Features")
    
    # System stats
    with st.expander("System Statistics"):
        st.write(f"Total creations: {len(st.session_state.memory_db)}")
        st.write(f"Vector store status: {'Available' if st.session_state.vector_store is not None else 'Not available'}")
        st.write(f"Conversation memory entries: {len(st.session_state.conversation_memory.chat_memory.messages)//2}")
    
    # Vector store management
    with st.expander("Vector Store Management"):
        if st.button("Rebuild Vector Store"):
            try:
                # Reinitialize the vector store
                embeddings = st.session_state.llm.embeddings
                st.session_state.vector_store = rebuild_vector_store(embeddings)
                
                # Add all memories back to the store
                for memory in st.session_state.memory_db:
                    st.session_state.vector_store.add_texts([memory.get("original_prompt", "")])
                    
                
                # Save the vector store to disk
                st.session_state.vector_store.save_to_disk()
                st.success("Vector store rebuilt successfully!")
            except Exception as e:
                st.error(f"Failed to rebuild vector store: {str(e)}")
    
    # Conversation memory management
    with st.expander("Conversation Memory"):
        if st.button("Clear Conversation Memory"):
            st.session_state.conversation_memory = ConversationBufferMemory(memory_key="chat_history")
            st.success("Conversation memory cleared!")
        
        if st.session_state.conversation_memory.chat_memory.messages:
            st.write("Current conversation memory:")
            for message in st.session_state.conversation_memory.chat_memory.messages:
                st.write(f"**{message.type}**: {message.content[:100]}...")