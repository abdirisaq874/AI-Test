import os
import streamlit as st
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import Dict, List

logger = logging.getLogger(__name__)

# Constants
VECTOR_DB_PATH = "output/vector_db"

def get_embeddings_model():
    """Initialize and return the embeddings model"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_or_load_vector_store():
    """Create a new vector store or load existing one"""
    embeddings = get_embeddings_model()
    
    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        logger.info(f"Loading vector store from {VECTOR_DB_PATH}")
        return FAISS.load_local(VECTOR_DB_PATH, embeddings)
    else:
        logger.info("Creating new vector store")
        return FAISS.from_documents([Document(page_content="initialization")], embeddings)

def load_vector_store():
    """Load the vector store or create a new one if it doesn't exist"""
    try:
        return create_or_load_vector_store()
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        embeddings = get_embeddings_model()
        return FAISS.from_documents([Document(page_content="initialization")], embeddings)

def save_vector_store(vector_store):
    """Save the vector store to disk"""
    try:
        vector_store.save_local(VECTOR_DB_PATH)
        logger.info(f"Vector store saved to {VECTOR_DB_PATH}")
    except Exception as e:
        logger.error(f"Error saving vector store: {str(e)}")

def add_to_vector_store(memory_item):
    """Add a memory item to the vector store"""
    try:
        if st.session_state.vector_store is None:
            st.session_state.vector_store = create_or_load_vector_store()
        
        # Create a document from the memory item
        # Ensure we have the original prompt in the metadata
        combined_text = f"{memory_item['original_prompt']} {memory_item.get('enhanced_prompt', '')}"
        metadata = {
            "id": memory_item["id"],
            "timestamp": memory_item["timestamp"],
            "image_path": memory_item["image_path"],
            "model_path": memory_item.get("model_path", ""),
            "original_prompt": memory_item["original_prompt"]  # Include original prompt in metadata
        }
        
        document = Document(page_content=combined_text, metadata=metadata)
        
        # Add to vector store
        st.session_state.vector_store.add_documents([document])
        
        # Save updated vector store
        save_vector_store(st.session_state.vector_store)
        logger.info(f"Added memory item {memory_item['id']} to vector store")
        
    except Exception as e:
        logger.error(f"Error adding to vector store: {str(e)}")
        st.error(f"Failed to add memory to vector store: {str(e)}")

def semantic_search(query: str, k: int = 5) -> List[Dict]:
    """Search memories based on semantic similarity using vector store"""
    try:
        if st.session_state.vector_store is None:
            st.session_state.vector_store = load_vector_store()
            
        if not query:
            return []
            
        # Get similar documents
        docs_with_scores = st.session_state.vector_store.similarity_search_with_score(query, k=k)
        
        # Extract results
        results = []
        for doc, score in docs_with_scores:
            # Skip initialization document
            if doc.metadata.get("id") != "initialization" and doc.metadata.get("id") is not None:
                results.append({
                    "id": doc.metadata.get("id"),
                    "score": float(score),
                    "content": doc.page_content,
                    "image_path": doc.metadata.get("image_path"),
                    "model_path": doc.metadata.get("model_path"),
                    "timestamp": doc.metadata.get("timestamp"),
                    "original_prompt": doc.metadata.get("original_prompt", "")
                })
        
        # If no results were found, return all memories
        if not results and st.session_state.memory_db:
            # Log this situation for debugging
            logger.info(f"No semantic results found for '{query}', returning regular memories")
            return st.session_state.memory_db[-k:] if len(st.session_state.memory_db) > k else st.session_state.memory_db
        
        logger.info(f"Found {len(results)} results for query: {query}")
        return results
        
    except Exception as e:
        logger.error(f"Error during semantic search: {str(e)}")
        st.error(f"Search error: {str(e)}")
        return []

def rebuild_vector_store():
    """Rebuild the vector store from scratch using all items in memory_db"""
    try:
        # Create a new vector store
        embeddings = get_embeddings_model()
        new_vector_store = FAISS.from_documents([Document(page_content="initialization", 
                                                         metadata={"id": "initialization"})], 
                                               embeddings)
        
        # Add all memory items to the new vector store
        for memory_item in st.session_state.memory_db:
            combined_text = f"{memory_item['original_prompt']} {memory_item.get('enhanced_prompt', '')}"
            metadata = {
                "id": memory_item["id"],
                "timestamp": memory_item["timestamp"],
                "image_path": memory_item["image_path"],
                "model_path": memory_item.get("model_path", ""),
                "original_prompt": memory_item["original_prompt"]
            }
            
            document = Document(page_content=combined_text, metadata=metadata)
            new_vector_store.add_documents([document])
        
        # Replace the old vector store
        st.session_state.vector_store = new_vector_store
        
        # Save the new vector store
        save_vector_store(st.session_state.vector_store)
        logger.info(f"Vector store rebuilt with {len(st.session_state.memory_db)} items")
        
        return True
    except Exception as e:
        logger.error(f"Error rebuilding vector store: {str(e)}")
        st.error(f"Failed to rebuild vector store: {str(e)}")
        return False