import os
import json
import uuid
from datetime import datetime
import logging
from typing import Dict, List, Optional
from vector_store import add_to_vector_store

logger = logging.getLogger(__name__)

# Constants
MEMORY_DIR = "output/memory"

# Memory functions
def save_to_memory(prompt: str, enhanced_prompt: str, image_path: str, model_path: Optional[str]) -> str:
    """Store creation in memory database"""
    memory_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    memory_item = {
        "id": memory_id,
        "timestamp": timestamp,
        "original_prompt": prompt,
        "enhanced_prompt": enhanced_prompt,
        "image_path": image_path,
        "model_path": model_path
    }
    
    # Add to in-memory database
    import streamlit as st
    st.session_state.memory_db.append(memory_item)
    
    # Add to vector store for semantic search
    add_to_vector_store(memory_item)
    
    # Save to disk for persistence
    with open(f"{MEMORY_DIR}/{memory_id}.json", "w") as f:
        json.dump(memory_item, f)
        
    logger.info(f"Creation saved to memory with ID: {memory_id}")
    return memory_id

def load_memory() -> List[Dict]:
    """Load all memories from disk"""
    memories = []
    
    if not os.path.exists(MEMORY_DIR):
        return []
        
    for filename in os.listdir(MEMORY_DIR):
        if filename.endswith(".json"):
            with open(f"{MEMORY_DIR}/{filename}", "r") as f:
                try:
                    memory_item = json.load(f)
                    memories.append(memory_item)
                except json.JSONDecodeError:
                    logger.error(f"Failed to load memory file: {filename}")
    
    logger.info(f"Loaded {len(memories)} memories from disk")
    return memories

def direct_memory_search(query: str) -> List[Dict]:
    """Search memories based on keyword matching"""
    results = []
    query = query.lower()
    
    import streamlit as st
    for memory in st.session_state.memory_db:
        if (query in memory["original_prompt"].lower() or 
            query in memory["enhanced_prompt"].lower()):
            results.append(memory)
    
    return results

def find_similar_creations(prompt: str, num_results: int = 3) -> List[Dict]:
    """Find similar creations based on semantic search"""
    from vector_store import semantic_search
    return semantic_search(prompt, k=num_results)