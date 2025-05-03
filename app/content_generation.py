import streamlit as st
import os
import base64
import logging
from typing import Tuple, Optional
from llm import enhance_prompt
from memory import save_to_memory

logger = logging.getLogger(__name__)

# Constants
IMAGES_DIR = "output/images"
MODELS_DIR = "output/models"

def generate_image(stub, enhanced_prompt: str) -> str:
    """Generate image from enhanced prompt using Openfabric app"""
    try:
        from config import TEXT_TO_IMAGE_APP_ID
        
        text_to_image_response = stub.call(
            TEXT_TO_IMAGE_APP_ID, 
            {'prompt': enhanced_prompt},
            'super-user'
        )
        
        generated_image = text_to_image_response.get('result')
        
        if not generated_image or len(generated_image) == 0:
            logger.error("Failed to generate image. Empty response from service.")
            return None
            
        # Save the image locally
        import uuid
        image_uuid = str(uuid.uuid4())
        image_path = f"{IMAGES_DIR}/{image_uuid}.png"
        
        with open(image_path, 'wb') as f:
            f.write(generated_image)
            
        logger.info(f"Image generated successfully and saved to {image_path}")
        return image_path
        
    except Exception as e:
        logger.error(f"Failed to generate image: {str(e)}")
        return None

def generate_3d_model(stub, image_path: str) -> str:
    """Generate 3D model from image using Openfabric app"""
    try:
        from config import IMAGE_TO_3D_APP_ID
        
        with open(image_path, 'rb') as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        image_to_3d_response = stub.call(
            IMAGE_TO_3D_APP_ID,
            {'input_image': encoded_image},
            'super-user'
        )
        
        model_3d = image_to_3d_response.get('generated_object')
        
        if model_3d and len(model_3d) > 0:
            import uuid
            model_uuid = str(uuid.uuid4())
            model_path = f"{MODELS_DIR}/{model_uuid}.glb"
            
            with open(model_path, 'wb') as f:
                f.write(model_3d)
                
            logger.info(f"3D model generated successfully and saved to {model_path}")
            return model_path
        else:
            logger.warning("Failed to generate 3D model. Empty response from service.")
            return None
                
    except Exception as e:
        logger.error(f"Failed to generate 3D model: {str(e)}")
        return None

def generate_complete_content(llm, stub, prompt: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Generate image and 3D model from prompt using all components"""
    # Step 1: Enhance prompt with LLM
    with st.spinner("Enhancing prompt with local LLM..."):
        enhanced_prompt = enhance_prompt(llm, prompt)
        st.info(f"Enhanced prompt: {enhanced_prompt}")
    
    # Step 2: Generate image from text
    with st.spinner("Generating image from text..."):
        image_path = generate_image(stub, enhanced_prompt)
        
        if image_path and os.path.exists(image_path):
            st.success("Image generated successfully!")
            st.image(image_path, caption="Generated Image")
        else:
            st.error("Failed to generate image.")
            return enhanced_prompt, None, None
    
    # Step 3: Generate 3D model from image
    with st.spinner("Converting image to 3D model..."):
        model_path = generate_3d_model(stub, image_path)
        
        if model_path and os.path.exists(model_path):
            st.success("3D model generated successfully!")
            
            # Create a streamlit-friendly approach to display the 3D model
            display_3d_model(model_path)
            
            # Provide a download link for the 3D model
            with open(model_path, "rb") as file:
                st.download_button(
                    label="Download 3D Model",
                    data=file,
                    file_name=f"{os.path.basename(model_path)}",
                    mime="application/octet-stream"
                )
        else:
            st.warning("Failed to generate 3D model.")
            model_path = None
    
    # Step 4: Add to memory
    if image_path:
        memory_id = save_to_memory(prompt, enhanced_prompt, image_path, model_path)
        st.success(f"Creation saved to memory with ID: {memory_id}")
        
        # Add to session history
        st.session_state.history.append({
            "id": memory_id,
            "prompt": prompt,
            "image_path": image_path,
            "model_path": model_path
        })
    
    return enhanced_prompt, image_path, model_path

def display_3d_model(model_path):
    """Display a 3D model"""
    try:
        # Convert the model file to base64
        with open(model_path, "rb") as file:
            model_bytes = file.read()
            model_base64 = base64.b64encode(model_bytes).decode('utf-8')
        
        file_extension = os.path.splitext(model_path)[1].lower()
        mime_type = {
            '.glb': 'model/gltf-binary',
            '.gltf': 'model/gltf+json',
            '.obj': 'model/obj',
            '.stl': 'model/stl'
        }.get(file_extension, 'application/octet-stream')
        
        # Create model viewer HTML
        html_content = f"""
        <script type="module" src="https://cdnjs.cloudflare.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
        <model-viewer 
            id="model-viewer" 
            camera-controls 
            auto-rotate 
            shadow-intensity="1" 
            style="width: 100%; height: 400px; background-color: #f0f0f0;"
            src="data:{mime_type};base64,{model_base64}">
        </model-viewer>
        """
        
        st.components.v1.html(html_content, height=450)
    except Exception as e:
        st.warning(f"Failed to display 3D model: {str(e)}")
        
        #  Fallback to simple viewer
        st.write("Using simplified 3D preview:")
        col1, col2 = st.columns([1, 1])
        
        # Display model type and size
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
        col1.metric("Model Format", os.path.splitext(model_path)[1].upper())
        col2.metric("Model Size", f"{file_size:.2f} MB")