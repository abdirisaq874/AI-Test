import streamlit as st
import logging

from typing import  List, Dict, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline



logger = logging.getLogger(__name__)

# Initialize LocalLLM
def init_llm():
    try:
        return LocalLLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except Exception as e:
        logger.error(f"Failed to initialize LocalLLM: {str(e)}")
        st.error(f"Failed to initialize LocalLLM: {str(e)}")
        return None

def enhance_prompt(llm, prompt: str) -> str:
    """Use the local LLM to enhance the prompt for better image generation"""
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that enhances image generation prompts by adding rich, concise visual details. Include elements like lighting, atmosphere, style, and perspective to make the prompts more vivid and useful for an AI image generator."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        enhanced_prompt = llm.generate_chat(messages)
        
        # Add to conversation memory
        st.session_state.conversation_memory.save_context(
            {"input": prompt},
            {"output": enhanced_prompt}
        )
        
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt
    except Exception as e:
        logger.error(f"Failed to enhance prompt: {str(e)}")
        return f"A detailed and artistic rendering of {prompt}"



class LocalLLM:
    """Class to handle interactions with locally running language models with conversation memory"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = None):
        """
        Initialize the local language model
        
        Args:
            model_name: Name of the model to use from Hugging Face
            device: Device to run the model on (cpu or cuda:0)
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Conversation memory
        self.conversation_history = []
        self.memory_window = 5  # Number of exchanges to remember
        
        # Determine model type based on name
        self.is_deepseek = "deepseek" in model_name.lower()
        self.is_tinyllama = "tinyllama" in model_name.lower()
        self.is_llama = "llama" in model_name.lower() and not self.is_tinyllama
        self.is_mistral = "mistral" in model_name.lower()
        
        # Determine device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Initializing local LLM {model_name} on {self.device}")
        
        try:
            # Set appropriate torch dtype based on device
            if self.device != "cpu":
                self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                self.torch_dtype = torch.float32
                
            # Create pipeline for easy generation
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device.startswith("cuda") else "cpu",
                trust_remote_code=True
            )
            
            # Get the tokenizer from the pipeline
            self.tokenizer = self.pipe.tokenizer
            
            self.logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise e

    def add_to_history(self, role: str, content: str):
        """
        Add a message to the conversation history
        
        Args:
            role: Role of the message sender (user or assistant)
            content: Content of the message
        """
        self.conversation_history.append({"role": role, "content": content})

        # log the conversation history
        logging.info(f"Conversation history is now: {self.conversation_history}")
        
        
        # Keep only the last N exchanges in memory
        if len(self.conversation_history) > self.memory_window * 2:
            self.conversation_history = self.conversation_history[-self.memory_window * 2:]
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")

    def format_prompt_with_history(self, prompt: str) -> str:
        """
        Format prompt with conversation history based on model type
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Formatted prompt with history context
        """
        # For DeepSeek models
        if self.is_deepseek:
            # Build history in DeepSeek format
            formatted_history = ""
            for message in self.conversation_history:
                role = "user" if message["role"] == "user" else "assistant"
                formatted_history += f"<|{role}|>\n{message['content']}\n"
                
            return formatted_history + f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        # For models with chat templates (TinyLlama, Llama, Mistral)
        elif self.is_tinyllama or self.is_llama or self.is_mistral:
            # For raw text generation, we'll still format with history
            formatted_history = ""
            for message in self.conversation_history:
                role = "User" if message["role"] == "user" else "Assistant"
                formatted_history += f"{role}: {message['content']}\n"
                
            return formatted_history + f"User: {prompt}\nAssistant:"
        
        # Default formatting for other models
        else:
            # Simple formatting for models without specific templates
            formatted_history = ""
            for message in self.conversation_history:
                role = "User" if message["role"] == "user" else "Assistant"
                formatted_history += f"{role}: {message['content']}\n"
                
            return formatted_history + f"User: {prompt}\nAssistant:"

    def extract_response(self, prompt: str, generated_text: str) -> str:
        """
        Extract the model's response from the generated text
        
        Args:
            prompt: Original prompt
            generated_text: Full generated text including prompt
            
        Returns:
            Extracted response
        """
        if self.is_deepseek:
            # Extract the part after <|assistant|>
            if "<|assistant|>" in generated_text:
                assistant_text = generated_text.split("<|assistant|>\n")[-1]
                # Remove any trailing model-specific tokens
                if "<|end|>" in assistant_text:
                    assistant_text = assistant_text.split("<|end|>")[0]
                return assistant_text.strip()
            else:
                # If no assistant marker, just return what comes after the prompt
                return generated_text[len(prompt):].strip()
        elif self.is_tinyllama or self.is_llama or self.is_mistral:
            # Check for assistant tag in chat format models
            if "<|assistant|>" in generated_text:
                assistant_parts = generated_text.split("<|assistant|>")
                if len(assistant_parts) > 1:
                    assistant_text = assistant_parts[-1]
                    # Remove any trailing tokens
                    if "</s>" in assistant_text:
                        assistant_text = assistant_text.split("</s>")[0]
                    return assistant_text.strip()
            # Default fallback - look for "Assistant:" marker
            if "Assistant:" in generated_text:
                return generated_text.split("Assistant:")[-1].strip()
            # Or just take what comes after the prompt
            return generated_text[len(prompt):].strip()
        else:
            # Default extraction for other models - look for "Assistant:" marker
            if "Assistant:" in generated_text:
                return generated_text.split("Assistant:")[-1].strip()
            # Or just take what comes after the prompt
            return generated_text[len(prompt):].strip()
            
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, 
                 with_memory: bool = True) -> str:
        """
        Generate text based on prompt, with optional memory usage
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            with_memory: Whether to use conversation history
            
        Returns:
            Generated text
        """
        try:
            # Format prompt according to model type and memory preference
            if with_memory and self.conversation_history:
                formatted_prompt = self.format_prompt_with_history(prompt)
            else:
                formatted_prompt = self.format_prompt(prompt)
            
            self.logger.info(f"Generating text with prompt: {formatted_prompt}")
            
            # Generate response
            response = self.pipe(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True
            )
            
            generated_text = response[0]["generated_text"]
            
            # Log the generated text
            self.logger.info(f"Generated text: {generated_text}")
            
            # Extract response based on model type
            assistant_response = self.extract_response(formatted_prompt, generated_text)
            
            # Add to conversation history if using memory
            if with_memory:
                self.add_to_history("user", prompt)
                self.add_to_history("assistant", assistant_response)
                
            return assistant_response
                
        except Exception as e:
            self.logger.error(f"Error during text generation: {str(e)}")
            return f"Error generating text: {str(e)}"
    
    def format_prompt(self, prompt: str) -> str:
        """
        Format prompt based on model type without using history
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Formatted prompt
        """
        if self.is_deepseek:
            # DeepSeek Coder/LLM format
            return f"<|user|>\n{prompt}\n<|assistant|>\n"
        elif self.is_tinyllama or self.is_llama or self.is_mistral:
            # Simple format for raw text generation
            return f"User: {prompt}\nAssistant:"
        else:
            # Default formatting for other models
            return f"User: {prompt}\nAssistant:"
    
    def generate_chat(self, messages: List[Dict[str, str]], 
                      max_new_tokens: int = 512, 
                      temperature: float = 0.7,
                      with_memory: bool = True) -> str:
        """
        Generate text using the chat template format
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            with_memory: Whether to use conversation history
            
        Returns:
            Generated assistant response
        """
        try:
            # Extract the current user message (assumed to be the last one)
            user_message = ""
            system_message = ""
            
            for message in messages:
                if message["role"] == "system":
                    system_message = message["content"]
                elif message["role"] == "user":
                    user_message = message["content"]
            
            # Handle DeepSeek models differently
            if self.is_deepseek:
                # Combine system and history for DeepSeek format
                full_history = self.conversation_history.copy() if with_memory else []
                
                # Add system message if present
                if system_message and not any(m["role"] == "system" for m in full_history):
                    full_history.insert(0, {"role": "system", "content": system_message})
                
                # Add the current user message
                if user_message:
                    full_history.append({"role": "user", "content": user_message})
                
                # Format the whole conversation
                formatted_prompt = ""
                for message in full_history:
                    role = "user" if message["role"] in ["user", "system"] else "assistant"
                    formatted_prompt += f"<|{role}|>\n{message['content']}\n"
                
                formatted_prompt += "<|assistant|>\n"
                
                # Generate with the formatted prompt
                response = self.pipe(
                    formatted_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    do_sample=True
                )
                
                generated_text = response[0]["generated_text"]
                assistant_response = self.extract_response(formatted_prompt, generated_text)
                
                # Add messages to history if using memory
                if with_memory:
                    if user_message:
                        self.add_to_history("user", user_message)
                    self.add_to_history("assistant", assistant_response)
                
                return assistant_response
            
            # For chat models with templates (TinyLlama, Llama, Mistral)
            if hasattr(self.tokenizer, "apply_chat_template"):
                # Combine provided messages with history if using memory
                full_messages = []
                
                # Add history first if using memory
                if with_memory and self.conversation_history:
                    for hist_message in self.conversation_history:
                        full_messages.append(hist_message)

                logging.info(f"Full messages before adding user message: {full_messages}")
                
                # Add system message if not in history
                if system_message and not any(m["role"] == "system" for m in full_messages):
                    full_messages.insert(0, {"role": "system", "content": system_message})
                
                # Add the current user message
                if user_message:
                    full_messages.append({"role": "user", "content": user_message})
                
                # Apply the chat template
                prompt = self.tokenizer.apply_chat_template(
                    full_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                self.logger.info(f"Generating chat response with prompt: {prompt}")
                
                # Generate response
                response = self.pipe(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    do_sample=True
                )
                
                generated_text = response[0]["generated_text"]
                assistant_response = self.extract_response(prompt, generated_text)
                
                # Add messages to history if using memory
                if with_memory:
                    if user_message:
                        self.add_to_history("user", user_message)
                    self.add_to_history("assistant", assistant_response)
                
                return assistant_response
            else:
                # Fallback for models without chat templates
                # Create a formatted prompt with history and current messages
                formatted_history = ""
                
                if with_memory and self.conversation_history:
                    for message in self.conversation_history:
                        role = "User" if message["role"] == "user" else "Assistant"
                        formatted_history += f"{role}: {message['content']}\n"
                
                # Add system message if present
                if system_message:
                    formatted_history = f"System: {system_message}\n{formatted_history}"
                
                # Add the current user message
                if user_message:
                    formatted_prompt = formatted_history + f"User: {user_message}\nAssistant:"
                else:
                    formatted_prompt = formatted_history + "Assistant:"
                
                # Generate with the formatted prompt
                response = self.pipe(
                    formatted_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    do_sample=True
                )
                
                generated_text = response[0]["generated_text"]
                assistant_response = self.extract_response(formatted_prompt, generated_text)
                
                # Add messages to history if using memory
                if with_memory:
                    if user_message:
                        self.add_to_history("user", user_message)
                    self.add_to_history("assistant", assistant_response)
                
                return assistant_response
                
        except Exception as e:
            self.logger.error(f"Error during chat generation: {str(e)}")
            return f"Error generating chat response: {str(e)}"
    
    def enhance_prompt(self, prompt: str) -> str:
        """
        Enhance a prompt to make it more detailed and visually descriptive
        
        Args:
            prompt: Original user prompt
            
        Returns:
            Enhanced prompt with more details
        """
        # Different enhancement approaches based on model type
        if self.is_deepseek:
            # Direct instruction for DeepSeek models
            enhancement_instruction = f"""
            Enhance this prompt for an AI image generator:
            "{prompt}"
            
            Make it more detailed and visually descriptive. Add specific details about:
            - Visual elements (colors, lighting, composition)
            - Style (realistic, cartoon, oil painting, etc.)
            - Mood and atmosphere
            
            Provide ONLY the enhanced prompt text.
            """
            
            try:
                enhanced = self.generate(enhancement_instruction, max_new_tokens=200, temperature=0.5, with_memory=False)
                
                # Clean up the response
                enhanced = enhanced.replace('"', '').replace("'", "").strip()
                
                # Validate output
                if len(enhanced) < 15 or "```" in enhanced:
                    return f"A highly detailed and artistic rendering of {prompt} with dramatic lighting, rich colors, and intricate details"
                    
                return enhanced
                
            except Exception as e:
                self.logger.error(f"Error enhancing prompt with DeepSeek: {str(e)}")
                return f"A highly detailed and artistic rendering of {prompt}"
        else:
            # Chat format for models that support it
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at enhancing image generation prompts with rich, descriptive details."
                },
                {
                    "role": "user",
                    "content": f"""Enhance this prompt for an AI image generator: "{prompt}"
                    
                    Make it more detailed and visually descriptive. Add specific details about:
                    - Visual elements (colors, lighting, composition)
                    - Style (realistic, cartoon, oil painting, etc.)
                    - Mood and atmosphere
                    
                    Provide ONLY the enhanced prompt text, no explanations or introductions."""
                }
            ]
            
            try:
                # Use the chat-specific generation without adding to history
                enhanced = self.generate_chat(messages, max_new_tokens=200, temperature=0.5, with_memory=False)
                
                # Clean up the response
                enhanced = enhanced.replace('"', '').replace("'", "").strip()
                
                # Validate output
                if len(enhanced) < 15 or "```" in enhanced or any(marker in enhanced for marker in ["<|", "|>"]):
                    return f"A highly detailed and artistic rendering of {prompt} with dramatic lighting, rich colors, and intricate details"
                    
                return enhanced
                
            except Exception as e:
                self.logger.error(f"Error enhancing prompt with chat model: {str(e)}")
                return f"A highly detailed and artistic rendering of {prompt}"
            
    def summarize_memory(self, memories: List[Dict[str, Any]], query: str) -> str:
        """
        Create a brief summary of relevant memories based on a query
        
        Args:
            memories: List of memory items
            query: User query
            
        Returns:
            Brief summary of memories related to the query
        """
        if not memories:
            return "No relevant past creations found."
            
        # Create a context with the memories
        memory_text = "\n".join([
            f"Memory {i+1}: {memory.get('original_prompt', '')} - {memory.get('enhanced_prompt', '')}"
            for i, memory in enumerate(memories[:3])  # Limit to top 3 memories
        ])
        
        if self.is_deepseek:
            # Direct instruction for DeepSeek models
            instruction = f"""
            Here are some past image creations:
            {memory_text}
            
            Current query: "{query}"
            
            Briefly summarize how these past creations are relevant to the current query in 2-3 sentences.
            """
            
            try:
                summary = self.generate(instruction, max_new_tokens=150, temperature=0.3, with_memory=False)
                return summary.strip()
            except Exception as e:
                self.logger.error(f"Error creating memory summary with DeepSeek: {str(e)}")
                return "Found some relevant past creations that might be useful for your current request."
        else:
            # Chat format for models that support it
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can summarize previous image creation memories."
                },
                {
                    "role": "user",
                    "content": f"""Here are some past image creations:
                    {memory_text}
                    Current query: "{query}"
                    Briefly summarize how these past creations are relevant to the current query in 2-3 sentences."""
                }
            ]
            
            try:
                summary = self.generate_chat(messages, max_new_tokens=150, temperature=0.3, with_memory=False)
                return summary.strip()
            except Exception as e:
                self.logger.error(f"Error creating memory summary with chat model: {str(e)}")
                return "Found some relevant past creations that might be useful for your current request."