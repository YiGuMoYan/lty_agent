
import os
import sys
from openai import OpenAI

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class LLMClient:
    def __init__(self):
        # Use config for model and URL
        self.model_name = config.MODEL_NAME
        
        # Ollama usually exposes OpenAI compatible API at /v1
        # config.OLLAMA_URL is likely '.../api/chat', we need base '.../v1' for OpenAI client
        # Let's derive it or hardcode standard if typically 11434
        
        if "api/chat" in config.OLLAMA_URL:
            base_url = config.OLLAMA_URL.replace("/api/chat", "/v1")
        else:
            base_url = config.OLLAMA_URL
            
        print(f"[LLMClient] Connecting to {base_url} (Model: {self.model_name})")
        
        self.client = OpenAI(
            api_key="ollama", # Dummy key required
            base_url=base_url
        )

    def chat_with_tools(self, messages, tools=None, tool_choice="auto"):
        """
        Chat completion with optional tool calling.
        """
        try:
            # Check if model string indicates support for tools? 
            # Most modern Ollama models (llama3, mistral, qwen) support it.
            # lty_v6:7b is likely Qwen based (user mentioned lty_qwen_emo previously).
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools if tools else None,
                tool_choice=tool_choice if tools else None,
                temperature=0.7 # Bring some creativity back for persona
            )
            return response.choices[0].message
        except Exception as e:
            print(f"[LLMClient] Error: {e}")
            # Fallback or retry logic could go here
            return None
