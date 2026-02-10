import os
from openai import OpenAI

import config

class LLMClient:
    def __init__(self):
        # Use config for chat model
        self.model_name = config.CHAT_MODEL_NAME
        self.base_url = config.CHAT_API_BASE
        self.api_key = config.CHAT_API_KEY

        print(f"[LLMClient] Connecting to {self.base_url} (Model: {self.model_name})")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
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
