import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to path to import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class QwenDriver:
    def __init__(self):
        self.api_key = config.GEN_API_KEY
        self.model_name = config.GEN_MODEL_NAME
        self.base_url = config.GEN_API_BASE

        if not self.api_key:
            # Fallback to check raw env if config didn't pick it up (though config should)
            self.api_key = os.getenv("GEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise ValueError("GEN_API_KEY not found in .env or config")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def search(self, query, model=None):
        """
        Executes a search-enabled query.
        """
        target_model = model or self.model_name
        try:
            completion = self.client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": query}],
                extra_body={
                    "enable_search": True,
                    "search_options": {
                        "search_strategy": "max"  # High performance search
                    }
                }
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Search Error: {e}")
            return None

    def extract_json(self, prompt, model=None):
        """
        Forces JSON output.
        """
        target_model = model or self.model_name
        try:
            completion = self.client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"JSON Error: {e}")
            return None

    def chat(self, messages, model=None):
        """
        Generic Chat Completion.
        """
        target_model = model or self.model_name
        try:
            completion = self.client.chat.completions.create(
                model=target_model,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Chat Error: {e}")
            return None

def test_driver():
    driver = QwenDriver()
    print("Testing Search...")
    print(driver.search("2024年洛天依有什么大事件？"))
    
    print("\nTesting JSON...")
    print(driver.extract_json("生成一个洛天依的简单档案，包含姓名和生日"))

if __name__ == "__main__":
    test_driver()
