import requests
import json
import os
import sys
from config import OLLAMA_URL, MODEL_NAME, PROMPT_PATH, DEBUG
from rag_core.engine import ResonanceEngine
from rag_core.intent import IntentResolver
from rag_core.tools import TOOLS_DEF
from utils.time_tools import inject_time_context

class ActionAgent:
    def __init__(self):
        self.engine = ResonanceEngine()
        self.resolver = IntentResolver()
        self.messages = []
        self.tools = TOOLS_DEF
        self.sys_prompt_base = self._load_prompt()

    def _load_prompt(self):
        try:
            with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return "You are Luo Tianyi."

    def call_ollama(self, sys_msg):
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "system", "content": sys_msg}] + self.messages,
            "tools": self.tools,
            "stream": False
        }
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=60)
            return r.json().get('message', {})
        except Exception as e:
            if DEBUG: print(f"  [System] Ollama API Error: {e}")
            return {"role": "assistant", "content": "（信号中断）..."}

    def chat_loop(self):
        print(f"=== Luo Tianyi (Resonance 3.0 Refactored) ===\n")
        
        # Dynamic Time Injection
        sys_msg = inject_time_context(self.sys_prompt_base)

        while True:
            try:
                u = input("User: ")
                if not u.strip(): continue
                if u.lower() in ['quit', 'exit']: break
                
                self.messages.append({"role": "user", "content": u})
                
                # --- Step 1: Intent Resolution (Bypass Logic) ---
                bypass, query, injection_header = self.resolver.resolve(u)
                
                if bypass:
                    if DEBUG: print(f"  [System] Auto-triggering search for '{query}'...")
                    res = self.engine.deep_search(query)
                    
                    # Inject Tool Result
                    full_injection = f"{injection_header}\n\n检索结果：\n{res}"
                    self.messages.append({"role": "system", "content": full_injection})
                    
                    # One-shot answer generation
                    resp = self.call_ollama(sys_msg)
                    final_ans = resp.get('content', '...')
                    self.messages.append({"role": "assistant", "content": final_ans})
                    print(f"Luo Tianyi: {final_ans}\n")
                    continue

                # --- Step 2: Standard Agent Loop ---
                loop = 0
                while loop < 3:
                    resp = self.call_ollama(sys_msg)
                    content = resp.get('content', '')
                    tool_calls = resp.get('tool_calls', [])
                    
                    # Handle Tools
                    if tool_calls:
                        self.messages.append(resp)
                        for tc in tool_calls:
                            func = tc['function']['name']
                            args = json.loads(tc['function']['arguments'])
                            query = args.get('query', u)
                            
                            if DEBUG: print(f"  [System] Tool Call: {func}('{query}')")
                            if func == "deep_resonate":
                                res = self.engine.deep_search(query)
                                self.messages.append({"role": "tool", "content": res, "name": func})
                        loop += 1
                        continue
                    
                    # Final Answer
                    self.messages.append({"role": "assistant", "content": content})
                    print(f"Luo Tianyi: {content}\n")
                    break

            except (KeyboardInterrupt, EOFError):
                print("\n[系统下线] 歌声渐远...")
                try: sys.exit(0)
                except: os._exit(0)
