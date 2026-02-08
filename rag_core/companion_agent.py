
import json
import os
from .llm_client import LLMClient
from .rag_tools import AVAILABLE_TOOLS
from .router import IntentRouter
from config import PROMPT_PATH

class CompanionAgent:
    def __init__(self):
        self.client = LLMClient()
        self.router = IntentRouter()
        self.history = []
        
        # 1. Load User's System Prompt
        try:
            with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
                base_prompt = f.read()
        except Exception as e:
            print(f"[CompanionAgent] Warning: Could not load prompt from {PROMPT_PATH}: {e}")
            base_prompt = "你是洛天依。"

        # --- Commercial Robustness: Inject Time Perception ---
        from datetime import datetime
        current_date = datetime.now().strftime("%Y年%m月%d日")
        # Prepend time context so the model knows "this year" vs "last year"
        time_context = f"【系统时间锚点：当前是 {current_date}】\n(请根据此时间判断'去年'、'今年'等相对时间词)\n\n"

        self.system_prompt = time_context + base_prompt
        self.history.append({"role": "system", "content": self.system_prompt})

    def chat(self, user_input):
        """
        Process user input using Explicit Routing -> Tool -> Response pipeline.
        """
        # 1. Route (The "Right Brain")
        # Don't add user input to history yet, wait until we have context
        # Pass a copy of history to avoid mutation, or just reference.
        # Note: self.history contains system prompt, user, assistant, tool calls.
        # We might want to filter for just user/assistant for the router context to save tokens?
        # But for now passing raw structure is fine, Router strips last 2 turns.
        route_result = self.router.route(user_input, history=self.history)
        
        tool_context = ""
        
        if route_result and route_result.get("tool"):
            func_name = route_result["tool"]
            func_args = route_result.get("args", {})
            print(f"[Companion] Router detected intent: {func_name}({func_args})")
            
            if func_name in AVAILABLE_TOOLS:
                function_to_call = AVAILABLE_TOOLS[func_name]
                try:
                    tool_result = function_to_call(**func_args)
                    
                    # --- Commercial Robustness: DeepSearch (Multi-hop) ---
                    # If we find specific entities (e.g. "Infinite Resonance") in the result,
                    # we must look them up immediately so the model knows what they are.

                    candidates = set()

                    # 1. Extract from JSON Result
                    parsed_res = []
                    try:
                        parsed_res = json.loads(tool_result)
                    except:
                        pass

                    if isinstance(parsed_res, list):
                        for item in parsed_res:
                            # Graph often returns strings or dicts with 'result'
                            if isinstance(item, str):
                                candidates.add(item)
                            elif isinstance(item, dict) and "result" in item:
                                candidates.add(item["result"])

                    # 2. Extract from Text via Regex (Books/Songs/Concerts)
                    # Matches: 「Name」, 《Title》, "Name"
                    import re
                    text_content = str(tool_result)
                    matches = re.findall(r'[「《](.*?)[」》]', text_content)
                    for m in matches:
                        candidates.add(m)

                    # Remove the original query itself (don't loop)
                    original_query = func_args.get('entity_name', '') or func_args.get('query', '')
                    if original_query in candidates:
                        candidates.remove(original_query)

                    # Limit candidates
                    final_candidates = list(candidates)[:2]

                    if final_candidates:
                        print(f"[Companion] DeepSearch detected entities: {final_candidates}. Triggering recursive lookup...")
                        from .rag_tools import search_knowledge_base

                        extra_info = ""
                        for entity in final_candidates:
                            if len(entity) < 2: continue

                            # Search KB
                            kb_res = search_knowledge_base(query=entity)

                            # Validate result is not empty/trivial
                            if kb_res and len(kb_res) > 50 and "[]" not in kb_res:
                                extra_info += f"\\n\\n【关联档案：{entity}】\\n{kb_res}"

                        if extra_info:
                             tool_result += extra_info

                    # --- End DeepSearch ---

                    # Data Validation: Check if result is empty JSON list/dict or error
                    is_empty = False
                    if not parsed_res and not isinstance(parsed_res, list): # Empty list is falsey
                         if isinstance(parsed_res, dict) and "status" in parsed_res and parsed_res["status"] == "not_found":
                             is_empty = True
                         elif len(parsed_res) == 0:
                             is_empty = True
                    
                    if is_empty:
                        # Try one last Hail Mary: Keyword Search in KB if Graph failed completely
                        if func_name == "query_knowledge_graph":
                             print(f"[Companion] Graph failed completely. Last resort: KB Search.")
                             from .rag_tools import search_knowledge_base
                             kb_res = search_knowledge_base(query=func_args.get('entity_name', ''))
                             if kb_res and kb_res != "[]":
                                 tool_context = f"\n\n【共鸣雷达补救】\n原图谱查询失败，但在档案库中发现：\n{kb_res}\n(请回答)"
                             else:
                                 tool_context = f"\n\n【共鸣雷达反馈】\n结果: 未找到任何相关数据。\n[系统指令] 严禁编造。"
                        else:
                             tool_context = f"\n\n【共鸣雷达反馈】\n结果: 未找到任何相关数据。\n[系统指令] 严禁编造。"
                    else:
                        tool_context = f"\n\n【共鸣雷达数据】\n工具调用: {func_name}\n检索结果: {tool_result}\n(请根据以上真实数据回答用户。)"
                        
                except Exception as e:
                    tool_context = f"\n\n【共鸣雷达报错】{str(e)}"
            else:
                 print(f"[Companion] Constructing response with tool result...")
        
        # 2. Chat (The "Left Brain")
        # Construct the actual message sent to the Persona Model
        # We inject the tool result simply as part of the user message or a system injection
        
        full_user_msg = user_input
        if tool_context:
            full_user_msg += tool_context
            
        self.history.append({"role": "user", "content": full_user_msg})
        
        print(f"[Companion] Generating response...")
        response_msg = self.client.chat_with_tools(self.history) # No tools passed here, just chat

        if response_msg:
             answer = response_msg.content
             # --- Commercial Robustness: Clean Output ---
             # Aggressively remove parentheses actions if the model still hallucinates them
             # Matches (text) or （text）
             import re
             answer = re.sub(r'[\(（][^\)）]+[\)）]', '', answer)
             # Clean up extra newlines created by removal
             answer = re.sub(r'\n\s*\n', '\n', answer).strip()
        else:
             answer = "（数据流中断...）"

        self.history.append({"role": "assistant", "content": answer})
        return answer
