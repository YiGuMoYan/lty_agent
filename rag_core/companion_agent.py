
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
            
        self.system_prompt = base_prompt
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
                    
                    # --- Commercial Robustness: Fallback Logic ---
                    # If Graph Search returns only "Category" info (weak result) but user asked for "Who wrote X" (specifics),
                    # we should Fallback to Vector Search.
                    
                    need_fallback = False
                    parsed_res = []
                    try:
                        parsed_res = json.loads(tool_result)
                    except:
                        pass
                    
                    if func_name == "query_knowledge_graph":
                        # Heuristic: If we found "DirectMatch" but it only tells us the Category (e.g. "Discography"),
                        # and implies we missed the Producer info (which is not in graph edges), trigger Vector Search.
                        # Real commercial logic would parse the QUESTION intention deeper. 
                        # For now, if result count is small or generic, we append Vector Search results.
                        
                        # Just ALWAYS append Vector Search for "Who/Detail" questions if Graph is just a Taxonomy.
                        # Actually, let's just create a "DoubleCheck" mechanism.
                        # If result is valid but maybe insufficient, we run search_knowledge_base in parallel?
                        # Simplest Commercial Fix: If result contains "Category:Discography" or "Category:Timeline", 
                        # also peek into the Vector DB using the entity name to get the "Content".
                        
                        if isinstance(parsed_res, list) and len(parsed_res) > 0:
                            # Check if it looks like a Taxonomy Node
                            first_hit = parsed_res[0]
                            # If it's just a node classification, we likely need the MD content.
                            # Let's auto-search KB for the same entity name.
                            print(f"[Companion] Auto-detecting need for detail... Triggering KB & Lyrics Fallback for '{func_args.get('entity_name', '')}'")
                            from .rag_tools import search_knowledge_base, search_lyrics
                            
                            # 1. Vector Search (Markdown)
                            kb_res = search_knowledge_base(query=func_args.get('entity_name', ''))
                            
                            # 2. Lyrics Search (Metadata) - Check if it's a song to get P-Master info
                            lyrics_res = search_lyrics(song_title=func_args.get('entity_name', ''))
                            
                            # Merge them
                            tool_result = f"【图谱概要】 {tool_result}\n"
                            if kb_res and kb_res != "[]":
                                tool_result += f"\n【详细档案 (KB)】 {kb_res}"
                            if lyrics_res and lyrics_res != "[]":
                                tool_result += f"\n【歌曲元数据 (LyricsDB)】 {lyrics_res}"

                    # --- End Fallback Logic ---

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
        else:
             answer = "（数据流中断...）"
             
        self.history.append({"role": "assistant", "content": answer})
        return answer
