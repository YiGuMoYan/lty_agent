
import json
from .llm_client import LLMClient
from .rag_tools import TOOLS_SCHEMA

class IntentRouter:
    def __init__(self):
        self.client = LLMClient()
        
    def route(self, user_query, history=None):
        """
        Determine if the query needs tools.
        Returns: { "tool": "name", "args": {...} } or None
        """
        import datetime
        current_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.datetime.now().year
        
        # Build Context String from last 5 turns
        context_str = "None"
        if history:
            # Get last user and assistant message (Commercial Standard: 5 turns)
            # Python slice is safe even if len < 5
            last_turns = history[-5:]
            context_str = json.dumps(last_turns, ensure_ascii=False)

        # Strict prompting for 7B models
        system_prompt = (
            f"You are an Intent Classifier. Current Date: {current_date_str}.\n"
            "Analyze the user's query and decide if it requires external knowledge. "
            "You MUST output valid JSON. Do not output markdown code blocks.\n\n"
            "CONTEXT AWARENESS: You are provided with recent chat history. Use it to resolve pronouns like 'it', 'he', 'that', 'the concert'.\n"
            f"Recent History: {context_str}\n\n"
            "CRITICAL: Resolve relative time (e.g. 'last year', 'this year') to ABSOLUTE years based on Current Date.\n"
            f"Example: If today is {current_year}, 'last year' = '{current_year - 1}'.\n"
            "CRITICAL: For 'entity_name', extract KEYWORDS, not full sentences. E.g. 'My concert' -> '演唱会'.\n\n"
            "Map user queries to one of these EXACT JSON formats:\n\n"
            "Analyze the user's query and decide if it requires external knowledge. "
            "You MUST output valid JSON. Do not output markdown code blocks.\n\n"
            "CRITICAL: Resolve relative time (e.g. 'last year', 'this year') to ABSOLUTE years based on Current Date.\n"
            f"Example: If today is {current_year}, 'last year' = '{current_year - 1}'.\n"
            "CRITICAL: For 'entity_name', extract KEYWORDS, not full sentences. E.g. 'My concert' -> '演唱会'.\n\n"
            "Map user queries to one of these EXACT JSON formats:\n\n"
            "1. IF searching for lyrics (by content or title):\n"
            "   {\"tool\": \"search_lyrics\", \"args\": {\"song_title\": \"Song Name\"}}\n"
            "   OR\n"
            "   {\"tool\": \"search_lyrics\", \"args\": {\"lyrics_snippet\": \"lyrics content\"}}\n\n"
            "2. IF searching for specific entity facts (Producer/Album/Brand) OR Events (Concert/Live):\n"
            "   {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"Keyword\", \"relation_type\": \"related_to\"}}\n"
            "   CRITICAL: If user asks about 'concerts', 'tours', 'live', use THIS tool, NOT search_lyrics.\n\n"
            "3. IF searching for general stories/lore/plots:\n"
            "   {\"tool\": \"search_knowledge_base\", \"args\": {\"query\": \"search keywords\"}}\n\n"
            "4. IF no tools needed (greeting/chat):\n"
            "   {\"tool\": null}\n\n"
            "Examples:\n"
            "User: 'What is the song meant for you?' -> {\"tool\": \"search_lyrics\", \"args\": {\"song_title\": \"为了你唱下去\"}}\n"
            "User: 'Who wrote 66CCFF?' -> {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"66CCFF\", \"relation_type\": \"composed_by\"}}\n"
            "User: 'What concert did you hold last year?' (if 2026) -> {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"2025\", \"relation_type\": \"happened_in\"}}\n"
            "User: 'Your own personal concert?' -> {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"演唱会\", \"relation_type\": \"concert\"}}\n"
            "User: 'Your own personal concert?' -> {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"演唱会\", \"relation_type\": \"concert\"}}\n"
            "User: 'Tell me about Stream of Light Concerto (流光协奏).' -> {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"流光协奏\", \"relation_type\": \"event_details\"}}\n"
            "User: 'Tell me more' (Context: discussing Stream of Light) -> {\"tool\": \"search_knowledge_base\", \"args\": {\"query\": \"Stream of Light concert details\"}}\n"
            "User: 'Where is it?' (Context: discussing Stream of Light) -> {\"tool\": \"search_knowledge_base\", \"args\": {\"query\": \"Stream of Light concert location\"}}\n"
            "User: 'And Dalabangba? (那达拉崩吧呢)' (Context: Who wrote Gou Zhi Qi Shi) -> {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"达拉崩吧\", \"relation_type\": \"composed_by\"}}\n"
            "User: 'Tell me about the execution.' -> {\"tool\": \"search_knowledge_base\", \"args\": {\"query\": \"execution\"}}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        print(f"[Router] Analyzing: {user_query}")
        try:
            # Force JSON mode if supported, or just rely on prompt
            response = self.client.client.chat.completions.create(
                model=self.client.model_name,
                messages=messages,
                response_format={"type": "json_object"}, 
                temperature=0.1 # Deterministic
            )
            content = response.choices[0].message.content
            print(f"[Router] Raw Logic: {content}")
            
            result = json.loads(content)
            if result.get("tool"):
                return result
            return None
            
        except Exception as e:
            print(f"[Router] Error: {e}")
            return None
