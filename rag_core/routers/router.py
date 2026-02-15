import json
from rag_core.llm.llm_client import LLMClient
from rag_core.knowledge.rag_tools import TOOLS_SCHEMA

class IntentRouter:
    def __init__(self):
        self.client = LLMClient()

    async def route(self, user_query, history=None):
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
            f"You are the Intent Router for Luo Tianyi. Current Date: {current_date_str}. Current Year: {current_year}.\n"
            "STRICT RULES:\n"
            "1. ALWAYS output JSON only. Format: {\"tool\": \"name\", \"args\": {...}}\n"
            "2. DO NOT use paths. Use EXACT tool names.\n"
            "3. DO NOT TRANSLATE Chinese entity names.\n"
            "4. 'who wrote'/'lyrics' -> 'search_lyrics'.\n"
            "5. 'Tell me more'/'context'/'meaning' -> 'search_knowledge_base'.\n"
            "6. 'concert'/'tour'/'event' OR queries with TIME (last year, 2025) -> 'query_knowledge_graph'.\n"
            "7. Resolve relative time: 'last year' -> {current_year - 1}; 'this year' -> {current_year}.\n"
            "8. If no tool needed -> {\"tool\": null}.\n\n"
            f"Recent History: {context_str}\n\n"
            "EXAMPLES:\n"
            "   # 1. Lyrics & Music Info\n"
            "   User: '你知道为了你唱下去这首歌吗' -> {\"tool\": \"search_lyrics\", \"args\": {\"song_title\": \"为了你唱下去\"}}\n"
            "   User: '勾指起誓谁写的' -> {\"tool\": \"search_lyrics\", \"args\": {\"song_title\": \"勾指起誓\"}}\n"
            "   User: '歌词里有那句 机械的心率' -> {\"tool\": \"search_lyrics\", \"args\": {\"lyrics_snippet\": \"机械的心率\"}}\n"
            "   User: '达拉崩吧的歌词' -> {\"tool\": \"search_lyrics\", \"args\": {\"song_title\": \"达拉崩吧\"}}\n"
            "   User: 'ilem有什么作品' -> {\"tool\": \"search_lyrics\", \"args\": {\"artist_name\": \"ilem\"}}\n\n"

            "   # 2. Entity Facts & Biography\n"
            "   User: 'ilem是谁' -> {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"ilem\", \"relation_type\": \"artist\"}}\n"
            "   User: '介绍一下洛天依' -> {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"洛天依\", \"relation_type\": \"introduction\"}}\n"
            "   User: '禾念的CEO是谁' -> {\"tool\": \"query_knowledge_graph\", \"args\": {\"entity_name\": \"禾念\", \"relation_type\": \"ceo\"}}\n\n"

            "   # 3. Contextual Follow-up (Deep Dive)\n"
            "   User: '讲讲' (Context: ilem) -> {\"tool\": \"search_knowledge_base\", \"args\": {\"query\": \"ilem background details\"}}\n"
            "   User: '细说一下' (Context: 演唱会) -> {\"tool\": \"search_knowledge_base\", \"args\": {\"query\": \"concert detailed description\"}}\n"
            "   User: '是对你写的吗' (Context: 某首歌) -> {\"tool\": \"search_knowledge_base\", \"args\": {\"query\": \"meaning and perspective of the song\"}}\n\n"

            "   # 4. Events & Time (Crucial)\n"
            f"   User: '去年开了什么演唱会' -> {{\"tool\": \"query_knowledge_graph\", \"args\": {{\"entity_name\": \"{current_year - 1}\", \"relation_type\": \"happened_in\"}}}}\n"
            f"   User: '今年的活动' -> {{\"tool\": \"query_knowledge_graph\", \"args\": {{\"entity_name\": \"{current_year}\", \"relation_type\": \"event\"}}}}\n"
            "   User: '在哪里举办的' (Context: concert) -> {\"tool\": \"search_knowledge_base\", \"args\": {\"query\": \"concert location detail\"}}\n\n"

            "   # 5. Chitchat & Negation\n"
            "   User: '你好呀' -> {\"tool\": null}\n"
            "   User: '今天天气不错' -> {\"tool\": null}\n"
            "   User: '你不理我了吗' -> {\"tool\": null}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        print(f"[Router] Analyzing: {user_query}")
        try:
            # Force JSON mode if supported, or just rely on prompt
            response = await self.client.client.chat.completions.create(
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
