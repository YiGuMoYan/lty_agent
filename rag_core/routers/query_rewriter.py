from rag_core.llm.llm_client import LLMClient
from rag_core.knowledge.alias_manager import AliasManager
from rag_core.utils.logger import logger

class QueryRewriter:
    def __init__(self):
        self.client = LLMClient.get_instance()
        self.alias_manager = AliasManager()

    async def rewrite(self, query: str, context: str = "") -> str:
        """
        Rewrite the query to be more search-friendly.
        1. Apply Alias Normalization (Fan slang -> Canonical)
        2. LLM Rewriting
        """
        # 1. Alias Normalization
        normalized_query = self.alias_manager.normalize(query)
        if normalized_query != query:
            logger.debug(f"[QueryRewriter] Alias Normalized: '{query}' -> '{normalized_query}'")

        # 2. LLM Rewriting
        system_prompt = (
            "You are a Query Rewriter for an information retrieval system about 'Luo Tianyi' (VSinger).\n"
            "Your goal is to optimize the user's query for a keyword/vector search engine.\n"
            "Rules:\n"
            "1. Remove conversational filler (e.g., 'Tell me about', 'Do you know').\n"
            "2. Expand ambiguous terms (e.g., 'that song' -> 'specific song name' if in context).\n"
            "3. Keep specific entity names EXACT.\n"
            "4. Output ONLY the rewritten query text. No quotes, no explanations.\n"
            "5. If the query is already specific, return it as is.\n"
            "6. Translate implicit references to explicit keywords."
        )

        user_prompt = f"Original Query: {normalized_query}\n"
        if context:
            user_prompt += f"Context: {context}\n"
        user_prompt += "Rewritten Query:"

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = await self.client.chat_with_tools(messages)
            if response and response.content:
                rewritten = response.content.strip()
                logger.debug(f"[QueryRewriter] '{normalized_query}' -> '{rewritten}'")
                return rewritten
            return normalized_query

        except Exception as e:
            logger.warning(f"[QueryRewriter] Failed to rewrite: {e}")
            return normalized_query
