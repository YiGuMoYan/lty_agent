from rag_core.llm.llm_client import LLMClient

class QueryRewriter:
    def __init__(self):
        self.client = LLMClient()

    async def rewrite(self, query: str, context: str = "") -> str:
        """
        Rewrite the query to be more search-friendly.
        Expand abbreviations, clarify ambiguous terms, and extract keywords.
        """
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

        user_prompt = f"Original Query: {query}\n"
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
                print(f"[QueryRewriter] '{query}' -> '{rewritten}'")
                return rewritten
            return query

        except Exception as e:
            print(f"[QueryRewriter] Failed to rewrite: {e}")
            return query
