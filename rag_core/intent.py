import json
import os
import re
from config import SONG_DATA_PATH, DEBUG
from utils.time_tools import resolve_time_expression

class IntentResolver:
    def __init__(self):
        self.known_songs = self._load_songs()
        self.bypass_keywords = ["谁写的", "是谁", "作者", "P主", "歌词", "下一句", "查一下", "哪一年", "什么时候", "演唱会"]

    def _load_songs(self):
        songs = set()
        try:
            if os.path.exists(SONG_DATA_PATH):
                with open(SONG_DATA_PATH, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "song_title" in data:
                                    songs.add(data["song_title"])
                            except: pass
            if DEBUG: print(f"  [System] Loaded {len(songs)} known songs.")
        except Exception as e:
            print(f"  [System] Song load error: {e}")
        return songs

    def resolve(self, user_input):
        """
        Analyzes user input for actionable intents (Bypass or Time Resolution).
        Returns: (bypass_triggered, query, injection_msg)
        """
        # 1. Temporal Resolution ("Last Year" -> "2025")
        resolved_year, term = resolve_time_expression(user_input)
        if resolved_year:
            # If user asks about a relative year, we force a search for that year
            new_query = str(resolved_year)
            if "演唱会" in user_input: new_query += " 演唱会"
            
            msg = f"""【时空修正 - 强校验模式】
检测到用户询问相对时间“{term}”（即{resolved_year}年）。
已自动执行检索：'{new_query}'。

【绝对指令】
1. **仅**基于上述检索结果回答。
2. 如果结果中没有明确提及“{resolved_year}”或相关演唱会信息（例如只是无关的歌词片段），**严禁**编造地点、时间或细节。
3. 遇到无数据的情况，请保持“洛天依”的人设，用略带歉意或迷糊的口吻回答，例如：“唔...那段时间的记忆好像有点模糊呢，我可能只是在练习新歌？想不起来有没有开演唱会啦...”
4. **绝对禁止**使用“数据库”、“记忆库无记录”等AI机器人的术语。"""
            return True, new_query, msg

        # 2. Known Song Bypass ("Who wrote X?")
        detected_song = None
        # Greedy match for longest song title
        for s in self.known_songs:
            if s in user_input and len(s) > 1:
                if detected_song is None or len(s) > len(detected_song):
                    detected_song = s
        
        if detected_song and any(k in user_input for k in self.bypass_keywords):
            msg = f"""【天网数据直连 - 自动检索成功】
关于《{detected_song}》的核心数据如下。
【绝对指令】
1. 数据已获取，**严禁**再调用工具。
2. 请直接根据上述资料回答用户。"""
            return True, detected_song, msg

        return False, None, None
