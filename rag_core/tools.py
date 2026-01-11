TOOLS_DEF = [
    {
        "type": "function",
        "function": {
            "name": "deep_resonate",
            "description": "深度共鸣检索。当【时空记忆残片】模糊、缺失或不确定时（例如：询问具体P主、完整歌词、早期历史），必须调用此功能来激活深层数据。严禁在信息不足时编造。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索关键词，如歌名、P主名或事件名。"}},
                "required": ["query"]
            }
        }
    }
]
