from pypinyin import lazy_pinyin, Style

def get_simplified_pinyin(text):
    """
    Converts text to simplified pinyin (first letter of each word).
    e.g., "为了你唱下去" -> "wlnycqx"
    """
    try:
        if not text: return ""
        pinyin_list = lazy_pinyin(text, style=Style.FIRST_LETTER)
        return "".join([p[0] for p in pinyin_list if p]).lower()
    except Exception:
        return ""

def clean_query(text):
    """
    Standardizes query text.
    """
    return text.replace("心率", "心律").replace("计协", "机械").strip()
