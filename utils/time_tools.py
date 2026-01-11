from datetime import datetime
import re

def get_current_year():
    return datetime.now().year

def resolve_time_expression(text):
    """
    Resolves relative time expressions like "去年" to specific years.
    Returns: (resolved_year_string, original_term) or (None, None)
    """
    current_year = get_current_year()
    
    mapping = {
        "去年": current_year - 1,
        "前年": current_year - 2,
        "今年": current_year,
        "明年": current_year + 1,
        "大前年": current_year - 3
    }
    
    # Check for keys in text
    for term, year in mapping.items():
        if term in text:
            return str(year), term
            
    return None, None

def inject_time_context(base_prompt):
    """
    Injects dynamic time context into the system prompt.
    """
    now = datetime.now()
    curr = now.year
    context = f"""
【系统时空同步单元】
- 当前精确时间：{now.strftime('%Y-%m-%d %H:%M:%S')}
- 当前年份：{curr}年
- 去年是：{curr-1}年
- 洛天依十周年是在：2022年
- 时间逻辑准则：任何早于本日期的年份均属于【过去】。严禁将过去年份描述为未来。
"""
    return context + base_prompt
