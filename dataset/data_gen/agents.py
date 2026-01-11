import json
import os
import sys

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm_driver import QwenDriver

class TaxonomyAgent:
    """Architect: Defines what needs to be researched based on SEARCH, not hallucination."""
    def __init__(self):
        self.driver = QwenDriver()

    def _scan_year(self, year):
        print(f"[Taxonomy] Scanning Timeline: {year}...")
        query = f"洛天依 {year}年 官方大事记 演唱会 知名歌曲 商业代言 获奖记录"
        try:
            raw_data = self.driver.search(query)
        except Exception:
            return []
            
        if not raw_data: return []

        prompt = f"""
        基于以下搜索结果，提取 {year} 年洛天依的核心里程碑事件。
        
        【搜索结果】
        {raw_data[:3000]}
        
        【筛选要求】
        1. **收录优先级**：
           - **大型演出**：Vsinger Live, BML, BW, 卫视春晚/跨年晚会 (格式: "参加[活动]，演唱《[曲目]》")
           - **商业成就**：官宣品牌代言、联动 (格式: "官宣成为[品牌]代言人")
           - **荣誉**：获得重要奖项。
        2. **排除**：普通的单曲发布（除非是《普通DISCO》这种出圈神曲）、单纯的生日贺图、非官方的小型活动。
        3. 严禁编造！如果没有重要大事，返回空列表。
        4. 返回JSON格式：{{"events": ["{year}年MM月: 事件描述", ...]}}
        """
        try:
            res = self.driver.extract_json(prompt, model="qwen-max")
            data = json.loads(res)
            return data.get("events", [])
        except Exception as e:
            print(f"  [Error] Parsing {year}: {e}")
            return []

    def _scan_domain(self, category, keywords, instruction):
        print(f"[Taxonomy] Scanning Domain: {category}...")
        query = f"{keywords} 详细列表"
        raw_data = self.driver.search(query)
        
        prompt = f"""
        基于搜索结果，列出关于“{category}”的具体话题清单。
        
        【搜索结果】
        {raw_data[:3000]}
        
        【特别指令】
        {instruction}

        【通用要求】
        1. 提取具体名词。
        2. 返回JSON格式：{{"topics": ["话题1", "话题2"]}}
        """
        try:
            res = self.driver.extract_json(prompt, model="qwen-max")
            data = json.loads(res)
            return data.get("topics", [])
        except Exception as e:
            print(f"  [Error] Parsing {category}: {e}")
            return []

    def generate_master_plan(self):
        import concurrent.futures
        
        taxonomy = []
        timeline_events = []
        
        # 1. Timeline Scan (2012-2026)
        years = list(range(2012, 2027))
        
        print(f"[Taxonomy] Starting parallel scan for {len(years)} years...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_year = {executor.submit(self._scan_year, year): year for year in years}
            for future in concurrent.futures.as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    events = future.result()
                    if events:
                        timeline_events.extend(events)
                except Exception as e:
                    print(f"  [Error] Year {year} generated exception: {e}")

        timeline_events.sort()

        taxonomy.append({
            "category": "Timeline_DeepDive",
            "subtopics": timeline_events
        })

        # 2. Domain Scans
        domains = [
            ("Discography_Famous", "洛天依 知名歌曲 殿堂曲 代表作 列表", "列出洛天依的知名代表作，包括‘神话曲’（千万播放）和‘殿堂曲’（百万播放）及粉丝公认的名曲。例如《三月雨》《66CCFF》《千年食谱颂》等。"),
            ("Discography_Albums", "洛天依 官方专辑 列表", "列出所有官方发行的专辑名称。"),
            ("Commercial_Deals", "洛天依 历年 品牌代言 联动", "列出所有代言的品牌名称或是联动的IP名称。"),
            ("Lore_World", "洛天依 官方设定 音之精 V5 AI 声库区别", "列出官方设定的核心名词（如音之精、天钿、共鸣）或声库版本名。"),
            ("Producers", "洛天依 著名P主 创作者 名单", "只列出**著名创作者的人名**（P主），例如ilem, 纯白, JUSF周存。**绝对不要**列出歌名！")
        ]
        
        print(f"[Taxonomy] Starting parallel scan for {len(domains)} domains...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_domain = {executor.submit(self._scan_domain, cat, kw, instr): cat for cat, kw, instr in domains}
            
            for future in concurrent.futures.as_completed(future_to_domain):
                cat = future_to_domain[future]
                try:
                    topics = future.result()
                    if topics:
                        taxonomy.append({"category": cat, "subtopics": topics})
                except Exception as e:
                    print(f"  [Error] Domain {cat} generated exception: {e}")
                
        return json.dumps({"taxonomy": taxonomy}, ensure_ascii=False, indent=2)

class AuthorAgent:
    """The Writer: Researches and writes the content."""
    def __init__(self):
        self.driver = QwenDriver()

    def draft(self, topic, category="General", feedback=None, previous_content=None):
        search_context = ""
        
        # 1. Search Phase
        if feedback:
            print(f"[Author] Refining '{topic}' based on feedback: {feedback[:50]}...")
            query = f"洛天依 {topic} {feedback}" 
            search_context = self.driver.search(query)
        else:
            print(f"[Author] Researching '{topic}' ({category})...")
            # --- Category-Specific Search Strategy ---
            if "Discography" in category:
                # Disambiguate songs (e.g., "Butterfly" -> "Luo Tianyi Song Butterfly")
                query = f"洛天依 歌曲《{topic}》 介绍 歌词 创作者 播放量"
            elif "Commercial" in category:
                # Focus on the collaboration/deal, not the brand itself
                query = f"洛天依 {topic} 代言 联动 活动 合作详情"
            elif "Interpersonal" in category:
                # Focus on relationship/interaction
                query = f"洛天依与{topic}的关系 互动 官方设定 合唱歌曲"
            elif "Producers" in category:
                query = f"洛天依 P主 {topic} 代表作 风格 采访"
            else:
                query = f"洛天依 {topic} 详细资料 时间 地点 事件"
                
            search_context = self.driver.search(query)
            
        if not search_context:
            return previous_content or "No info found."

        # 2. Writing Phase
        # --- Category-Specific Prompt Constraints ---
        special_instructions = ""
        if "Commercial" in category:
            special_instructions = """
            **【商业及联动特别规定】**
            1. **核心聚焦**：只描述洛天依参与的部分（如：定制形象、联名产品、广告曲、线下活动）。
            2. **剔除无关**：严禁大段引用该品牌的企业历史、无关产品介绍。如果搜到的是“肯德基的历史”，请忽略。
            3. **若是同名品牌**：确认为洛天依代言的那个品牌（例如 '清风' 是纸巾还是其他），如果搜到无关品牌，请注明无法确认。
            """
        elif "Discography" in category:
            special_instructions = """
            **【歌曲条目特别规定】**
            1. **消歧义**：必须确信这是洛天依演唱的歌曲。如果同名歌曲是其他歌手的（如《蝴蝶》有很多版），只写洛天依版。
            2. **基本信息**：必须包含 P主（创作者）、投稿时间、大致播放量级。
            """
            
        prompt_content = f"""
        你是一位严谨的档案管理员。请基于参考资料，撰写或修改一篇关于“{topic}”的百科档案。
        
        【参考资料 (来自最新搜索)】
        {search_context}
        
        【上一版草稿】
        {previous_content if previous_content else "(无)"}
        
        【修改要求】
        {feedback if feedback else "这是初稿，请全面、详尽地撰写。"}
        
        {special_instructions}
        
        **【绝对准则 - 违反将导致系统崩溃】**
        1. **证据优先**：你写下的每一个事实（日期、地点、人物、曲名）都必须在【参考资料】中有明确出处。
        2. **严禁编造**：如果【参考资料】中没有列出详情，**绝对不要**自己编造！请直接写“暂无详细数据”。
        3. **宁缺毋滥**：搜到什么写什么，搜不到就留空。
        4. **格式规范**：使用标准 Markdown。
        
        请输出一份**真实、可信**的档案。
        """
        
        messages = [{"role": "user", "content": prompt_content}]
        return self.driver.chat(messages)

class CriticAgent:
    """The Red Team: Finds flaws."""
    def __init__(self):
        self.driver = QwenDriver()

    def review(self, topic, draft):
        print(f"[Critic] Reviewing '{topic}'...")
        prompt = f"""
        你是一位极其挑剔的百科全书主编。请评审这篇关于“{topic}”的草稿。
        
        【草稿内容】
        {draft[:4000]}
        
        【评审标准】
        1. 是否缺失关键事实（如具体日期、地点、关键人物）？
        2. 是否有模糊不清的描述（如“多首歌曲”、“很多粉丝”）？要求具体数字或列表。
        3. 格式是否规范 Markdown？

        如果草稿完美，请返回 JSON: {{"status": "PASS", "feedback": "无"}}
        如果存在缺陷，请返回 JSON: {{"status": "FAIL", "feedback": "指具体的缺失点，例如：缺失演唱会曲目单..."}}
        """
        return self.driver.extract_json(prompt)


class ArchivistAgent:
    """Scribe: Writes the final file."""
    def __init__(self, output_root="dataset/knowledge_base"):
        self.output_root = output_root
        if not os.path.exists(output_root):
            os.makedirs(output_root)

    def exists(self, category, topic):
        safe_cat = "".join(x for x in category if x.isalnum() or x in "-_")
        safe_topic = "".join(x for x in topic if x.isalnum() or x in "-_")
        file_path = os.path.join(self.output_root, safe_cat, f"{safe_topic}.md")
        return os.path.exists(file_path)

    def archive(self, category, topic, content):
        if not content:
            print(f"[Archivist] No content for {topic}, skipping.")
            return

        # Sanitize filename
        safe_cat = "".join(x for x in category if x.isalnum() or x in "-_")
        safe_topic = "".join(x for x in topic if x.isalnum() or x in "-_")
        
        dir_path = os.path.join(self.output_root, safe_cat)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        file_path = os.path.join(dir_path, f"{safe_topic}.md")
        
        md_content = f"""---
category: {category}
topic: {topic}
source: Qwen-Plus-Search-Engine
---

# {topic}

{content}
"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"[Archivist] Saved to {file_path}")
