# 洛天依歌词数据整理

## 项目简介

这是一个用于整理洛天依歌词数据的工具集，包括数据清理、分析和优化功能。

## 数据结构

### 原始数据结构

```json
{
  "song_title": "歌曲标题",
  "p_masters": ["生产者1", "生产者2"],
  "lyrics": "歌词内容（包含元数据）"
}
```

### 处理后数据结构

```json
{
  "song_title": "歌曲标题",
  "p_masters": ["生产者1", "生产者2"],
  "lyrics": "原始歌词内容",
  "raw_lyrics": "原始歌词内容（与lyrics相同，保留用于回溯）",
  "cleaned_lyrics": "清理后的歌词内容（移除了元数据和版权信息）",
  "song_metadata": {
    "作词": "作词者",
    "作曲": "作曲者",
    "编曲": "编曲者",
    "制作人": "制作人",
    "混音": "混音师",
    "母带": "母带处理者",
    "美工": "美工",
    "配唱制作人": "配唱制作人",
    "监制": "监制"
  }
}
```

## 整理结果

### 数据统计

| 项目 | 数量 |
|------|------|
| 原始歌曲总数 | 708首 |
| 唯一歌曲数 | 706首 |
| 重复歌曲数 | 2首 |
| 处理后歌曲数 | 707首 |
| 平均歌词长度 | 573字符 |

### 顶级生产者

1. 洛天依: 154首歌曲
2. JUSF周存: 50首歌曲
3. 洛天依Official: 45首歌曲
4. 阿良良木健: 27首歌曲
5. ilem: 26首歌曲
6. 纯白: 24首歌曲
7. COP: 17首歌曲
8. WOVOP: 13首歌曲
9. 闹闹丶: 12首歌曲
10. 赛亚♂sya: 11首歌曲

## 使用说明

### 数据整理

运行数据整理脚本：

```bash
python clean_data.py
```

该脚本会：
1. 加载`lyrics.jsonl`文件
2. 分析数据统计信息
3. 移除重复歌曲
4. 清理歌词，分离元数据
5. 保存处理后的数据到`cleaned_lyrics.jsonl`文件

### 加载处理后的数据

```python
import json

def load_processed_data(file_path):
    """加载处理后的数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# 使用示例
data = load_processed_data('cleaned_lyrics.jsonl')
print(f"加载了 {len(data)} 首歌曲")

# 获取第一首歌曲的信息
first_song = data[0]
print(f"歌曲标题: {first_song['song_title']}")
print(f"生产者: {', '.join(first_song['p_masters'])}")
print(f"作词: {first_song['song_metadata'].get('作词', '未知')}")
print(f"作曲: {first_song['song_metadata'].get('作曲', '未知')}")
print(f"清理后的歌词: {first_song['cleaned_lyrics'][:100]}...")
```

### 搜索歌曲

```python
def search_songs(data, keyword, field='song_title'):
    """搜索歌曲"""
    results = []
    keyword = keyword.lower()
    
    for song in data:
        if keyword in str(song.get(field, '')).lower():
            results.append(song)
    
    return results

# 使用示例
results = search_songs(data, '洛天依', field='p_masters')
print(f"找到 {len(results)} 首洛天依参与的歌曲")

results = search_songs(data, 'ilem', field='song_metadata')
print(f"找到 {len(results)} 首ilem参与创作的歌曲")
```

## 后续建议

1. **数据丰富化**：考虑添加更多元数据字段，如发行日期、歌曲类型、BPM等
2. **数据分类**：根据歌曲风格、主题或生产者对数据进行分类
3. **数据可视化**：创建可视化图表，展示歌曲数量、生产者分布等
4. **API服务**：考虑将数据导入到数据库或创建API服务，方便查询和使用
5. **定期更新**：定期添加新的洛天依歌曲到数据集中

## 文件说明

- `lyrics.jsonl`: 原始歌词数据
- `cleaned_lyrics.jsonl`: 处理后的歌词数据
- `clean_data.py`: 数据整理脚本
- `README.md`: 项目说明文档

## 技术栈

- Python 3.6+
- 标准库：json, re, os, collections

## 许可证

本项目仅供学习和研究使用，所有歌词和歌曲版权归原作者所有。
