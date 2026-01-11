#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
洛天依歌词数据整理工具
用于清理、分析和优化歌词JSONL数据
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Any


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    加载JSONL文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        JSON对象列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"解析错误: {e}，行内容: {line}")
    return data


def save_jsonl_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    保存为JSONL文件
    
    Args:
        data: JSON对象列表
        file_path: 文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def analyze_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析数据
    
    Args:
        data: JSON对象列表
        
    Returns:
        分析结果
    """
    analysis = {
        "total_songs": len(data),
        "unique_songs": 0,
        "duplicate_songs": 0,
        "empty_titles": 0,
        "empty_lyrics": 0,
        "average_lyric_length": 0,
        "top_producers": defaultdict(int),
        "song_lengths": []
    }
    
    # 统计唯一歌曲
    unique_titles = set()
    
    for song in data:
        # 检查标题
        title = song.get("song_title", "").strip()
        if not title:
            analysis["empty_titles"] += 1
        
        # 检查歌词
        lyrics = song.get("lyrics", "").strip()
        if not lyrics:
            analysis["empty_lyrics"] += 1
        else:
            analysis["song_lengths"].append(len(lyrics))
        
        # 统计生产者
        producers = song.get("p_masters", [])
        for producer in producers:
            analysis["top_producers"][producer] += 1
        
        # 统计唯一歌曲
        unique_titles.add(title)
    
    analysis["unique_songs"] = len(unique_titles)
    analysis["duplicate_songs"] = analysis["total_songs"] - analysis["unique_songs"]
    
    # 计算平均歌词长度
    if analysis["song_lengths"]:
        analysis["average_lyric_length"] = sum(analysis["song_lengths"]) / len(analysis["song_lengths"])
    
    # 排序生产者
    analysis["top_producers"] = dict(sorted(analysis["top_producers"].items(), key=lambda x: x[1], reverse=True))
    
    return analysis


def remove_duplicates(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    移除重复歌曲
    
    Args:
        data: JSON对象列表
        
    Returns:
        去重后的JSON对象列表
    """
    seen = set()
    unique_data = []
    
    for song in data:
        title = song.get("song_title", "").strip()
        # 使用标题和歌词内容的组合作为唯一标识
        identifier = f"{title}|{song.get('lyrics', '')[:100]}"
        
        if identifier not in seen:
            seen.add(identifier)
            unique_data.append(song)
    
    return unique_data


def clean_lyrics(lyrics: str) -> Dict[str, Any]:
    """
    清理歌词，分离元数据和实际歌词
    
    Args:
        lyrics: 原始歌词
        
    Returns:
        包含清理后歌词和元数据的字典
    """
    # 定义元数据正则表达式（使用多行匹配，确保完整匹配行）
    metadata_patterns = {
        "作词": r"^作词\s*[:：]\s*(.+)$",
        "作曲": r"^作曲\s*[:：]\s*(.+)$",
        "编曲": r"^编曲\s*[:：]\s*(.+)$",
        "制作人": r"^制作人\s*[:：]\s*(.+)$",
        "混音": r"^混音\s*[:：]\s*(.+)$",
        "母带": r"^母带\s*[:：]\s*(.+)$",
        "美工": r"^美工\s*[:：]\s*(.+)$",
        "配唱制作人": r"^配唱制作人\s*[:：]\s*(.+)$",
        "监制": r"^监制\s*[:：]\s*(.+)$",
    }
    
    metadata = {}
    cleaned_lines = []
    is_metadata = False
    
    # 逐行处理歌词
    lines = lyrics.splitlines()
    for line in lines:
        stripped_line = line.strip()
        
        # 检查是否为元数据行
        metadata_found = False
        for key, pattern in metadata_patterns.items():
            match = re.match(pattern, stripped_line)
            if match:
                metadata[key] = match.group(1).strip()
                metadata_found = True
                is_metadata = True
                break
        
        # 如果是版权信息，跳过
        if stripped_line == "（版权所有，未经许可请勿使用）":
            continue
        
        # 如果不是元数据行，添加到清理后的歌词中
        if not metadata_found:
            cleaned_lines.append(stripped_line)
    
    # 移除连续的空行
    final_cleaned = []
    for line in cleaned_lines:
        if line or not final_cleaned or final_cleaned[-1]:
            final_cleaned.append(line)
    
    # 合并为字符串
    cleaned_lyrics = '\n'.join(final_cleaned).strip()
    
    return {
        "raw_lyrics": lyrics,
        "cleaned_lyrics": cleaned_lyrics,
        "metadata": metadata
    }


def process_songs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    处理歌曲数据
    
    Args:
        data: JSON对象列表
        
    Returns:
        处理后的JSON对象列表
    """
    processed_data = []
    
    for song in data:
        processed_song = song.copy()
        
        # 清理歌词
        if "lyrics" in processed_song:
            cleaned = clean_lyrics(processed_song["lyrics"])
            processed_song["raw_lyrics"] = cleaned["raw_lyrics"]
            processed_song["cleaned_lyrics"] = cleaned["cleaned_lyrics"]
            processed_song["song_metadata"] = cleaned["metadata"]
        
        processed_data.append(processed_song)
    
    return processed_data


def main():
    """主函数"""
    # 文件路径
    input_file = "lyrics.jsonl"
    output_file = "cleaned_lyrics.jsonl"
    
    # 加载数据
    print(f"正在加载数据: {input_file}")
    data = load_jsonl_file(input_file)
    print(f"加载完成，共 {len(data)} 首歌曲")
    
    # 分析数据
    print("\n正在分析数据...")
    analysis = analyze_data(data)
    print("数据分析结果:")
    print(f"  总歌曲数: {analysis['total_songs']}")
    print(f"  唯一歌曲数: {analysis['unique_songs']}")
    print(f"  重复歌曲数: {analysis['duplicate_songs']}")
    print(f"  空标题歌曲数: {analysis['empty_titles']}")
    print(f"  空歌词歌曲数: {analysis['empty_lyrics']}")
    print(f"  平均歌词长度: {analysis['average_lyric_length']:.2f} 字符")
    
    print("\n  顶级生产者:")
    for i, (producer, count) in enumerate(list(analysis['top_producers'].items())[:10], 1):
        print(f"    {i}. {producer}: {count} 首歌曲")
    
    # 移除重复数据
    print(f"\n正在移除重复数据...")
    unique_data = remove_duplicates(data)
    print(f"去重完成，剩余 {len(unique_data)} 首歌曲")
    
    # 处理歌曲数据
    print("\n正在处理歌曲数据...")
    processed_data = process_songs(unique_data)
    print(f"处理完成")
    
    # 保存处理后的数据
    print(f"\n正在保存处理后的数据: {output_file}")
    save_jsonl_file(processed_data, output_file)
    print(f"保存完成")
    
    # 生成统计报告
    print("\n数据整理完成！")
    print(f"原始数据: {len(data)} 首歌曲")
    print(f"处理后数据: {len(processed_data)} 首歌曲")
    print(f"移除重复: {len(data) - len(processed_data)} 首歌曲")
    print(f"\n建议:")
    print("1. 检查处理后的数据，确保歌词和元数据分离正确")
    print("2. 考虑添加更多元数据字段，如发行日期、歌曲类型等")
    print("3. 可以根据歌曲标题或生产者对数据进行分类")
    print("4. 后续可以考虑将数据导入到数据库或创建API服务")


if __name__ == "__main__":
    main()
