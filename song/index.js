import Meting from '@meting/core';
import fs from 'fs';
import path from 'path';

// 获取当前文件所在目录的绝对路径
const __filename = new URL(import.meta.url).pathname;
// 修复Windows路径问题，移除前导斜杠
const normalizedPath = __filename.startsWith('/') ? __filename.slice(1) : __filename;
const __dirname = path.dirname(normalizedPath);

// 文件路径 - 注意：用户指定的是lyrics.jsonl，而不是song_lyrics.jsonl
const LYRICS_FILE_PATH = path.join(__dirname, 'lyrics.jsonl');

// 初始化Meting实例
const meting = new Meting('netease');
meting.format(true);

// 清理歌曲名称，移除括号及其内容
function cleanSongTitle(title) {
    // 移除所有括号及其内容，包括中英文括号
    return title.replace(/\(.*?\)|（.*?）/g, '').trim();
}

// 加载已存在的歌曲标题
async function loadExistingSongs() {
    const existingSongs = new Set();
    
    // 检查lyrics.jsonl文件是否存在
    if (fs.existsSync(LYRICS_FILE_PATH)) {
        try {
            const content = fs.readFileSync(LYRICS_FILE_PATH, 'utf-8');
            const lines = content.split('\n').filter(line => line.trim());
            
            for (const line of lines) {
                try {
                    const songData = JSON.parse(line.trim());
                    const songTitle = songData.song_title || '';
                    if (songTitle) {
                        // 清理歌曲名称后添加到集合中
                        const cleanedTitle = cleanSongTitle(songTitle);
                        existingSongs.add(cleanedTitle);
                    }
                } catch (error) {
                    // 忽略解析错误的行
                    console.error(`解析歌词文件行失败: ${error.message}`);
                    continue;
                }
            }
        } catch (error) {
            console.error(`读取歌词文件失败: ${error.message}`);
        }
    }
    
    return existingSongs;
}

// 清理歌词，移除时间标签
function cleanLyrics(lyrics) {
    if (!lyrics) return '';
    
    // 移除各种格式的时间标签：[00:00.00], [00:00.000], [00:00]
    let cleaned = lyrics.replace(/\[\d{2}:\d{2}(\.\d{2,3})?\]/g, '').trim();
    
    // 移除可能存在的空行
    cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
    
    return cleaned;
}

// 获取单首歌曲的歌词
async function getSongLyric(song) {
    try {
        // 获取歌词
        const lyricResult = await meting.lyric(song.id);
        const lyricData = JSON.parse(lyricResult);
        
        // 如果有歌词且不是纯音乐
        if (lyricData.lyric && lyricData.lyric.trim() && !lyricData.lyric.includes('纯音乐，请欣赏')) {
            const cleanedLyrics = cleanLyrics(lyricData.lyric);
            
            // 处理p_masters字段，确保它是一个字符串数组
            let p_masters = [];
            if (song.artist) {
                // 检查artist是否已经是数组，如果是则直接使用，如果不是则包装成数组
                if (Array.isArray(song.artist)) {
                    // 确保数组中的元素都是字符串，且没有嵌套数组
                    p_masters = song.artist.flat().filter(item => typeof item === 'string');
                } else {
                    p_masters = [String(song.artist)];
                }
            }
            
            return {
                song_title: song.name,
                p_masters: p_masters,
                lyrics: cleanedLyrics
            };
        } else {
            return null; // 跳过纯音乐或无歌词的歌曲
        }
    } catch (error) {
        console.error(`获取歌曲 ${song.name} 歌词失败: ${error.message}`);
        return null;
    }
}

// 写入歌词到文件 - 每次获得歌词都会立即调用此函数保存到文件
function writeLyricToFile(lyricData) {
    if (!lyricData || !lyricData.lyrics) {
        console.log(`跳过无效歌词数据: ${lyricData?.song_title || '未知歌曲'}`);
        return;
    }
    
    try {
        const line = JSON.stringify(lyricData, null, 0) + '\n';
        // 使用同步写入确保歌词立即保存到文件
        fs.appendFileSync(LYRICS_FILE_PATH, line, 'utf-8');
        console.log(`已成功保存歌曲到文件: ${lyricData.song_title}`);
    } catch (error) {
        console.error(`保存歌曲到文件失败: ${lyricData.song_title}, 错误: ${error.message}`);
    }
}

// 搜索洛天依歌曲并获取歌词
async function searchLuoTianyiSongs(initialPage = 1, pageCount = 1) {
    console.log('开始搜索洛天依歌曲...');
    
    // 加载已存在的歌曲
    const existingSongs = await loadExistingSongs();
    console.log(`已存在 ${existingSongs.size} 首歌曲`);
    
    // 如果jsonl文件不存在，创建一个空文件
    if (!fs.existsSync(LYRICS_FILE_PATH)) {
        fs.writeFileSync(LYRICS_FILE_PATH, '', 'utf-8');
        console.log(`已创建歌词文件: ${LYRICS_FILE_PATH}`);
    }
    
    const searchQuery = '洛天依';
    let totalSongsFound = 0;
    let totalNewSongsAdded = 0;
    
    // 遍历指定的页数
    for (let page = initialPage; page < initialPage + pageCount; page++) {
        console.log(`\n正在搜索第 ${page} 页...`);
        
        try {
            // 搜索歌曲
            const searchResult = await meting.search(searchQuery, { page: page, limit: 10 });
            const songs = JSON.parse(searchResult);
            
            console.log(`第 ${page} 页找到 ${songs.length} 首歌曲`);
            totalSongsFound += songs.length;
            
            // 过滤出新歌曲
            const newSongs = songs.filter(song => {
                const cleanedSongName = cleanSongTitle(song.name);
                return !existingSongs.has(cleanedSongName);
            });
            console.log(`第 ${page} 页有 ${newSongs.length} 首新歌曲`);
            
            // 逐一处理新歌曲 - 每次获得歌词都会立即保存到文件
            for (const song of newSongs) {
                console.log(`\n正在处理歌曲: ${song.name}`);
                
                // 获取歌曲歌词
                const lyricData = await getSongLyric(song);
                
                // 立即保存歌词到文件（如果有有效歌词）
                if (lyricData) {
                    writeLyricToFile(lyricData);
                    totalNewSongsAdded++;
                }
                
                // 将新歌曲清理后的名称添加到已存在集合中，避免重复处理
                const cleanedSongName = cleanSongTitle(song.name);
                existingSongs.add(cleanedSongName);
                
                // 避免请求过快，添加延迟
                console.log('添加500ms延迟，避免请求过快...');
                await new Promise(resolve => setTimeout(resolve, 500));
            }
        } catch (error) {
            console.error(`搜索第 ${page} 页失败: ${error.message}`);
            continue;
        }
    }
    
    console.log(`\n搜索完成！`);
    console.log(`总共找到 ${totalSongsFound} 首歌曲`);
    console.log(`新添加了 ${totalNewSongsAdded} 首歌曲到歌词文件中`);
    console.log(`当前歌词文件中共有 ${existingSongs.size} 首歌曲`);
}

// 全局变量设置
const initialPage = 21; // 初始页，可修改
const pageCount = 40; // 要加载的页数，可修改

// 主函数
async function main() {
    console.log(`设置：初始页 = ${initialPage}, 加载页数 = ${pageCount}`);
    
    await searchLuoTianyiSongs(initialPage, pageCount);
}

// 执行主函数
main().catch(error => {
    console.error('程序执行失败:', error);
    process.exit(1);
});
