import Meting from '@meting/core';
import fs from 'fs';
import path from 'path';

// 获取当前文件所在目录的绝对路径
const __filename = new URL(import.meta.url).pathname;
const normalizedPath = __filename.startsWith('/') ? __filename.slice(1) : __filename;
const __dirname = path.dirname(normalizedPath);

// 文件路径
const LYRICS_FILE_PATH = path.join(__dirname, 'lyrics.jsonl');

// 初始化Meting实例
const meting = new Meting('netease');
meting.format(true);

// 从lyrics.jsonl文件中加载已存在的歌曲
async function loadExistingSongs() {
    const existingSongs = new Set();
    
    // 检查文件是否存在
    if (!fs.existsSync(LYRICS_FILE_PATH)) {
        console.log('lyrics.jsonl文件不存在，将创建新文件');
        return existingSongs;
    }
    
    try {
        const content = fs.readFileSync(LYRICS_FILE_PATH, 'utf-8');
        const lines = content.split('\n');
        
        for (const line of lines) {
            if (!line.trim()) continue;
            
            try {
                const songData = JSON.parse(line.trim());
                if (songData.song_title) {
                    existingSongs.add(songData.song_title);
                }
            } catch (error) {
                console.error(`解析行失败: ${line}`);
                continue;
            }
        }
        
        console.log(`已加载 ${existingSongs.size} 首已存在的歌曲`);
        return existingSongs;
    } catch (error) {
        console.error(`加载现有歌曲失败: ${error.message}`);
        return existingSongs;
    }
}

// 获取单首歌曲的歌词
async function getSongLyric(song) {
    try {
        const lyricResult = await meting.lyric(song.id);
        const lyricData = JSON.parse(lyricResult);
        
        if (lyricData.lyric && lyricData.lyric.trim()) {
            return {
                success: true,
                lyric: lyricData.lyric
            };
        } else {
            return {
                success: false,
                message: '未找到歌词'
            };
        }
    } catch (error) {
        return {
            success: false,
            message: `获取歌词失败: ${error.message}`
        };
    }
}

// 清理歌词，删除时间标签
function cleanLyrics(lyrics) {
    if (!lyrics) return '';
    
    // 使用正则表达式删除时间标签
    const cleaned = lyrics.replace(/\[\d{2}:\d{2}(:\d{2})?\.\d{2,3}?\]/g, '');
    
    // 去除多余的换行和空格
    return cleaned
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0)
        .join('\n');
}

// 搜索洛天依的歌曲并获取歌词
async function searchLuoTianyiSongs(initialPage = 1, pageCount = 1) {
    console.log(`开始搜索洛天依的歌曲，初始页: ${initialPage}，页数: ${pageCount}`);
    
    // 加载已存在的歌曲
    const existingSongs = await loadExistingSongs();
    
    // 统计信息
    let totalSongs = 0;
    let newSongs = 0;
    let skippedSongs = 0;
    
    // 遍历每一页
    for (let page = initialPage; page < initialPage + pageCount; page++) {
        console.log(`\n正在处理第 ${page} 页...`);
        
        try {
            // 搜索洛天依的歌曲
            const searchResult = await meting.search('洛天依', { page, limit: 10 });
            const songs = JSON.parse(searchResult);
            
            if (!Array.isArray(songs) || songs.length === 0) {
                console.log(`第 ${page} 页没有找到歌曲`);
                continue;
            }
            
            totalSongs += songs.length;
            
            // 遍历搜索结果
            for (const song of songs) {
                const songTitle = song.name;
                console.log(`\n检查歌曲: ${songTitle}`);
                
                // 检查歌曲是否已存在
                if (existingSongs.has(songTitle)) {
                    console.log(`歌曲 ${songTitle} 已存在，跳过`);
                    skippedSongs++;
                    continue;
                }
                
                // 获取歌曲歌词
                const lyricResult = await getSongLyric(song);
                
                if (lyricResult.success) {
                    // 清理歌词
                    const cleanedLyric = cleanLyrics(lyricResult.lyric);
                    
                    if (cleanedLyric.includes('纯音乐，请欣赏')) {
                        console.log(`歌曲 ${songTitle} 是纯音乐，跳过`);
                        continue;
                    }
                    
                    // 构建歌曲数据
                    const songData = {
                        song_title: songTitle,
                        p_masters: song.artist,
                        lyrics: cleanedLyric
                    };
                    
                    // 写入文件
                    const line = JSON.stringify(songData, null, 0) + '\n';
                    fs.appendFileSync(LYRICS_FILE_PATH, line, 'utf-8');
                    
                    console.log(`成功获取并保存歌曲 ${songTitle} 的歌词`);
                    newSongs++;
                    
                    // 更新已存在歌曲集合
                    existingSongs.add(songTitle);
                    
                    // 避免请求过快，添加延迟
                    await new Promise(resolve => setTimeout(resolve, 500));
                } else {
                    console.log(`获取歌曲 ${songTitle} 歌词失败: ${lyricResult.message}`);
                }
            }
        } catch (error) {
            console.error(`处理第 ${page} 页时出错: ${error.message}`);
            continue;
        }
    }
    
    // 输出统计信息
    console.log(`\n搜索完成！`);
    console.log(`总搜索歌曲数: ${totalSongs}`);
    console.log(`新增歌曲数: ${newSongs}`);
    console.log(`跳过已存在歌曲数: ${skippedSongs}`);
    console.log(`最终歌曲总数: ${existingSongs.size}`);
}

// 执行搜索
// 支持通过命令行参数设置初始页和页数
const args = process.argv.slice(2);
const initialPage = parseInt(args[0]) || 1;
const pageCount = parseInt(args[1]) || 1;

searchLuoTianyiSongs(initialPage, pageCount).catch(error => {
    console.error('搜索过程中出错:', error);
    process.exit(1);
});
