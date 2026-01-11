import fs from 'fs';
import path from 'path';

// 获取当前文件所在目录的绝对路径
const __filename = new URL(import.meta.url).pathname;
// 修复Windows路径问题，移除前导斜杠
const normalizedPath = __filename.startsWith('/') ? __filename.slice(1) : __filename;
const __dirname = path.dirname(normalizedPath);

// 文件路径
const INPUT_FILE_PATH = path.join(__dirname, 'song_lyrics.jsonl');
const OUTPUT_FILE_PATH = path.join(__dirname, 'lyrics.jsonl');

// 处理歌词，删除时间标签
function cleanLyrics(lyrics) {
    if (!lyrics) return '';

    // 使用正则表达式删除时间标签，格式如 [00:00.00] 或 [00:00:00]
    const cleaned = lyrics.replace(/\[\d{2}:\d{2}(:\d{2})?\.\d{2,3}?\]/g, '');

    // 去除多余的换行和空格
    return cleaned
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0)
        .join('\n');
}

// 处理JSONL文件
function processJsonlFile() {
    try {
        // 读取输入文件内容
        const content = fs.readFileSync(INPUT_FILE_PATH, 'utf-8');
        const lines = content.split('\n');

        // 处理每一行
        const processedLines = [];

        for (const line of lines) {
            if (!line.trim()) continue;

            try {
                const songData = JSON.parse(line.trim());

                // 检查是否找到歌词
                const lyrics = songData.lyrics || songData.get('lyrics', '') || '';

                // 跳过没有找到歌词的条目
                if (lyrics === '未找到歌词' || lyrics.includes('获取歌词失败') || lyrics === '纯音乐，请欣赏') {
                    continue;
                }

                // 整理歌词，删除时间标签
                const cleanedLyrics = cleanLyrics(lyrics);

                // 如果整理后的歌词为空或包含"纯音乐，请欣赏"，跳过
                if (!cleanedLyrics.trim() || cleanedLyrics.includes('纯音乐，请欣赏')) {
                    continue;
                }

                // 更新歌词
                songData.lyrics = cleanedLyrics;

                // 只保留需要的字段
                const cleanedSongData = {
                    song_title: songData.song_title,
                    p_masters: songData.p_masters && songData.p_masters.length > 0 ? [songData.p_masters[0]] : [],
                    lyrics: songData.lyrics
                };

                // 添加到处理结果
                processedLines.push(JSON.stringify(cleanedSongData, null, 0));
            } catch (error) {
                // 忽略解析错误的行
                console.error(`解析行失败: ${error.message}`);
                continue;
            }
        }

        // 写入输出文件
        fs.writeFileSync(OUTPUT_FILE_PATH, processedLines.join('\n') + '\n', 'utf-8');

        console.log(`处理完成！`);
        console.log(`输入文件行数: ${lines.length}`);
        console.log(`输出文件行数: ${processedLines.length}`);
        console.log(`已删除: ${lines.length - processedLines.length} 行`);
    } catch (error) {
        console.error(`处理文件失败: ${error.message}`);
    }
}

// 执行处理
processJsonlFile();
