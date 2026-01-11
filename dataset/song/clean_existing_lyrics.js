import fs from 'fs';
import path from 'path';

// 获取当前文件所在目录的绝对路径
const __filename = new URL(import.meta.url).pathname;
const normalizedPath = __filename.startsWith('/') ? __filename.slice(1) : __filename;
const __dirname = path.dirname(normalizedPath);

// 文件路径
const LYRICS_FILE_PATH = path.join(__dirname, 'lyrics.jsonl');
const TEMP_FILE_PATH = path.join(__dirname, 'lyrics_temp.jsonl');

// 清理歌词，移除时间标签
function cleanLyrics(lyrics) {
    if (!lyrics) return '';
    
    // 移除各种格式的时间标签：[00:00.00], [00:00.000], [00:00]
    let cleaned = lyrics.replace(/\[\d{2}:\d{2}(\.\d{2,3})?\]/g, '').trim();
    
    // 移除可能存在的空行
    cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
    
    return cleaned;
}

// 清理p_masters字段，确保它是一个字符串数组
function cleanP_masters(p_masters) {
    let cleaned = [];
    
    if (Array.isArray(p_masters)) {
        // 展平数组，确保没有嵌套数组
        const flattened = p_masters.flat();
        // 过滤出字符串类型的元素
        cleaned = flattened.filter(item => typeof item === 'string');
    } else if (typeof p_masters === 'string') {
        // 如果是字符串，包装成数组
        cleaned = [p_masters];
    } else if (p_masters) {
        // 其他类型，转换为字符串并包装成数组
        cleaned = [String(p_masters)];
    }
    
    return cleaned;
}

// 清理现有歌词文件
async function cleanExistingLyrics() {
    console.log('开始清理现有歌词文件...');
    
    // 检查文件是否存在
    if (!fs.existsSync(LYRICS_FILE_PATH)) {
        console.error('歌词文件不存在！');
        return;
    }
    
    let totalLines = 0;
    let processedLines = 0;
    let skippedLines = 0;
    
    // 读取文件内容
    const content = fs.readFileSync(LYRICS_FILE_PATH, 'utf-8');
    const lines = content.split('\n').filter(line => line.trim());
    
    totalLines = lines.length;
    console.log(`共找到 ${totalLines} 行数据`);
    
    // 处理每一行，收集到数组中
    const processedLinesArray = [];
    
    for (const line of lines) {
        try {
            // 解析JSON
            const songData = JSON.parse(line.trim());
            
            // 清理歌词
            if (songData.lyrics) {
                songData.lyrics = cleanLyrics(songData.lyrics);
            }
            
            // 清理p_masters字段
            songData.p_masters = cleanP_masters(songData.p_masters);
            
            // 添加到处理结果数组
            processedLinesArray.push(songData);
            processedLines++;
        } catch (error) {
            console.error(`处理行失败: ${error.message}`);
            skippedLines++;
            continue;
        }
    }
    
    console.log(`处理完成！`);
    console.log(`总共处理 ${totalLines} 行`);
    console.log(`成功处理 ${processedLines} 行`);
    console.log(`跳过 ${skippedLines} 行`);
    
    // 直接写入回原文件（使用同步写入确保安全）
    const outputContent = processedLinesArray.map(line => JSON.stringify(line, null, 0)).join('\n') + '\n';
    fs.writeFileSync(LYRICS_FILE_PATH, outputContent, 'utf-8');
    
    console.log(`\n已成功写入清理后的歌词到文件！`);
    console.log(`文件路径: ${LYRICS_FILE_PATH}`);
}

// 执行清理
cleanExistingLyrics().catch(error => {
    console.error('清理歌词文件失败:', error);
    process.exit(1);
});