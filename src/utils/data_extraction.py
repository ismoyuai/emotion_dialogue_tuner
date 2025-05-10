import json
import random
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent

# 读取JSON文件
with open(root_path / "data/LCCC-base-split/LCCC-base_test.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# 收集所有唯一句子
processed_sentences = set()

# 遍历所有对话组和句子
for dialog_group in data:
    for sentence in dialog_group:
        # 清洗数据并加入集合
        cleaned = sentence.replace(' ', '').replace('"', '')
        processed_sentences.add(cleaned)

# 转换为列表并验证数据量
unique_sentences = list(processed_sentences)
if len(unique_sentences) < 1000:
    raise ValueError(f"错误：唯一句子只有{len(unique_sentences)}条，不足1000条")

# 随机采样1000条不重复数据
selected = random.sample(unique_sentences, 1000)

# 写入文件
with open(root_path / "data/cleaned_output.txt", 'w', encoding='utf-8') as f:
    for sentence in selected:
        f.write(sentence + '\n')