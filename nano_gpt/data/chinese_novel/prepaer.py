import os
import numpy as np
import tiktoken
novel_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
enc = tiktoken.get_encoding("gpt2")
train_output_file = os.path.join(os.path.dirname(__file__), 'train.bin')
val_output_file = os.path.join(os.path.dirname(__file__), 'val.bin')
if __name__ == '__main__':
    # 读取小说文件
    with open(novel_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 简单分割成 train 和 val（例如，按字符数分割，val 占 10%）
    val_ratio = 0.1
    split_idx = int(len(text) * (1 - val_ratio))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # 定义分词函数
    def tokenize_text(text_chunk):
        ids = enc.encode_ordinary(text_chunk)
        ids.append(enc.eot_token)  # 添加结束 token
        return ids

    # 分词
    train_ids = tokenize_text(train_text)
    val_ids = tokenize_text(val_text)

    # 保存为二进制文件
    def save_to_bin(ids, filename):
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(filename)
        print(f"Saved {len(ids)} tokens to {filename}")

    save_to_bin(train_ids, train_output_file)
    save_to_bin(val_ids, val_output_file)
