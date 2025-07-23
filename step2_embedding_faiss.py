# step2_embedding_faiss.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Step1の結果からテキストリストを再利用
from step1_read_csv import all_texts

# モデル読み込み（ローカル・軽量・無料）
model = SentenceTransformer('all-MiniLM-L6-v2')

# 埋め込み（ベクトル化）
print("[INFO] テキスト数:", len(all_texts))
print("[INFO] Embeddingを開始します。少々お待ちください...")
embeddings = model.encode(all_texts, show_progress_bar=True)

# FAISSに登録
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# 保存（index + テキスト対応表）
faiss.write_index(index, "index.faiss")

with open("texts.json", "w", encoding="utf-8") as f:
    json.dump(all_texts, f, ensure_ascii=False, indent=2)

print("[SUCCESS] EmbeddingとFAISS保存が完了しました。")
