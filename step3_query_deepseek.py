# step3_query_deepseek.py

import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# --- 設定 ---
API_KEY = "sk-e2cbfc4ef2b6470dad1168efd9ee403b"
API_URL = "https://api.deepseek.com/v1/chat/completions"

# --- モデルとデータ読み込み ---
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("index.faiss")
with open("texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

# --- クエリ入力 ---
query = input("質問を入力してください：")
query_embedding = model.encode([query])

# --- FAISS検索（上位3件） ---
D, I = index.search(np.array(query_embedding).astype('float32'), k=3)
retrieved_texts = [texts[i] for i in I[0]]

# --- Deepseekへのリクエスト ---
prompt = "以下の不動産情報を参考に、質問に対して簡潔に日本語で回答してください。\n\n"
prompt += "\n\n".join(retrieved_texts)
prompt += f"\n\n質問：{query}\n\n回答："

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "あなたは優秀な不動産アドバイザーです。"},
        {"role": "user", "content": prompt}
    ]
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print("[INFO] Deepseek APIへ問い合わせ中...")
response = requests.post(API_URL, headers=headers, json=payload)
result = response.json()

# --- 出力 ---
print("\n[回答]")
print("\n[DEBUG] Deepseek レスポンス内容：")
print(json.dumps(result, indent=2, ensure_ascii=False))
