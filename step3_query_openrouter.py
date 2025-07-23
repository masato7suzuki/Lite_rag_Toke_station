# step3_query_openrouter.py

import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# --- OpenRouter API設定 ---
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-r1:free"

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

# --- プロンプト作成 ---
prompt = "以下の不動産情報を参考に、質問に対して簡潔に日本語で回答してください。\n\n"
prompt += "\n\n".join(retrieved_texts)
prompt += f"\n\n質問：{query}\n\n回答："

payload = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "system", "content": "あなたは不動産の専門アドバイザーです。"},
        {"role": "user", "content": prompt}
    ]
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- API呼び出し ---
print("[INFO] OpenRouter経由でDeepseekに問い合わせ中...")
response = requests.post(API_URL, headers=headers, json=payload)
result = response.json()

# --- 結果出力 ---
if 'choices' in result:
    print("\n[回答]")
    print(result['choices'][0]['message']['content'])
else:
    print("\n[DEBUG] エラー発生：")
    print(json.dumps(result, indent=2, ensure_ascii=False))
