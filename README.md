# Lite_rag_Toke_station
Real estate finder in Toke station.
# Lite RAG Toolkit

A lightweight Retrieval-Augmented Generation (RAG) implementation using OpenAI, Streamlit, and basic vector storage.

軽量なRAG（検索拡張生成）ツールキットであり、OpenAI、Streamlit、および基本的なベクトルストレージを利用しています。

---

## Features

- Minimal setup with Python
- File upload and document parsing
- Local vector embedding
- Streamlit-based frontend
- OpenAI API integration

## 主な機能

- Pythonによる最小構成
- ファイルアップロードと文書解析
- ローカルでのベクトル埋め込み処理
- Streamlitを用いたインターフェース
- OpenAI APIとの統合

---

## How to Use

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 使用方法

1. このリポジトリをクローンする
2. 必要なパッケージをインストール：`pip install -r requirements.txt`
3. アプリを起動：`streamlit run app.py`

---

## Configuration

Set your OpenAI API key in `.env` file:

## Backend Options (Embedding and Chat)

You can choose between:

- ✅ DeepSeek API (Free, tested)
- ☑️ OpenAI API via OpenRouter (Optional, needs key)
