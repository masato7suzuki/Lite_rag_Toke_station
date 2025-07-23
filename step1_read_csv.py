import pandas as pd

file_data = r"C:\Users\daito\Desktop\緑区フォルダー\JR-Sotobo line_Toke_20053_20244.csv"
file_note = r"C:\Users\daito\Desktop\緑区フォルダー\土気説明.csv"

df_data = pd.read_csv(file_data, encoding='cp932')
df_note = pd.read_csv(file_note, encoding='cp932')

texts_data = df_data.fillna('').astype(str).apply(lambda row: ' '.join(row), axis=1).tolist()
texts_note = df_note.fillna('').astype(str).apply(lambda row: ' '.join(row), axis=1).tolist()

all_texts = texts_note + texts_data

for i, t in enumerate(all_texts[:3]):
    print(f"[{i}]: {t[:100]}...")
