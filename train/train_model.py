import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

DATA_PATH = "train/data.csv"
MODEL_OUTPUT = "service/model.onnx"

def train():
    if not os.path.exists(DATA_PATH):
        print(f"[-] Файл {DATA_PATH} не найден. Создаю шаблон...")
        pd.DataFrame(columns=['text', 'label']).to_csv(DATA_PATH, index=False)
        return

    df = pd.read_csv(DATA_PATH)
    if df.empty or len(df) < 2:
        print("[-] Недостаточно данных.")
        return

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(solver='liblinear'))
    ])

    print("[*] Обучение...")
    pipeline.fit(df['text'], df['label'])
    
    initial_type = [('input', StringTensorType([None, 1]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)

    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    with open(MODEL_OUTPUT, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[+] Успех! Модель экспортирована: {MODEL_OUTPUT}")

if __name__ == "__main__":
    train()
