# AI-Arbitrator (MLpyst)

![Rust](https://img.shields.io/badge/rust-v1.75+-orange.svg)
![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![ONNX](https://img.shields.io/badge/ONNX-v1.16-white.svg)

Современный скелет для высоконагруженных ML-решений. Обучение на Python (Data Science), инференс на Rust (Backend). 

Этот проект идеально подходит для автоматизации техподдержки, модерации чатов и решения споров в C2C-сделках.

---

## Почему это эффективно?

1.  **Python для DS**: Обучаем модель на `scikit-learn` или `PyTorch` в привычной среде.
2.  **Rust для Backend**: Используем `Axum` + `ONNX Runtime` для обработки 1000+ запросов в секунду с минимальной задержкой.
3.  **ONNX**: Универсальный формат обмена моделями. Никакого Python в продакшене!

---

## Структура проекта

*   **/train**: Скрипты на Python для генерации данных, обучения и экспорта в `.onnx`.
*   **/service**: Высокопроизводительный API на Rust, который принимает текст и выдает предсказание.

---

## 🛠 Быстрый запуск

### 1. Подготовка модели (Python)

```bash
cd train
pip install -r requirements.txt
# Положите ваш датасет в train/data.csv (столбцы text, label)
python train_model.py
```

### 2. Запуск сервиса (Rust)

```bash
cd service
# Перенесите model.onnx в корень папки service (скрипт сделает это сам)
cargo run --release
```

### 3. Тестирование API

Сервис принимает POST-запросы на `http://localhost:3000/predict`.

```bash
curl -X POST http://localhost:3000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": "Этот аккаунт - скам! Верни мои деньги!"}'
```

---

## Docker (Production Ready)

Проект включает `Dockerfile` для быстрой контейнеризации. Сборка мультистейдж: билд на Rust-образе, рантайм на легковесном Debian.

```bash
docker build -t ai-arbitrator .
docker run -p 3000:3000 ai-arbitrator
```

---

## Лицензия

MIT License
