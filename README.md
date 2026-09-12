# Mini-LLM — decoder-only Transformer, обученный с нуля

Настоящая autoregressive language model: `tokens → embeddings → transformer → logits → next-token prediction`.
Никаких баз «вопрос→ответ», retrieval, шаблонов и готовых весов. Ответ генерируется собственными весами токен за токеном.

## Архитектура

- Token embeddings + RoPE (или learned positional, переключается в конфиге)
- Multi-Head Self-Attention с causal mask (поддержан GQA через `num_kv_heads`), pre-norm блоки
- SwiGLU FFN, RMSNorm/LayerNorm, residual connections, N блоков
- LM head + causal CrossEntropy loss со сдвигом `input[t] → target[t+1]`
- Генерация: цикл `prompt → tokenize → model → sample → append → model → …` (`model/generation.py`)

Пресеты (`config/*.json`):

| preset | vocab | hidden | layers | heads | ctx | params |
|---|---|---|---|---|---|---|
| micro (CPU smoke) | 2000 | 128 | 4 | 4 | 256 | ~1M |
| tiny (старт, free Colab) | 8000 | 256 | 6 | 8 | 512 | ~9M |
| small | 16000 | 384 | 8 | 8 | 512 | ~25M |
| medium | 24000 | 512 | 12 | 8 | 1024 | ~110M |
| base (путь к Qwen-уровню, T4, много сессий) | 24000 | 768 | 12 | 12 | 1024 | ~150M |

## Структура

```
├── config/{config,tiny,small,medium,micro}.json
├── tokenizer/train_tokenizer.py
├── model/{config,model,generation}.py
├── training/{train,dataset,collator}.py
├── scripts/{build_data,evaluate}.py
├── inference/chat.py
├── tests/test_generation.py
├── colab/train_colab.ipynb
└── data/{conversations.jsonl, pretrain_corpus.txt, conversations_text.txt}
```

## 1. Установка

```bash
python -m pip install -r requirements.txt
```

## 2. Подготовка датасета

```bash
python scripts/build_data.py
# -> data/conversations.jsonl (JSONL {"messages":[{"role","content"}]}), pretrain_corpus.txt
```

Формат строки `conversations.jsonl`:
```json
{"messages":[{"role":"user","content":"Привет"},{"role":"assistant","content":"Привет! Чем могу помочь сегодня?"}]}
```

Свой датасет: положите свой JSONL в тот же формат и укажите путь в `data.sft_file` конфига.

## 3. Обучение токенизатора (Byte-Level BPE, RU+EN)

```bash
python -m tokenizer.train_tokenizer --config config/tiny.json
# проверка примеров RU/EN печатается в конце; файлы -> tokenizer/out_tiny/tokenizer.json
```

Спецтокены: `<PAD> <UNK> <BOS> <EOS> <USER> <ASSISTANT>`.

## 4. Обучение локально

```bash
# быстрая проверка пайплайна на CPU (секунды-минуты):
python -m tokenizer.train_tokenizer --config config/micro.json
python -m training.train --config config/micro.json --stage sft

# локальное демо с качеством (CPU, ~15-25 мин): претрейн + 6 эпох SFT
python -m training.train --config config/micro.json --stage pretrain
python -m training.train --config config/micro_full.json --stage sft  # resume с претрейна
python -m inference.chat --checkpoint checkpoints/micro_full/export
# готовые веса micro-демо уже лежат в checkpoints/micro_full/export — чат работает без обучения

# основная конфигурация:
python -m training.train --config config/tiny.json --stage pretrain  # stage 1 (опционально)
python -m training.train --config config/tiny.json --stage sft       # stage 2, assistant-only loss
```

Используются: mixed precision (bf16/fp16 на CUDA), gradient accumulation, gradient clipping,
cosine scheduler с warmup, checkpoint saving (`last.pt`/`best.pt`), resume, eval split, сохранение лучшего чекпоинта.

## 5. Обучение в Google Colab

1. Откройте `colab/train_colab.ipynb` в Colab (GPU: T4).
2. Выполняйте ячейки по порядку: GPU check → install → data → tokenizer → sanity → pretrain → sft → evaluate → tests.
3. Ячейка упаковки скачает `mini-llm-tiny.zip` (export + tokenizer + config).

## 6. Чекпоинты

`checkpoints/<run>/`: `last.pt`, `best.pt`, `export/{config.json, model.safetensors}`, `tokenizer.json`, `run_config.json`.

## 7. Скачивание модели из Colab

Последняя ячейка ноутбука создаёт и скачивает `mini-llm-tiny.zip`. Распакуйте рядом с проектом.

## 8. Чат (без переобучения)

```bash
python -m inference.chat --checkpoint checkpoints/tiny_run/export
# опции: --temperature 0.8 --top-k 50 --top-p 0.9 --repetition-penalty 1.1 --max-new-tokens 128
```

Шаблон: `<USER>\nвопрос\n<ASSISTANT>\n` — модель продолжает после `<ASSISTANT>`, стоп на `<EOS>`.

## 9. Изменение размера модели

Отредактируйте `config/*.json` (`model`: `hidden_size`, `num_layers`, `num_heads`, `context_length`, `vocab_size`;
`train`: `batch_size`, `learning_rate`, `epochs`, `gradient_accumulation_steps`, `weight_decay`) или скопируйте пресет.

## 10. Изменение датасета

Замените/дополните `data/conversations.jsonl` (и `pretrain_corpus.txt` для stage 1), переобучите токенизатор, если сильно меняется язык/домен.

## 11. Продолжение обучения

```bash
python -m training.train --config config/tiny.json --stage sft --resume checkpoints/tiny_run/last
```

## 12. Оценка

```bash
python scripts/evaluate.py --checkpoint checkpoints/tiny_run/export   # RU/EN набор из 14 промптов
python -m tests.test_generation --checkpoint checkpoints/tiny_run/export  # 8 проверок «это LLM, а не поиск»
```

## Ограничения маленькой модели

- 9–25M параметров не удерживают широкие факты; ответы короткие и иногда общие.
- Качество растёт с объёмом/разнообразием данных и stage-1 претрейном; tiny без претрейна — разговорный минимум.
- Длинные рассуждения и редкие знания — за пределами tiny/small; для них нужен больший корпус и `medium`.
