"""Build diverse conversational dataset + pretraining corpus.

Outputs:
  data/conversations.jsonl      - {"messages": [{"role": "user"|"assistant", "content": str}, ...]}
  data/conversations_text.txt   - plain text dump (for tokenizer training)
  data/pretrain_corpus.txt      - plain LM corpus (stage-1 pretraining)

Design: many distinct topics, several paraphrases per question, several
different answers per repeated question (so the model learns language
patterns, not 1:1 memorization).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 1234
N_TARGET = 1400  # total dialogs

# ---------------------------------------------------------------- pools
# Each entry: (topic, [(user_variant, [assistant_variants...])])
POOLS: list[tuple[str, list[tuple[str, list[str]]]]] = [
 ("greeting", [
  ("Привет", ["Привет! Рад тебя видеть. Чем могу помочь сегодня?",
              "Здравствуйте! Что хотите узнать?",
              "Привет-привет! Спрашивайте, помогу чем смогу."]),
  ("Привет! Как дела?", ["Привет! У меня всё хорошо, я же программа. А у вас как дела? Чем помочь?",
                          "Здравствуйте! Дела отлично. Что вас интересует сегодня?"]),
  ("Hello", ["Hello! How can I help you today?",
             "Hi there! What would you like to know?"]),
  ("How are you?", ["I'm doing well, thank you! How can I assist you today?",
                     "All good here. What can I do for you?"]),
  ("Добрый вечер", ["Добрый вечер! Чем могу быть полезен?",
                    "Здравствуйте! Рад вас видеть. Спрашивайте."]),
  ("Что нового?", ["Я постоянно учусь новому. А у вас что нового? Могу рассказать про науку, код или языки.",
                   "Много всего: в мире технологий каждый день что-то происходит. Что вам интересно?"]),
 ]),
 ("about_self", [
  ("Расскажи о себе.", ["Я — небольшая языковая модель, обученная предсказывать следующий токен текста. Могу отвечать на вопросы, объяснять темы и помогать с кодом.",
                         "Я компактный ИИ-ассистент: принимаю текст, превращаю его в токены и генерирую продолжение по своим весам. Спросите что-нибудь!"]),
  ("Tell me about yourself.", ["I'm a small autoregressive language model. I take tokens as input and predict the next tokens, one step at a time.",
                                "I'm a compact AI assistant trained from scratch to generate text. Ask me anything!"]),
  ("Кто ты?", ["Я — мини-LLM: нейросеть-трансформер, которая генерирует ответы токен за токен.",
               "Я искусственный интеллект, маленькая языковая модель. Помогаю с вопросами и задачами."]),
  ("What can you do?", ["I can answer questions, explain concepts, help with code and math, and chat in Russian or English.",
                         "I can explain topics, write and debug simple code, solve basic math, and hold a conversation."]),
 ]),
 ("programming", [
  ("Что такое Python?", ["Python — это язык программирования общего назначения с простым синтаксисом. Его используют для веба, анализа данных, автоматизации и ИИ.",
                          "Python — популярный язык: читаемый синтаксис, большая экосистема библиотек. Хорош как первый язык."]),
  ("What is Python?", ["Python is a general-purpose programming language known for its simple syntax and wide range of applications, from web to data science.",
                        "Python is a readable, high-level language with a huge ecosystem. Great as a first language."]),
  ("Как изучать программирование?", ["Начните с базового синтаксиса: переменные, условия, циклы, функции. Затем делайте маленькие проекты каждый день и читайте чужой код.",
                                      "План: 1) синтаксис и типы данных, 2) циклы и функции, 3) структуры данных, 4) свой проект. Главное — практика по 30-60 минут ежедневно."]),
  ("How can I learn programming?", ["Start with basic syntax — variables, conditions, loops, functions — then build small projects. Consistency beats long rare sessions.",
                                     "Pick one language, learn fundamentals, then practice daily on tiny projects and read other people's code."]),
  ("Что такое функция?", ["Функция — это именованный блок кода, который принимает входные данные и возвращает результат. Она помогает не повторять код.",
                           "Функция упаковывает логику: вызвал с аргументами — получил результат. Пример на Python: def square(x): return x*x."]),
  ("What is a variable?", ["A variable is a named place to store a value, like x = 5. You can read it and change it later.",
                            "Think of a variable as a labeled box holding data your program can reuse."]),
  ("Объясни, что такое цикл.", ["Цикл повторяет действия: for проходит по коллекции, while повторяется пока условие истинно. Важно не забыть условие выхода.",
                                 "Цикл for в Python: for i in range(5): print(i) — напечатает числа от 0 до 4."]),
  ("What is a list in Python?", ["A list is an ordered collection: nums = [1, 2, 3]. You can index it, slice it, append items and loop over it.",
                                  "Lists hold sequences of values and support indexing, slicing and methods like append and pop."]),
  ("Что такое API?", ["API — это интерфейс, через который программы общаются: набор функций или HTTP-методов с описанными входами и выходами.",
                       "API позволяет одной программе пользоваться возможностями другой, не зная её внутреннего устройства."]),
  ("How do I debug code?", ["Read the error message carefully, reproduce the bug minimally, add prints or use a debugger, fix one thing at a time and re-test.",
                              "Narrow it down: isolate the failing input, check assumptions with prints, form a hypothesis, test the fix."]),
  ("Что такое Git?", ["Git — система контроля версий: фиксирует изменения кода коммитами, позволяет ветвиться и объединять работу нескольких людей.",
                       "С Git вы сохраняете историю проекта, откатываетесь к старым версиям и работаете в ветках без страха сломать основу."]),
  ("Explain recursion.", ["Recursion is when a function calls itself on a smaller subproblem, with a base case that stops. Example: factorial(n) = n * factorial(n-1), factorial(0) = 1.",
                            "A recursive solution has two parts: the base case and the step that shrinks the problem toward it."]),
 ]),
 ("math", [
  ("Сколько будет 7 умножить на 8?", ["7 × 8 = 56. Можно посчитать как 7×4=28, затем удвоить: 56.",
                                       "Получится 56."]),
  ("What is 15% of 200?", ["15% of 200 is 30: 200 × 0.15 = 30.",
                             "30. You can compute 10% (20) plus 5% (10)."]),
  ("Что такое простое число?", ["Простое число делится только на 1 и на себя: 2, 3, 5, 7, 11, 13 и так далее.",
                                 "Это число больше 1 без делителей кроме 1 и себя. Например, 17 — простое, а 18 — нет."]),
  ("What is the Pythagorean theorem?", ["For a right triangle, a² + b² = c², where c is the hypotenuse.",
                                          "The square of the hypotenuse equals the sum of squares of the legs."]),
  ("Реши уравнение: 2x + 3 = 11", ["2x = 8, значит x = 4. Проверка: 2·4+3 = 11. Верно.",
                                    "Вычитаем 3: 2x = 8, делим на 2: x = 4."]),
  ("What is a fraction?", ["A fraction like 3/4 means 3 parts out of 4 equal parts. The top is the numerator, the bottom the denominator.",
                             "It represents division: 3/4 = 0.75."]),
  ("Объясни, что такое процент.", ["Процент — это сотая часть: 25% от 200 = 50. Чтобы найти p% от числа, умножьте число на p/100.",
                                    "1% — одна сотая. 50% — половина, 100% — целое."]),
  ("What is 12 squared?", ["12² = 144.", "144 — twelve times twelve."]),
 ]),
 ("science", [
  ("Объясни, что такое космос.", ["Космос — это огромное пространство, в котором находятся звёзды, планеты, галактики и другие небесные объекты. Большая часть его — вакуум.",
                                   "Космосом называют всё за пределами атмосферы Земли: звёзды, планеты, газ, пыль и тёмную материю. Он расширяется со времени Большого взрыва."]),
  ("Explain what space is.", ["Space is the vast expanse containing stars, planets, galaxies and other celestial objects, mostly vacuum.",
                                "Space is everything beyond Earth's atmosphere. It is enormous, mostly empty, and still expanding."]),
  ("Что такое искусственный интеллект?", ["Искусственный интеллект — это системы, которые выполняют задачи, требующие мышления: распознавание, прогнозирование, генерация текста. Языковые модели предсказывают следующий токен по контексту.",
                                           "ИИ — область computer science о машинах, имитирующих интеллект. Один из подходов — нейросети-трансформеры, обученные на больших текстах."]),
  ("What is artificial intelligence?", ["AI is the field of building systems that perform tasks requiring intelligence: perception, prediction, language. Language models predict the next token given context.",
                                         "Artificial intelligence studies machines that mimic intelligent behavior, from vision to text generation."]),
  ("Почему небо голубое?", ["Солнечный свет рассеивается молекулами воздуха; короткие синие волны рассеиваются сильнее — это рэлеевское рассеяние.",
                              "Атмосфера сильнее рассеивает синий свет, поэтому днём небо голубое, а на закате — красное."]),
  ("Why is the sky blue?", ["Sunlight scatters off air molecules; short blue wavelengths scatter most — Rayleigh scattering.",
                              "The atmosphere scatters blue light in all directions, so the sky looks blue."]),
  ("Что такое фотосинтез?", ["Фотосинтез — процесс, которым растения превращают свет, воду и CO2 в глюкозу и кислород.",
                               "Растения улавливают энергию света хлорофиллом и запасают её в сахарах, выделяя кислород."]),
  ("What is gravity?", ["Gravity is the attraction between masses. Newton described it as a force; Einstein as curvature of spacetime.",
                          "It keeps planets in orbit and pulls objects toward Earth at about 9.8 m/s²."]),
  ("Что такое атом?", ["Атом — мельчайшая единица вещества: ядро из протонов и нейтронов плюс электроны вокруг.",
                        "Всё вокруг состоит из атомов. Их тип определяет элемент: водород, кислород, железо и т.д."]),
  ("How does rain form?", ["Water evaporates, cools and condenses on tiny particles into droplets. When droplets grow heavy, they fall as rain.",
                             "Warm moist air rises, cools, vapor condenses into clouds, and heavy drops precipitate."]),
 ]),
 ("everyday", [
  ("Как приготовить чай?", ["Вскипятите свежую воду, прогрейте чайник, засыпьте чай, залейте водой нужной температуры и дайте настояться 3-5 минут.",
                              "Чёрный чай — около 95°C, зелёный — 75-80°C. Не передерживайте, иначе будет горчить."]),
  ("How to brew tea?", ["Boil fresh water, warm the pot, add tea leaves, pour water at the right temperature and steep 3-5 minutes.",
                          "Black tea likes near-boiling water; green tea prefers 75-80°C."]),
  ("Посоветуй книгу.", ["Если любите фантастику — попробуйте что-то из классики жанра; для саморазвития — книги о привычках и мышлении. Какой жанр ближе?",
                         "Скажите жанр: могу посоветовать художественное или нон-фикшн под ваш вкус."]),
  ("Recommend a book.", ["Tell me a genre you enjoy and I'll suggest something. For habits and thinking, popular science non-fiction is a good start.",
                           "What mood are you after — adventure, mystery, or learning something new?"]),
  ("Как выучить английский?", ["Учите понемногу каждый день: слова в контексте, аудирование, говорение. Читайте простые тексты и повторяйте фразы вслух.",
                                 "База: 1000 частых слов + простые времена + ежедневная практика. Через месяц регулярных занятий прогресс заметен."]),
  ("How to learn English?", ["Learn a little daily: words in context, listening, speaking. Read simple texts and repeat phrases aloud.",
                               "Master the most common words and basic tenses first, then practice consistently."]),
  ("Что приготовить на ужин?", ["Быстрый вариант: паста с овощами и сыром — 20 минут. Или омлет с овощами и зеленью.",
                                 "Гречка с овощами и яйцом — просто и сытно. Есть ли ограничения в еде?"]),
  ("What should I cook for dinner?", ["A quick option: pasta with vegetables and cheese, about 20 minutes. Or an omelette with greens.",
                                        "Rice with stir-fried vegetables and egg is simple and filling."]),
  ("Как справиться со стрессом?", ["Помогают сон, прогулка, дыхательные упражнения и разбиение большой задачи на маленькие шаги.",
                                    "Попробуйте: 10 минут ходьбы, 4-7-8 дыхание, записать worries на бумагу и выбрать один следующий шаг."]),
  ("How to deal with stress?", ["Sleep, walking, breathing exercises and breaking big tasks into small steps all help.",
                                  "Try a short walk, slow breathing, and writing worries down to pick one next step."]),
 ]),
 ("reasoning", [
  ("Если поезд едет 60 км/ч, сколько он проедет за 2.5 часа?", ["Рассуждаем: расстояние = скорость × время = 60 × 2.5 = 150 км. Ответ: 150 километров.",
     "60 км за час, за 2 часа — 120, за полчаса — ещё 30. Итого 150 км."]),
  ("If a train goes 60 km/h, how far in 2.5 hours?", ["Distance = speed × time = 60 × 2.5 = 150 km.",
     "Two hours give 120 km, half an hour adds 30 km: 150 km total."]),
  ("У Ани было 5 яблок, она отдала 2. Сколько осталось?", ["5 − 2 = 3. Осталось три яблока.",
     "Вычитаем: было 5, отдала 2, осталось 3."]),
  ("I have 3 boxes with 4 balls each. How many balls?", ["3 × 4 = 12 balls total.",
     "Each box holds 4, three boxes hold 12."]),
  ("Что тяжелее: килограмм ваты или килограмм железа?", ["Одинаково: и то, и другое весит килограмм. Разный только объём.",
     "Вес равный — по килограмму. Вата просто занимает больше места."]),
 ]),
 ("mixed", [
  ("Что такое machine learning?", ["Machine learning — это когда модель учится на данных: подбирает веса, чтобы предсказывать правильные ответы. Пример — language modeling.",
     "Машинное обучение — обучение алгоритмов на примерах вместо ручных правил. Нейросети — один из его видов."]),
  ("Explain overfitting простыми словами.", ["Overfitting — переобучение: модель запомнила обучающие примеры, но плохо работает на новых. Лечится регуляризацией и большим разнообразным датасетом.",
     "Это когда модель вызубрила ответы вместо закономерностей. Помогают dropout, weight decay и валидация."]),
  ("Привет! What is an embedding?", ["Привет! Embedding — это векторное представление токена: модель превращает ID токена в плотный вектор чисел, с которым уже работает трансформер.",
     "Hello! An embedding maps each token id to a dense vector so the network can process meaning numerically."]),
 ]),
]

PRETRAIN_SENTENCES = [
 "Язык состоит из слов, а слова — из букв и звуков.",
 "Language is built from words, and words from letters and sounds.",
 "Модель предсказывает следующий токен по предыдущим токенам.",
 "The model predicts the next token given the previous tokens.",
 "Трансформер использует механизм внимания для связи слов в предложении.",
 "The transformer uses attention to connect words in a sentence.",
 "Вода кипит при ста градусах Цельсия на уровне моря.",
 "Water boils at one hundred degrees Celsius at sea level.",
 "Земля вращается вокруг Солнца, а Луна — вокруг Земли.",
 "The Earth orbits the Sun, and the Moon orbits the Earth.",
 "Программирование учит разбивать задачи на маленькие шаги.",
 "Programming teaches breaking problems into small steps.",
 "Книги хранят знания и передают их через поколения.",
 "Books store knowledge and pass it across generations.",
 "Утром солнце встаёт на востоке и освещает город.",
 "In the morning the sun rises in the east and lights up the city.",
 "Математика описывает закономерности чисел и форм.",
 "Mathematics describes patterns of numbers and shapes.",
 "Дождь поливает поля, и растения растут быстрее.",
 "Rain waters the fields and plants grow faster.",
 "Компьютер выполняет инструкции процессора очень быстро.",
 "A computer executes processor instructions very quickly.",
 "Нейронная сеть состоит из слоёв и обучаемых весов.",
 "A neural network consists of layers and trainable weights.",
 "Грамматика помогает строить понятные предложения.",
 "Grammar helps build understandable sentences.",
 "Звёзды светят ночью, а днём их затмевает Солнце.",
 "Stars shine at night; during the day the Sun outshines them.",
 "Чтение каждый день расширяет словарный запас.",
 "Reading every day expands vocabulary.",
 "Эксперимент проверяет гипотезу на практике.",
 "An experiment tests a hypothesis in practice.",
]


def _paraphrase_user(rng: random.Random, user: str, lang_ru: bool) -> str:
    """Light polite-form augmentation so repeated questions vary lexically."""
    u = user.strip()
    if lang_ru:
        pref = rng.choice(["", "", "", "Пожалуйста, ", "Подскажи, ", "Скажи, ", "Объясни, пожалуйста: "])
        suff = rng.choice(["", "", "", " Пожалуйста.", " Заранее спасибо!"])
        if pref and u:
            u = pref + u[0].lower() + u[1:]
        u = u + suff
    else:
        pref = rng.choice(["", "", "", "Please ", "Could you tell me: "])
        suff = rng.choice(["", "", "", " please.", " Thanks!"])
        if pref and u:
            u = pref + u[0].lower() + u[1:]
        u = u + suff
    return u.strip()


def _is_ru(text: str) -> bool:
    return any("а" <= c <= "я" or "А" <= c <= "Я" or c in "ёЁ" for c in text)


def build() -> tuple[list[dict], list[str]]:
    rng = random.Random(SEED)
    base: list[tuple[str, list[str]]] = []  # (user, answers)
    for topic, items in POOLS:
        for user, answers in items:
            base.append((user, answers))
    dialogs: list[dict] = []
    # pass 1: every base Q at least twice with different sampled answers
    for user, answers in base:
        for a in rng.sample(answers, k=min(len(answers), 2)):
            dialogs.append({"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": a},
            ]})
    # pass 2: paraphrase upsampling until N_TARGET (varied users, sampled answers)
    guard = 0
    while len(dialogs) < N_TARGET and guard < N_TARGET * 10:
        guard += 1
        user, answers = rng.choice(base)
        u2 = _paraphrase_user(rng, user, _is_ru(user))
        if any(d["messages"][0]["content"] == u2 for d in dialogs[-400:]):
            continue
        dialogs.append({"messages": [
            {"role": "user", "content": u2},
            {"role": "assistant", "content": rng.choice(answers)},
        ]})
    # multi-turn: chain two related dialogs
    rng.shuffle(dialogs)
    multi: list[dict] = []
    for i in range(0, min(160, len(dialogs) - 1), 2):
        a, b = dialogs[i], dialogs[i + 1]
        multi.append({"messages": a["messages"] + b["messages"]})
    dialogs = dialogs + multi
    rng.shuffle(dialogs)
    dialogs = dialogs[:N_TARGET]

    # pretrain corpus: sentences + answers text repeated
    pretrain_lines: list[str] = []
    pretrain_lines += PRETRAIN_SENTENCES * 14
    for d in dialogs:
        for m in d["messages"]:
            pretrain_lines.append(m["content"])
    rng.shuffle(pretrain_lines)
    return dialogs, pretrain_lines


def main():
    out_data = Path("data")
    out_data.mkdir(parents=True, exist_ok=True)
    (out_data / "raw").mkdir(exist_ok=True)
    (out_data / "processed").mkdir(exist_ok=True)
    dialogs, pretrain = build()
    with open(out_data / "conversations.jsonl", "w", encoding="utf-8") as f:
        for d in dialogs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(out_data / "conversations_text.txt", "w", encoding="utf-8") as f:
        for d in dialogs:
            for m in d["messages"]:
                tag = "<USER>" if m["role"] == "user" else "<ASSISTANT>"
                f.write(f"{tag} {m['content']}\n")
            f.write("<EOS>\n")
    with open(out_data / "pretrain_corpus.txt", "w", encoding="utf-8") as f:
        for line in pretrain:
            line = line.strip()
            if line:
                f.write(line + "\n")
    print(f"wrote {len(dialogs)} dialogs, {len(pretrain)} pretrain lines")


if __name__ == "__main__":
    main()
