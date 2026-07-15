"""Synthetic domain corpus of Russian image-generation prompts.

Real open corpora of *Russian* prompts are scarce (prompt datasets are almost
all English; the HF community translated only ~500). So we generate a domain
corpus from templates using **grammatically pre-agreed fragments**, so the
agreement model is never trained on wrong Russian:

  * subjects are built as `adjective(agreed) + noun` — the adjective is inflected
    to the noun's gender/number (via `adj_nominatives`), teaching prompt-style
    adjective→noun agreement over domain vocabulary;
  * settings / styles / lighting / quality / camera fragments are hand-written
    correct phrases (mostly prepositional, so their case is fixed).

The result is realistic prompt running text. It is fed to the context model
(domain agreement + phrasing) and its token counts are merged into the
vocabulary (domain-appropriate frequencies). Deterministic given the seed.

    python -m src.prompt_corpus         # preview a few generated prompts
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from .domain_lexicon import adj_nominatives

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "prompt_corpus.txt"

# nouns by gender/number (nominative). Gender must be correct — agreement depends on it.
NOUNS = {
    "m": ["портрет", "воин", "замок", "лес", "корабль", "город", "дракон", "робот",
          "киборг", "маяк", "рыцарь", "тигр", "волк", "орёл", "конь", "мост", "храм",
          "дворец", "пейзаж", "закат", "океан", "водопад", "сад", "особняк", "самурай",
          "автомобиль", "космонавт", "старик", "ангел", "демон", "единорог"],
    "f": ["девушка", "женщина", "принцесса", "кошка", "птица", "роза", "река", "улица",
          "башня", "крепость", "деревня", "планета", "галактика", "звезда", "луна",
          "фея", "ведьма", "воительница", "балерина", "долина", "пещера", "вершина",
          "пустыня", "комната", "маска", "корона"],
    "n": ["небо", "море", "облако", "дерево", "поле", "озеро", "здание", "существо",
          "чудовище", "солнце", "лицо", "окно", "зеркало", "пламя", "оружие", "кольцо"],
    "pl": ["горы", "облака", "звёзды", "руины", "цветы", "деревья", "волны", "крылья",
           "доспехи", "огни", "здания", "птицы", "воины", "корабли"],
}

ADJECTIVES = [
    "детализированный", "реалистичный", "красивый", "молодой", "старый", "древний",
    "туманный", "тёмный", "яркий", "мрачный", "величественный", "таинственный",
    "волшебный", "футуристический", "неоновый", "золотой", "серебряный", "огненный",
    "ледяной", "могучий", "изящный", "суровый", "роскошный", "старинный", "одинокий",
    "гигантский", "прекрасный", "загадочный", "мифический", "эпический",
]

# hand-written correct phrases (fixed case) --------------------------------
SETTINGS = ["в тёмном лесу", "на закате", "в неоновом свете", "среди гор",
            "у старого моста", "в густом тумане", "на фоне заката", "под звёздным небом",
            "в глубоком космосе", "на городской улице", "в заброшенном замке",
            "на вершине горы", "в цветущем саду", "у морского берега", "в ночном городе",
            "посреди пустыни", "в снежных горах", "на фоне полной луны"]
STYLES = ["масляная живопись", "цифровое искусство", "акварельный рисунок", "концепт-арт",
          "в стиле киберпанк", "в стиле фэнтези", "аниме стиль", "пиксель-арт",
          "3d рендер", "фотореализм", "стиль барокко", "минимализм", "сюрреализм",
          "в стиле стимпанк", "японская гравюра"]
LIGHTING = ["мягкое освещение", "драматический свет", "золотой час", "объёмный свет",
            "неоновая подсветка", "рассеянный свет", "контровой свет", "кинематографичный свет"]
QUALITY = ["высокая детализация", "8k разрешение", "реалистичные тени", "чёткий фокус",
           "профессиональная фотография", "гиперреализм", "трендовый на artstation",
           "невероятная детализация", "резкие детали", "студийное качество"]
CAMERA = ["широкоугольный объектив", "макросъёмка", "вид сверху", "крупный план",
          "портретная съёмка", "динамичный ракурс", "вид снизу"]


def _subject(rng: random.Random) -> str:
    gender = rng.choices(["m", "f", "n", "pl"], weights=[4, 4, 2, 2])[0]
    noun = rng.choice(NOUNS[gender])
    k = rng.choices([1, 2], weights=[3, 2])[0]     # 1-2 agreeing adjectives
    adjs = rng.sample(ADJECTIVES, k)
    forms = [adj_nominatives(a)[gender] for a in adjs]
    return " ".join(forms + [noun])


def generate(n: int = 80_000, seed: int = 3) -> list[str]:
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        parts = [_subject(rng)]
        for pool, p in ((SETTINGS, 0.6), (STYLES, 0.5), (LIGHTING, 0.4),
                        (QUALITY, 0.5), (CAMERA, 0.3)):
            if rng.random() < p:
                parts.append(rng.choice(pool))
        prompts.append(", ".join(parts))
    return prompts


def write(n: int = 80_000, seed: int = 3, out: Path = OUT) -> int:
    prompts = generate(n, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    return len(prompts)


def sentences(path: Path = OUT):
    """Yield token lists (for the context model), like tatoeba_sentences."""
    import re
    word = re.compile(r"[а-яё]+(?:-[а-яё]+)*")
    if not path.exists():
        write()
    with path.open(encoding="utf-8") as f:
        for line in f:
            toks = word.findall(line.lower())
            if toks:
                yield toks


def word_counts(path: Path = OUT) -> Counter:
    """Token frequencies in the corpus (to merge into the vocabulary)."""
    c: Counter = Counter()
    for toks in sentences(path):
        c.update(toks)
    return c


if __name__ == "__main__":
    for p in generate(12):
        print(" •", p)
