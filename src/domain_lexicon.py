"""Domain lexicon for image-generation prompts.

OpenSubtitles under-represents descriptive prompt vocabulary (e.g.
"детализированный" is rank ~260k; "акварельный"/"минималистичный" are absent).
So we inject a curated list of common Russian prompt terms and generate their
inflected forms with a small rule-based adjective inflector (regular hard / soft
/ stressed paradigms). These forms are merged into the vocabulary with a modest
synthetic frequency, so the completer always offers the right endings for prompt
words — and their agreement works via the same suffix statistics.

This is data, not a model: it complements the learned n-gram / context models.
"""
from __future__ import annotations

VELAR = set("кгх")
HUSH = set("жшчщ")

# --- common prompt adjectives (masculine nominative lemma) ----------------
PROMPT_ADJECTIVES = """
детализированный высокодетализированный ультрадетализированный реалистичный
гиперреалистичный фотореалистичный кинематографичный кинематографический
минималистичный абстрактный сюрреалистичный футуристический винтажный неоновый
акварельный масляный пастельный монохромный контрастный объёмный глянцевый
матовый зернистый размытый симметричный геометрический органический готический
барочный психоделический киберпанковый стимпанковый фэнтезийный эпический
драматический атмосферный туманный дождливый солнечный закатный звёздный лунный
огненный серебряный изумрудный бирюзовый малиновый багровый лазурный пышный
изящный величественный старинный роскошный таинственный мистический волшебный
сказочный живописный портретный панорамный широкоугольный студийный
профессиональный яркий мягкий тёплый холодный чёткий гладкий нежный суровый
брутальный элегантный гламурный мрачный светлый тёмная_no древний утренний
вечерний ночной ледяной золотой
""".split()

# --- meta / framing nouns: describe the artefact itself. Very common when
# writing image prompts ("изображение девушки", "на картинке"), but rare in the
# conversational OpenSubtitles corpus, so they need a stronger domain prior than
# ordinary prompt nouns to compete at short prefixes. ------------------------
META_NOUNS = """
изображение картинка иллюстрация рисунок фотография арт сцена кадр вид визуализация
""".split()

# --- common prompt nouns (base form; existing vocab handles most cases) ---
PROMPT_NOUNS = """
портрет пейзаж закат рассвет натюрморт силуэт композиция текстура освещение
тень отражение градиент фон передний_план задний_план ракурс перспектива
детализация детали мазок штрих палитра оттенок блик туман дымка облако облака
горы море океан лес поле река город улица замок башня храм лицо глаза волосы
кожа крылья доспехи корона дракон единорог робот киборг космос галактика
планета звёзды луна солнце неон стекло металл мрамор дерево вода огонь лёд
""".split()

SYNTHETIC_BASE = 500   # oblique/agreeing forms: available, chosen by context
SYNTHETIC_NOM = 1200   # nominative (citation) form: the sensible no-context default
SYNTHETIC_META = 6000  # framing nouns: core prompt vocabulary, must out-rank
                       # rare-in-subtitles homographs at short prefixes


def inflect_adjective(lemma: str) -> set[str]:
    """Regular Russian adjective paradigm (nominal long forms, no short/animacy)."""
    if lemma.endswith("ый"):
        kind, stem = "hard", lemma[:-2]
    elif lemma.endswith("ой"):
        kind, stem = "stressed", lemma[:-2]
    elif lemma.endswith("ий"):
        stem = lemma[:-2]
        kind = "velar" if (stem and stem[-1] in VELAR | HUSH) else "soft"
    else:
        return {lemma}
    if not stem:
        return {lemma}

    forms = {lemma}
    if kind == "soft":  # синий, летний, древний
        for e in ["ий", "его", "ему", "им", "ем", "яя", "ей", "юю",
                  "ее", "ие", "их", "ими"]:
            forms.add(stem + e)
        return forms

    last = stem[-1]
    velar, hush = last in VELAR, last in HUSH
    y = "и" if (velar or hush) else "ы"
    o = "о" if kind == "stressed" else ("е" if hush else "о")  # stressed -ой keeps о
    masc_nom = "ой" if kind == "stressed" else ("ий" if (velar or hush) else "ый")
    forms.update({
        stem + masc_nom,
        stem + o + "го", stem + o + "му", stem + y + "м", stem + o + "м",
        stem + "ая", stem + o + "й", stem + "ую",
        stem + o + "е",
        stem + y + "е", stem + y + "х", stem + y + "ми",
    })
    return forms


def _stem_kind(lemma: str):
    if lemma.endswith("ый"):
        return lemma[:-2], "hard"
    if lemma.endswith("ой"):
        return lemma[:-2], "stressed"
    if lemma.endswith("ий"):
        stem = lemma[:-2]
        return stem, ("velar" if (stem and stem[-1] in VELAR | HUSH) else "soft")
    return None, None


def adj_nominatives(lemma: str) -> dict[str, str]:
    """Nominative forms by gender/number: {m, f, n, pl}. Used to build agreeing
    noun phrases for the prompt corpus."""
    stem, kind = _stem_kind(lemma)
    if not stem:
        return {"m": lemma, "f": lemma, "n": lemma, "pl": lemma}
    if kind == "soft":
        return {"m": stem + "ий", "f": stem + "яя", "n": stem + "ее", "pl": stem + "ие"}
    last = stem[-1]
    velar, hush = last in VELAR, last in HUSH
    y = "и" if (velar or hush) else "ы"
    m = "ой" if kind == "stressed" else ("ий" if (velar or hush) else "ый")
    n = "ее" if hush else "ое"
    return {"m": stem + m, "f": stem + "ая", "n": stem + n, "pl": stem + y + "е"}


def domain_forms() -> dict[str, int]:
    """All prompt word forms -> synthetic count."""
    out: dict[str, int] = {}
    for lemma in PROMPT_ADJECTIVES:
        if lemma.endswith("_no"):
            continue
        for f in inflect_adjective(lemma):
            out[f] = max(out.get(f, 0), SYNTHETIC_BASE)
        out[lemma] = SYNTHETIC_NOM  # masc nominative = default when no context
    for noun in PROMPT_NOUNS:
        if "_" in noun:
            continue
        out[noun] = max(out.get(noun, 0), SYNTHETIC_NOM)
    for noun in META_NOUNS:
        out[noun] = max(out.get(noun, 0), SYNTHETIC_META)
    return out


if __name__ == "__main__":
    d = domain_forms()
    print(f"{len(d)} domain forms")
    for lemma in ["детализированный", "футуристический", "золотой", "древний"]:
        print(lemma, "->", sorted(inflect_adjective(lemma)))
