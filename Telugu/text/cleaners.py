import re
import unicodedata
from .numbers import normalize_numbers

try:
    from unidecode import unidecode
except ImportError:
    def unidecode(text):
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

_whitespace_re = re.compile(r"\s+")

_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

# Telugu spoken number lookup for dates, times, ordinals, and 0-99
_TE_ONES = {
    0: "సున్నా",
    1: "ఒకటి",
    2: "రెండు",
    3: "మూడు",
    4: "నాలుగు",
    5: "ఐదు",
    6: "ఆరు",
    7: "ఏడు",
    8: "ఎనిమిది",
    9: "తొమ్మిది",
    10: "పది",
    11: "పదకొండు",
    12: "పన్నెండు",
    13: "పదమూడు",
    14: "పద్నాలుగు",
    15: "పదిహేను",
    16: "పదహారు",
    17: "పదిహేడు",
    18: "పద్దెనిమిది",
    19: "పందొమ్మిది",
    20: "ఇరవై",
    30: "ముప్పై",
    40: "నలభై",
    50: "యాభై",
    60: "అరవై",
    70: "డెబ్బై",
    80: "ఎనభై",
    90: "తొంభై",
}

_TE_TENS_PREFIX = {
    20: "ఇరవై",
    30: "ముప్పై",
    40: "నలభై",
    50: "యాభై",
    60: "అరవై",
    70: "డెబ్బై",
    80: "ఎనభై",
    90: "తొంభై",
}

_TE_HUNDREDS = {
    1: "వంద",
    2: "రెండు వందల",
    3: "మూడు వందల",
    4: "నాలుగు వందల",
    5: "ఐదు వందల",
    6: "ఆరు వందల",
    7: "ఏడు వందల",
    8: "ఎనిమిది వందల",
    9: "తొమ్మిది వందల",
}

_TE_THOUSANDS = {
    1: "వెయ్యి",
    2: "రెండు వేల",
    3: "మూడు వేల",
    4: "నాలుగు వేల",
    5: "ఐదు వేల",
    6: "ఆరు వేల",
    7: "ఏడు వేల",
    8: "ఎనిమిది వేల",
    9: "తొమ్మిది వేల",
}

# Mapping from Telugu dependent vowel signs (Gunintalu) to canonical independent vowel phonemes (Achulu)
_MATRA_TO_VOWEL = {
    "\u0C3E": "ఆ",  # ా -> ఆ (aa)
    "\u0C3F": "ఇ",  # ి -> ఇ (i)
    "\u0C40": "ఈ",  # ీ -> ఈ (ii)
    "\u0C41": "ఉ",  # ు -> ఉ (u)
    "\u0C42": "ఊ",  # ూ -> ఊ (uu)
    "\u0C43": "ఋ",  # ృ -> ఋ (r)
    "\u0C44": "ౠ",  # ౄ -> ౠ (rr)
    "\u0C46": "ఎ",  # ె -> ఎ (e)
    "\u0C47": "ఏ",  # ే -> ఏ (ee)
    "\u0C48": "ఐ",  # ై -> ఐ (ai)
    "\u0C4A": "ఒ",  # ొ -> ఒ (o)
    "\u0C4B": "ఓ",  # ో -> ఓ (oo)
    "\u0C4C": "ఔ",  # ౌ -> ఔ (au)
}

_VIRAMA = "\u0C4D"  # ్ (suppresses inherent vowel 'అ')
_INHERENT_VOWEL = "అ"


def is_telugu_consonant(ch: str) -> bool:
    return "\u0C15" <= ch <= "\u0C39" or ch in ("\u0C58", "\u0C59")


def is_telugu_independent_vowel(ch: str) -> bool:
    return "\u0C05" <= ch <= "\u0C14" or ch in ("\u0C60", "\u0C61")


def is_telugu_modifier(ch: str) -> bool:
    # Anusvara (ం), Visarga (ః), Chandrabindu (ઁ)
    return ch in ("\u0C01", "\u0C02", "\u0C03")


def telugu_word_to_phones(word: str) -> list:
    """
    Converts a normalized Telugu word into a sequence of acoustic phonemes:
    - Consonants (Hallulu) emit their base consonant phone plus inherent vowel 'అ'
      UNLESS followed by a dependent vowel sign (Gunintam) or Virama ('్').
    - Dependent vowel signs emit the corresponding canonical vowel phone (Achulu).
    - Virama ('్') suppresses the inherent vowel (handling consonant clusters & geminates like 'ట్టు').
    - Anusvara ('ం') and Visarga ('ః') are preserved as distinct coda phones.
    """
    phones = []
    chars = [c for c in word if "\u0C00" <= c <= "\u0C7F"]
    n = len(chars)
    i = 0
    while i < n:
        ch = chars[i]
        if is_telugu_independent_vowel(ch):
            phones.append(ch)
            i += 1
        elif is_telugu_consonant(ch):
            phones.append(ch)
            if i + 1 < n and chars[i + 1] == _VIRAMA:
                # Virama suppresses inherent vowel
                i += 2
            elif i + 1 < n and chars[i + 1] in _MATRA_TO_VOWEL:
                phones.append(_MATRA_TO_VOWEL[chars[i + 1]])
                i += 2
            else:
                # Inherent vowel 'అ'
                phones.append(_INHERENT_VOWEL)
                i += 1
        elif ch in _MATRA_TO_VOWEL:
            phones.append(_MATRA_TO_VOWEL[ch])
            i += 1
        elif ch == _VIRAMA:
            i += 1
        elif is_telugu_modifier(ch):
            phones.append(ch)
            i += 1
        else:
            phones.append(ch)
            i += 1
    return phones


def telugu_text_to_phones(text: str) -> list:
    """
    Converts a Telugu sentence into a list of phoneme tokens with 'sp' at punctuation/clause breaks.
    """
    cleaned = telugu_cleaners(text)
    tokens = re.findall(r"[\u0C00-\u0C7F]+|[,.;?!]", cleaned)
    phones = []
    for tok in tokens:
        if tok in ",.;?!":
            if phones and phones[-1] != "sp":
                phones.append("sp")
        else:
            w_phones = telugu_word_to_phones(tok)
            if w_phones:
                phones.extend(w_phones)
    if phones and phones[-1] == "sp":
        phones.pop()
    return phones


def int_to_telugu(n: int) -> str:
    if n < 0:
        return "మైనస్ " + int_to_telugu(-n)
    if n in _TE_ONES:
        return _TE_ONES[n]
    if n < 100:
        tens = (n // 10) * 10
        rem = n % 10
        return f"{_TE_TENS_PREFIX[tens]} {_TE_ONES[rem]}"
    if n < 1000:
        h = n // 100
        rem = n % 100
        if rem == 0:
            return "వంద" if h == 1 else _TE_HUNDREDS[h].replace("వందల", "వందలు")
        h_word = "నూట" if h == 1 else _TE_HUNDREDS[h]
        return f"{h_word} {int_to_telugu(rem)}"
    if n < 10000:
        th = n // 1000
        rem = n % 1000
        if rem == 0:
            return "వెయ్యి" if th == 1 else _TE_THOUSANDS[th].replace("వేల", "వేలు")
        return f"{_TE_THOUSANDS[th]} {int_to_telugu(rem)}"
    return " ".join(_TE_ONES[int(d)] for d in str(n))


def expand_telugu_numbers(text: str) -> str:
    for i, te_d in enumerate("౦౧౨౩౪౫౬౭౮౯"):
        text = text.replace(te_d, str(i))

    def _replace_time(m):
        hh = int(m.group(1))
        mm = int(m.group(2))
        if mm == 0:
            return int_to_telugu(hh)
        return f"{int_to_telugu(hh)} {int_to_telugu(mm)}"

    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", _replace_time, text)

    def _replace_ordinal(m):
        num = int(m.group(1))
        word = int_to_telugu(num)
        if word.endswith("డు"):
            return word[:-2] + "డవ"
        if word.endswith("టి"):
            return "మొదటి"
        return word + "వ"

    text = re.sub(r"(\d+)వ\b", _replace_ordinal, text)
    text = re.sub(r"\b\d+\b", lambda m: int_to_telugu(int(m.group(0))), text)
    return text


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def telugu_cleaners(text):
    """Pipeline for Telugu Unicode text, including NFC normalization and Telugu number/time expansion."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = expand_telugu_numbers(text)
    text = collapse_whitespace(text)
    return text.strip()
