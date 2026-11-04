"""
Defines the set of symbols used in text input to the Telugu FastSpeech 2 model.
Includes full Telugu Unicode block (U+0C00 to U+0C7F) + silence tokens + ARPAbet fallback.
"""

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Complete Telugu Unicode characters (U+0C00 to U+0C7F):
# Includes Anusvara, Visarga, Independent Vowels (Achulu), Consonants (Hallulu),
# Dependent Vowel Signs (Gunintalu), Virama/Halant (U+0C4D), and Telugu digits/signs.
_telugu_chars = [chr(cp) for cp in range(0x0C00, 0x0C80)]
_telugu_phones = ["@" + ch for ch in _telugu_chars]

_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _telugu_chars
    + _telugu_phones
    + _arpabet
    + _pinyin
    + _silences
)
