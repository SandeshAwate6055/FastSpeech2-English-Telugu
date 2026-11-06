""" from https://github.com/keithito/tacotron """

import re

try:
    import inflect
    _inflect = inflect.engine()
except ImportError:
    _inflect = None

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _fallback_num_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _fallback_num_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rem = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rem == 0 else " " + _fallback_num_to_words(rem))
    if n < 1000000:
        rem = n % 1000
        return _fallback_num_to_words(n // 1000) + " thousand" + ("" if rem == 0 else " " + _fallback_num_to_words(rem))
    return " ".join(_ONES[int(d)] for d in str(n))


_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_time_re = re.compile(r"\b([0-9]{1,2}):([0-9]{2})\s*(am|pm|a\.m\.|p\.m\.)?\b", re.IGNORECASE)
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_time(m):
    hh = int(m.group(1))
    mm = int(m.group(2))
    meridiem = m.group(3)
    hh_w = _fallback_num_to_words(hh)
    if mm == 0:
        mm_w = ""
    elif mm < 10:
        mm_w = " oh " + _fallback_num_to_words(mm)
    else:
        mm_w = " " + _fallback_num_to_words(mm)
    mer_w = ""
    if meridiem:
        clean_m = meridiem.replace(".", "").lower()
        mer_w = " " + " ".join(list(clean_m))
    return f"{hh_w}{mm_w}{mer_w}"


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m):
    if _inflect is not None:
        return _inflect.number_to_words(m.group(0))
    num = int(re.sub(r"(st|nd|rd|th)$", "", m.group(0)))
    return _fallback_num_to_words(num)


def _expand_number(m):
    num = int(m.group(0))
    if _inflect is None:
        if 2010 <= num < 2100:
            return _fallback_num_to_words(num // 100) + " " + _fallback_num_to_words(num % 100)
        return _fallback_num_to_words(num)
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(
                num, andword="", zero="oh", group=2
            ).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_time_re, _expand_time, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
