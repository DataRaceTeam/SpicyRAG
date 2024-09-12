import re
from typing import List, Optional

gost_pattern = r'ГОСТ.*?\b\d{4}\b'
number_pattern = r'№\s*(\d+(?:-[a-zA-Zа-яА-Я]+)?)\.'
date_pattern = r"\d{2}\.\d{2}\.\d{4}"


def extract_gosts(text: str) -> Optional[List[str]]:
    return re.findall(gost_pattern, text)


def extract_npa_number(text: str) -> str:
    match_number = re.search(number_pattern, text)
    return match_number.group(1) if match_number else None


def extract_date(text: str) -> str:
    match_date = re.search(date_pattern, text)
    return match_date.group() if match_date else None
