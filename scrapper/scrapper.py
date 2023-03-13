import requests

from typing import List

from bs4 import BeautifulSoup


def get_div_texts_from_url(url: str) -> List[str]:
    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")
    divs = soup.find_all("div")
    texts = [div.text.strip() for div in divs]

    return [text for text in texts if text != ""]