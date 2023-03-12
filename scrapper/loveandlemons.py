import requests
import argparse

from typing import List, Dict
from warnings import warn

import pandas as pd

from bs4 import BeautifulSoup
from tqdm import tqdm


def trim_str(string):
    return f"{string}".replace("\n", "")


def get_div_texts_from_url(url: str) -> List[str]:
    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")
    divs = soup.find_all("div")
    texts = [div.text.strip() for div in divs]

    return [text for text in texts if text != ""]


def label_indicators(df: pd.DataFrame, on: str, values: Dict):
    for name, value in values.items():
        df[name] = df[on] == value


def extract_page(url: str, ingredients: str, instructions: str) -> pd.DataFrame:
    texts = get_div_texts_from_url(url)
    df = pd.DataFrame({"text": texts})

    values = {
        "ingredients": ingredients,
        "instructions": instructions,
    }

    label_indicators(df, on="text", values=values)

    return df


def validate_page_df(df: pd.DataFrame, url=None, columns=("ingredients", "instructions")) -> None:
    missing = [col for col in columns if df[col].sum == 0]

    for missing_col in missing:
        warning = f"Could not find a paragraph containing the given {missing_col}. "
        if url:
            warning += f"URL: {url}"
        warn(warning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='LoveAndLemons scrapper',
        description='Scraps pages from LoveAndLemons.com according to the provided csv. '
                    'For each paragraph, it indicates if it contains the recipe or the ingredients themselves.',
        )

    parser.add_argument('--input_csv', default="../loaveandlemons_dataset.csv", help="The provided labeled csv")
    parser.add_argument('--output_csv', default="../loaveandlemons_raw.csv", help="Destination to save the output")

    args = parser.parse_args()

    targets = pd.read_csv(args.input_csv)

    targets["ingredients"] = targets["ingredients"].apply(trim_str)
    targets["instructions"] = targets["instructions"].apply(trim_str)

    def extract_row(row):
        df = extract_page(row["url"], row["ingredients"], row["instructions"])
        validate_page_df(df, row["url"])

        return df

    tqdm.pandas(desc="Scrapping webpages!")
    extracted = targets.progress_apply(extract_row, axis=1)
    df = pd.concat(list(extracted))

    df.to_csv(args.output_csv)
