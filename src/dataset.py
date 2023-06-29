import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from utils import Label


@dataclass
class DataItem:
    id: str
    title: str
    content: str
    published: datetime
    url: str
    label: Label


@dataclass
class Dataset:
    inner: List[DataItem]

    def as_pandas(self):
        return pd.DataFrame(
            data=self.inner,
            columns=("id", "title", "content", "published", "url", "label")
        )


class DatasetLoader:
    def __init__(self, base_path="data"):
        self.base_path = base_path

    def load_fakenewsnet(self, path="fakenewsnet/politifact", drop_empty_title=True, drop_empty_text=True, drop_unknown_publish=True):
        path = Path(self.base_path).joinpath(path)
        if not path.exists():
            raise "Dataset path does not exist"
        dataset = []
        for path, label in [(path.joinpath("real"), Label.REAL), (path.joinpath("fake"), Label.FAKE)]:
            for id in os.listdir(path):
                with open(path.joinpath(id)) as f:
                    f_json = json.load(f)
                    url = f_json.get("url")
                    dt_exact = f_json.get("article", {}).get("published")
                    if f := f_json.get("publish_date"):
                        dt_date = int(f)
                    else:
                        dt_date = None
                    date = try_parse_datetime(url, dt_exact, dt_date)
                    if not date and drop_unknown_publish:
                        continue
                    item = DataItem(
                        id.removesuffix(".json"),
                        f_json["title"],
                        f_json["text"],
                        date,
                        url,
                        label
                    )
                    if (drop_empty_title and not item.title) or (drop_empty_text and not item.content):
                        continue
                    dataset.append(item)
        return Dataset(dataset)


def try_parse_datetime(url: str, dt_exact: Optional[str], dt_date: Optional[int]) -> Optional[datetime]:
    if dt_exact:
        return datetime.strptime(dt_exact, "%a %b %d %Y %H:%M:%S GMT+0000 (UTC)")
    elif dt_date:
        return datetime.fromtimestamp(dt_date)
    elif mat := re.search(r"https://web.archive.org/web/(\d+)/", url):
        # We have a a lot of web archive urls so try get a date from here
        dt = mat.group(1)
        return datetime.strptime(dt, "%Y%m%d%H%M%S")

    return None


if __name__ == "__main__":
    df = DatasetLoader().load_fakenewsnet(drop_unknown_date=True).as_pandas()
    print(df)
    print()
    print(df["label"].value_counts())