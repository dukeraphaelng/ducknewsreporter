import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from utils import Label


@dataclass
class DataItem:
    id: str
    title: str
    content: str
    url: str
    label: Label


@dataclass
class Dataset:
    inner: List[DataItem]

    def as_pandas(self):
        return pd.DataFrame(
            data=self.inner,
            columns=("id", "title", "content", "url", "label")
        )


class DatasetLoader:
    def __init__(self, base_path="data"):
        self.base_path = base_path

    def load_fakenewsnet(self, path="fakenewsnet/politifact", drop_empty_title=True, drop_empty_text=True):
        path = Path(self.base_path).joinpath(path)
        if not path.exists():
            raise "Dataset path does not exist"
        dataset = []
        for path, label in [(path.joinpath("real"), Label.REAL), (path.joinpath("fake"), Label.FAKE)]:
            for id in os.listdir(path):
                with open(path.joinpath(id)) as f:
                    f_json = json.load(f)
                    item = DataItem(
                        id.removesuffix(".json"),
                        f_json["title"],
                        f_json["text"],
                        f_json["url"],
                        label
                    )
                    if (drop_empty_title and not item.title) or (drop_empty_text and not item.content):
                        continue
                    dataset.append(item)
        return Dataset(dataset)
        

if __name__ == "__main__":
    df = DatasetLoader().load_fakenewsnet().as_pandas()
    print(df)
    print()
    print(df["label"].value_counts())