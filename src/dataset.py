import csv
import json
import multiprocessing as mp
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from newspaper import Article, article

from utils import Label


@dataclass
class FakeNewsNetItem:
    id: str
    title: str
    content: str
    published: datetime
    url: str
    label: Label

    ctx1_url: Optional[str] = None
    ctx1_title: Optional[str] = None
    ctx1_content: Optional[str] = None

    ctx2_url: Optional[str] = None
    ctx2_title: Optional[str] = None
    ctx2_content: Optional[str] = None

    ctx3_url: Optional[str] = None
    ctx3_title: Optional[str] = None
    ctx3_content: Optional[str] = None


@dataclass
class FakeNewsNetDataset:
    inner: List[FakeNewsNetItem]

    def as_pandas(self):
        df = pd.DataFrame(
            data=self.inner,
            columns=("id", "title", "content", "published", "url", "label",
                     "ctx1_url", "ctx1_title", "ctx1_content",
                     "ctx2_url", "ctx2_title", "ctx2_content",
                     "ctx3_url", "ctx3_title", "ctx3_content")
        )
        return df.fillna(value=pd.NA)


@dataclass
class ContextItem:
    def download(art: Tuple[str, str, str, str]):
        foreign_id, a1_url, a2_url, a3_url = art
        def do_download(a: str):
            if not a:
                return None
            try:
                art = Article(a)
                art.download()
                if art.download_state == article.ArticleDownloadState.SUCCESS:
                    art.parse()
                    art = ContextItem.Article.from_newspaper(art)
                    return art
            except:
                # Print a logging message??
                pass
            return None
        
        # TODO: These can be downloaded in parallel
        a1 = do_download(a1_url)
        a2 = do_download(a2_url)
        a3 = do_download(a3_url)

        return ContextItem(
            foreign_id,
            a1,
            a2,
            a3,
        )
    
    def as_dict(self):
        return dict(
            foreign_id=self.foreign_id,
            article1=self.article1.as_dict() if self.article1 else None,
            article2=self.article2.as_dict() if self.article2 else None,
            article3=self.article3.as_dict() if self.article3 else None,
        )
    
    def from_dict(d: Dict[str, str]):
        return ContextItem(
            d["foreign_id"],
            # Rest are not rested so we can just use unpacking
            ContextItem.Article(**d["article1"]) if d["article1"] else None,
            ContextItem.Article(**d["article2"]) if d["article2"] else None,
            ContextItem.Article(**d["article3"]) if d["article3"] else None,
        )

    @dataclass
    class Article:
        def from_newspaper(a: Article):
            return ContextItem.Article(
                a.title,
                a.text,
                a.url,
            )
        
        def as_dict(self):
            return dict(
                title=self.title,
                content=self.content,
                url=self.url,
            )

        title: str
        content: str
        url: str

    foreign_id: str
    article1: Optional[Article]
    article2: Optional[Article]
    article3: Optional[Article]


class DatasetLoader:
    def __init__(self, base_path="data"):
        self.base_path = base_path
    
    def download_context_articles(self,
                                  threads: Optional[int]=None,
                                  csv_path="fakenewsnet/politifact/context.csv",
                                  write_path="fakenewsnet/politifact/context"):
        """Downloads the up to 3 articles from a csv that describes a linked
        foreign id and 3 html article urls. WARNING: May take a long time.

        Args:
            threads: (Optional[int], optional): Number of cpu threads to use when downloading. Default to use all available.
            csv_path (str, optional): Path to the csv.
            write_path (str, optional): Path to cache the articles out to.
        """
        manifest: List[Tuple[str, str, str, str]] = []
        with open(Path(self.base_path).joinpath(csv_path)) as f:
            f_csv = csv.reader(f)
            for line in f_csv:
                a1 = line[2]
                a2 = line[3]
                a3 = line[4]
                if not a1 and not a2 and not a3:
                    continue
                # line[1] is the context keywords which we can ignore
                manifest.append((line[0], a1, a2, a3))
        dataset: Dict[str, ContextItem] = {}
        
        base_path = Path(self.base_path).joinpath(write_path)
        os.makedirs(base_path, exist_ok=True)
        with mp.Pool(threads) as pool:
            for it in pool.map(ContextItem.download, manifest):
                dataset[it.foreign_id] = it
                path = base_path.joinpath(f"{it.foreign_id}.json")
                with open(path, "w") as f:
                    json.dump(it.as_dict(), f)
        return dataset


    def load_fakenewsnet(self,
                         path="fakenewsnet/politifact",
                         join_context=True,
                         drop_empty_title=True,
                         drop_empty_text=True,
                         drop_unknown_publish=True):
        base_path = Path(self.base_path).joinpath(path)
        if not base_path.exists():
            raise "Dataset path does not exist"
        dataset = []
        for path, label in [(base_path.joinpath("real"), Label.REAL), (base_path.joinpath("fake"), Label.FAKE)]:
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
                    item = FakeNewsNetItem(
                        id.removesuffix(".json"),
                        f_json["title"],
                        f_json["text"],
                        date,
                        url,
                        label
                    )
                    if (drop_empty_title and not item.title) or (drop_empty_text and not item.content):
                        continue
                    if join_context:
                        ctx_path = base_path.joinpath("context").joinpath(id)
                        if ctx_path.exists():
                            with open(ctx_path) as f_ctx:
                                ctx = ContextItem.from_dict(json.load(f_ctx))
                            if ctx.article1:
                                item.ctx1_url = ctx.article1.url
                                item.ctx1_title = ctx.article1.title
                                item.ctx1_content = ctx.article1.content
                            if ctx.article2:
                                item.ctx2_url = ctx.article2.url
                                item.ctx2_title = ctx.article2.title
                                item.ctx2_content = ctx.article2.content
                            if ctx.article3:
                                item.ctx3_url = ctx.article3.url
                                item.ctx3_title = ctx.article3.title
                                item.ctx3_content = ctx.article3.content
                    dataset.append(item)
        return FakeNewsNetDataset(dataset)


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