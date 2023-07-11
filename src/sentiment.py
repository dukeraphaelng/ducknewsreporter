from dataclasses import dataclass
from typing import Optional

import pandas as pd
from textblob import TextBlob


@dataclass
class Sentiment:
    """Value between -1 and 1 describing whether the document is positive or negative in emotion"""
    avg_polarity: Optional[float] = None
    """Value between 0 and 1 describing whether the document is
    objective/factual* (0) or subjective/opinionated (1).
    
    *Note the "factual" here is a language feature that doesn't tell us whether
    the document is real or fake news yet."""
    avg_subjectivity: Optional[float] = None

    def as_pandas(self, prefix):
        p = self.avg_polarity if self.avg_polarity is not None else pd.NA
        s = self.avg_subjectivity if self.avg_subjectivity is not None else pd.NA
        return pd.Series((p, s),
                         (f"{prefix}_polarity", f"{prefix}_subjectivity"))


class TextBlobSentimentExtractor:
    def __init__(self):
        pass

    def extract_sentiment_opt(self, input: Optional[str]) -> Optional[Sentiment]:
        """Extracts the sentiment using the popular
        [TextBlob](https://textblob.readthedocs.io/) Extractor. Can accept
        optional values such as python None or pandas NaType

        Args:
            input (Optional[str]): The input to extract sentiment from

        Returns:
            Optional[Sentiment]: The sentiment of the text
        """
        if input is None:
            return Sentiment()
        if not isinstance(input, str) and pd.isnull(input):
            return Sentiment()
        return self.extract_sentiment(input)

    def extract_sentiment(self, input: str) -> Sentiment:
        """Extracts the sentiment using the popular [TextBlob](https://textblob.readthedocs.io/) Extractor.

        Args:
            input (str): The input to extract sentiment from

        Returns:
            Sentiment: The sentiment of the text
        """
        blob = TextBlob(input)
        return Sentiment(
            blob.sentiment.polarity,
            blob.sentiment.subjectivity
        )
