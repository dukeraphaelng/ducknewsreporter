from typing import Any, List, Tuple

from gensim import corpora
from gensim.models import LdaModel

from preprocess import Preprocessor


class TextRankExtractor:
    def __init__(self):
        # Punctuation is removed since we don't want punctuation to be a possible keyword.
        self.preprocessor = Preprocessor(remove_punctuation=True)
    
    def extract_main_themes(self, content: str,
                            num_words=10,
                            num_topics=1,
                            ignore_numerical_keywords=True) -> Tuple[List[Any], str]:
        """Extract the main themes out of the content in a news article.

        Args:
            content (str): The news article
            num_words (int, optional): The number of keywords to consider per topic. Defaults to 10.
            num_topics (int, optional): The number of topics to consider. Defaults to 1.
            ignore_numerical_keywords (bool, optional): Whether to return
            keywords which are just numbers. Keywords which are just dates or
            money values can sometimes not be useful. Defaults to False.

        Returns:
            Tuple[List[Any], str]: A tuple consisting of the main themes
            extracted from the content and the string summary of length
            num_words of the topics.
        """
        documents = list(map(self.preprocessor.preprocess_and_tokenize, content.split("\n")))
        dictionary = corpora.Dictionary(documents)

        corpus = list(map(dictionary.doc2bow, documents))

        lda_model = LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=20,
            # Reproducible
            random_state=42
        )

        main_themes = lda_model.show_topics(
            num_topics=num_topics,
            num_words=num_words,
            formatted=False
        )

        all_keywords: List[Tuple[str, float]] = []
        for topic_id, words in main_themes:
            all_keywords.extend(words)
        
        # Sort by decreasing probability
        all_keywords.sort(key=lambda p: p[1], reverse=True)

        # Deduplicate keywords from all topics and select by probability. Only take
        # num_words amount of words.
        keywords = []
        for word, proba in all_keywords:
            if word in keywords:
                # Deduplication
                continue
            if ignore_numerical_keywords and word.isnumeric():
                continue
            keywords.append(word)
            if len(keywords) >= num_words:
                break

        return main_themes, " ".join(keywords)


if __name__ == "__main__":
    import sys
    print("Text to extract context from: ", end="", flush=True)
    content = sys.stdin.read()
    main_themes, summary_string = TextRankExtractor().extract_main_themes(content)
    print(main_themes)
    print(summary_string)
