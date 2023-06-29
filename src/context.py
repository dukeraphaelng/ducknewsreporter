import re

from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords

# Can only be called once to make it thread-safe
stopwords.ensure_loaded()

class TextRankExtractor:

    def extract_main_themes(self, content: str, num_keywords=10):
        documents = content.split('\n')

        texts = [[token for token in document.lower().split()] for document in documents]

        stop_words = stopwords.words("english")
        texts = [[token for token in text if token not in stop_words] for text in texts]

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        num_topics = 1
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)

        main_themes = lda_model.print_topics(num_topics=num_topics, num_words=10)

        all_themes = []
        for theme_tuple in main_themes:
            _, theme_tuple = theme_tuple
            theme_tuple = re.sub(r"[^a-zA-Z\'\’]", " ", theme_tuple)
            
            for theme in theme_tuple.split(" "):
                if theme:
                    all_themes.append(theme)

        return main_themes, " ".join(all_themes[:num_keywords])


if __name__ == "__main__":
    import sys
    print("Text to extract context: ", end="")
    content = sys.stdin.read()
    main_themes, summary_string = TextRankExtractor().extract_main_themes(content, num_keywords=10)
    print(main_themes)
    print(summary_string)
