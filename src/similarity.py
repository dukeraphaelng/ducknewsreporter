from typing import Any, List, Tuple

import gensim.downloader
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import Preprocessor


class SimilarityModel():
    def __init__(self, embedding_type, dataset = None, ngram_range = (1,1), word2vec_type = 'word2vec-google-news-300'):
        """Perform textual relevance calculation

        SimilarityModel calculates the average distance between a
        target document and a list of similar documents. The higher
        the value the more likely are the two documents similar. We
        provide two similarity metrics: cosine distance and word
        matching.
        
        Parameters
        ----------
        embedding_type : string
            Must be 'tfidf' or 'word2vec'
        dataset : array_like (optional, default=None)
            Array of shape (n_samples, n_words). Each sample is a tokenised
            list of words. if embedding_type is 'word2vec', dataset is ignored.
        ngram_range: tuple_like (optional, default=(1,1))
            ngram range of the TF-IDF
        word2vec_type: string (optional, default='word2vec-google-news-300')
            Must be a valid gensim word2vec model. Only required when
            using 'word2vec' as embedding_type

        Returns
        -------
        tr : SimilarityModel()
            A Textual Relevance instance

        Methods
        -------
        consine_dist(prediction, contexts)
            Calculate the average cosine distance of the prediction document
            and all the context documents (min: 0, max: 1)
        word_match(prediction, contexts)
            Calculate the average summed word match of prediction document
            and all the context documents

        Examples
        --------
        >>> pp = Preprocessor()
        >>> ds = DatasetLoader().load_fakenewsnet(drop_if_less_than_num_contexts=3)
        >>> df = ds.as_pandas()
        >>> df["content"] = df["content"].apply(pp.preprocess_and_tokenize)
        >>> tfidf = SimilarityModel('tfidf', df.content)
        >>> tfidf.cosine_dist(df['content'].iloc[0], [
            \ pp.preprocess_and_tokenize(df['ctx2_content'].iloc[0]), 
            \ pp.preprocess_and_tokenize(df['ctx3_content'].iloc[0])])
        0.19219322243197196 
        
        Reference
        -----
        Alsuliman, F., Bhattacharyya, S., Slhoub, K., Nur, N. and Chambers, 
        C.N., 2022, June. Social Media vs. News Platforms: A Cross-analysis for 
        Fake News Detection Using Web Scraping and NLP. In Proceedings of the 
        15th International Conference on PErvasive Technologies Related to 
        Assistive Environments (pp. 190-196).
        """
        if embedding_type not in ['tfidf', 'word2vec']:
            raise ValueError(f'Word embedding type {embedding_type} is invalid. Valid types are tfidf and word2vec')
        self.embedding_type = embedding_type

        if self.embedding_type == 'tfidf':
            self.vectorizer = self.__get_embedding_tfidf(dataset, ngram_range)
        else:
            # Model must be in list(gensim.downloader.info()['models'].keys())
            self.__word2vec = gensim.downloader.load(word2vec_type)
            self.__word2vec_dim = gensim.downloader.info()['models'][word2vec_type]['parameters']['dimension']
            self.vectorizer = self.__get_embedding_word2vec()
    
    def __get_embedding_tfidf(self, dataset, ngram_range):
        '''Train a sklearn TfidfVectorizer on a tokenised dataset'''        
        # Experiment with different TfidVectorizer parameters here or use literature review results
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, ngram_range=ngram_range)
        vectorizer.fit(dataset)

        # https://stackoverflow.com/questions/24440332/numpy-scipy-sparse-matrix-to-vector
        return lambda docs: [mat.A.T.flatten() for mat in vectorizer.transform(docs)]
    
    def __get_embedding_word2vec(self):
        '''Return a lambda function to convert a tokenised dataset into matrices of word2vec'''
        return lambda docs: [self.__get_average_word2vec_vector(doc) for doc in docs]
        
    def __get_average_word2vec_vector(self, words):
        '''Reduce word2vec tokenised text (words) to average vector
        
        Reference
        ---------
        Sitikhu, P., Pahi, K., Thapa, P. and Shakya, S., 2019, November. 
        A comparison of semantic similarity methods for maximum human interpretability. 
        In 2019 artificial intelligence for transforming business and society (AITB) 
        (Vol. 1, pp. 1-4). IEEE.
        https://arxiv.org/pdf/1910.09129v1.pdf 
        '''
        val = np.zeros(self.__word2vec_dim,)
        for word in words:
            try:
                val += self.__word2vec[word]
            except KeyError:
                # Vectors of non-existent words are ignored
                continue
        # To calculate the document vector we take the average of all vectors
        return val / len(words)
    
    def cosine_dist(self, prediction, contexts):
        """
        Parameters
        ----------
        prediction : array_like
            Array of shape (n_words, ). The document to compare against its
            contexts
        contexts : array_like
            Array of shape (n_samples, n_words). Each sample is a tokenised
            list of words. Each sample is a context document of the prediction

        Returns
        -------
        dist : float
            average cosine distance between 0 and 1 of prediction against all
            contexts. 0 means there is no relationship between two vectors, 1
            means two vectors are the same
        """
        predict_vec = self.vectorizer([prediction])[0]
        context_vecs = self.vectorizer(contexts)

        # https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
        return np.mean([1 - spatial.distance.cosine(predict_vec, context_vec) for context_vec in context_vecs])

    def word_appearance(self, prediction, contexts):
        '''Calculate word appearance
        Sum the number of unique prediction words in the context document
        Divided by the total number of unique words in the context doc

        Parameters
        ----------
        prediction : array_like
            Array of shape (n_words, ). The document to compare against its
            contexts
        contexts : array_like
            Array of shape (n_samples, n_words). Each sample is a tokenised
            list of words. Each sample is a context document of the prediction

        Returns
        -------
        dist : float
            average value of word appearance of the prediction doc against each
            context doc. 0 means there is no relationship between two vectors, 1
            means two vectors are the same
        '''
        unique_predict_words = set(prediction)
        out = []
        for context in contexts:
            unique_context_words = set(context)
            common_words = unique_predict_words.intersection(unique_context_words)
            out.append(len(common_words) / len(unique_context_words))

        return np.mean(out)

    def matching_score(self, prediction, contexts):
        '''Calculate the L1 norm vector value for the unique common words in the prediction
        doc and context doc divided by the L1 norm vector value of the unique words in
        context document. Note we can only use TF-IDF vectorizer.

        Parameters
        ----------
        prediction : array_like
            Array of shape (n_words, ). The document to compare against its
            contexts
        contexts : array_like
            Array of shape (n_samples, n_words). Each sample is a tokenised
            list of words. Each sample is a context document of the prediction

        Returns
        -------
        dist : float
            average value of matching score of the prediction doc against each
            context doc. 0 means there is no relationship between two vectors, 1
            means two vectors are the same
        '''
        unique_predict_words = set(prediction)
        out = []
        for context in contexts:
            unique_context_words = set(context)
            common_words = unique_predict_words.intersection(unique_context_words)
            # Assumes TF-IDF vectorizer with vector value >= 0
            out.append(np.sum(self.vectorizer(common_words)) / np.sum(self.vectorizer(unique_context_words)))

        return np.mean(out)


class LDASummaryExtractor:
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
    main_themes, summary_string = LDASummaryExtractor().extract_main_themes(content)
    print(main_themes)
    print(summary_string)
