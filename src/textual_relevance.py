import numpy as np
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader

class TextualRelevance():
    def __init__(self, embedding_type, dataset = None, word2vec_type = 'word2vec-google-news-300'):
        """Perform textual relevance calculation

        TextualRelevance calculates the average distance between a
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
        word2vec_type: string (optional, default='word2vec-google-news-300')
            Must be a valid gensim word2vec model. Only required when
            using 'word2vec' as embedding_type

        Returns
        -------
        tr : TextualRelevance()
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
        >>> tfidf = TextualRelevance('tfidf', df.content)
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
            self.vectorizer = self.__get_embedding_tfidf(dataset)
        else:
            # Model must be in list(gensim.downloader.info()['models'].keys())
            self.__word2vec = gensim.downloader.load(word2vec_type)
            self.__word2vec_dim = gensim.downloader.info()['models'][word2vec_type]['parameters']['dimension']
            self.vectorizer = self.__get_embedding_word2vec()
    
    def __get_embedding_tfidf(self, dataset):
        '''Train a sklearn TfidfVectorizer on a tokenised dataset'''        
        # Experiment with different TfidVectorizer parameters here or use literature review results
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
        vectorizer.fit(dataset)

        # https://stackoverflow.com/questions/24440332/numpy-scipy-sparse-matrix-to-vector
        return lambda docs: [mat.A.T.flatten() for mat in vectorizer.transform(docs)]
    
    def __get_embedding_word2vec(self):
        '''Return a lambda function to convert a tokenised dataset into matrices of word2vec'''
        return lambda docs: [self.__get_average_word2vec_vector(doc) for doc in docs]
        
    def __get_average_word2vec_vector(self, words):
        '''Reduce word2vec tokenised text (words) to average vector
        
        Reference
        -----
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
    
    def word_match(self, prediction, contexts):
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
            average value of summed vectors of common words between the
            prediction and each context document (between 0 and inf). 0 means 
            there is no relationship, higher values means more similar words.
        """
        common_words_list = [set(prediction).intersection(set(context)) for context in contexts]
        return np.average(np.sum(self.vectorizer(common_words_list), axis=1))