import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel
from xgboost import XGBClassifier

from dataset import DatasetLoader
from preprocess import Preprocessor
from sentiment import TextBlobSentimentExtractor
from textual_relevance import TextualRelevance


class Pipeline:
    def __init__(self, similarity=True, sentiment=False):
        self.similarity = similarity
        self.sentiment = sentiment
    
    def load_dataset_from_file(self, directory: str):
        base = Path(directory)
        X = np.fromfile(base.joinpath("X.numpy"))
        y = np.fromfile(base.joinpath("y.numpy"))
        return (X, y)

    def load_dataset(self, quiet=False, save: Optional[str]=None):
        """Performs the following work in order:
         - Load the dataset and join context
         - Extracts sentiment features
         - Preprocesses and tokenizes the content
         - Extracts BERT features
         - Extracts similarity features
         - Concatenates all above features to a numpy array
         - Scales all features so that they are between 0 and 1

        Returns:
            (X: np.ndarray, y: np.ndarray): A numpy X and y with all the features concatenated
        """
        logging.basicConfig(format='\x1b[1;36m%(asctime)s: %(message)s\u001b[0m', level=logging.INFO, datefmt='%Y-%m-%dT%H:%M:%SZ')
        logging.getLogger().disabled = quiet

        # Load dataset
        df = DatasetLoader().load_horne2017_fakenewsdata(drop_if_less_than_num_contexts=1).as_pandas()

        # Sentiment features (handles unprocessed text)
        if self.sentiment:
            logging.info("Sentiment extraction...")
            sentiment = TextBlobSentimentExtractor()
            def extract_sentiment(row):
                s_content = sentiment.extract_sentiment(row["content"])
                delta_subjectivity = []
                delta_polarity = []
                for key in ("ctx1_content", "ctx2_content", "ctx3_content"):
                    s = sentiment.extract_sentiment_opt(row[key])
                    if s.avg_subjectivity is not None:
                        delta_subjectivity.append(s_content.avg_subjectivity - s.avg_subjectivity)
                        delta_polarity.append(s_content.avg_polarity - s.avg_polarity)
                subjectivity = sum(delta_subjectivity) / len(delta_subjectivity)
                polarity = sum(delta_polarity) / len(delta_polarity)
                return pd.Series((subjectivity, polarity), ("subjectivity", "polarity"))
            df = pd.concat([df, df.apply(extract_sentiment, axis=1)], axis=1)
            logging.info("Sentiment extraction...done")
        else:
            logging.info("Sentiment extraction...skipped")

        # Preprocess data
        logging.info("Preprocessing and tokenization...")
        preprocessor = Preprocessor()
        for content in ("content", "ctx1_content", "ctx2_content", "ctx3_content"):
            df[content] = df[content].apply(preprocessor.preprocess_and_tokenize_opt)
        logging.info("Preprocessing and tokenization...done")
        
        # BERT Features
        logging.info("BERT feature extraction...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = TFBertModel.from_pretrained("bert-base-uncased")

        df["content_bert_tokens"] = df["content"].apply(lambda art: tokenizer(" ".join(art)[:512], return_tensors="tf"))
        df["content_bert"] = df["content_bert_tokens"].apply(lambda x: model(x).pooler_output.numpy()[0])
        logging.info("BERT feature extraction...done")

        # Similarity features
        if self.similarity:
            logging.info("Similarity comparison...")
            tfidf_1_2 = TextualRelevance("tfidf", df["content"], ngram_range=(1, 2))
            def extract_similarity(row):
                contents = []
                for context in [row["ctx1_content"], row["ctx2_content"], row["ctx3_content"]]:
                    if isinstance(context, list):
                        contents.append(context)
                cosine_dist = tfidf_1_2.cosine_dist(row["content"], contents)
                word_app = tfidf_1_2.word_appearance(row["content"], contents)
                matching = tfidf_1_2.matching_score(row["content"], contents)
                harmonic_mean = 3 / ((1 / cosine_dist) + (1 / word_app) + (1 / matching))
                return harmonic_mean
            df["tf_idf_1_2_harmonic_mean"] = df.apply(extract_similarity, axis=1)
            logging.info("Similarity comparison...done")
        else:
            logging.info("Similarity comparison...skipped")

        # Convert to numpy
        X = np.vstack(df["content_bert"])
        if self.similarity:
            X = np.hstack((X, df["tf_idf_1_2_harmonic_mean"].to_numpy().reshape((-1, 1))))
        if self.sentiment:
            X = np.hstack((X, df["subjectivity"].to_numpy().reshape((-1, 1))))
            X = np.hstack((X, df["polarity"].to_numpy().reshape((-1, 1))))
        y = df["label"].apply(int).to_numpy()

        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 0.99))
        X = scaler.fit_transform(X)

        if save:
            base = Path(save)
            X.tofile(base.joinpath("X.numpy"))
            y.tofile(base.joinpath("y.numpy"))

        return (X, y)


class MachineLearningClassifier:
    def __init__(self, random_state=42):
        self._names = [
            "Logistic Regression",
            "SVC",
            "Decision Tree",
            "KNN",
            "Gaussian NB",
            "Random Forest",
            "AdaBoost",
            "XGBoost",
        ]
        self._classifiers = [
            GridSearchCV(
                LogisticRegression(max_iter=1000, random_state=random_state),
                {"C": [0.8, 1, 1.2], "solver": ["lbfgs", "liblinear"]},
                n_jobs=-1
            ),
            GridSearchCV(
                SVC(random_state=random_state),
                {"C": [0.8, 1, 1.2], "kernel": ["linear", "poly", "rbf", "sigmoid"]},
                n_jobs=-1
            ),
            GridSearchCV(
                DecisionTreeClassifier(random_state=random_state),
                {"criterion": ["gini", "entropy"], "max_depth": [3, 5, 7, 9, None], "max_features": [0.3, "sqrt", 1.0], "min_samples_split": [2, 3, 4]},
                n_jobs=-1
            ),
            GridSearchCV(
                KNeighborsClassifier(),
                {"n_neighbors": [3, 5, 7, 9]},
                n_jobs=-1
            ),
            GridSearchCV(
                GaussianNB(),
                {},
                n_jobs=-1
            ),
            GridSearchCV(
                RandomForestClassifier(random_state=random_state, n_jobs=-1),
                {"max_features": [0.3, "sqrt", 1.0], "n_estimators": [100, 300, 500, 700]},
                n_jobs=-1
            ),
            GridSearchCV(
                AdaBoostClassifier(random_state=random_state),
                {"learning_rate": [0.8, 1.0, 1.2], "n_estimators": [30, 50, 70]},
                n_jobs=-1
            ),
            GridSearchCV(
                XGBClassifier(objective="binary:logistic", random_state=random_state),
                {"eta": [0.2, 0.3, 0.4, 0.5], "max_depth": [2, 4, 6, 8, 10], "lambda": [1, 1.2, 1.5]},
                n_jobs=-1
            ),
        ]
    
    def fit(self, X, y, quiet=False):
        for clf in tqdm(self._classifiers, "Fitting models", disable=quiet):
            clf.fit(X, y)
        self.fitted = True
    
    def best_params(self):
        if not self.fitted:
            raise Exception("Models have not been fitted yet")
        results = {}
        for name, clf in zip(self._names, self._classifiers):
            results[name] = clf.best_params_
        return results

    def predict(self, X) -> Dict[str, Any]:
        if not self.fitted:
            raise Exception("Models have not been fitted yet")
        results = {}
        for name, clf in zip(self._names, self._classifiers):
            result = clf.predict(X)
            results[name] = result
        return results


if __name__ == "__main__":
    pipe = Pipeline()
    print(pipe.load_dataset())