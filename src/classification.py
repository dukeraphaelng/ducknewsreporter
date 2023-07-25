import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel
from xgboost import XGBClassifier

from dataset import DatasetLoader
from non_latent_features import NonLatentFeatures
from preprocess import Preprocessor
# Sentiment comes from NonLatentFeatures now
# from sentiment import TextBlobSentimentExtractor
from textual_relevance import TextualRelevance


class DataNormalizer:
    def __init__(self, needs_standard_scaler, needs_min_max_scaler):
        self.needs_standard_scaler = needs_standard_scaler
        self.needs_min_max_scaler = needs_min_max_scaler
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    
    def __getstate__(self):
        return (
            self.needs_standard_scaler,
            self.needs_min_max_scaler,
            self.standard_scaler,
            self.min_max_scaler
        )

    def __setstate__(self, tup):
        (
            self.needs_standard_scaler,
            self.needs_min_max_scaler,
            self.standard_scaler,
            self.min_max_scaler
        ) = tup

    def fit_transform(self, data):
        data[:, self.needs_standard_scaler] = self.standard_scaler.fit_transform(data[:, self.needs_standard_scaler])
        data[:, self.needs_min_max_scaler] = self.min_max_scaler.fit_transform(data[:, self.needs_min_max_scaler])
        return data
    
    def transform(self, data):
        data[:, self.needs_standard_scaler] = self.standard_scaler.transform(data[:, self.needs_standard_scaler])
        data[:, self.needs_min_max_scaler] = self.min_max_scaler.transform(data[:, self.needs_min_max_scaler])
        return data


class Pipeline:
    _DIVERSITY_KEYS = ["div_NOUN_sum", "div_NOUN_percent", "div_VERB_sum", "div_VERB_percent", "div_ADJ_sum", "div_ADJ_percent", "div_ADV_sum", "div_ADV_percent", "div_LEX_sum", "div_LEX_percent", "div_CONT_sum", "div_CONT_percent", "div_FUNC_sum", "div_FUNC_percent"]
    _PRONOUN_KEYS = ["pron_FPS_sum", "pron_FPS_percent", "pron_FPP_sum", "pron_FPP_percent", "pron_STP_sum", "pron_STP_percent"]
    _QUANTITY_KEYS = ["quant_NOUN_sum", "quant_NOUN_percent", "quant_VERB_sum", "quant_VERB_percent", "quant_ADJ_sum", "quant_ADJ_percent", "quant_ADV_sum", "quant_ADV_percent", "quant_PRON_sum", "quant_PRON_percent", "quant_DET_sum", "quant_DET_percent", "quant_NUM_sum", "quant_NUM_percent", "quant_PUNCT_sum", "quant_PUNCT_percent", "quant_SYM_sum", "quant_SYM_percent", "quant_PRP_sum", "quant_PRP_percent", "quant_PRP$_sum", "quant_PRP$_percent", "quant_WDT_sum", "quant_WDT_percent", "quant_CD_sum", "quant_CD_percent", "quant_VBD_sum", "quant_VBD_percent", "quant_STOP_sum", "quant_STOP_percent", "quant_LOW_sum", "quant_LOW_percent", "quant_UP_sum", "quant_UP_percent", "quant_NEG_sum", "quant_NEG_percent", "quant_QUOTE_sum", "quant_NP_sum", "quant_CHAR_sum", "quant_WORD_sum", "quant_SENT_sum", "quant_SYLL_sum"]
    _SENTIMENT_KEYS = ["senti_!_sum", "senti_!_percent", "senti_?_sum", "senti_?_percent", "senti_CAPS_sum", "senti_CAPS_percent", "senti_POL_sum", "senti_SUBJ_sum"]
    _AVERAGE_KEYS = ["avg_chars_per_word_sum", "avg_words_per_sent_sum", "avg_claus_per_sent_sum", "avg_puncts_per_sent_sum"]
    _MEDIAN_SYNTAX_TREE_KEYS = ["med_st_ALL_sum", "med_st_NP_sum"]
    _READABILITY_KEYS = ["read_gunning-fog_sum", "read_coleman-liau_sum", "read_dale-chall_sum", "read_flesch-kincaid_sum", "read_linsear-write_sum", "read_spache_sum", "read_automatic_sum", "read_flesch_sum"]

    @dataclass
    class NonLatentConfig:
        diversity: bool = True
        pronoun: bool = True
        quantity: bool = True
        sentiment: bool = True
        average: bool = True
        median_syntax_tree: bool = True
        readability: bool = True

        def build_keys(self):
            keys = []
            if self.diversity:
                keys.extend(Pipeline._DIVERSITY_KEYS)
            if self.pronoun:
                keys.extend(Pipeline._PRONOUN_KEYS)
            if self.quantity:
                keys.extend(Pipeline._QUANTITY_KEYS)
            if self.sentiment:
                keys.extend(Pipeline._SENTIMENT_KEYS)
            if self.average:
                keys.extend(Pipeline._AVERAGE_KEYS)
            if self.median_syntax_tree:
                keys.extend(Pipeline._MEDIAN_SYNTAX_TREE_KEYS)
            if self.readability:
                keys.extend(Pipeline._READABILITY_KEYS)
            return keys

        def build_drop_keys(self):
            keys = []
            if not self.diversity:
                keys.extend(Pipeline._DIVERSITY_KEYS)
            if not self.pronoun:
                keys.extend(Pipeline._PRONOUN_KEYS)
            if not self.quantity:
                keys.extend(Pipeline._QUANTITY_KEYS)
            if not self.sentiment:
                keys.extend(Pipeline._SENTIMENT_KEYS)
            if not self.average:
                keys.extend(Pipeline._AVERAGE_KEYS)
            if not self.median_syntax_tree:
                keys.extend(Pipeline._MEDIAN_SYNTAX_TREE_KEYS)
            if not self.readability:
                keys.extend(Pipeline._READABILITY_KEYS)
            return keys

    def __init__(self, similarity=True, non_latent: Optional[NonLatentConfig]=NonLatentConfig()):
        self.similarity = similarity
        self.non_latent = non_latent
    
    def load_dataset_from_file(self, dir = "data/Horne2017_FakeNewsData/Buzzfeed"):
        base = Path(dir)
        sets = {}
        for name in ["train", "valid", "test"]:
            df = pd.read_csv(base.joinpath(f"features_{name}.csv"))
            y = df["label"].to_numpy()
            labels_to_drop = ["label"]
            if not self.similarity:
                labels_to_drop.append("tf_idf_1_2_harmonic_mean")
            if self.non_latent:
                labels_to_drop.extend(self.non_latent.build_drop_keys())
            else:
                # Delete all keys
                labels_to_drop.extend(Pipeline.NonLatentConfig().build_keys())
            X = df.drop(labels_to_drop, axis=1).to_numpy()
            sets[name] = (X, y)
        X_train, y_train = sets["train"]
        X_valid, y_valid = sets["valid"]
        X_test, y_test = sets["test"]
        return (X_train, X_valid, X_test, y_train, y_valid, y_test)

    def load_dataset(self, random_state=42, quiet=False, save_dir: Optional[str]=None):
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
        logging.basicConfig(format="\x1b[1;36m%(asctime)s: %(message)s\u001b[0m", level=logging.INFO, datefmt="%Y-%m-%dT%H:%M:%SZ")
        logging.getLogger().disabled = quiet

        # Load dataset
        df = DatasetLoader().load_horne2017_fakenewsdata(drop_if_less_than_num_contexts=1).as_pandas()

        # Non latent features (requires limited preprocessing)
        if self.non_latent:
            logging.info("Non latent feature extraction...")
            preprocessor_ascii = Preprocessor(
                lowercase=False,
                remove_non_ascii=True,
                remove_punctuation=False,
                lemmatization=False,
                remove_stopwords=False
            )
            def extract_non_latent(row):
                content = preprocessor_ascii.preprocess(row["content"])
                non_latent_dict = NonLatentFeatures(content).output_all()
                return pd.Series(non_latent_dict.values(), non_latent_dict.keys())
            df = pd.concat([df, df.apply(extract_non_latent, axis=1)], axis=1)
            logging.info("Non latent feature extraction...done")
        else:
            logging.info("Non latent feature extraction...skipped")

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
        feature_labels = [f"bert_{n}" for n in range(len(df["content_bert"][0]))]
        X = np.vstack(df["content_bert"])
        if self.similarity:
            feature_labels.append("tf_idf_1_2_harmonic_mean")
            X = np.hstack((X, df["tf_idf_1_2_harmonic_mean"].to_numpy().reshape((-1, 1))))
        if self.non_latent:
            non_latent_keys = self.non_latent.build_keys()
            feature_labels.extend(non_latent_keys)
            X = np.hstack((X, df[non_latent_keys].to_numpy()))
        y = df["label"].apply(int).to_numpy()

        # Datasets: 60, 20, 20 split on train, valid, test
        X1, X_test, y1, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
        X_train, X_valid, y_train, y_valid = train_test_split(X1, y1, test_size=0.25, random_state=random_state, stratify=y1)

        # Scale features
        needs_standard_scaler = []
        needs_min_max_scaler = []
        for i, label in enumerate(feature_labels):
            if label.endswith("_sum"):
                needs_standard_scaler.append(i)
            else:
                needs_min_max_scaler.append(i)
        
        normalizer = DataNormalizer(needs_standard_scaler, needs_min_max_scaler)
        X_train = normalizer.fit_transform(X_train)
        X_valid = normalizer.transform(X_valid)
        X_test = normalizer.transform(X_test)

        if save_dir:
            base = Path(save_dir)
            labels = ["label", *feature_labels]
            for (name, XX, yy) in [("train", X_train, y_train), ("valid", X_valid, y_valid), ("test", X_test, y_test)]:
                df_out = pd.DataFrame(np.hstack((yy.reshape((-1, 1)), XX)), columns=labels)
                df_out.to_csv(base.joinpath(f"features_{name}.csv"), index=False)
            with open(base.joinpath("scaler.pickle"), "wb") as f:
                pickle.dump(normalizer, f)

        return (X_train, X_valid, X_test, y_train, y_valid, y_test)


class MachineLearningClassifier:
    def __init__(self, random_state=42):
        self.logistic_regression = LogisticRegression(max_iter=1000, random_state=random_state)
        self.svc = SVC(random_state=random_state)
        self.decision_tree = DecisionTreeClassifier(random_state=random_state)
        self.xgboost = XGBClassifier(objective="binary:logistic", random_state=random_state)

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
                self.logistic_regression,
                {"C": [0.8, 1, 1.2], "solver": ["lbfgs", "liblinear"]},
                n_jobs=-1
            ),
            GridSearchCV(
                self.svc,
                {"C": [0.8, 1, 1.2], "kernel": ["linear", "poly", "rbf", "sigmoid"]},
                n_jobs=-1
            ),
            GridSearchCV(
                self.decision_tree,
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
                self.xgboost,
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