### Silence all warnings. They aren't related to this pipeline.
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import logging

logging.set_verbosity_error()
###

import pickle
from datetime import datetime

import numpy as np
from newspaper import Article
from transformers import BertTokenizer, TFBertModel

from classification import DataNormalizer, Pipeline
from context import TextRankExtractor
from non_latent_features import NonLatentFeatures
from preprocess import Preprocessor
from textual_relevance import TextualRelevance

# Preload our BERT models so our program doesn't hang for a very long time
# during inference
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")


def info(msg):
    now = datetime.now()
    print(f"\x1b[1;36m{now:%Y-%m-%dT%H:%M:%SZ}: {msg}\u001b[0m")

def get_article(prompt, log_msg):
    url = input(prompt)
    article = Article(url)
    article.download()
    article.parse()
    info(log_msg)
    title = article.title
    content = article.text
    print(f"-----HEADLINE-----\n{title}\n-----CONTENT-----\n{content[:200]}...\n------------------")
    return content


# Get the initial article
input_content = get_article(
    prompt  = "Enter the URL of input article to classify: ",
    log_msg = "Extracting input article..."
)

# Get the summary
info("Extracting summary...")
summary_ext = TextRankExtractor()
_probabilities, summary = summary_ext.extract_main_themes(input_content, num_words=6)
info("Main themes...extracted\n(Please enter into a PageRank algorithm)")
print(f"\n{'-'*len(summary)}\n\n{summary}\n\n{'-'*len(summary)}\n")

# Get 3 articles
ctx1_content = get_article(
    prompt  = "Enter the URL of Context article 1: ",
    log_msg = "Extracting Context article 1..."
)
ctx2_content = get_article(
    prompt  = "Enter the URL of Context article 2: ",
    log_msg = "Extracting Context article 2..."
)
ctx3_content = get_article(
    prompt  = "Enter the URL of Context article 3: ",
    log_msg = "Extracting Context article 3..."
)

# Preprocessing
info("Preprocessing...")
preprocessor = Preprocessor()
input_content_tokens = preprocessor.preprocess_and_tokenize(input_content)
ctx_tokens = [
    preprocessor.preprocess_and_tokenize(ctx1_content),
    preprocessor.preprocess_and_tokenize(ctx2_content),
    preprocessor.preprocess_and_tokenize(ctx3_content),
]
info("Preprocessing...done")

# Similarity features
info("Similarity feature extraction...")
tfidf_1_2 = TextualRelevance("tfidf", input_content_tokens, ngram_range=(1, 2))
cosine_dist = tfidf_1_2.cosine_dist(input_content_tokens, ctx_tokens)
word_app = tfidf_1_2.word_appearance(input_content_tokens, ctx_tokens)
matching = tfidf_1_2.matching_score(input_content_tokens, ctx_tokens)
similarity_score = 3 / ((1 / cosine_dist) + (1 / word_app) + (1 / matching))
info("Similarity feature extraction...done")

# Non-latent features
info("Non latent feature extraction...")
preprocessor_ascii = Preprocessor(
    lowercase=False,
    remove_non_ascii=True,
    remove_punctuation=False,
    lemmatization=False,
    remove_stopwords=False
)
input_content_only_ascii = preprocessor_ascii.preprocess(input_content)
non_latent_features = NonLatentFeatures(input_content_only_ascii).output_all()
wanted_non_latent_keys = Pipeline.NonLatentConfig().build_keys()
non_latent_features = [v for (k, v) in non_latent_features.items() if k in wanted_non_latent_keys]
info("Non latent feature extraction...done")

# BERT Feature extraction
info("BERT feature extraction...")
bert_input = " ".join(input_content_tokens)[:512] # Truncate first 512 tokens
bert_features = bert_model(bert_tokenizer(bert_input, return_tensors="tf")).pooler_output.numpy()[0]
info("BERT feature extraction...done")

# Concatenate features
features = np.array([
    *bert_features,
    similarity_score,
    *non_latent_features
]).reshape((1, -1))

# Normalize all features - the pickled normalizer has only been trained on our
# training data.
with open("data/Horne2017_FakeNewsData/Buzzfeed/normalizer.pickle", "rb") as f:
    normalizer: DataNormalizer = pickle.load(f)
features = normalizer.transform(features)

print(features)