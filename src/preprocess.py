import re
import string
from typing import List, Optional

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Can only be called once to make it thread-safe
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


class Preprocessor:
    def __init__(
        self,
        lowercase=True,
        remove_non_ascii=True,
        remove_punctuation=False,
        lemmatization=True,
        remove_stopwords=True,
    ):
        """Performs preprocessing and tokenizes the input

        Args:
            lowercase (bool, optional): Convert all ascii characters to lowercase. Defaults to True.
            remove_non_ascii (bool, optional): Removes all non ascii characters from the input. Defaults to True.
            remove_punctuation (bool, optional): Removes all punctuation. Defaults to False.
            lemmatization (bool, optional): Use lemmatization, converting all words to their base form. Defaults to True.
            remove_stopwords (bool, optional): Removes stopwords. Defaults to True.
        """
        self._lowercase = lowercase
        self._remove_non_ascii = remove_non_ascii
        self._remove_punctuation = remove_punctuation
        if lemmatization:
            self._lemmatizer = WordNetLemmatizer()
        else:
            self._lemmatizer = None
        self._remove_stopwords = remove_stopwords
    
    def preprocess(self, input: str) -> Optional[str]:
        """Preprocess a string

        Args:
            input (str): The string to preprocess

        Returns:
            str: Preprocessed string
        """
        # Remove newlines
        
        if input is None:
            return input
        if not isinstance(input, str) and pd.isnull(input):
            return input

        input.replace("\n", " ")

        if self._remove_non_ascii or self._remove_punctuation:
            def include_char(c):
                if self._remove_non_ascii and c not in string.printable:
                    return False
                if self._remove_punctuation and c in string.punctuation:
                    return False
                return True
            input = "".join(ch for ch in input if include_char(ch))
        
        if self._lemmatizer:
            input = " ".join(map(self._lemmatizer.lemmatize, input.split()))

        if self._lowercase:
            input = input.lower()

        if self._remove_stopwords:
            words = stopwords.words("english")
            input = " ".join(w for w in input.split() if w not in words)
        
        return input
    
    def tokenize(self, input: str) -> List[str]:
        """Tokenize the input by splitting on punctuation. Punctuation will be
        considered a token by themselves.

        Args:
            input (str): The string to tokenize

        Returns:
            List[str]: A list of tokens
        """
        return re.findall(r"\w+|[^\s\w]+", input)
    
    def tokenize_opt(self, input: str) -> Optional[List[str]]:
        if input is None:
            return input
        if not isinstance(input, str) and pd.isnull(input):
            return input
        return self.tokenize(input)        
    
    def preprocess_and_tokenize(self, input: str) -> List[str]:
        return self.tokenize(self.preprocess(input))
        
    def preprocess_and_tokenize_opt(self, input: Optional[str]) -> Optional[List[str]]:
        """Same as preprocess_and_tokenize but it can accept optional values such as python None or pandas NaType

        Args:
            input (Optional[str]): The input to preprocess and tokenize

        Returns:
            Optional[List[str]]: Tokenized string
        """
        if input is None:
            return input
        if not isinstance(input, str) and pd.isnull(input):
            return input
        return self.preprocess_and_tokenize(input)


if __name__ == "__main__":
    import sys
    print("Text to extract preprocess: ", end="")
    sys.stdout.flush()
    input = sys.stdin.read()
    processor = Preprocessor()
    tokens = processor.preprocess_and_tokenize(input)
    print(tokens)