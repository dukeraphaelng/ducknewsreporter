from collections import Counter
import string

import numpy as np

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')

from readability import Readability
import syllables
import contractions

def num_punc(words):
    count_ = 0
    for word in words:
        if word in string.punctuation:
            count_ += 1
    return count_

# REVIEW ALL NLTK POS USING nltk.help.upenn_tagset()

class SemanticNonLatentFeatures():
    '''
    Reference:
        Zhou, X. and Zafarani, R., 2020. A survey of fake news: 
        Fundamental theories, detection methods, and opportunities. 
        ACM Computing Surveys (CSUR), 53(5), pp.1-40.
        
        Horne, B. and Adali, S., 2017, May. This just in: Fake news 
        packs a lot in title, uses simpler, repetitive content in 
        text body, more similar to satire than real news. In 
        Proceedings of the international AAAI conference on web 
        and social media (Vol. 11, No. 1, pp. 759-766).
        
        Garg, S. and Sharma, D.K., 2022. Linguistic features based 
        framework for automatic fake news detection. Computers & 
        Industrial Engineering, 172, p.108432.
    '''
    def __init__(self, raw):
        # Do contractions
        self.raw = ' '.join(contractions.fix(word) for word in raw.split())

        # From stackoverflow
        # Without punctuation
        self.clean_tokenizer = RegexpTokenizer(r'\w+')

        # Can't get clean tokens
        self.tags = nltk.pos_tag(self.text)
        
        self.run_loop_words = False
        
        self.readability = Readability(self.raw)
        
        # ADJ, ADP, ADV, CONJ, DET, NOUN, NUM, PRT, VERB, ., X
        self.pos_count = Counter(tag for word, tag in self.tags)
        total = sum(self.pos_count.values())        
        self.pos_percent = dict((tag, float(count_)/total) for tag, count_ in self.pos_count.items())
    
    # Lazy getter
    def get_sents(self):
        if not self.words:
            self.sents = nltk.sen_tokenize(self.raw)
        return self.sents
    
    def get_origin_words(self):
        if not self.origin_words:
            self.origin_tokens = nltk.word_tokenize(self.raw)
            self.origin_words = nltk.Text(self.origin_tokens)
        return self.origin_words
    
    def get_clean_words(self):
        # No punctuation
        if not self.clean_words:
            self.clean_tokens = self.clean_tokenizer.tokenize(self.raw)
            self.clean_words = nltk.Text(self.clean_tokens)
        return self.clean_words

    def puncs_per_sent(self):
        if not self.getter_puns_per_sent:
            self.getter_puns_per_sent = [num_punc(sent) for sent in self.get_sents()]
        return self.getter_puns_per_sent
    
    def loop_words(self):
        if not self.run_loop_words:
            self.upper = 0
            self.lower = 0
            self.all_caps = 0
            self.quotations = 0
            self.punctuation = 0
            self.question_marks = 0
            self.exclamation_marks = 0
            self.stop_count = 0
            
            self.first_person_singular = 0
            self.first_person_plural = 0
            self.second_third_person = 0
            
            # negations
            # NO = self.post_tag('DT)
            # NEVER, NOT = self.post_tag('RB')
            
            stops = set(stopwords.words('english'))

            for token in self.origin_words:
                if token in string.punctuation:
                    self.punctuation += 1
                    if token in ('"', "'"):
                        self.quotations += 1
                    elif token == '?':
                        self.question_marks += 1
                    elif token == '!':
                        self.exclamation_marks +=1
                else:
                    if token[0].isupper():
                        self.upper += 1
                        if token.isupper():
                            self.all_caps += 1
                    elif token.islower():
                        self.lower += 1

                    if token in stops:
                        self.stop_count += 1
                    
                    lower_token = token.lower()
                    if token in ('i', 'me', 'my', 'mine', 'myself'):
                        self.first_person_singular += 1
                    elif token in ('we', 'us', 'our', 'ourselves'):
                        self.first_person_plural += 1
                    elif token in ('you', 'your', 'he', 'she', 'it', 'him', 'his', 'her', 'they', 'their', 'them'):
                        self.second_third_person += 1

            self.run_loop_words = True
    
    def get_tag_count(self, arr):
        return [self.pos_count(tag) for tag in arr]
    
    def diversity(self, type_):
        '''
        Diversity
        
        Args:
            type_: lexical, content, function, noun, verb, adj, adv
        '''
        # POS COUNTS:
        # https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/
        if type_ == 'lexical':
            # Not including punctuation
            unique_words = len(set(self.get_clean_words()))
            all_words = len(self.get_clean_words())
            return unique_words / all_words
        elif type_ == 'content':
            # Cannot calculate it like this, must get unique words

            # https://pronuncian.com/content-and-function-words#:~:text=Content%20words%20are%20usually%20nouns,focus%20his%20or%20her%20attention.
            return np.sum(self.get_tag_count(['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'])) / len(self.get_clean_words())
        elif type_ == 'function':
            # https://pronuncian.com/content-and-function-words#:~:text=Content%20words%20are%20usually%20nouns,focus%20his%20or%20her%20attention.
            # auxiliary verbs, preps, conjuncs, det, pronoun
            return np.sum(self.get_tag_count(['CC', 'DT', 'IN', 'MD', 'PRP', 'PRP$', 'WP', 'WP$', 'WDT'])) / len(self.get_clean_words())
        elif type_ == 'noun':
            # Cannot calculate it like this, must get unique words
            return np.sum(self.get_tag_count(['NN', 'NNS', 'NNP', 'NNPS'])) / len(self.get_clean_words())
        elif type_ == 'verb':
            # Cannot calculate it like this, must get unique words
            return np.sum(self.get_tag_count(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])) / len(self.get_clean_words())
        elif type_ == 'adj':
            # Cannot calculate it like this, must get unique words
            return np.sum(self.get_tag_count(['JJ', 'JJR', 'JJS'])) / len(self.get_clean_words())
        elif type_ == 'adv':
            # Cannot calculate it like this, must get unique words
            return np.sum(self.get_tag_count(['RB', 'RBR', 'RBS'])) / len(self.get_clean_words())
        else:        
            raise ValueError(f'Invalid type: {type_}')
    
    def pronoun(self, type_):
        '''
        Non-immediacy
        
        Args:
            type_: first_person_singular, first_person_plural, second_third_person
        '''
        if type_ == 'first_person_singular':
            # I, me, my, mine, myself
            self.loop_words()
            return self.first_person_singular
        elif type_ == 'first_person_plural':
            # we us our, ourselves
            self.loop_words()
            return self.first_person_plural
        elif type_ == 'second_third_person':
            # you your he she it his him her they their them
            self.loop_words()
            return self.second_third_person
    
    def quantity(self, type_, reduce_='percent'):
        '''
        Quantity
        
        Args:
            type_: quotations, words, sents, chars, 
            noun_phrases, lower, upper, adv, det, nouns, adj, 
            articles, negations, syllables, verbs, analytic, comparison, 
            punctuations, wh_determinants, cardinals, personal_pronouns,
            posessive_pronouns, past_tense, proper_nouns, verb_phrases, stop_words

            reduce_: count, percent
        '''                    
        if type_== 'nouns':
            return np.sum(self.get_tag_count(['NN', 'NNS', 'NNP', 'NNPS']))
        elif type_== 'proper_nouns':
            return np.sum(self.get_tag_count(['NNP', 'NNPS']))
        elif type_ == 'verb':
            return np.sum(self.get_tag_count(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']))
        elif type_ == 'adj':
            return np.sum(self.get_tag_count(['JJ', 'JJR', 'JJS']))
        elif type_ == 'adv':
            return np.sum(self.get_tag_count(['RB', 'RBR', 'RBS']))       
        elif type_== 'articles':
            return self.pos_count('DT')
        elif type_== 'det':
            return self.pos_count('PDT')
        elif type_== 'personal_pronouns':
            return self.pos_count('PRP')
        elif type_== 'possessive_pronouns':
            return self.pos_count('PRP$')
        elif type_== 'wh_determinants':
            return self.pos_count('WDT')
        elif type_== 'past_tense':
            return self.pos_count('VBD')
        elif type_== 'cardinals':
            return self.pos_count('CD')
        elif type_== 'chars':
            return len(self.raw)
        elif type_== 'words':
            return len(self.get_clean_words())
        elif type_== 'sents':
            return len(self.get_sents())
        elif type_== 'punctuations':
            self.loop_words()
            return self.punctuation
        elif type_ == 'quotations':
            self.loop_words()
            return self.quotations
        elif type_== 'lower':
            self.loop_words()
            return self.lower
        elif type_== 'upper':
            self.loop_words()
            return self.upper
        elif type_== 'negations':
            pass
        elif type_== 'noun_phrases':
            pass
        elif type_== 'verb_phrases':
            pass
        elif type_== 'syllables':
            return syllables.estimate(self.raw)
        elif type_== 'stop_words':
            self.loop_words()
            return self.stop_count
        else:
            raise ValueError(f'Invalid type: {type_}')

        # Others: WDT: wh-determiner, WP: wh-pronoun, WP%: possessive wh-pronoun, WRB: wh-adverb

    def sentiment(self, type_, reduce_='percent'):
        '''
        Sentiment
        
        Args:
            type_: anx-ang-sad, exclamation_mark, neg_word, pos_word,
            polarity, subjectivity, avg_neg_senti, avg_pos_senti, all_caps
            reduce_: count, percent 
        '''
        
        if type_ == 'exclamation_mark':
            self.loop_words()
            return self.exclamation_marks
        elif type_ == 'question_mark':
            self.loop_words()
            return self.question_marks
        elif type_ == 'all_caps':
            self.loop_words()
            return self.all_caps
        elif type_ == 'polarity':
            pass
        elif type_ == 'subjectivity':
            pass
        else:
            raise ValueError(f'Invalid type: {type_}')

    # COMPLETE
    def readability(self, type_):
        '''
        Readability
        
        Args:
            type_: gunning-fog, automatic, coleman-liau, dale-chall,
            flesch, linsear-write, spache, flesch-kincaid
            
        Reference:
            https://pypi.org/project/py-readability-metrics/
        '''
        if type_ == 'gunning-fog':
            res = self.readability.gunning_fog()
            return res.score, res.grade_level
        elif type_ == 'automatic':
            res = self.readability.ari()
            return res.score, res.grade_levels, res.ages
        elif type_ == 'coleman-liau':
            res = self.readability.coleman_liau()
            return res.score, res.grade_level
        elif type_ == 'dale-chall':
            res = self.readability.dale_chall()
            return res.score, res.grade_levels
        elif type_ == 'flesch':
            res = self.readability.flesch()
            return res.score, res.ease, res.grade_levels
        elif type_ == 'flesch-kincaid':
            res = self.readability.flesch_kincaid()
            return res.score, res.grade_level
        elif type_ == 'linsear-write':
            res = self.readability.linsear_write()
            return res.score, res.grade_level
        elif type_ == 'spache':
            res = self.readability.spache()
            return res.score, res.grade_level
        else:
            raise ValueError(f'Invalid type: {type_}')

    # TODO: claus_per_sent
    def average(self, type_):
        '''
        Complexity
        
        Args:
            type_: chars_per_word, words_per_sent, claus_per_sent, puncs_per_sent
        '''
        
        if type_ == 'chars_per_word':
            return np.mean([len(word) for word in self.get_clean_words()])
        elif type_ == 'words_per_sent':
            return np.mean([len(sent) for sent in self.get_sents()])
        elif type_ == 'claus_per_sen':
            pass
            # UNIMPLEMENTED
            # https://stackoverflow.com/questions/31790259/counting-sentence-clauses-with-nltk
            # # self.sents = None
            
            # def num_claus(sent):
            #     pass

            # np.mean([num_claus(sent) for sent in sents])
        elif type_ == 'puncs_per_sen':
            return np.mean(self.puncs_per_sent())
        else:
            raise ValueError(f'Invalid type: {type_}')

    # median depth & number of
    def syntax_tree(self, type_):
        '''
        Syntax Tree
        
        Args:
            type_: all, noun_phrase, verb_phrase
        '''
        if type_ == 'all':
            pass
        elif type_ == 'noun_phrase':
            pass
        elif type_ == 'verb_phrase':
            pass
        else:
            raise ValueError(f'Invalid type: {type_}')