## spaCy Installation
# pip install -U pip setuptools wheel
# pip install -U spacy
# python -m spacy download en_core_web_sm

## Other dependencies
# pip3 install spacytextblob contractions syllables py-readability-metrics
# python -m textblob.download_corpora
# python -m spacy download en_core_web_sm

import numpy as np

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from readability import Readability
from readability.exceptions import ReadabilityException
import syllables
import contractions

class NonLatentFeatures():
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
        # Expand contractions
        self.raw = ' '.join(contractions.fix(word) for word in raw.split())

        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe('spacytextblob')

        self.doc = nlp(self.raw)
    
        self.run_loop = False
        self.counter = {}
        self.unique = {}
        self.avg_chars_per_word = []
        self.arr = {}
        
        # For readability indices
        self.readability = Readability(self.raw)        

    def _get_tag_count(self, arr):
        return [self.counter.get(tag, 0) for tag in arr]

    def _get_num_clauses(self):
        '''
        https://subscription.packtpub.com/book/data/9781838987312/2/ch02lvl1sec13/splitting-sentences-into-clauses
        '''
        def find_root_of_sentence(doc):
            root_token = None
            for token in doc:
                if (token.dep_ == "ROOT"):
                    root_token = token
            return root_token
        
        def find_other_verbs_count(doc, root_token):
            other_verbs_count = 0
            for token in doc:
                ancestors = list(token.ancestors)
                if (token.pos_ == "VERB" and len(ancestors) == 1\
                    and ancestors[0] == root_token):
                    other_verbs_count += 1
            return other_verbs_count

        num_clauses = 0
        for sent in self.doc.sents:
            root_token = find_root_of_sentence(sent)
            other_verbs_count = find_other_verbs_count(self.doc, root_token)
            # +1 for root token
            num_clauses += (1 + other_verbs_count)
        return num_clauses

    def _loop_runner(self):
        if not self.run_loop:
            for token in self.doc:
                self.unique[token.pos_] = self.unique.get(token.pos_, set())
                self.unique[token.pos_].add(token.lemma_)
                
                self.counter[token.pos_] = self.counter.get(token.pos_, 0) + 1
                self.counter[token.tag_] = self.counter.get(token.tag_, 0) + 1
                
                if token.is_alpha:
                    self.arr['chars_per_word'] = self.arr.get('chars_per_word', [])
                    self.arr['chars_per_word'].append(len(token))
                    
                    if token.pos_ == 'PRON':
                        lower_token = token.text.lower()
                        if lower_token in ('i', 'me', 'my', 'mine', 'myself'):
                            self.counter['FPS'] = self.counter.get('FPS', 0) + 1
                        elif lower_token in ('we', 'us', 'our', 'ourselves'):
                            self.counter['FPP'] = self.counter.get('FPP', 0) + 1
                        elif lower_token in ('you', 'your', 'he', 'she', 'it', 'him', 'his', 'her', 'they', 'their', 'them'):
                            self.counter['STP'] = self.counter.get('STP', 0) + 1
                    elif token.text.lower() in ('no', 'never', 'not'):
                        self.counter['NEG'] = self.counter.get('NEG', 0) + 1
                    if token.is_stop:
                        self.counter['STOP'] = self.counter.get('STOP', 0) + 1

                    if token.text[0].isupper():
                        self.counter['UP'] = self.counter.get('UP', 0) + 1
                        if token.text.isupper():
                            self.counter['CAPS'] = self.counter.get('CAPS', 0) + 1
                    elif token.text.islower():
                        self.counter['LOW'] = self.counter.get('LOW', 0) + 1
                else:
                    if token.text in ('!', '?'):
                        self.counter[token.text] = self.counter.get(token.text, 0) + 1

        self.run_loop = True
    
    def diversity(self, type_, reduce_):
        '''
        Diversity
        
        Args:
            type_: lexical, content, function, noun, verb, adj, adv
        '''
        
        self._loop_runner()
        
        # ARGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'LEX', 'CONT', 'FUNC']
        
        if reduce_ not in ('sum', 'percent'):
            raise ValueError(f'Invalid reduction method: {reduce_}')
                    
        arr = []
        if type_ in ('NOUN', 'VERB', 'ADJ', 'ADV'):
            arr = self.unique.get(type_, [])
            
            if reduce_ == 'sum':
                return len(arr)
            else:
                if self.counter.get(type_, 0) == 0:
                    return 0
                else:
                    return len(arr) / self.counter.get(type_, 0)
                
        elif type_ in ('LEX', 'CONT', 'FUNC'):
            filtered_keys = []
            # https://pronuncian.com/content-and-function-words#:~:text=Content%20words%20are%20usually%20nouns,focus%20his%20or%20her%20attention.
            if type_ == 'LEX':
                filtered_keys = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'VERB']
                # Can do np.sum or np.avg on this value or divide them by the len(self.doc)
            elif type_ == 'CONT':
                filtered_keys = ['ADJ', 'ADV', 'ADJ', 'NOUN', 'PROPN', 'VERB']
            elif type_ == 'FUNC':
                filtered_keys = ['AUX', 'ADP', 'CONJ', 'CCONJ', 'DET', 'PRON', 'SCONJ']

            arr = [len(self.unique.get(k, [])) for k in filtered_keys]
            
            if reduce_ == 'sum':
                return np.sum(arr)
            else:
                # total = np.sum(self._get_tag_count(filtered_keys))
                # if total == 0:
                #     return 0
                # return len(arr) / total
                return np.sum(arr) / len(self.doc)

        else:
            raise ValueError(f'Invalid type: {type_}')
            
    def pronoun(self, type_, reduce_):
        '''
        Non-immediacy
        
        Args:
            type_: FPS, FPP, STP
            first-person-singular, first-person,plural, second-third-person
        '''
        self._loop_runner()

        if reduce_ not in ('sum', 'percent'):
            raise ValueError(f'Invalid reduction method: {reduce_}')

        if type_ in ('FPS', 'FPP', 'STP'):
            if reduce_ == 'sum':
                return self.counter.get(type_, 0)
            else:
                return self.counter.get(type_, 0) / len(self.doc)
        else:
            raise ValueError(f'Invalid type: {type_}')
    
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
        self._loop_runner()
        
        if reduce_ not in ('sum', 'percent'):
            raise ValueError(f'Invalid reduction method: {reduce_}')
        
        # POS-related
        if type_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'NUM', 'PUNCT', 'SYM'] + ['PRP', 'PRP$', 'WDT', 'CD', 'VBD'] + ['STOP', 'LOW', 'UP', 'NEG']:
            # POS (UPOS) or TAG (English POS) or others
            sum_ = self.counter.get(type_, 0)
        else:
            if reduce_ == 'percent':
                raise ValueError(f'Invalid reduction method: {reduce_}')
            
            if type_ == 'QUOTE':
                sum_ = np.sum(self._get_tag_count(["``", '""', "''"]))

            elif type_ == 'NP':
                sum_ = len(list(self.doc.noun_chunks))

            # Simple Count
            elif type_== 'CHAR':
                sum_ = len(self.raw)
            elif type_== 'WORD':
                sum_ = len(self.doc)
            elif type_== 'SENT':
                sum_ = len(list(self.doc.sents))

            # Using Library
            elif type_== 'SYLL':
                sum_ = syllables.estimate(self.raw)

            else:
                raise ValueError(f'Invalid type: {type_}')
        
        # CAN ONLY BE ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'NUM', 'PUNCT', 'SYM'] + ['PRP', 'PRP$', 'WDT', 'CD', 'VBD'] + ['STOP', 'LOW', 'UP', 'NEG']
        if reduce_ == 'sum':
            return sum_
        else:
            return sum_ / len(self.doc)

    def sentiment(self, type_, reduce_='percent'):
        '''
        Sentiment
        
        Args:
            type_: anx-ang-sad, exclamation_mark, neg_word, pos_word,
            polarity, subjectivity, avg_neg_senti, avg_pos_senti, all_caps
            reduce_: count, percent 
        '''
        self._loop_runner()
        
        if reduce_ not in ('sum', 'percent'):
            raise ValueError(f'Invalid reduction method: {reduce_}')
        
        if type_ in ['!','?', 'CAPS']:
            if reduce_ == 'sum':
                return self.counter.get(type_, 0)
            else:
                return self.counter.get(type_, 0) / len(self.doc)
        else:
            if reduce_ == 'percent':
                raise ValueError(f'Invalid reduction method: {reduce_}')
            
            if type_ == 'POL':
                # Polarity
                return self.doc._.blob.polarity
            elif type_ == 'SUBJ':
                # Subjectivity
                return self.doc._.blob.subjectivity
            else:
                raise ValueError(f'Invalid type: {type_}')

    def average(self, type_):
        '''
        Complexity
        
        Args:
            type_: chars_per_word, words_per_sent, claus_per_sent, puncts_per_sent
        '''
        self._loop_runner()        
        if type_ == 'chars_per_word':
            return np.mean(self.arr['chars_per_word'])
        elif type_ == 'words_per_sent':
            return np.mean([len(sent) for sent in self.doc.sents])
        elif type_ == 'puncts_per_sent':
            self.arr['puncts_per_sent'] = [0] * len(list(self.doc.sents))
            for i, sent in enumerate(self.doc.sents):
                self.arr['puncts_per_sent'].append(0)
                for j in range(sent.start, sent.end):
                    if self.doc[j].pos_ == 'PUNCT':
                        self.arr['puncts_per_sent'][i] += 1
            return np.mean(self.arr['puncts_per_sent'])
        elif type_ == 'claus_per_sent':
            return np.mean(self._get_num_clauses())
        else:
            raise ValueError(f'Invalid type: {type_}')

    def syntax_tree(self, type_):
        '''
        Syntax Tree
        
        Args:
            type_: all, noun_phrase, verb_phrase
        '''
        # https://stackoverflow.com/questions/64591644/how-to-get-height-of-dependency-tree-with-spacy
        def walk_tree(node, depth):
            if node.n_lefts + node.n_rights > 0:
                return max(walk_tree(child, depth + 1) for child in node.children)
            else:
                return depth

        if type_ == 'ALL':
            depths_arr = []
            for sent in self.doc.sents:
                depths_arr.append(walk_tree(sent.root, 0))
            
            return np.median(depths_arr)
        elif type_== 'NP':
            depths_arr = []
            for sent in self.doc.sents:
                for noun_phrase in sent.noun_chunks:
                    depths_arr.append(walk_tree(noun_phrase.root, 0))
            return np.median(depths_arr)
        # elif type_== 'VP':
        #     vps = []
        #     for sent in self.doc.sents:
        #         # https://devpress.csdn.net/python/63045ba97e6682346619a782.html
        #         pattern = r'<VERB>?<ADV>*<VERB>+'
        #         doc = textacy.Doc(sent, lang='en_core_web_sm')
        #         lists = textacy.extract.token_matches(doc, pattern)
        #         vps.concat(lists)

        #     if reduce_ == 'median_depth':
        #         depths_arr = []
        #         for sent in self.doc.sents:
        #             for noun_phrase in sent.noun_chunks:
        #                 depths_arr.append(walk_tree(noun_phrase.root, 0))
        #         return np.median(depths_arr)            
        #     else:
        #         return len(vps)
        else:
            raise ValueError(f'Invalid type: {type_}')

    def readability_(self, type_):
        '''
        Readability
        
        Args:
            type_: gunning-fog, coleman-liau, dale-chall,
            flesch-kincaid, linsear-write, spache, automatic, flesch 
            
        Reference:
            https://pypi.org/project/py-readability-metrics/
        '''
        res = None
        
        if type_ not in ['gunning-fog', 'coleman-liau', 'dale-chall', 'flesch-kincaid', 'linsear-write', 'spache', 'automatic', 'flesch']:
            raise ValueError(f'Invalid type: {type_}')

        try:
            if type_ == 'gunning-fog':
                res = self.readability.gunning_fog()
            elif type_ == 'coleman-liau':
                res = self.readability.coleman_liau()
            elif type_ == 'flesch-kincaid':
                res = self.readability.flesch_kincaid()
            elif type_ == 'linsear-write':
                res = self.readability.linsear_write()
            elif type_ == 'spache':
                res = self.readability.spache()
            elif type_ == 'dale-chall':
                res = self.readability.dale_chall()
            elif type_ == 'automatic':
                res = self.readability.ari()
            elif type_ == 'flesch':
                res = self.readability.flesch()
            
            # if type_ == 'automatic':
            #     return res.score, res.grade_levels, res.ages
            # elif type_ == 'flesch':
            #     return res.score, res.grade_levels, res.ease
            # elif type_ == 'dale-chall':
            #     return res.score, res.grade_levels
            # else:
            #     return res.score, res.grade_level
            
            return res.score
        
        # Due to doc size <= 100
        except ReadabilityException:
            return 0

    def output_all(self):
        feats = {}
        
        for arg in ['NOUN', 'VERB', 'ADJ', 'ADV', 'LEX', 'CONT', 'FUNC']:
            for reduce_ in ('sum', 'percent'):
                feats['div_' + arg + '_' + reduce_] = self.diversity(arg, reduce_)

        for arg in ['FPS', 'FPP', 'STP']:
            for reduce_ in ('sum', 'percent'):
                feats['pron_' + arg + '_' + reduce_] = self.pronoun(arg, reduce_)

        for arg in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'NUM', 'PUNCT', 'SYM'] + ['PRP', 'PRP$', 'WDT', 'CD', 'VBD'] + ['STOP', 'LOW', 'UP', 'NEG']:
            for reduce_ in ('sum', 'percent'):
                feats['quant_' + arg + '_' + reduce_] = self.quantity(arg, reduce_)
        for arg in ['QUOTE', 'NP', 'CHAR', 'WORD', 'SENT', 'SYLL']:
            feats['quant_' + arg + '_' + 'sum'] = self.quantity(arg, 'sum')

        for arg in ['!','?', 'CAPS']:
            for reduce_ in ('sum', 'percent'):
                feats['senti_' + arg + '_' + reduce_] = self.sentiment(arg, reduce_)
        for arg in ['POL', 'SUBJ']:
            feats['senti_' + arg + '_' + 'sum'] = self.sentiment(arg, 'sum')

        for arg in ['chars_per_word', 'words_per_sent', 'claus_per_sent', 'puncts_per_sent']:
            feats['avg_' + arg + '_' + 'sum'] = self.average(arg)

        for arg in ['ALL', 'NP']:
            feats['med_st_' + arg + '_' + 'sum'] = self.syntax_tree(arg)

        for arg in ['gunning-fog', 'coleman-liau', 'dale-chall', 'flesch-kincaid', 'linsear-write', 'spache', 'automatic', 'flesch']:
            feats['read_' + arg + '_' + 'sum'] = self.readability_(arg)

        return feats

if __name__ == "__main__":
    text = "I, you, we, $$, !?, HAHA, Claim: Nearly 5 million uncounted California ballots cast for Bernie Sanders were found on Hillary Clinton's private email server by the F.B.I. Rating: About this rating False\n\nFake news occasionally proves itself to be more amusing than annoying (i.e., actual satire rather than an attempt to fool readers with sensationalist clickbait), and an August 2016 article from the Nevada County Scooper hit that mark \u2014 deftly blending political controversies over Democratic presidential nominee Hillary Clinton's use of a private e-mail server, suggestions of election fraud, and claims by supporters of rival Bernie Sanders that their candidate had been cheated out of the nomination by a partisan Democratic National Committee into one short article that began:\n\nDemocratic nominee Hillary Clinton is in hot water again after nearly 5 million uncounted California electronic ballots were found on her email server by the F.B.I. The majority of those ballots cast were by Bernie Sanders supporters. The election commission has an emergency meeting scheduled for tomorrow morning to discuss a possible vote recount which will include the newly-found ballots. Ms. Clinton has already come under fire during this election cycle over using her private email server for personal use, including storing and sending classified information with it. There has been a formal investigation, but no charges have been filed in that case. Some of the charges are facing several charges.\n\n\n\nAlas for those who might wish to believe that some or all of the above is true, the Nevada County Scooper's \"Manifesto\" notes that the site is all about fake news and satire:"
    doc = NonLatentFeatures(text)

    # FIX THESE UP
    print('\nDiversity')
    for arg in ['NOUN', 'VERB', 'ADJ', 'ADV', 'LEX', 'CONT', 'FUNC']:
        for reduce_ in ('sum', 'percent'):
            print(arg, reduce_, ': ', doc.diversity(arg, reduce_))

    print('\nPronoun')
    for arg in ['FPS', 'FPP', 'STP']:
        for reduce_ in ('sum', 'percent'):
            print(arg, reduce_, ': ', doc.pronoun(arg, reduce_))

    print('\nQuantity')
    for arg in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'NUM', 'PUNCT', 'SYM'] + ['PRP', 'PRP$', 'WDT', 'CD', 'VBD'] + ['STOP', 'LOW', 'UP', 'NEG']:
        for reduce_ in ('sum', 'percent'):
            print(arg, reduce_, ': ', doc.quantity(arg, reduce_))        
    for arg in ['QUOTE', 'NP', 'CHAR', 'WORD', 'SENT', 'SYLL']:
        print(arg, 'sum', ': ', doc.quantity(arg, 'sum'))

    print('\nSentiment')
    for arg in ['!','?', 'CAPS']:
        for reduce_ in ('sum', 'percent'):
            print(arg, reduce_, ': ', doc.sentiment(arg, reduce_))
    for arg in ['POL', 'SUBJ']:
        print(arg, 'sum', ': ', doc.sentiment(arg, 'sum'))

    print('\nAverage')
    for arg in ['chars_per_word', 'words_per_sent', 'claus_per_sent', 'puncts_per_sent']:
        print(arg, ': ', doc.average(arg))

    print('\nSyntaxTreeMedianDepth')
    for arg in ['ALL', 'NP']:
        print(arg, ': ', doc.syntax_tree(arg))

    print('\nReadability')
    for arg in ['gunning-fog', 'coleman-liau', 'dale-chall', 'flesch-kincaid', 'linsear-write', 'spache', 'automatic', 'flesch']:
        print(arg, ': ', doc.readability_(arg))
        
    print(doc.output_all())
    
