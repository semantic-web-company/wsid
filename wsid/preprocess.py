import unicodedata
import re
from collections import Counter

import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer

latin_letters = {}
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = LancasterStemmer()
REPL_HOLDER = '__replacement_{}__'
REPL_PATTERN = re.compile(REPL_HOLDER.format('(?P<id>\d+)'))


def preprocess(texts, excluded_tokens=(),
               remove_nonletters=True, remove_shortwords=True,
               make_lower=True, remove_stopwords=True,
               stem=False, digits2token=True,
               min_df=0, max_df=1.):
    """

    :param list[str] texts: texts
    :param list[str] excluded_tokens: excluded tokens
    :param bool remove_nonletters: NOT IMPLEMENTED!
    :param bool remove_shortwords: remove words under 3 chars
    :param bool make_lower: cast to lower
    :param bool remove_stopwords: nltk english stopwords
    :param bool stem: Lancaster stemmer is used
    :param bool digits2token: all digits replaced to "DIGIT"
    :param int min_df: minimum number of documents where each token should appear.
             the infrequent tokens are filtered out.
    :param float max_df: maximum fraction of documents where tokens could appear.
    :return: corpus of texts and vocabulary with frequencies
    :rtype: (list[str], Counter)
    """
    corpus = []
    repl_list = []
    vocabulary = Counter()
    df = Counter()
    max_df *= len(texts)

    excluded_and_digit = list(excluded_tokens) + ['DIGIT']
    for txt in texts:
        if excluded_tokens:
            txt, replaced = replace_in_str(txt, excluded_tokens)
            repl_list.append(replaced)

        if make_lower:
            txt = txt.lower()
        if remove_shortwords:
            txt = ' '.join(re.findall(r'\w{3,}', txt))
        if digits2token:
            txt = re.sub(r'(?<=[\s.!?])\d+(?=[\s.!?])', 'DIGIT', txt)
        tokens = word_tokenize(txt)
        if remove_stopwords:
            tokens = [token for token in tokens if token not in stopwords]
        if stem:
            tokens = [stemmer.stem(t) for t in tokens]
        vocabulary.update(tokens)
        df.update(set(tokens))
        corpus.append(tokens)

    for token in excluded_and_digit:
        df[token] = max_df - 1
    all_tokens = set()
    for i in range(len(corpus)):
        tokens = corpus[i]
        if excluded_tokens:
            tokens = restore_in_list(tokens, repl_list[i])
        tokens = [w for w in tokens
                  if df[w] >= min_df
                  if df[w] <= max_df]
        txt = ' '.join(tokens)
        corpus[i] = txt
        all_tokens |= set(tokens)
    for token in set(vocabulary) - set(all_tokens):
        del vocabulary[token]
    return corpus, vocabulary


def replace_in_list(tokens_list, tokens_to_replace):
    ans = tokens_list[:]
    repl_list = []
    set_tokens_to_replace = set(tokens_to_replace)
    for i_token, token in enumerate(ans):
        if token in set_tokens_to_replace:
            repl_list.append(token)
            ans[i_token] = REPL_HOLDER.format(len(repl_list) - 1)
    return ans, repl_list


def restore_in_list(tokens_list, tokens_to_restore):
    ans = tokens_list[:]
    for i_token, token in enumerate(ans):
        match = REPL_PATTERN.match(token)
        if match:
            ans[i_token] = tokens_to_restore[int(match.group('id'))]
    return ans


def replace_in_str(text, tokens):
    ans = text[:]
    repl_list = []
    for token in tokens:
        token_found = True
        while token_found:
            match = re.search(r'(?<=\b){}(?=\b)'.format(token), ans)
            if match is not None:
                span = match.span()
                repl_list.append(ans[span[0]:span[1]])
                ans = (ans[:span[0]] +
                       REPL_HOLDER.format(len(repl_list)-1) +
                       ans[span[1]:])
            else:
                token_found = False
    return ans, repl_list


def restore_in_str(replaced_text, tokens):
    replacement_found = True
    ans = replaced_text[:]
    while replacement_found:
        match = re.search(REPL_PATTERN, ans)
        if match is not None:
            span = match.span()
            ans = (ans[:span[0]] +
                   tokens[int(match.group('id'))] +
                   ans[span[1]:])
        else:
            replacement_found = False
    return ans
































# def preprocess(texts, forms, canonical_entity, min_df=3, max_df=0.67):
#     from collections import Counter
#     corpus = []
#     vocabulary = Counter()
#     df = Counter()
#     max_df *= len(texts)
#     for txt in texts:
#         txt = txt.lower()
#         for form in forms:
#             txt = re.sub(r'(?<=\b){}\w{{0,2}}(?=\b)'.format(form),
#                          canonical_entity, txt)
#             assert not re.findall(r'(?<=\b){}\w{{1,2}}(?=\b)'.format(form),
#                                   txt)
#         txt = ' '.join(
#             re.findall(r'\w{{3,}}|{}'.format(canonical_entity), txt)
#         )
#         txt = re.sub(r'(?<=[\s.!?])\d+(?=[\s.!?])', 'DIGIT', txt)
#         tokens = [
#             (token if token not in [canonical_entity, 'DIGIT'] else token)
#             for token in txt.split()
#             ]
#         vocabulary.update(tokens)
#         df.update(set(tokens))
#         assert any(0 < df[w] for w in tokens)
#         corpus.append(tokens)
#     for token in [canonical_entity, 'DIGIT']:
#         df[token] = max_df - 1
#     all_tokens = set()
#     for i in range(len(corpus)):
#         tokens = [w for w in corpus[i]
#                   if df[w] >= min_df
#                   if df[w] < max_df]
#         txt = ' '.join(tokens)
#         corpus[i] = txt
#         all_tokens |= set(tokens)
#     for token in set(vocabulary) - set(all_tokens):
#         del vocabulary[token]
#     return corpus, vocabulary




























### OLD STUFF


class CoocException(Exception):
    pass


def is_latin(uchr):
    try:
        return latin_letters[uchr]
    except KeyError:
        return latin_letters.setdefault(uchr, 'LATIN' in unicodedata.name(uchr))


def roman_words(unistr):
    """
    Get all the western words for the given unistr
    """
    ans = ''
    for uchr in unistr:
        if uchr.isalpha():
            if is_latin(uchr):
                ans += uchr
        else:
            ans += uchr
    return ans  # ''.join(uchr for uchr in unistr if uchr.isalpha() if is_latin(uchr))


def eliminate_stopwords(text):
    return ' '.join([word for word in word_tokenize(text)
                     if word not in stopwords])


def eliminate_shortwords(text):
    return ' '.join(re.findall(r'\w{3,}', text))


def eliminate_symbols(text):
    return re.sub('[^a-zA-Z0-9_\s]', '', text)


def remove_nonletters(word):
    return ''.join(re.findall('\w+', word))


# class CoVectorizer(CountVectorizer):
#     def __init__(self, w, proximity_func=None, scoring='unbiased_dice'):
#         def linear_proximity(x):
#             return (w - abs(x) + 0.5) * 2 / w
#         assert isinstance(w, int)
#         self.w = w
#         if proximity_func is None:
#             proximity_func = linear_proximity
#         assert callable(proximity_func)
#         self.proximity_func = proximity_func
#         self.scoring = scoring
#         super().__init__()
#
#     def _count_vocab(self, processed_texts, y):
#         w = self.w
#         co_terms_dict = defaultdict(dict)
#
#         for category in np.unique(y):
#             category_texts = [processed_texts[i]
#                               for i in range(len(processed_texts))
#                               if y[i] == category]
#             co_terms_dict[category] = cooc.get_co(
#                 category_texts, w,
#                 method=self.scoring,
#                 proximity_func=self.proximity_func,
#                 forms_dict=self.forms_dict
#             )
#             try:
#                 co_terms_dict[category] = co_terms_dict[category][
#                     self.forms_dict[self.entity_forms[0]]
#                 ]
#             except:
#                 print(co_terms_dict[category])
#                 co_terms_dict[category] = \
#                     co_terms_dict[category][self.entity_forms[0]]
#         if not co_terms_dict:
#             raise CoocException('No cooccurrent terms found.')
#         self.co_terms = co_terms_dict
#
#         all_tokens = set(sum([list(d.keys()) for d in co_terms_dict.values()],
#                             []))
#         vocabulary = {counter_term: ind
#                       for ind, counter_term in enumerate(all_tokens)}
#
#         self.vocabulary = vocabulary
#
#     @staticmethod
#     def preprocess_text(text, nonletters=True, shortwords=True,
#                         symbols=True, lower=True, stopwords=True,
#                         exclude_tokens=(), stem=None, digits2token=True):
#         ans_text = text[:]
#
#         # OLD word by word implementation
#         # words = word_tokenize(ans_text)
#         # processed_words = []
#         # for word in words:
#         #     if word in exclude_tokens:
#         #         processed_word = word
#         #     else:
#         #         processed_word = word
#         #         if nonletters:
#         #             processed_word = remove_nonletters(processed_word)
#         #         if shortwords:
#         #             processed_word = eliminate_shortwords(processed_word)
#         #         if symbols:
#         #             processed_word = eliminate_symbols(processed_word)
#         #         if lower:
#         #             processed_word = processed_word.lower()
#         #         if stopwords:
#         #             processed_word = eliminate_stopwords(processed_word)
#         #         if stem:
#         #             processed_word = stemmer.stem(processed_word)
#         #         if digits2token:
#         #             txt = re.sub(r'(?<=[\s.!?])\d+(?=[\s.!?])', 'DIGIT', txt)
#         #     if processed_word:
#         #         processed_words.append(processed_word)
#         # ans_text = ' '.join(processed_words)
#         # return ans_text
#
#     def fit_transform(self, raw_documents, y, entity_forms,
#                       forms_dict=None):
#         processed_texts = [self.preprocess_text(raw_documents[i])
#                            for i in range(len(raw_documents))]
#         processed_forms = [self.preprocess_text(entity_forms[i])
#                            for i in range(len(entity_forms))]
#         new_forms_dict = {ef: processed_forms[0] for ef in processed_forms}
#         if forms_dict is None:
#             self.forms_dict = new_forms_dict
#         else:
#             self.forms_dict = {self.preprocess_text(x):
#                                    self.preprocess_text(forms_dict[x])
#                                for x in forms_dict}
#             self.forms_dict.update(new_forms_dict)
#         self.entity_forms = processed_forms
#         self._validate_vocabulary()
#         self._count_vocab(processed_texts, y)
#         x = self.transform(processed_texts, classes=y)
#         return x
#
#     def fit(self, raw_documents, y, entity_forms,
#             forms_dict=None):
#         self.fit_transform(raw_documents, y, entity_forms, forms_dict)
#         return self
#
#     def transform(self, raw_documents, classes=None):
#         co_terms = self.co_terms
#         if classes is None:
#             all_co_terms = dict()
#             for category in co_terms.keys():
#                 all_co_terms.update(co_terms[category])
#         w = self.w
#         assert isinstance(w, int)
#
#         x_data = array.array('f')
#         j_indices = array.array('i')
#         indptr = array.array('i')
#         indptr.append(0)
#
#
#         for i, doc in enumerate(raw_documents):
#             processed_doc = self.preprocess_text(doc)
#             local_words, _ = cooc.get_t2t_proximities(
#                 processed_doc, w=self.w,
#                 proximity_func=self.proximity_func
#             )
#             for entity_form in self.entity_forms[1:]:
#                 for token in local_words[entity_form]:
#                     local_words[self.entity_forms[0]][token] += \
#                         local_words[entity_form][token]
#             local_words = local_words[self.entity_forms[0]]
#             if classes is not None:
#                 for word, score in local_words.items():
#                     if word in co_terms[classes[i]]:
#                         x_data.append(score *
#                                       co_terms[classes[i]][word])
#                         j_indices.append(self.vocabulary[word])
#             else:
#                 for word, score in local_words.items():
#                     if word in all_co_terms:
#                         x_data.append(score * all_co_terms[word])
#                         j_indices.append(self.vocabulary[word])
#
#             indptr.append(len(j_indices))
#         x = sp.csr_matrix((x_data, j_indices, indptr),
#                           shape=(len(indptr) - 1, len(self.vocabulary)),
#                           dtype=float)
#         x.sum_duplicates()
#
#         return x
#
#     @property
#     def form_features(self):
#         if self.form_features_ is None:
#             raise ValueError('Form features are not set yet. '
#                              'Probably, the vectorizer is not fitted yet')
#         return self.form_features_
#
#     @form_features.setter
#     def form_features(self, x):
#         self.form_features_ = x
#
#     @property
#     def vocabulary(self):
#         if self.vocabulary_ is None:
#             return None
#         else:
#             return dict(self.vocabulary_)
#
#     @vocabulary.setter
#     def vocabulary(self, x):
#         if x is not None:
#             assert isinstance(x, dict)
#             self.vocabulary_ = x
#         else:
#             self.vocabulary_ = x
#
#     @property
#     def w(self):
#         if self.w_ is None:
#             raise ValueError('Window size w is not set yet. '
#                              'Probably, the vectorizer is not fitted yet')
#         return self.w_
#
#     @w.setter
#     def w(self, value):
#         self.w_ = value
#
#     @property
#     def co_terms(self):
#         if self.co_terms_ is None:
#             raise ValueError('Cooccurrent terms are not set yet. '
#                              'Probably, the vectorizer is not fitted yet')
#         return dict(self.co_terms_)
#
#     @co_terms.setter
#     def co_terms(self, value):
#         self.co_terms_ = value
