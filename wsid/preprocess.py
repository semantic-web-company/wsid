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
