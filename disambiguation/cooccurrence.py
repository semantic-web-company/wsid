import functools
import logging
import re
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse
from nltk import word_tokenize

__all__ = ['_calculate_z_score',
           'get_co',
           'get_t2t_proximities',
           'get_binom_scores']

module_logger = logging.getLogger(__name__)


def _calculate_z_score(tokens_counter, term, p_co, relevant_tokens):
    """
    Calculate co-occurrence of entity and term. The calculated score is the Z
    score, see (http://resources.esri.com/help/9.3/ArcGISDesktop/com/Gp_ToolRef/
    spatial_statistics_toolbox/what_is_a_z_score_what_is_a_p_value.htm)

    :param tokens_counter: Counter object counting words' appearances
    :param term: the term for which the cooccurrence with the entity is
    calculated
    :param p_co: probability of cooccurrence by accident
    :param relevant_tokens: words inside windows for all occurrences of
    entitites with scores (dict)
    :return: score calculated based on binomial distribution
    """
    token_count = tokens_counter[term]
    assert token_count > 0
    exp_co_count = p_co * token_count
    std_dev = (token_count*p_co*(1 - p_co)) ** .5
    if std_dev == 0:
        score = 0
    elif std_dev > 0:
        co_count = relevant_tokens[term]
        score = (co_count - exp_co_count) / std_dev
    else:
        assert False
    return score


def get_t2t_proximities(tokens,
                        w,
                        proximity_func=lambda x: 1,
                        frequent_tokens=None,
                        t2t_cos=None):
    """
    Get the relevant tokens in the sense of proximity. Extracts all the tokens
    that are within the window w from the entity and calculates their proximity
    score.

    :param text: the processed text (one at a time).
    :param entity_pattern: everything that matches the pattern is believed
    to be the entity. Be sure to only find whole words.
    :param w: window size.
    :param proximity_func: proximity function.
    :return: relevant_tokens_score: {token: proximity score}
    :return: tokens: all the tokens from the text after processing
    """
    if frequent_tokens is None:
        frequent_tokens = set(tokens)
    # n_tokens = len(frequent_tokens)
    if t2t_cos is None:
        t2t_cos = defaultdict(lambda: defaultdict(int))
    for this_ind, this_token in enumerate(tokens):
        if this_token in frequent_tokens:
            for i in range(max(0, this_ind - w), this_ind):
                token_ith = tokens[i]
                if token_ith in frequent_tokens:
                    distance = i - this_ind
                    prox_score = proximity_func(distance)
                    # token ith to current token
                    t2t_cos[this_token][token_ith] += prox_score
                    # current token to token ith
                    t2t_cos[token_ith][this_token] += prox_score
    return t2t_cos


def get_binom_scores(relevant_tokens_score,
                     entity_occurs,
                     n_all_tokens,
                     all_tokens_counter,
                     threshold,
                     w,
                     **kwargs):
    """
    Get binomial z scores for relevant tokens.

    :param relevant_tokens_score: {token: score} scores of individual tokens.
    Score can either reflect the proximity to the entity or simply the number
    of occurrences.
    :param all_tokens: list of all tokens
    :param relevant_tokens_count: number of relevant tokens altogether
    :return: {token: z_score} dictionary
    """
    p_co = (2*entity_occurs*w / n_all_tokens
            if n_all_tokens > 2*entity_occurs*w
            else 1)
    assert 0 < p_co <= 1
    co_tokens = dict()
    for relevant_token in relevant_tokens_score:
        score = _calculate_z_score(all_tokens_counter, relevant_token,
                                   p_co, relevant_tokens_score)
        if score >= threshold:
            co_tokens[relevant_token] = score
    return co_tokens


def get_dice_scores(relevant_tokens_score,
                    entity_occurs,
                    n_all_tokens,
                    all_tokens_counter,
                    threshold,
                    *args,
                    **kwargs):
    """
    Get dice scores for cooccurrent tokens.

    :param relevant_tokens_score: {token: score} scores of individual tokens.
    Score can either reflect the proximity to the entity or simply the number
    of occurrences.
    :param all_tokens: list of all tokens
    :param threshold: a threshold on dice score
    :return: {token: dice_score} dictionary
    """
    co_tokens = dict()
    for relevant_token in relevant_tokens_score:
        co_score = (relevant_tokens_score[relevant_token] /
                    (entity_occurs + all_tokens_counter[relevant_token]))
        if co_score >= threshold:
            co_tokens[relevant_token] = co_score
    return co_tokens


def get_unbiased_dice_scores(relevant_tokens_score,
                             entity_occurs,
                             n_all_tokens,
                             all_tokens_counter,
                             threshold,
                             w,
                             **kwargs):
    """
    Get unbiased dice scores for cooccurrent tokens. Take expected cooccurrence
    count into account. expected_co_count = B * p_co,
    score = 2*(AB - expected_co_count) / (A + B)

    :param relevant_tokens_score: {token: score} scores of individual tokens.
    Score can either reflect the proximity to the entity or simply the number
    of occurrences.
    :param entity_occurs: integer: number of times entity occurs
    :param n_all_tokens: total number of tokens
    :param all_tokens_counter: counter of all tokens
    :param threshold: a threshold on dice score
    :param w: size of the window
    :return: {token: unbiased_dice_score} dictionary
    """
    co_tokens = dict()
    for relevant_token in relevant_tokens_score:
        A = entity_occurs
        B = all_tokens_counter[relevant_token]
        AB = relevant_tokens_score[relevant_token]
        exp_co_count = 2*A*B*w / n_all_tokens # len(all_tokens)
        co_score = (AB - exp_co_count) / (A + B)
        if co_score >= threshold:
            co_tokens[relevant_token] = co_score
    return co_tokens


def dict2rcd(t2t_dict, t2i):
    """
    Transform dict of dict into sparse matrix.
    :param t2t_dict: dict of dicts
    :param t2i: token to index dictionary
    :return: rows, cols, data lists
    """
    rows = []
    cols = []
    data = []
    if t2t_dict:
        for k0, d0 in t2t_dict.items():
            k0_ind = t2i[k0]
            for k1, v in d0.items():
                rows.append(k0_ind)
                cols.append(t2i[k1])
                data.append(v)
        # ans = scipy.sparse.csr_matrix((data, (rows, cols)),
        #                               shape=(len(t2i), len(t2i)))
        # return ans
    # else:
    #     return scipy.sparse.csr_matrix((len(t2i), len(t2i)))
    return rows, cols, data


def get_co_dict(t2t_prox, token2ind, ind2token, token):
    """
    Get cooc dict from sparse matrix for the specified token.
    :param t2t_prox: sparse cooc matrix
    :param token2ind: token 2 index dict
    :param ind2token: index 2 token dict
    :param token: the token
    :return: dict of cooc for the token
    """
    token_ind = token2ind[token]
    token_row = t2t_prox.getrow(token_ind).tocoo()
    cols = token_row.col
    data = token_row.data
    ans = {
        ind2token[cols_i]: data_i for cols_i, data_i in zip(cols, data)
    }
    # for col in token_row.nonzero()[1]:
    #     ans[ind2token[col]] = token_row[0, col]
    return ans


def get_co(texts,
           w,
           method='binom',
           proximity_func=lambda x: 1,
           threshold=0,
           forms_dict=None,
           entity=None,
           min_tf=None,
           max_df=None,
           dict_size_limit=5000,
           **kwargs):
    assert len(texts) and not isinstance(texts, str)
    cache = functools.lru_cache()
    proximity_func = cache(proximity_func)
    texts_tokens, all_tokens_counter, df_tokens = get_tokens_and_counts(
        texts=texts, forms_dict=forms_dict
    )
    # filter out rare tokens if min term freq is provided
    if min_tf is not None:
        tokens_counter = {
            token: freq
            for token, freq in all_tokens_counter.items()
            if freq >= min_tf
        }
    else:
        tokens_counter = dict(all_tokens_counter)
    # filter out frequent token if max df is provided
    if max_df is not None:
        df_limit = len(texts)*max_df
        tokens_counter = {
            token: freq
            for token, freq in tokens_counter.items()
            if freq <= df_limit
        }
    else:
        tokens_counter = dict(tokens_counter)
    module_logger.info('Number of tokens: {}'.format(len(tokens_counter)))
    tokens_set = set(tokens_counter)

    t2t_prox = scipy.sparse.csr_matrix((len(tokens_set), len(tokens_set)))
    token2ind = {token: i for i, token in enumerate(tokens_set)}
    t2t_cos = defaultdict(lambda: defaultdict(float))
    # accumulate token to token proximities over all tokenized texts
    for i, tokens in enumerate(texts_tokens):
        t2t_cos = get_t2t_proximities(tokens, w, proximity_func, tokens_set,
                                      t2t_cos=t2t_cos)
        if len(t2t_cos) > dict_size_limit:
            rows, cols, data = dict2rcd(t2t_cos, token2ind)
            t2t_prox += scipy.sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(len(token2ind), len(token2ind))
            )
            t2t_cos.clear()
        # if len(t2t_cos) > dict_size_limit:
        #     t2t_text_sparse = dict2csr(t2t_cos, token2ind)
        #     t2t_prox += t2t_text_sparse

        if i % round(len(texts_tokens)/10) == 0:
            module_logger.info('{} out of {} done'.format(i, len(texts_tokens)))
    else:
        rows, cols, data = dict2rcd(t2t_cos, token2ind)
        t2t_prox += scipy.sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(token2ind), len(token2ind))
        )
        t2t_cos.clear()
        # t2t_text_sparse = dict2csr(t2t_cos, token2ind)
        # t2t_prox += t2t_text_sparse
        module_logger.info('All done')

    n_all_tokens = sum(all_tokens_counter.values())
    if method == 'binom':
        func_method = get_binom_scores
        params = [n_all_tokens, all_tokens_counter, threshold, w]
    elif method == 'dice':
        func_method = get_dice_scores
        params = [n_all_tokens, all_tokens_counter, threshold]
    elif method == 'unbiased_dice':
        func_method = get_unbiased_dice_scores
        params = [n_all_tokens, all_tokens_counter, threshold, w]
    else:
        raise Exception('Method {} is not supported'.format(method))

    rows = []
    cols = []
    data = []
    ind2token = {ind: token for token, ind in token2ind.items()}
    for token, token_ind in token2ind.items():
        token_cos_dict = get_co_dict(t2t_prox, token2ind, ind2token, token)
        token_params = [token_cos_dict, all_tokens_counter[token]] + params
        co_scores_dict = func_method(*token_params, **kwargs)
        for k1, v in co_scores_dict.items():
            rows.append(token_ind)
            cols.append(token2ind[k1])
            data.append(v)
        co_scores_dict.clear()
        token_cos_dict.clear()
    co_scores = scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(tokens_set), len(tokens_set)))
    return co_scores, token2ind, ind2token


def get_tokens_and_counts(texts, forms_dict=None):
    texts_tokens = []
    all_tokens = []
    df_tokens = []
    # tokenize and count tokens
    for text in texts:
        if forms_dict:
            for form in forms_dict.keys():
                subed_text = re.sub(form, forms_dict[form], text)
        else:
            subed_text = text
        tokens = word_tokenize(subed_text)
        texts_tokens.append(tokens)
        all_tokens += tokens
        df_tokens += set(tokens)
    all_tokens_counter = Counter(all_tokens)
    df_tokens = Counter(df_tokens)
    return texts_tokens, all_tokens_counter, df_tokens


def get_co_text_averaged(texts,
                         w,
                         method='binom',
                         proximity_func=lambda x: 1,
                         threshold=0,
                         forms_dict=None,
                         entity=None,
                         **kwargs):
    assert len(texts) and not isinstance(texts, str)
    if method == 'binom':
        func_method = get_binom_scores
    elif method == 'dice':
        func_method = get_dice_scores
    elif method == 'unbiased_dice':
        func_method = get_unbiased_dice_scores
    else:
        raise Exception('Method {} is not supported'.format(method))
    co_scores = defaultdict(lambda: defaultdict(list))
    for i in range(len(texts)):
        if forms_dict:
            for form in forms_dict.keys():
                subed_text = re.sub(form, forms_dict[form], texts[i])
        else:
            subed_text = texts[i]
        t2t_cos_text, all_new_tokens = get_t2t_proximities(
            subed_text, w, proximity_func,
        )
        new_tokens_counter = Counter(all_new_tokens)
        for token in t2t_cos_text:
            token_params = [
                t2t_cos_text[token], new_tokens_counter[token],
                all_new_tokens, new_tokens_counter, threshold, w
            ]
            res = func_method(*token_params, **kwargs)
            for token2 in res:
                co_scores[token][token2].append(res[token2])
    for token1 in co_scores:
        for token2 in co_scores[token1]:
            co_scores[token1][token2] = np.mean(co_scores[token1][token2])

    return co_scores
