import re
from collections import Counter, defaultdict
import numpy as np

from nltk import word_tokenize

__all__ = ['_calculate_z_score',
           'get_co',
           'get_relevant_tokens',
           'get_binom_scores']


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


def get_relevant_tokens(text,
                        w,
                        proximity_func=lambda x: 1,
                        entity=None):
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
    tokens = word_tokenize(text)

    if entity is None:
        t2t_cos = defaultdict(lambda: defaultdict(int))
        for this_ind, this_token in enumerate(tokens):
            for i in range(max(0, this_ind - w), this_ind):
                token_ith = tokens[i]
                distance = i - this_ind
                # token ith to current token
                t2t_cos[this_token][token_ith] += proximity_func(distance)
                # current token to token ith
                t2t_cos[token_ith][this_token] += proximity_func(distance)
        return t2t_cos, tokens
    else:
        cos = defaultdict(int)
        for this_ind, this_token in enumerate(tokens):
            if this_token == entity:
                for i in range(max(0, this_ind - w),
                               min(this_ind + w, len(tokens))):
                    if i == this_ind: continue
                    token_ith = tokens[i]
                    distance = abs(i - this_ind)
                    # token ith to current token
                    cos[token_ith] += proximity_func(distance)
        return cos, tokens


def get_binom_scores(relevant_tokens_score,
                     entity_occurs,
                     all_tokens,
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
    p_co = (2*entity_occurs*w / len(all_tokens)
            if len(all_tokens) > 2*entity_occurs*w
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
                    all_tokens,
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
                             all_tokens,
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
    :param all_tokens: list of all tokens
    :param relevant_tokens_count: number of relevant tokens altogether
    :param threshold: a threshold on dice score
    :return: {token: unbiased_dice_score} dictionary
    """
    co_tokens = dict()
    for relevant_token in relevant_tokens_score:
        A = entity_occurs
        B = all_tokens_counter[relevant_token]
        AB = relevant_tokens_score[relevant_token]
        exp_co_count = 2*A*B*w / len(all_tokens)
        co_score = (AB - exp_co_count) / (A + B)
        if co_score >= threshold:
            co_tokens[relevant_token] = co_score
    return co_tokens


def get_co(texts,
           w,
           method='binom',
           proximity_func=lambda x: 1,
           threshold=0,
           forms_dict=None,
           entity=None,
           **kwargs):
    assert len(texts) and not isinstance(texts, str)
    all_tokens = []
    t2t_cos = defaultdict(lambda: defaultdict(int))
    cos = defaultdict(int)
    for i in range(len(texts)):
        if forms_dict:
            for form in forms_dict.keys():
                subed_text = re.sub(form, forms_dict[form], texts[i])
        else:
            subed_text = texts[i]
        if entity is None:
            t2t_cos_text, all_new_tokens = \
                get_relevant_tokens(
                    subed_text, w, proximity_func,
                )
            all_tokens += all_new_tokens
            for token1 in t2t_cos_text:
                for token2 in t2t_cos_text[token1]:
                    t2t_cos[token1][token2] += t2t_cos_text[token1][token2]
        else:
            cos_text, all_new_tokens = \
                get_relevant_tokens(
                    subed_text, w, proximity_func,
                    entity=entity
                )
            all_tokens += all_new_tokens
            for token in cos_text:
                cos[token] += cos_text[token]

    all_tokens_counter = Counter(all_tokens)
    if method == 'binom':
        func_method = get_binom_scores
        params = [all_tokens, all_tokens_counter, threshold, w]
    elif method == 'dice':
        func_method = get_dice_scores
        params = [all_tokens, all_tokens_counter, threshold]
    elif method == 'unbiased_dice':
        func_method = get_unbiased_dice_scores
        params = [all_tokens, all_tokens_counter, threshold, w]
    else:
        raise Exception('Method {} is not supported'.format(method))
    co_scores = dict()
    if entity is None:
        for token in t2t_cos:
            token_params = [t2t_cos[token], all_tokens_counter[token]] + params
            co_scores[token] = func_method(*token_params, **kwargs)
    else:
        token_params = [cos, all_tokens_counter[entity]] + params
        kwargs['entity'] = entity
        co_scores = func_method(*token_params, **kwargs)

    return co_scores


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
        t2t_cos_text, all_new_tokens = get_relevant_tokens(
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
