import functools
import logging
import os
import pickle
import re
import uuid
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse
import array
from decouple import config
from nltk import word_tokenize
from diskcache import FanoutCache

__all__ = ['get_co',
           'get_t2t_proximities',
           'texts2tokens']

module_logger = logging.getLogger(__name__)


class IncrementalSparseMatrix(object):
    def __init__(self, shape, dtype):
        if dtype is np.int32:
            type_flag = 'i'
        elif dtype is np.int64:
            type_flag = 'l'
        elif dtype is np.float32:
            type_flag = 'f'
        elif dtype is np.float64:
            type_flag = 'd'
        else:
            raise Exception(f'type {dtype} not supported.')
        self.dtype = dtype
        self.shape = shape
        self.type_flag = type_flag

        self.rows = array.array('i')
        self.cols = array.array('i')
        self.data = array.array(type_flag)
        self.matrix = scipy.sparse.coo_matrix(shape, dtype=dtype)

    def append(self, i, j, v):
        m, n = self.shape
        if (i >= m or j >= n):
            raise Exception('Index out of bounds')
        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)
        self._check_buffer()

    def extend(self, is_, js, vs):
        assert len(is_) == len(js)
        assert len(vs) == len(js)
        self.rows.extend(is_)
        self.cols.extend(js)
        self.data.extend(vs)
        self._check_buffer()

    def _flush(self):
        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)
        self.matrix += scipy.sparse.coo_matrix((data, (rows, cols)),
                                               shape=self.shape)
        self.matrix.sum_duplicates()
        self.matrix.eliminate_zeros()
        self.rows = array.array('i')
        self.cols = array.array('i')
        self.data = array.array(self.type_flag)
        module_logger.debug('Matrix flushed.')

    def _check_buffer(self, buffer_limit=config('RAM_LIMIT_GB', cast=int, default=1)*(10**7)):
        if len(self.data) > buffer_limit:
            self._flush()

    def tocoo(self):
        self._flush()
        return self.matrix

    def tocsr(self):
        self._flush()
        return self.matrix.tocsr()

    def __len__(self):
        return len(self.data)


def load_cos(cos_path=config('COS_STORE_PATH', default=None),
             t2i_path=config('TOKEN2IND_PATH', default=None),
             i2t_path=config('IND2TOKEN_PATH', default=None)):
    cos = scipy.sparse.load_npz(cos_path)
    with open(t2i_path, 'rb') as f:
        t2i = pickle.load(f)
    with open(i2t_path, 'rb') as f:
        i2t = pickle.load(f)
    return cos, t2i, i2t


def save_cos(cos, t2i, i2t,
             cos_path=config('COS_STORE_PATH', default=None),
             t2i_path=config('TOKEN2IND_PATH', default=None),
             i2t_path=config('IND2TOKEN_PATH', default=None)):
    scipy.sparse.save_npz(cos_path, cos)
    with open(t2i_path, 'wb') as f:
        pickle.dump(t2i, f)
    with open(i2t_path, 'wb') as f:
        pickle.dump(i2t, f)


def get_t2t_proximities(tokens,
                        token2ind,
                        w,
                        proximity_func=lambda x: 1,
                        return_dict=False):
    """
    Get the relevant tokens in the sense of proximity. Extracts all the tokens
    that are within the window w from the entity and calculates their proximity
    score.

    :param bool return_dict: if return should be dict.
    :param dict[str, int] token2ind:
    :param list[str] tokens:
    :param int w: window size.
    :param (int) -> float proximity_func: proximity function.
    :rtype: (list[int], list[int], list[float]) or dict[str, dict[str, int]]
    """
    if not return_dict:
        ans_rows = []
        ans_cols = []
        ans_data = []
    else:
        t2t_cos = defaultdict(lambda: defaultdict(int))

    for this_ind, this_token in enumerate(tokens):
        if this_token in token2ind:
            this_token_vocab_ind = token2ind[this_token]
            for i in range(max(0, this_ind - w), this_ind):
                ith_token = tokens[i]
                if ith_token in token2ind:
                    distance = i - this_ind
                    prox_score = proximity_func(distance)
                    if not return_dict:
                        ith_token_vocab_ind = token2ind[ith_token]
                        # current token to token ith
                        ans_rows.append(this_token_vocab_ind)
                        ans_cols.append(ith_token_vocab_ind)
                        ans_data.append(prox_score)
                        # ith token to current token
                        ans_cols.append(this_token_vocab_ind)
                        ans_rows.append(ith_token_vocab_ind)
                        ans_data.append(prox_score)
                    else:
                        # token ith to current token
                        t2t_cos[this_token][ith_token] += prox_score
                        # current token to token ith
                        t2t_cos[ith_token][this_token] += prox_score
    if not return_dict:
        return ans_rows, ans_cols, ans_data
    else:
        return t2t_cos


def get_dice_scores(t2t_prox,
                    token2ind,
                    ind2token,
                    entity_str,
                    n_all_tokens,
                    all_tokens_counter,
                    threshold,
                    w,
                    **kwargs):
    """
    Get unbiased dice scores for cooccurrent tokens. Take expected cooccurrence
    count into account. expected_co_count = B * p_co,
    score = 2*(AB - expected_co_count) / (A + B)

    :param dict[str, int] token2ind:
    :param dict[int, str] ind2token:
    :param scipy.sparse.csr_matrix t2t_prox:
    :param str entity_str: number of times entity occurs
    :param int n_all_tokens: total number of tokens
    :param collections.Counter all_tokens_counter: counter of all tokens
    :param float threshold: a threshold on dice score
    :param int w: size of the window
    :return: rows, cols and values of co-occurrences
    :rtype: (list[int], list[int], list[float])
    """
    e_ind = token2ind[entity_str]
    e_row = t2t_prox.getrow(e_ind).tocoo()
    cols = e_row.col
    data = e_row.data
    entity_occurs = all_tokens_counter[entity_str]
    ans_rows = []
    ans_cols = []
    ans_data = []
    for relevant_token_ind, AB in zip(cols, data):
        A = entity_occurs
        relevant_token_str = ind2token[relevant_token_ind]
        B = all_tokens_counter[relevant_token_str]
        co_score = AB / (A + B)
        if co_score >= threshold:
            ans_rows.append(e_ind)
            ans_cols.append(relevant_token_ind)
            ans_data.append(co_score)
    return ans_rows, ans_cols, ans_data


def get_unbiased_dice_scores(t2t_prox,
                             token2ind,
                             ind2token,
                             entity_str,
                             n_all_tokens,
                             all_tokens_counter,
                             threshold,
                             w,
                             **kwargs):
    """
    Get unbiased dice scores for cooccurrent tokens. Take expected cooccurrence
    count into account. expected_co_count = B * p_co,
    score = 2*(AB - expected_co_count) / (A + B)

    :param dict[str, int] token2ind:
    :param dict[int, str] ind2token:
    :param scipy.sparse.csr_matrix t2t_prox:
    :param str entity_str: number of times entity occurs
    :param int n_all_tokens: total number of tokens
    :param collections.Counter all_tokens_counter: counter of all tokens
    :param float threshold: a threshold on dice score
    :param int w: size of the window
    :return: rows, cols and values of co-occurrences
    :rtype: (list[int], list[int], list[float])
    """
    e_ind = token2ind[entity_str]
    e_row = t2t_prox.getrow(e_ind).tocoo()
    cols = e_row.col
    data = e_row.data
    entity_occurs = all_tokens_counter[entity_str]
    ans_rows = []
    ans_cols = []
    ans_data = []
    for relevant_token_ind, AB in zip(cols, data):
        A = entity_occurs
        relevant_token_str = ind2token[relevant_token_ind]
        B = all_tokens_counter[relevant_token_str]
        # AB = data[relevant_token_ind]
        exp_co_count = 2*A*B*w / n_all_tokens # len(all_tokens)
        co_score = (AB - exp_co_count) / (A + B)
        if co_score >= threshold:
            ans_rows.append(e_ind)
            ans_cols.append(relevant_token_ind)
            ans_data.append(co_score)
    return ans_rows, ans_cols, ans_data


storage_folder_co = config('STORAGE_FOLDER', default='/tmp/diskcache')
cache_co = FanoutCache(storage_folder_co, eviction_policy='none')
@cache_co.memoize(tag='get_co')
def get_co(texts_or_path,
           w,
           input_type='collection',
           method='unbiased_dice',
           proximity='const',
           threshold=0,
           forms_dict=None,
           necessary_tokens=None,
           min_df=None,
           max_df=None,
           tokenize=word_tokenize,
           **kwargs):
    """
    Iterate over texts and compute co-occurrences for each term.

    :param (str) -> list[str] tokenize:
    :param (list[str] or None) necessary_tokens: tokens to be included even
        if the DF scores are outside of (min_df, max_df)
    :param (list[str] or str) texts_or_path: iter of texts or a path to texts
    :param int w: window size
    :param str input_type: "file_path", "folder_path" or "collection".
    :param str method:
    :param str proximity: "linear" or "const"
    :param float threshold:
    :param dict[str, list[str]] or None forms_dict: Alternative labels (synonyms)
    :param int or None min_df: minimum doc freq in abs counts
    :param float or None max_df: maximum doc freq as a ratio of the size of the corpus
    :return: COs matrix, token name to its index in the COs matrix and inverse dict
    :rtype: (scipy.sparse.csr_matrix, dict[str, int], dict[int, str])
    """
    module_logger.info(f'Start get_co function')
    module_logger.info(f'Storage folder: {storage_folder_co}')

    if proximity == 'const':
        proximity_func = lambda x: 1
    elif proximity == 'linear':
        proximity_func = lambda x: (w - abs(x) + 0.5) * 2 / w
    else:
        raise ValueError(f'Proximity "{proximity}" is not supported.')
    cache = functools.lru_cache()
    proximity_func = cache(proximity_func)

    texts_tokens_iter, token2ind, all_tokens_counter, len_texts = \
        texts2tokens(texts_or_path=texts_or_path,
                     input_type=input_type,
                     tokenize=tokenize,
                     forms_dict=forms_dict, min_df=min_df, max_df=max_df)
    if necessary_tokens is not None:
        for token in necessary_tokens:
            if token not in token2ind:
                token2ind[token] = len(token2ind)
    module_logger.info(f'Number of tokens: {sum(all_tokens_counter.values())}')
    module_logger.info(f'Total unique tokens: {len(token2ind)}')

    t2t_prox_constructor = IncrementalSparseMatrix(
        shape=(len(token2ind), len(token2ind)),
        dtype=np.float32)
    # accumulate token to token proximities over all tokenized texts
    _10proc_texts = round(len_texts / 10)
    for i, tokens in enumerate(texts_tokens_iter):
        new_rows, new_cols, new_data = get_t2t_proximities(
            tokens, token2ind, w, proximity_func)
        t2t_prox_constructor.extend(new_rows, new_cols, new_data)
        if _10proc_texts > 0 and i % _10proc_texts == 0:
            s = f'{i+1} out of {len_texts} texts done ' \
                f'({round(i/_10proc_texts)*10}%). ' \
                f'{len(t2t_prox_constructor.matrix.data)} edges.'
            module_logger.info(s)
    else:
        t2t_prox = t2t_prox_constructor.tocsr()
        module_logger.info('All done')

    n_all_tokens = sum(all_tokens_counter.values())
    params = (n_all_tokens, all_tokens_counter, threshold, w)
    if method == 'dice':
        func_method = get_dice_scores
    elif method == 'unbiased_dice':
        func_method = get_unbiased_dice_scores
    else:
        raise Exception('Method {} is not supported'.format(method))

    ind2token = {ind: token for token, ind in token2ind.items()}
    _10proc_tokens = round(len(token2ind) / 10)
    co_scores_constructor = IncrementalSparseMatrix(
        shape=(len(token2ind), len(token2ind)),
        dtype=np.float32)
    for i, (token, token_ind) in enumerate(token2ind.items()):
        token_params = (t2t_prox, token2ind, ind2token, token) + params
        new_rows, new_cols, new_data = func_method(*token_params, **kwargs)
        co_scores_constructor.extend(new_rows, new_cols, new_data)
        if _10proc_tokens > 0 and i % _10proc_tokens == 0:
            s = f'{i + 1} out of {len(token2ind)} tokens done ' \
                f'({round(i / _10proc_tokens)*10}%). ' \
                f'{len(co_scores_constructor.matrix.data)} edges.'
            module_logger.info(s)
    else:
        co_scores = co_scores_constructor.tocsr()
        module_logger.info('All done')
    module_logger.info(f'Total unique tokens: {len(token2ind)}, '
                       f'total COs: {len(co_scores.data)}')
    return co_scores, token2ind, ind2token


def texts2tokens(texts_or_path,
                 input_type='collection',
                 tokenize=word_tokenize,
                 forms_dict=None,
                 min_df=None, max_df=None):
    """

    :param (str) -> list[str] tokenize:
    :param (list[str] or str) texts_or_path: iter of texts or a path to texts
    :param str input_type: "file_path", "folder_path" or "collection".
    :param dict[str, list[str]] or None forms_dict: Alternative labels (synonyms)
    :param int or None min_df: minimum doc freq in abs counts
    :param float or None max_df: maximum doc freq as a ratio of the size of the corpus
    :return: texts_tokens_iter, token2ind, all_tokens_counter
    :rtype: (iter[str], dict[str, int], Counter[str, int], int)
    """
    all_tokens_counter, df_tokens, len_texts, texts_tokens_fn = \
        get_tokens_and_counts(texts_or_path=texts_or_path,
                              tokenize=tokenize,
                              input_type=input_type,
                              forms_dict=forms_dict)
    texts_tokens_iter = iter_texts_tokens(texts_tokens_fn)
    # filter out rare tokens if min term freq is provided
    if min_df is not None:
        df_tokens_filter = {
            token: freq
            for token, freq in df_tokens.items()
            if freq >= min_df
        }
    else:
        df_tokens_filter = dict(df_tokens)
    module_logger.debug(f'min_df = {min_df}, tokens before: {len(df_tokens)}, tokens after = {len(df_tokens_filter)}')
    # filter out frequent token if max df is provided
    if max_df is not None:
        df_limit = len_texts * max_df
        df_tokens_filter = {
            token: freq
            for token, freq in df_tokens_filter.items()
            if freq <= df_limit
        }
    else:
        df_tokens_filter = dict(df_tokens_filter)
    module_logger.debug(f'max_df = {max_df}, tokens after = {len(df_tokens_filter)}')
    tokens_set = set(df_tokens_filter)
    token2ind = {token: i for i, token in enumerate(tokens_set)}
    return texts_tokens_iter, token2ind, all_tokens_counter, len_texts


def iter_texts_tokens(dump_file_name):
    with open(dump_file_name) as f:
        for line in f:
            yield line.split(', ')


storage_folder_tokens_counts = os.path.join(storage_folder_co,
                                            'tokens_and_counts')
cache_tc = FanoutCache(storage_folder_tokens_counts, eviction_policy='none')
@cache_tc.memoize(tag='get_tokens_and_counts')
def get_tokens_and_counts(texts_or_path, tokenize, input_type, forms_dict=None):
    """

    :param (list[str] or str) texts_or_path: iter of texts or a path to texts
    :param (str) -> list[str] tokenize:
    :param str input_type: "file_path", "folder_path" or "collection".
    :param dict[str, list[str]] forms_dict: Alternative labels (synonyms)
    """
    def iter_from_folder(path):
        all_file_names = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if os.path.isfile(os.path.join(path, fname))]
        for file_path in all_file_names:
            with open(file_path, 'rb') as f:
                f_text = f.read()
            yield f_text.decode('ascii', errors='ignore')

    def iter_from_file(path):
        with open(path, 'rb') as f:
            for line in f:
                yield line.decode('ascii', errors='ignore')

    if input_type is not 'collection':
        if input_type is 'file_path':
            texts = iter_from_file(texts_or_path)
        elif input_type is 'folder_path':
            texts = iter_from_folder(texts_or_path)
        else:
            raise ValueError(f'Input type "{input_type}" is not supported.')
    else:
        texts = texts_or_path

    all_tokens_counter = Counter()
    df_tokens = Counter()
    # dump_file = NamedTemporaryFile(mode='w+', delete=False)
    filename = str(uuid.uuid4())
    dump_file_name = os.path.join(storage_folder_co, filename)
    with open(dump_file_name, mode='w+') as dump_file:
        # tokenize and count tokens
        for i, text in enumerate(texts):
            if forms_dict:
                for form in forms_dict.keys():
                    subed_text = re.sub(form, forms_dict[form], text)
            else:
                subed_text = text
            tokens = tokenize(subed_text)
            tokens_s = ', '.join(tokens) + '\n'
            dump_file.write(tokens_s)
            # texts_tokens.append(tokens)
            for t in tokens:
                all_tokens_counter[t] += 1
            for t in set(tokens):
                df_tokens[t] += 1
            # all_tokens += tokens
            # df_tokens += set(tokens)
    # dump_file.close()
    # temp_file_name = dump_file.name
    len_texts = i+1
    return all_tokens_counter, df_tokens, len_texts, dump_file_name
