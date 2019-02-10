import os
import re
import cProfile
from collections import Counter
import math

import wsid.preprocess as preproc
import doc_to_text.doc_to_text as doc2text
from wsid.cooccurrence import *
from wsid.cooccurrence import get_tokens_and_counts

testdata_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')


def get_co_tokens(cos, t2i, i2t, target):
    t_id = t2i[target]
    co_terms = {i2t[x]: cos[t_id, x] for x in cos.getrow(t_id).indices}
    return co_terms


class TestCooccurrence:
    def setUp(self):
        self.text = ' A '*15 + ' c '*20000 + (' A B ' + ' c '*50)*10 + \
                    (' A ' + ' c '*10 + ' B ' + ' c '*50)*20 + ' B '*30 + \
                    ' c'*8200
        self.sample_text = doc2text.doc_to_text(
            os.path.join(testdata_path, 'AppBody-Sample-English.docx'))
        self.cinnamon_text = doc2text.doc_to_text(
            os.path.join(testdata_path, 'cinnamon.doc'))
        self.python_text = doc2text.doc_to_text(
            os.path.join(testdata_path, 'python_text.txt'))

    def test_get_co(self):
        entity = 'title'
        cos, t2i, i2t = get_co(
            [self.sample_text, self.sample_text, self.sample_text],
            20
        )
        nf_text = preproc.eliminate_shortwords(
            preproc.eliminate_stopwords(self.sample_text)
        )
        nf_cos, nft2i, nfi2t = get_co([nf_text] * 3, 20)
        fn_text = preproc.eliminate_stopwords(
            preproc.eliminate_shortwords(self.sample_text)
        )
        fn_cos, fnt2i, fni2t = get_co([fn_text] * 5, 20)
        print('Sample english co terms: {}'.format(fn_cos))
        assert 'skilled' in t2i
        assert 'skilled' in nft2i
        assert 'skilled' in fnt2i

    def test_relevant_words_proximity(self):
        entity = 'powder'
        w = 20
        cinnamon_text = preproc.eliminate_shortwords(
            preproc.eliminate_symbols(self.cinnamon_text.lower())
        )
        texts_tokens_iter, token2ind, all_tokens_counter, len_texts = \
            texts2tokens(texts_or_path=[cinnamon_text])
        relevant_words = \
            get_t2t_proximities(
                next(texts_tokens_iter), token2ind, w,
                # proximity_func=lambda x: (w - abs(x) + 0.5) * 2 / w,
                return_dict=True
            )
        relevant_words = relevant_words[entity]
        assert len(relevant_words) <= 2*w*cinnamon_text.count(entity), (len(relevant_words), cinnamon_text.count(entity))

    def test_relevant_words_symmetry(self):
        entity = 'title'
        sample_text = preproc.eliminate_shortwords(
            preproc.eliminate_symbols(self.sample_text.lower())
        )
        w = 20
        texts_tokens_iter, token2ind, all_tokens_counter, len_texts = \
            texts2tokens(texts_or_path=[sample_text])
        relevant_words_score = \
            get_t2t_proximities(
                next(texts_tokens_iter), token2ind, w,
                proximity_func=lambda x: (w - abs(x) + 0.5) * 2 / w,
                return_dict=True
            )
        for rel_word in relevant_words_score[entity]:
            assert (relevant_words_score[entity][rel_word] ==
                    relevant_words_score[rel_word][entity])

    def test_dice_co(self):
        entity = 'powder'
        w = 20
        cos, t2i, i2t = get_co([self.cinnamon_text], w,
                               method='dice',
                               threshold=0.1)
        co_terms = get_co_tokens(cos, t2i, i2t, entity)
        vals = set(co_terms.values())
        assert all(0.099 <= x <= 1 for x in vals), [x
                                                  for x in vals
                                                  if not 0.099 <= x <= 1]

    def test_unbiased_dice_co_limits(self):
        th = 0.01
        w = 20
        cos, t2i, i2t = get_co(
            [self.cinnamon_text], w,
            method='unbiased_dice',
            threshold=th
        )
        assert all(th <= x <= 1 for x in cos.data)

    def test_unbiased_dice_co_symmetry(self):
        th = 0.0
        entity = r'powder'
        cinnamon_text = preproc.eliminate_shortwords(
            preproc.eliminate_symbols(self.cinnamon_text.lower())
        )
        w = 20
        cos, t2i, i2t = get_co(
            [cinnamon_text], w,
            method='unbiased_dice',
            threshold=th
        )
        co_terms_e = get_co_tokens(cos, t2i, i2t, entity)
        for term in co_terms_e:
            # print(entity, term, end=' ')
            # print(term_co[entity], co_terms[term])
            co_terms_t = get_co_tokens(cos, t2i, i2t, term)
            assert math.isclose(co_terms_e[term],
                                co_terms_t[entity])

    def test_unbiased_dice_co_symmetry2(self):
        th = 0.01
        entity = r'the'
        cinnamon_text = preproc.eliminate_shortwords(
            preproc.eliminate_symbols(self.cinnamon_text.lower())
        )
        w = 20
        cos, t2i, i2t = get_co(
            [cinnamon_text], w,
            method='unbiased_dice',
            threshold=th
        )
        co_terms_e = get_co_tokens(cos, t2i, i2t, entity)
        for term in co_terms_e:
            # print(entity, term, end=' ')
            # print(term_co[entity], co_terms[term])
            co_terms_t = get_co_tokens(cos, t2i, i2t, term)
            assert math.isclose(co_terms_e[term],
                                co_terms_t[entity])

    def test_unbiased_dice_co_term_frequency_change(self):
        entity = 'powder'
        w = 20
        a_freq = self.cinnamon_text.count(' a ')
        has_freq = self.cinnamon_text.count(' has ')
        cos, t2i, i2t = get_co(
            [self.cinnamon_text], w,
            method='unbiased_dice',
            threshold=-1
        )
        co_terms = get_co_tokens(cos, t2i, i2t, entity)
        cos, t2i, i2t = get_co(
            [self.cinnamon_text, 'a ' * a_freq * 9], w,
            method='unbiased_dice',
            threshold=0
        )
        co_terms_10as = get_co_tokens(cos, t2i, i2t, entity)
        cos, t2i, i2t = get_co(
            [self.cinnamon_text, 'has ' * has_freq * 9], w,
            method='unbiased_dice',
            threshold=-1
        )
        co_terms_10has = get_co_tokens(cos, t2i, i2t, entity)
        assert 'a' not in co_terms_10as
        assert 'has' in co_terms_10has and 'has' in co_terms
        assert co_terms_10has['has'] / co_terms['has'] < 10

    def test_unbiased_dice_co_triplicate_docs(self):
        entity = 'powder'
        cinnamon_text = preproc.eliminate_shortwords(
            preproc.eliminate_symbols(self.cinnamon_text.lower())
        )
        w = 20
        cos, t2i, i2t = get_co(
            [cinnamon_text], w,
            method='unbiased_dice',
            threshold=0.0
        )
        co_terms1 = get_co_tokens(cos, t2i, i2t, entity)
        cos, t2i, i2t = get_co(
            [cinnamon_text]*10, w,
            method='unbiased_dice',
            threshold=0.0
        )
        co_terms10 = get_co_tokens(cos, t2i, i2t, entity)
        ratios = []
        for term in co_terms1:
            print(term, co_terms10[term], co_terms1[term],
                  co_terms10[term] / co_terms1[term])
            ratios.append(co_terms10[term] / co_terms1[term])
            assert 0.8 < co_terms10[term] / co_terms1[term] < 1.2
