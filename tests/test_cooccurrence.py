import os
import re
import cProfile
from collections import Counter
import math

import disambiguation.preprocess as preproc
import doc_to_text.doc_to_text as doc2text
from disambiguation.cooccurrence import *

testdata_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')


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

    def test_calculate(self):
        words = self.text.split()
        words_counter = Counter(words)
        relevant_words, words = \
            get_t2t_proximities(self.text, 10, proximity_func=lambda x: 1)
        relevant_words = relevant_words['A']
        entity_occurs = len([x for x in words if x == 'A'])
        p_co = sum(relevant_words.values()) / (len(words) - entity_occurs)
        score = _calculate_z_score(words_counter, 'B', p_co, relevant_words)
        print('A and B cooc score = ', score)
        assert score > 5
        assert score < 10

    def test_get_co(self):
        entity = 'title'
        co_terms = get_co(
            [self.sample_text, self.sample_text, self.sample_text],
            20
        )[entity]
        nf_text = preproc.eliminate_shortwords(
            preproc.eliminate_stopwords(self.sample_text)
        )
        nf_co_terms = get_co([nf_text] * 3, 20)
        fn_text = preproc.eliminate_stopwords(
            preproc.eliminate_shortwords(self.sample_text)
        )
        fn_co_terms = get_co([fn_text] * 5, 20)[entity]
        print('Sample english co terms: {}'.format(fn_co_terms))
        assert 'skilled' in co_terms
        assert 'skilled' in nf_co_terms
        assert 'skilled' in fn_co_terms

    def test_relevant_words_proximity(self):
        entity = 'powder'
        w = 20
        cinnamon_text = preproc.eliminate_shortwords(
            preproc.eliminate_symbols(self.cinnamon_text.lower())
        )
        relevant_words, words = \
            get_t2t_proximities(
                cinnamon_text, w,
                proximity_func=lambda x: (w - abs(x) + 0.5) * 2 / w
            )
        relevant_words = relevant_words[entity]
        assert len(relevant_words) >= 2*w + cinnamon_text.count(entity)
        print(relevant_words.values())
        print(relevant_words.items())
        for word, score in relevant_words.items():
            if not 0 < score <= w*words.count(word):
                print(score, word, words.count(word))
                assert 0

    def test_relevant_words_symmetry1(self):
        entity = 'the'
        cinnamon_text = preproc.eliminate_shortwords(
            preproc.eliminate_symbols(self.cinnamon_text.lower())
        )
        w = 20
        relevant_words_score, words = \
            get_t2t_proximities(
                cinnamon_text, w,
                proximity_func=lambda x: (w - abs(x) + 0.5) * 2 / w
            )
        for rel_word in relevant_words_score[entity]:
            assert (relevant_words_score[entity][rel_word] ==
                    relevant_words_score[rel_word][entity])

    def test_relevant_words_symmetry2(self):
        entity = 'title'
        sample_text = preproc.eliminate_shortwords(
            preproc.eliminate_symbols(self.sample_text.lower())
        )
        w = 20
        relevant_words_score, words = \
            get_t2t_proximities(
                sample_text, w,
                proximity_func=lambda x: (w - abs(x) + 0.5) * 2 / w
            )
        for rel_word in relevant_words_score[entity]:
            assert (relevant_words_score[entity][rel_word] ==
                    relevant_words_score[rel_word][entity])


    def test_binom_co_with_proximity(self):
        entity = 'powder'
        w = 20
        co_terms = get_co([self.cinnamon_text], w,
                          with_informativeness=False)[entity]
        co_terms_prox = get_co(
            [self.cinnamon_text], w,
            proximity_func=lambda x: (w - abs(x) + .5) * 2 / w,
            with_informativeness=False
        )[entity]
        print('Co terms: ', co_terms)
        print('Co terms proximity: ', co_terms_prox)
        assert len(co_terms_prox) < len(co_terms)


    def test_dice_co(self):
        entity = 'powder'
        w = 20
        co_terms = get_co(
            [self.cinnamon_text], w,
            method='dice',
            threshold=0.1
        )[entity]
        print('Co terms: ', co_terms)
        assert all(0.1 <= x <= 1 for x in co_terms.values())

    def test_unbiased_dice_co_limits(self):
        th = 0.01
        entity = 'powder'
        w = 20
        co_terms = get_co(
            [self.cinnamon_text], w,
            method='unbiased_dice',
            threshold=th
        )[entity]
        print(co_terms)
        print(max(co_terms.values()))
        assert all(th <= x <= 1 for x in co_terms.values())

    def test_unbiased_dice_co_symmetry(self):
        th = 0.0
        entity = r'powder'
        cinnamon_text = preproc.eliminate_shortwords(
            preproc.eliminate_symbols(self.cinnamon_text.lower())
        )
        w = 20
        co_terms = get_co(
            [cinnamon_text], w,
            method='unbiased_dice',
            threshold=th
        )
        for term in co_terms[entity]:
            # print(entity, term, end=' ')
            # print(term_co[entity], co_terms[term])
            assert math.isclose(co_terms[entity][term],
                                co_terms[term][entity])

    def test_unbiased_dice_co_symmetry2(self):
        th = 0.01
        entity = r'the'
        cinnamon_text = preproc.eliminate_shortwords(
            preproc.eliminate_symbols(self.cinnamon_text.lower())
        )
        w = 20
        co_terms = get_co(
            [cinnamon_text], w,
            method='unbiased_dice',
            threshold=th
        )
        for term in co_terms[entity]:
            # print(entity, term, end=' ')
            # print(term_co[entity], co_terms[term])
            assert math.isclose(co_terms[entity][term],
                                co_terms[term][entity])

    def test_unbiased_dice_co_term_frequency_change(self):
        entity = 'powder'
        w = 20
        a_freq = self.cinnamon_text.count(' a ')
        has_freq = self.cinnamon_text.count(' has ')
        co_terms = get_co(
            [self.cinnamon_text], w,
            method='unbiased_dice',
            threshold=-1
        )[entity]
        co_terms_10as = get_co(
            [self.cinnamon_text, 'a ' * a_freq * 9], w,
            method='unbiased_dice',
            threshold=0
        )[entity]
        co_terms_10has = get_co(
            [self.cinnamon_text, 'has ' * has_freq * 9], w,
            method='unbiased_dice',
            threshold=-1
        )[entity]
        assert 'a' not in co_terms_10as
        print(co_terms_10has['has'], co_terms['has'])
        assert co_terms_10has['has'] / co_terms['has'] < 10

    def test_unbiased_dice_co_co_frequency_change(self):
        # How do I test this???
        pass

    def test_unbiased_dice_co_triplicate_docs(self):
        entity = 'powder'
        cinnamon_text = preproc.eliminate_shortwords(
            preproc.eliminate_symbols(self.cinnamon_text.lower())
        )
        w = 20
        co_terms1 = get_co(
            [cinnamon_text], w,
            method='unbiased_dice',
            threshold=0.0
        )[entity]
        co_terms10 = get_co(
            [cinnamon_text]*10, w,
            method='unbiased_dice',
            threshold=0.0
        )[entity]
        ratios = []
        for term in co_terms1:
            print(term, co_terms10[term], co_terms1[term],
                  co_terms10[term] / co_terms1[term])
            ratios.append(co_terms10[term] / co_terms1[term])
            assert 0.8 < co_terms10[term] / co_terms1[term] < 1.2
