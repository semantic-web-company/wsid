# __author__ = 'artreven'
# Use with Nosetests

import os

from nltk import word_tokenize

dirname = os.path.dirname(os.path.realpath(__file__))
datapath = dirname + '/data/'

from wsid.preprocess import *


class TestTokensSubstitute:
    def setUp(self):
        self.text = """
    lorem Ipsum is simply dummy text of the printing and typesetting industry. 
    Lorem Ipsum has been the industry's standard dummy text ever since the 
    1500s, when an unknown printer took a galley of type and scrambled it to 
    make a type specimen book. It has survived not only five centuries, but 
    also the leap into electronic typesetting, remaining essentially unchanged. 
    It was popularised in the 1960 with the release of Letraset sheets 
    containing Lorem Ipsum passages, and more recently with desktop publishing 
    software like Aldus PageMaker including versions of lorem Ipsum."""
        self.exclude_tokens = ['lorem', 'a', 'an']
        self.tokens_list = word_tokenize(self.text)

    def test_replace_tokens_in_str(self):
        ans, tokens = replace_in_str(text=self.text, tokens=self.exclude_tokens)
        assert set(tokens) == {'lorem', 'a', 'an'}
        assert len(tokens) == 5

    def test_restore_tokens_in_str(self):
        ans, tokens = replace_in_str(text=self.text, tokens=self.exclude_tokens)
        restored = restore_in_str(ans, tokens)
        assert restored == self.text

    def test_replace_tokens_in_list(self):
        ans, tokens = replace_in_list(tokens_list=self.tokens_list,
                                      tokens_to_replace=self.exclude_tokens)
        assert set(tokens) == {'lorem', 'a', 'an'}
        assert len(tokens) == 5

    def test_restore_tokens_in_list(self):
        ans, tokens = replace_in_list(tokens_list=self.tokens_list,
                                      tokens_to_replace=self.exclude_tokens)
        restored = restore_in_list(tokens_list=ans,
                                   tokens_to_restore=tokens)
        assert restored == self.tokens_list

    def test_replace_token_nothing_found(self):
        ans, tokens = replace_in_str(text=self.text, tokens=['abcd'])
        assert not tokens
        assert ans == self.text

    def test_remove_short(self):
        corpus, vocab = preprocess([self.text], [], remove_nonletters=False,
                         remove_shortwords=True,
                         make_lower=False, remove_stopwords=False,
                         stem=False, digits2token=False,
                         min_df=0, max_df=1.)
        ans = corpus[0]
        assert all([len(x) > 2 for x in ans.split()])

    def test_remove_stopwords(self):
        corpus, vocab = preprocess([self.text], [], remove_nonletters=False,
                         remove_shortwords=False,
                         make_lower=False, remove_stopwords=True,
                         stem=False, digits2token=False,
                         min_df=0, max_df=1.)
        ans = corpus[0]
        assert 'the' not in ans

    def test_digits2token(self):
        corpus, vocab = preprocess([self.text], [], remove_nonletters=False,
                         remove_shortwords=False,
                         make_lower=False, remove_stopwords=False,
                         stem=False, digits2token=True,
                         min_df=0, max_df=1.)
        ans = corpus[0]
        assert 'DIGIT' in ans

    def test_lower(self):
        corpus, vocab = preprocess([self.text], ['Lorem'],
                                   remove_nonletters=False,
                                   remove_shortwords=False,
                                   make_lower=True, remove_stopwords=False,
                                   stem=False, digits2token=False,
                                   min_df=0, max_df=1.)
        ans = corpus[0]
        assert ans
        assert 'Lorem' in ans, ans





# class TestCase:
#     def setUp(self):
#         self.greek_english_text = "Το Lorem Ipsum είναι απλά ένα κείμενο χωρίς νόημα για τους επαγγελματίες της" \
#                                   "τυπογραφίας και στοιχειοθεσίας. Το Lorem Ipsum είναι το επαγγελματικό πρότυπο όσον" \
#                                   " αφορά το κείμενο χωρίς νόημα, από τον 15ο αιώνα, όταν ένας ανώνυμος τυπογράφος " \
#                                   "πήρε ένα δοκίμιο και ανακάτεψε τις λέξεις για να δημιουργήσει ένα δείγμα βιβλίου. " \
#                                   "Όχι μόνο επιβίωσε πέντε αιώνες, αλλά κυριάρχησε στην ηλεκτρονική στοιχειοθεσία, " \
#                                   "παραμένοντας με κάθε τρόπο αναλλοίωτο. Έγινε δημοφιλές τη δεκαετία του '60 με την " \
#                                   "έκδοση των δειγμάτων της Letraset όπου περιελάμβαναν αποσπάσματα του Lorem Ipsum, " \
#                                   "και πιο πρόσφατα με το λογισμικό ηλεκτρονικής σελιδοποίησης όπως το Aldus " \
#                                   "PageMaker που περιείχαν εκδοχές του Lorem Ipsum."
#         self.arabic_english_text = '"الشكلي منذ القرن الخامس عشر عندما قامت مطبعة مجهولة برص مجموعة من الأحرف بشكل عشوائي أخذتها من نص، لتكوّن كتيّب بمثابة دليل أو مرجع شكلي لهذه الأحرف. خمسة قرون من الزمن لم تقضي على هذا النص، بل انه حتى صار مستخدماً وبشكله الأصلي في الطباعة والتنضيد الإلكتروني. انتشر بشكل كبير في ستينيّات هذا القرن مع إصدار رقائق "ليتراسيت" (Letraset) البلاستيكية تحوي مقاطع من هذا النص، وعاد لينتشر مرة أخرى مؤخراَ مع ظهور برامج النشر الإلكتروني مثل "ألدوس بايج مايكر" (Aldus PageMaker) والتي حوت أيضاً على نسخ من '
#         self.cinnamon_text = CoVectorizer.preprocess_text(doc2text.doc_to_text(datapath + 'cinnamon.doc'))
#         assert len(self.cinnamon_text) > 0
#         pass
#
#     def test_greek(self):
#         roman_part = roman_words(self.greek_english_text)
#         assert len(roman_part) < len(self.greek_english_text)
#         assert 'Lorem' in roman_part
#
#     def test_arabic(self):
#         roman_part = roman_words(self.arabic_english_text)
#         assert len(roman_part) < len(self.arabic_english_text)
#         assert 'Aldus' in roman_part
