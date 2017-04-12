# __author__ = 'artreven'
# Use with Nosetests

import os

import sklearn.feature_extraction.text as sktext
dirname = os.path.dirname(os.path.realpath(__file__))
datapath = dirname + '/data/'

from disambiguation.preprocess import *
import doc_to_text.doc_to_text as doc2text


def test_eliminate_stopwords():
    filename = 'cinnamon.doc'
    text = doc2text.doc_to_text(datapath + filename)
    clarified = eliminate_stopwords(text)
    assert len(clarified) < len(text)


def test_eliminate_short():
    filename = 'cinnamon.doc'
    text = doc2text.doc_to_text(datapath + filename)
    cleaned = eliminate_shortwords(text)
    assert len(cleaned) < len(text)


class TestCase:
    def setUp(self):
        self.greek_english_text = "Το Lorem Ipsum είναι απλά ένα κείμενο χωρίς νόημα για τους επαγγελματίες της" \
                                  "τυπογραφίας και στοιχειοθεσίας. Το Lorem Ipsum είναι το επαγγελματικό πρότυπο όσον" \
                                  " αφορά το κείμενο χωρίς νόημα, από τον 15ο αιώνα, όταν ένας ανώνυμος τυπογράφος " \
                                  "πήρε ένα δοκίμιο και ανακάτεψε τις λέξεις για να δημιουργήσει ένα δείγμα βιβλίου. " \
                                  "Όχι μόνο επιβίωσε πέντε αιώνες, αλλά κυριάρχησε στην ηλεκτρονική στοιχειοθεσία, " \
                                  "παραμένοντας με κάθε τρόπο αναλλοίωτο. Έγινε δημοφιλές τη δεκαετία του '60 με την " \
                                  "έκδοση των δειγμάτων της Letraset όπου περιελάμβαναν αποσπάσματα του Lorem Ipsum, " \
                                  "και πιο πρόσφατα με το λογισμικό ηλεκτρονικής σελιδοποίησης όπως το Aldus " \
                                  "PageMaker που περιείχαν εκδοχές του Lorem Ipsum."
        self.arabic_english_text = '"الشكلي منذ القرن الخامس عشر عندما قامت مطبعة مجهولة برص مجموعة من الأحرف بشكل عشوائي أخذتها من نص، لتكوّن كتيّب بمثابة دليل أو مرجع شكلي لهذه الأحرف. خمسة قرون من الزمن لم تقضي على هذا النص، بل انه حتى صار مستخدماً وبشكله الأصلي في الطباعة والتنضيد الإلكتروني. انتشر بشكل كبير في ستينيّات هذا القرن مع إصدار رقائق "ليتراسيت" (Letraset) البلاستيكية تحوي مقاطع من هذا النص، وعاد لينتشر مرة أخرى مؤخراَ مع ظهور برامج النشر الإلكتروني مثل "ألدوس بايج مايكر" (Aldus PageMaker) والتي حوت أيضاً على نسخ من '
        self.cinnamon_text = CoVectorizer.preprocess_text(doc2text.doc_to_text(datapath + 'cinnamon.doc'))
        assert len(self.cinnamon_text) > 0
        pass

    def test_greek(self):
        roman_part = roman_words(self.greek_english_text)
        assert len(roman_part) < len(self.greek_english_text)
        assert 'Lorem' in roman_part

    def test_arabic(self):
        roman_part = roman_words(self.arabic_english_text)
        assert len(roman_part) < len(self.arabic_english_text)
        assert 'Aldus' in roman_part


class TestMyVectorizer:
    def setUp(self):
        self.X_texts = ["Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
                        "It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).",
                        "There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text. All the Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, making this the first true generator on the Internet. It uses a dictionary of over 200 Latin words, combined with a handful of model sentence structures, to generate Lorem Ipsum which looks reasonable."]
        self.other_text = 'Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.'
        self.c_vec = sktext.CountVectorizer()
        entity = r'Lorem'
        self.w = 3
        self.my_vec = CoVectorizer(self.w, proximity_func=lambda x: self.w - abs(x))
        self.c_X = self.c_vec.fit_transform(self.X_texts)
        y = [1, 0, 1]
        self.my_X = self.my_vec.fit_transform(self.X_texts, y,
                                          entity_forms=[entity]*len(self.X_texts))


    def test_fit(self):
        assert self.my_X.shape[1] < self.c_X.shape[1]
        assert 'ipsum' in self.my_vec.vocabulary
        ind_ipsum = self.my_vec.vocabulary_['ipsum']
        assert all([self.my_X.getrow(i).todense()[0, ind_ipsum] > 0 for i in range(self.my_X.shape[0])])

    def test_transform(self):
        new_X = self.my_vec.transform([self.other_text])
        assert 'ipsum' in self.my_vec.vocabulary
        ind_ipsum = self.my_vec.vocabulary['ipsum']
        assert new_X.todense()[0, ind_ipsum] > 0