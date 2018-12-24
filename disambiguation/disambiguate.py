import numpy as np
from disambiguation import cooccurrence as cooc


def classify_texts(corpus_texts, sense_clusters, entity, w):
    prediction = []
    for i in range(len(corpus_texts)):
        doc = corpus_texts[i]
        categ, conf, distr, evidences = cluster_text(
            doc, sense_clusters,
            entity=entity,
            w=w
        )
        prediction.append([categ, conf])
    return prediction


def cluster_text(text, senses, entity, w=20):
    if len(senses) < 2:
        return 0, 1, [], []
    else:
        t2t_cos = cooc.get_t2t_proximities(
            text.split(), w,
            # proximity_func=lambda x: (w - abs(x) + .5) / w,
        )
        context = t2t_cos[entity]
        distr = [sum(sense[x] * context[x] for x in context if x in sense)
                 for sense in senses]
        evidences = [[x for x in context if x in sense] for sense in senses]
        if not any(distr):
            print(text)
            print(context)
            print('Not possible to decide on category: no evidence!')

        result = np.argmax(distr)
        sorted_distr = sorted(distr, reverse=True)
        delta = (
            (sorted_distr[0] - sorted_distr[1]) / sorted_distr[1]
            if sorted_distr[1] else 0
        )
        conf = 1 - 1 / (1 + delta)
        return result, conf, distr, evidences
