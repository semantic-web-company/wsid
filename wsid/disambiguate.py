import numpy as np
from wsid import cooccurrence as cooc


# def classify_texts(corpus_texts, sense_clusters, entity, w):
#     prediction = []
#     for i in range(len(corpus_texts)):
#         doc = corpus_texts[i]
#         categ, conf, distr, evidences = cluster_text(
#             doc, sense_clusters,
#             entity=entity,
#             w=w
#         )
#         prediction.append([categ, conf])
#     return prediction


def cluster_text(text, hubs, senses, entity, w=20):
    if len(senses) < 2:
        return 0, 1, [], []
    else:
        texts_tokens, token2ind, all_tokens_counter, len_texts = cooc.texts2tokens([text])
        tokens = next(texts_tokens)
        t2t_cos = cooc.get_t2t_proximities(tokens, token2ind, w,
                                           return_dict=True)
        context = t2t_cos[entity]

        distr = [sum(sense[x] * context[x] for x in context if x in sense) # hub[1] *
                 for hub, sense in zip(hubs, senses)]
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


def disambiguate(text, model, entity=None, start_end_inds=None, w=10):
    """

    :param str text:
    :param wsid.induce.utils.InducedModel model:
    :param (str or None) entity:
    :param ((int, int) or None) start_end_inds:
    :param int w:
    :return: distribution over model senses
    :rtype: (list[float], list[list[str]])
    """
    if entity is None:
        assert start_end_inds is not None
        entity = text[start_end_inds[0]:start_end_inds[1]]
    texts_tokens, token2ind, all_tokens_counter, len_texts = cooc.texts2tokens([text])
    tokens = next(texts_tokens)
    t2t_cos = cooc.get_t2t_proximities(tokens, token2ind, w,
                                       return_dict=True)
    context = t2t_cos[entity]
    evidences = [[x for x in context if x in sense.cluster]
                 for sense in model.senses]
    distr = [sum(sense.cluster[x] * context[x] for x in ev)
             for sense, ev in zip(model.senses, evidences)]
    # assert any(distr), context
    return distr, evidences
