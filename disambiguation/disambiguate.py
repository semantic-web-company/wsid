from.induce import cluster_text

def classify_texts(hubs, corpus, sense_clusters, canonical_form, w):
    """

    :return:
    """
    broader_senses_inds = [i for i, hub in enumerate(hubs)
                           if hub[2] == 1]
    prediction = []
    for i in range(len(corpus)):
        doc = corpus[i]
        categ, conf, distr, evidences = cluster_text(
            doc, sense_clusters,
            entity=canonical_form,
            w=w
        )
        if categ in broader_senses_inds:
            categ = 'this'
        else:
            categ = 'other'
        prediction.append([categ, conf])
    return prediction