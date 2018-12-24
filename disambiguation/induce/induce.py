import logging
import math
import time
import numpy as np
from functools import reduce

import networkx as nx
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import disambiguation.cooccurrence as cooc

from . import hyperlex

module_logger = logging.getLogger(__name__)

# w = 35
th_co = 0.0
# n_cos_limit = 5000


def induce(texts, canonical_entity_form, method='hyperlex',
           proximity='linear', cooc_method='unbiased_dice', w=15,
           broaders=None,
           cos=None, token2ind=None, ind2token=None):
    """
    Cluster the set of texts in order to identify different meanings of the
    entity.

    :param texts: input texts
    :param canonical_entity_form: how the entity is represented in texts
    :param method: method to use for induction. Currently only HyperLex is
        implemented
    :param proximity: ['linear', 'constant']
        proximity for cooccurrences calculation.
        Default: linear.
    :param cooc_method: method for calculating cooccurrences,
        see get_co description.
        Default: 'unbiased_dice'
    :param w: window size for cooccurrences
    :return: hubs, clusters, collocation graph, PageRank
    """
    def get_cos_inds(term_ind):
        term_row = cos.getrow(term_ind).indices
        return term_row

    start = time.time()
    if cos is None:
        new_texts = texts[:]
        if proximity == 'linear':
            prox_func = lambda x: 2*(w - abs(x) + 0.5) / w
        elif proximity == 'const':
            prox_func = lambda x: 1
        else:
            raise Exception('Proximity {} is not known/implemented'.format(
                proximity
            ))
        cos, token2ind, ind2token = cooc.get_co(
            new_texts, w, method=cooc_method,
            threshold=th_co,
            proximity_func=prox_func
        )
    co_time = time.time()
    module_logger.info('Cos done in {:0.3f}s'.format(co_time - start))

    # get 2nd order COs
    entity_ind = token2ind[canonical_entity_form]
    # bin COs matrix to get the edges
    entity_row = cos.getrow(entity_ind)
    order_2_cos_inds = set(list(entity_row.indices) + [entity_ind])
    neigh_divider = 20
    neighbors_th = len(order_2_cos_inds) / neigh_divider
    module_logger.info('Neighbors limit = ' + str(round(neighbors_th)))
    # 2nd order COs binary matrix to get 2nd order COs.
    cos_bin = scipy.sparse.csr_matrix(
        ([1] * len(cos.data), cos.indices, cos.indptr),
        shape=cos.shape
    )
    entity_row_bin = scipy.sparse.csr_matrix(
        ([1] * len(entity_row.data), entity_row.indices, entity_row.indptr),
        shape=entity_row.shape
    )
    entity_2_bin_row = entity_row_bin.dot(cos_bin)
    module_logger.info('2nd order matrix done, {}*{} = {}'.format(
        entity_row_bin.shape, cos_bin.shape, entity_2_bin_row.shape
    ))
    order_2_cos_inds |= {
        entity_2_bin_row.indices[i]
        for i in range(len(entity_2_bin_row.indices))
        if entity_2_bin_row.data[i] >= neighbors_th
    }
    order_2_cos = [ind2token[x] for x in order_2_cos_inds]
    V = set(order_2_cos)
    assert canonical_entity_form in V
    order2_time = time.time()
    module_logger.info(('Order2 done in {:0.3f}s'.format(order2_time - co_time)))
    module_logger.info('Number of 2nd order coocs: {}'.format(len(order_2_cos)))

    co_co_E = []
    entity_co_edges = []
    # create a list of indices of tokens that are in the vertices set V
    V_inds = [token2ind[x] for x in V]
    cos_inds2V_inds = {x: i for i, x in enumerate(V_inds)}
    entity_ind_V = cos_inds2V_inds[entity_ind]
    cos_V = cos[:, V_inds]
    # prepare a graph to populate
    G = nx.DiGraph()
    for e, co1 in enumerate(V):
        co1_ind = token2ind[co1]
        # normalize so that the sum of all edges outcoming from a token in V to other tokens in V is 1
        denom = cos_V.getrow(co1_ind).sum()
        co1_row_denom = (cos_V.getrow(co1_ind) / denom).toarray()[0]
        co1_cos = {ind2token[V_inds[i]] for i in cos_V.getrow(co1_ind).indices}
        if co1 != canonical_entity_form:
            # G.add_weighted_edges_from(
            #     (co1, co2, co1_row_denom[cos_inds2V_inds[token2ind[co2]]])
            #     for co2 in co1_cos
            #     if co1_row_denom[cos_inds2V_inds[token2ind[co2]]] > 1/neighbors_th
            # )
            for co2 in co1_cos:
                co_score = co1_row_denom[cos_inds2V_inds[token2ind[co2]]]
                if co_score > 2.5/len(order_2_cos_inds):
                    G.add_edge(co1, co2, weight=co_score)
            entity_co_edges += [
                (co1, canonical_entity_form, co1_row_denom[entity_ind_V])]
            # co_co_E += [
            #     (co1, co2, co1_row_denom[cos_inds2V_inds[token2ind[co2]]])
            #     for co2 in co1_cos]
        else:
            co_entity_edges = [
                (co1, co2, co1_row_denom[cos_inds2V_inds[token2ind[co2]]])
                for co2 in co1_cos]
    assert len(co_entity_edges) - len(entity_co_edges) <= 1, (len(co_entity_edges),
                                                              len(entity_co_edges))
    co_co_time = time.time()
    module_logger.info('Co2co done in {:0.3f}s'.format(co_co_time - order2_time))

    # COs are prepared now
    # G = nx.DiGraph()
    # G.add_nodes_from(V)
    # G.add_weighted_edges_from(co_co_E)
    # for x in G:
    #     assert 0.9 < sum(G[x][n]['weight'] for n in G[x]) < 1.1, (x, sum(G[x][n]['weight'] for n in G[x]))

    # G_no_entity = G.copy()
    # G_no_entity = G
    G.remove_node(canonical_entity_form)
    module_logger.info('Number of nodes: {}, number of edges: {}'.format(
        len(G.nodes()), len(G.edges())
    ))
    pr = nx.pagerank_scipy(G, tol=1.0e-10, max_iter=300)
    G.add_weighted_edges_from(co_entity_edges)
    G.add_weighted_edges_from(entity_co_edges)
    #
    pr_time = time.time()
    module_logger.info('Pagerank done in {:0.3f}s'.format(pr_time - co_co_time))

    if method == 'hyperlex':
        if broaders is None:
            hubs, clusters = hyperlex.get_hubs(
                G, canonical_entity_form,
                th_hub=1. / math.log(len(texts)),
                pr=pr
            )
        else:
            hubs, clusters = hyperlex.get_hubs_with_broaders(
                G, canonical_entity_form,
                th_hub=1. / math.log(len(texts)),
                pr=pr,
                broaders=broaders
            )
    else:
        raise Exception()
    module_logger.info('Hubs done in {:0.3f}s'.format(time.time() - pr_time))
    return hubs, clusters, G, pr, cos


def plot_graph_degrees(G):
    import matplotlib.pyplot as plt
    degree_sequence = sorted(G.degree().values(),
                             reverse=True)  # degree sequence
    weight_sequence = sorted(G.degree(weight='weight').values(),
                             reverse=True)  # weight sequence
    plt.figure(1)
    plt.subplot(211)
    plt.plot(degree_sequence, 'b-', marker='o')
    plt.title("Degree rank plot")
    plt.ylabel("degree")
    plt.xlabel("rank")

    plt.subplot(212)
    plt.plot(weight_sequence, 'b-', marker='o')
    plt.title("Weight rank plot")
    plt.ylabel("weight")
    plt.xlabel("rank")
    plt.show()


def get_graph_nodes_attributes(sense_clusters, colors, pr,
                               limit_per_cluster=10):
    nodes_colors = dict()
    nodes_weights = dict()
    for i in range(len(sense_clusters)):
        sense_cluster = sense_clusters[i]
        top_items = sorted(sense_cluster.items(),
                           key=lambda x: x[1],
                           reverse=True)[:limit_per_cluster]
        top_words = {x[0] for x in top_items}
        for word in top_words:
            if word in nodes_colors:
                if nodes_weights[word] <= sense_cluster[word]:
                    nodes_colors[word] = colors[i]
                    nodes_weights[word] = sense_cluster[word]
            else:
                nodes_colors[word] = colors[i]
                nodes_weights[word] = sense_cluster[word]
    c = 0
    for node, node_pr in sorted(pr.items(), key=lambda x: x[1],
                                reverse=True):
        if node not in nodes_colors:
            nodes_colors[node] = 'gray'
            nodes_weights[node] = node_pr
            c += 1
        if c >= limit_per_cluster:
            break
    return nodes_colors, nodes_weights


def make_plotly_fig(G, node_weights, nodes_colors, fig_title='',
                    show_fig=False):
    import plotly.offline as po
    import plotly.graph_objs as go

    pos = nx.drawing.nx_agraph.graphviz_layout(G)
    edges = list(G.edges(data='weight'))
    # scale width
    edges_width = MinMaxScaler(feature_range=(0, 7)).fit_transform(
        np.array([x[2] for x in edges]).reshape(-1, 1)
    )

    edge_trace = go.Scatter(
        x=[],
        y=[],
        mode='lines'
    )
    for i in range(len(edges)):
        x0, y0 = pos[edges[i][0]]
        x1, y1 = pos[edges[i][1]]
        width = edges_width[i][0]

        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
        edge_trace['hoverinfo'] = 'none'
        edge_trace['line']['width'] = width
        edge_trace['line']['color'] = '#888'


    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=go.scatter.Marker(
            color=[],
            size=[],
            line=dict(width=2)
        )
    )

    annotations = []
    max_weight = max(node_weights.values())
    for node in G.nodes():
        x, y = pos[node]
        node_info = '{}, weight: {:.2e}'.format(
            str(node),
            node_weights[node]
        )
        scaled_size = 40 * node_weights[node] / max_weight

        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['marker']['color'] += (nodes_colors[node],)
        node_trace['marker']['size'] += (scaled_size,)
        node_trace['text'] += (node_info,)
        ann_dict = {
            'x': x, 'y': y, 'text': str(node), 'showarrow': False
        }
        annotations.append(ann_dict)

    fig = go.Figure(data=go.Data([edge_trace, node_trace]),
                    layout=go.Layout(
                        title=fig_title,
                        titlefont=dict(size=16),
                        showlegend=False,
                        width=900,
                        height=650,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=annotations,
                        xaxis=go.layout.XAxis(showgrid=False, zeroline=False,
                                              showticklabels=False),
                        yaxis=go.layout.YAxis(showgrid=False, zeroline=False,
                                              showticklabels=False))
                    )
    if not show_fig:
        output = 'div'
    else:
        output = 'file'
    return po.plot(fig, filename='networkx.html',
                   auto_open=show_fig,
                   output_type=output)


def compare_senses(senses):
    vocab = list(reduce(lambda x, y: set(x)|set(y), senses, set()))
    sense_vecs = []
    for sense in senses:
        sense_vec = []
        for t in vocab:
            sense_vec.append(sense[t] if t in sense else 0)
        sense_vecs.append(np.array(sense_vec))
    sim_matrix = get_sim_matrix(sense_vecs, sense_vecs)
    return vocab, sense_vecs, sim_matrix


def get_sim_matrix(vecs1, vecs2):
    sim_matrix = [
        [
            cosine_similarity(
                vec1.reshape(1, -1),
                vec2.reshape(1, -1)
            )
            for vec1 in vecs1
        ]
        for vec2 in vecs2
    ]
    return sim_matrix


def get_top_confidence_items(clustering_results, min_limit=5, conf_th=0.9):
    """
    produces a list of clusters containing document indices with top confident
    documents.

    :param clustering_results: list of (category, confidence)
    """
    ordered = sorted(
        [(i, x[0], x[1]) for i, x in enumerate(clustering_results)],
        key=lambda y: y[2],
        reverse=True
    )
    categs = {x for _, x, conf in ordered}
    top_conf_items = {
        categ: [] for categ in categs
        }
    for i, x, conf in ordered:
        if len(top_conf_items[x]) < min_limit or conf > conf_th:
            top_conf_items[x].append(i)
    return top_conf_items
