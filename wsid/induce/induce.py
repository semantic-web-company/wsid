import functools
import logging
import math
import time

import networkx as nx
import numpy as np
from functools import reduce

# import networkx as nx
import igraph as ig
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import wsid.cooccurrence as cooc

from . import hyperlex

module_logger = logging.getLogger(__name__)

th_co = 0.0


def induce(cos, token2ind, ind2token, entity_str,
           method='hyperlex', broader_groups=()):
    """
    Cluster co-occurrences of the entity to identify different meanings.

    :param scipy.sparse.csr_matrix cos: sparse matrix of token co-occurrences
    :param dict[str, int] token2ind: map from token string to its index in cos
    :param dict[int, str] ind2token: inverse of :token2ind:
    :param str entity_str: how the entity is represented in :token2ind:
    :param list[list[str]] broader_groups:
    :return: hubs, clusters, EntityCoOccurrences object
    """

    start = time.time()
    if method == 'hyperlex':
        # import cProfile
        # s = '''hubs, clusters, e_cos = hyperlex.get_hubs(
        #     entity_str, cos, token2ind, ind2token,
        #     th_hub=1./np.log10(len(cos.nonzero()[0])),
        #     broader_groups=broader_groups
        # )'''
        # cProfile.runctx(s, locals=locals(), globals=globals())
        hubs, clusters, e_cos = hyperlex.get_hubs(
            entity_str, cos, token2ind, ind2token,
            th_hub=1./np.log10(len(cos.nonzero()[0])),  # TODO: check if this th is good
            broader_groups=broader_groups
        )
    else:
        raise Exception()
    module_logger.info(
        'Hubs done in {:0.3f}s, total hubs: {}'.format(time.time() - start,
                                                       len(hubs)))
    return hubs, clusters, e_cos


# def plot_graph_degrees(G):
#     import matplotlib.pyplot as plt
#     degree_sequence = sorted(G.degree().values(),
#                              reverse=True)  # degree sequence
#     weight_sequence = sorted(G.degree(weight='weight').values(),
#                              reverse=True)  # weight sequence
#     plt.figure(1)
#     plt.subplot(211)
#     plt.plot(degree_sequence, 'b-', marker='o')
#     plt.title("Degree rank plot")
#     plt.ylabel("degree")
#     plt.xlabel("rank")
#
#     plt.subplot(212)
#     plt.plot(weight_sequence, 'b-', marker='o')
#     plt.title("Weight rank plot")
#     plt.ylabel("weight")
#     plt.xlabel("rank")
#     plt.show()


# def get_graph_nodes_attributes(sense_clusters, colors, pr,
#                                limit_per_cluster=10):
#     nodes_colors = dict()
#     nodes_weights = dict()
#     for i in range(len(sense_clusters)):
#         sense_cluster = sense_clusters[i]
#         top_items = sorted(sense_cluster.items(),
#                            key=lambda x: x[1],
#                            reverse=True)[:limit_per_cluster]
#         top_words = {x[0] for x in top_items}
#         for word in top_words:
#             if word in nodes_colors:
#                 if nodes_weights[word] <= sense_cluster[word]:
#                     nodes_colors[word] = colors[i]
#                     nodes_weights[word] = sense_cluster[word]
#             else:
#                 nodes_colors[word] = colors[i]
#                 nodes_weights[word] = sense_cluster[word]
#     c = 0
#     for node, node_pr in sorted(pr.items(), key=lambda x: x[1],
#                                 reverse=True):
#         if node not in nodes_colors:
#             nodes_colors[node] = 'gray'
#             nodes_weights[node] = node_pr
#             c += 1
#         if c >= limit_per_cluster:
#             break
#     return nodes_colors, nodes_weights


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
