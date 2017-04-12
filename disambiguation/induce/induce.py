import logging
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import disambiguation.cooccurrence as cooc

from . import hyperlex

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# w = 35
th_co = 0.0
# n_cos_limit = 5000


def induce(texts, canonical_entity_form, method='hyperlex',
           proximity='linear', cooc_method='unbiased_dice', w=35):
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
    start = time.time()
    new_texts = texts[:]
    if proximity == 'linear':
        prox_func = lambda x: (w - abs(x) + 0.5) / w
    elif proximity == 'const':
        prox_func = lambda x: 1
    else:
        raise Exception('Proximity {} is not known/implemented'.format(
            proximity
        ))
    cos = cooc.get_co(
        new_texts, w, method=cooc_method,
        threshold=th_co,
        proximity_func=prox_func
    )
    co_time = time.time()
    logger.info('Cos done in {:0.3f}s'.format(co_time - start))

    entity_co = cos[canonical_entity_form]
    set_e_co = set(entity_co.keys())
    neighbors_th = len(set_e_co) / 20
    logger.info('Neighbors limit = ' + str(round(neighbors_th)))
    order_2_cos = set_e_co.copy()
    for term in entity_co:
        for x in set(cos[term]) - order_2_cos:
            if len(set(cos[x]) & set_e_co) > neighbors_th:
                order_2_cos.add(x)
    # order_2_cos = {x for term in entity_co for x in cos[term]
    #                if (x in set_e_co or
    #                    len(set(cos[x]) & set_e_co) > neighbors_th)}
    # order_2_cos = entity_co
    V = set(order_2_cos)
    assert canonical_entity_form in order_2_cos
    order2_time = time.time()
    logger.info(('Order2 done in {:0.3f}s'.format(order2_time - co_time)))
    logger.info('Number of 2nd order coocs: {}'.format(len(order_2_cos)))

    co_co_E = []
    for co1 in V:
        denom = sum(cos[co1].values())
        for co2 in set(cos[co1]) & set(V):
            co_co_E.append((co1, co2, cos[co1][co2]/denom))
    co_co_time = time.time()
    logger.info('Co2co done in {:0.3f}s'.format(co_co_time - order2_time))

    # COs are prepared now
    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(co_co_E)
    G_no_entity = G.copy()
    G_no_entity.remove_node(canonical_entity_form)
    pr = nx.pagerank_scipy(G_no_entity)
    logger.info('Number of nodes: {}, number of edges: {}'.format(
        len(G.nodes()), len(G.edges())
    ))
    #
    pr_time = time.time()
    logger.info('Pagerank done in {:0.3f}s'.format(pr_time - co_co_time))

    if method == 'hyperlex':
        hubs, clusters = hyperlex.get_hubs(
            G, canonical_entity_form,
            th_hub=1. / math.log(len(texts)),
            pr=pr
        )
    else:
        raise Exception()
    logger.info('Hubs done in {:0.3f}s'.format(time.time() - pr_time))
    return hubs, clusters, G, pr, cos


def get_graph_figure(G, pagerank, nodesize_mult, edgewidth_mult, entity_name):
    fig = plt.figure()
    pos = nx.drawing.nx_agraph.graphviz_layout(G)
    nodesize = [pagerank[v] * nodesize_mult for v in G.nodes()
                if not v == entity_name]
    edgewidth = [x[2] * edgewidth_mult for x in G.edges(data='weight')]
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edgewidth, edge_color='m')
    nx.draw_networkx_nodes(G, pos, node_size=nodesize, alpha=0.4)
    nx.draw_networkx_labels(G, pos, fontsize=12)
    plt.title('Word Senses Net: '.format(entity_name))
    plt.axis('off')
    return fig


def get_plt2py_graph_div(G, pagerank, entity_name):
    start = time.time()
    import plotly.tools as tls
    import plotly.offline as po
    top_nodes = sorted(pagerank.items(),
                       key=lambda x: x[1],
                       reverse=True)[:50]
    G_50 = G.subgraph([x[0] for x in top_nodes])
    g_fig = get_graph_figure(G_50, pagerank, 2, 5, entity_name)
    plotly_fig = tls.mpl_to_plotly(g_fig)
    logger.info('Graph figure done in {:0.3f}s'.format(time.time() - start))
    return po.plot(plotly_fig, auto_open=False, output_type='div')


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


def make_plotly_fig(G, node_weights, nodes_colors, fig_title='',
                    show_fig=False):
    import plotly.offline as po
    import plotly.graph_objs as go

    pos = nx.drawing.nx_agraph.graphviz_layout(G)
    edges = G.edges(data='weight')
    # scale width
    edges_width = MinMaxScaler(feature_range=(0, 7)).fit_transform(
        [x[2] for x in edges]
    )

    edge_trace = go.Scatter(
        x=[],
        y=[],
        mode='lines'
    )
    for i in range(len(edges)):
        x0, y0 = pos[edges[i][0]]
        x1, y1 = pos[edges[i][1]]
        width = edges_width[i]

        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]
        edge_trace['hoverinfo'] = 'none'
        edge_trace['line']['width'] = width
        edge_trace['line']['color'] = '#888'


    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=go.Marker(
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

        node_trace['x'].append(x)
        node_trace['y'].append(y)
        node_trace['marker']['color'].append(nodes_colors[node])
        node_trace['marker']['size'].append(scaled_size)
        node_trace['text'].append(node_info)
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
                        xaxis=go.XAxis(showgrid=False, zeroline=False,
                                       showticklabels=False),
                        yaxis=go.YAxis(showgrid=False, zeroline=False,
                                       showticklabels=False))
                    )
    if not show_fig:
        output = 'div'
    else:
        output = 'file'
    return po.plot(fig, filename='networkx.html',
                   auto_open=show_fig,
                   output_type=output)


def cluster_text(text, senses, entity, w=20):
    if len(senses) < 2:
        return 0, 1
    else:
        context, _ = cooc.get_relevant_tokens(
            text, w,
            proximity_func=lambda x: (w - abs(x) + .5) / w,
            entity=entity
        )
        distr = [sum(sense[x] * context[x] for x in context if x in sense)
                 for sense in senses]
        if not any(distr):
            print('Not possible to decide on category: no evidence!')

        result = np.argmax(distr)
        sorted_distr = sorted(distr, reverse=True)
        delta = (
            (sorted_distr[0] - sorted_distr[1]) / sorted_distr[1]
            if sorted_distr[1] else 0
        )
        conf = 1 - 1 / (1 + delta)
        return result, conf


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
