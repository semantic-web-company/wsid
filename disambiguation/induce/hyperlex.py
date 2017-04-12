import numpy as np

import networkx as nx


def get_hubs(G, entity_form, th_hub, pr=None):
    """
    Pick up the sense hubs from the CO graph G

    :param G: CO graph
    :param th_hub: threshold for adding new hub
    :return: List[(str, float)], List[str]
    """
    entity_co = {
        t: G[entity_form][t]['weight']
        for t in G[entity_form]
        if t != entity_form
    }
    G_wo_entity = G.copy()
    G_wo_entity.remove_node(entity_form)
    if pr is None:
        pr = nx.pagerank_scipy(G_wo_entity)
    left_nodes = {x: 1 for x in pr}
    pr_co = {x: (pr[x] * entity_co[x])
             for x in entity_co}
    if entity_form in pr_co:
        del pr_co[entity_form]
    max_pr_co = max(pr_co.values())
    hubs = []
    clusters = []
    ranking = sorted(pr_co.items(), key=lambda y: y[1], reverse=True)

    for node, score in ranking:
        if left_nodes[node] > 0.9:
            if score / max_pr_co > th_hub:
                original_neighbors = set(G_wo_entity.neighbors(node))
                original_neighbors_co = sum(
                    G[node][n]['weight'] for n in original_neighbors
                )
                left_neighbors_co = sum(
                    G[node][n]['weight'] * left_nodes[n]
                    for n in original_neighbors
                )
                n_contrib = (left_neighbors_co / original_neighbors_co
                             if original_neighbors else 0)
                value = (n_contrib *
                         (pr_co[node] / max_pr_co)
                         if original_neighbors else 0)
                if value > th_hub:
                    left_nodes[node] = 0
                    hubs.append((node, value, n_contrib))
                    node_w = entity_co[node]
                    cluster = {node: pr[node] * node_w}
                    for x in original_neighbors:
                        involvement = np.sqrt(
                            sum(G[x][n]['weight']
                                for n in original_neighbors | {node}
                                if n in G[x])
                        )
                        left_nodes[x] -= involvement
                        cluster[x] = (pr[x] * node_w * involvement)
                    clusters.append(cluster)
            else:
                break

    return hubs, clusters