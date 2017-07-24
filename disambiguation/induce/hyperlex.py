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


def get_hubs_with_broaders(G, entity_form, th_hub, broaders, pr=None):
    """
    Pick up the sense hubs from the CO graph G

    :param G: CO graph
    :param entity_form: surface form of the entity as it is found in G
    :param th_hub: threshold for adding new hub
    :param broaders: broaders that describe the target sense
    :return: List[(str, float)], List[str]
    """

    def get_co_broader_ranking(G, pr):
        co_score_func = lambda x, y: (1 if x == y else
                                      (G[x][y]['weight'] if y in G[x] else 0))
        pr_co_br = {
            x: pr[x] * sum(co_score_func(x, broader)
                           for broader in [entity_form] + broaders)
            for x in G_wo_entity
        }
        co_br_ranking = sorted(pr_co_br.items(), key=lambda y: y[1], reverse=True)
        return co_br_ranking

    entity_co = {
        term: G[entity_form][term]['weight']
        for term in G[entity_form]
        if term != entity_form
    }
    G_wo_entity = G.copy()
    G_wo_entity.remove_node(entity_form)
    max_co = max(G[x][y]['weight'] for x in G for y in G[x])
    if pr is None:
        pr = nx.pagerank_scipy(G_wo_entity)
    left_nodes = {x: 1 for x in pr}

    co_br_ranking = get_co_broader_ranking(G, pr)
    hubs = []
    clusters = []
    score = 0
    while (left_nodes[co_br_ranking[0][0]] > 0.5 and
           co_br_ranking[0][1] > score * 0.5):
        br_hub, score = co_br_ranking.pop(0)
        left_nodes[br_hub] = 0
        br_neighbors = set(G_wo_entity.neighbors(br_hub)) - {br_hub}
        print(len(br_neighbors))
        cluster = {br_hub: pr[br_hub]}
        for x in pr:
            involvement = (
                sum(G[x][n]['weight']
                    for n in br_neighbors
                    if n in G[x])
                +
                sum(G[x][n]['weight']
                    for n in (set(broaders) & set(G)) | {br_hub}
                    if n in G[x])
            )
            # left_nodes[x] = 0
            left_nodes[x] -= involvement
            cluster[x] = (pr[br_hub] * involvement)
            # cluster[x] = (pr[x] * node_w * involvement)
        clusters.append(cluster)
        hubs.append((br_hub, score, 1))
    for hub in hubs:
        print(hub)
    print(co_br_ranking[0][0], co_br_ranking[0][1], left_nodes[co_br_ranking[0][0]])


    pr_co = {
        x: pr[x] * entity_co[x] for x in entity_co
    }
    max_pr_co = max(pr_co.values())
    ranking = sorted(pr_co.items(), key=lambda y: y[1], reverse=True)
    #
    # hubs = []
    # clusters = []

    # ranking = [(x[0], max_pr_co) for x in br_hubs] + ranking
    for node, score in ranking:
        # print(node, score)
        # print(left_nodes[node])
        # print(pr[node])
        # print()
        # input()

        if left_nodes[node] > 0.6:
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
                         (score / max_pr_co)
                         if original_neighbors else 0)
                if value > th_hub:
                    left_nodes[node] = 0
                    hubs.append((node, value, 0))
                    # node_w = entity_co[node]
                    # cluster = {node: pr[node] * node_w}
                    cluster = {node: pr[node]}
                    for x in pr:
                        involvement = (
                            sum(G[x][n]['weight']
                                for n in original_neighbors
                                if n in G[x])
                            +
                            (G[x][node]['weight']
                             if node in G[x]
                             else 0)
                        )
                        # left_nodes[x] = 0
                        left_nodes[x] -= involvement
                        cluster[x] = (pr[node] * involvement)
                        # cluster[x] = (pr[x] * node_w * involvement)
                        # print(involvement)
                    clusters.append(cluster)
            else:
                break
        # print()
        # print(node, score)
        # print(sum(co_score_func(node, broader) for broader in broaders))
        # print(hubs)

    return hubs, clusters
