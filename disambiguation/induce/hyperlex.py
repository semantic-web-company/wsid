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

    def get_ranking(G, scoring_func):
        scoring_dict = {x: scoring_func(x) for x in G}
        ranking = sorted(scoring_dict.items(), key=lambda y: y[1], reverse=True)
        return ranking

    entity_co = {
        term: G[term][entity_form]['weight']
        for term in G[entity_form]
        if term != entity_form
    }
    G_wo_entity = G.copy()
    G_wo_entity.remove_node(entity_form)

    if pr is None:
        pr = nx.pagerank_scipy(G_wo_entity, tol=1.0e-10, max_iter=300)
    pr_co = {x: pr[x] * entity_co[x] for x in entity_co}
    max_pr_co = max(pr_co.values())
    left_nodes = {x: 1 for x in pr}
    node_contrib = lambda x: (
        sum(G_wo_entity[x][n]['weight'] * left_nodes[n]
            for n in G_wo_entity.neighbors(x)) /
        sum(G_wo_entity[x][n]['weight'] for n in G_wo_entity.neighbors(x))
    )

    broaders_G = set(broaders) & set(G)
    assert broaders_G
    co_score_func = lambda x, y: (1 if x == y else
                                  (G[x][y]['weight'] if y in G[x] else 0))
    pr_br_co_scoring = lambda x: pr[x] * (
        co_score_func(x, entity_form) +
        sum(co_score_func(x, broader) for broader in broaders_G) /
        len(broaders_G)
    )
    ranking = get_ranking(G_wo_entity, pr_br_co_scoring)
    hubs = []
    clusters = []
    max_pr_co_br = ranking[0][1]
    scoring_func = lambda x: (
        (pr[x] * (
            co_score_func(x, entity_form) +
            sum(co_score_func(x, broader) for broader in broaders_G) /
            len(broaders_G)
        ) / max_pr_co_br) * (left_nodes[x] * node_contrib(x))**2
    )
    ranking = get_ranking(G_wo_entity, scoring_func)

    while True:
        br_hub, score = ranking.pop(0)
        if left_nodes[br_hub] > 0.75:
            if score > th_hub:
                hub_core = {br_hub} | (set(broaders) & set(G))
                hub_neighbors = set(G_wo_entity.neighbors(br_hub))
                hub_neighbors_involvement = {
                    n:
                        sum(G[n][z]['weight'] for z in set(G_wo_entity[n]) & hub_neighbors) /
                        sum(G[n][z]['weight'] for z in set(G_wo_entity[n]))
                    for n in hub_neighbors
                }
                cluster = {node: pr[node] for node in hub_core}
                inv = dict()
                for n in hub_core:
                    left_nodes[n] = 0
                for x in set(pr) - hub_core:
                    x_neigh = set(G_wo_entity.neighbors(x))
                    common_neigh = (x_neigh & hub_neighbors) - hub_core - {x}
                    involvement = np.sqrt(
                        sum(
                            G_wo_entity[x][n]['weight'] * hub_neighbors_involvement[n]
                            for n in common_neigh
                        ) +
                        sum(
                            G_wo_entity[x][n]['weight'] / len(hub_core - {br_hub})
                            for n in ((hub_core & x_neigh) - {br_hub})
                        ) +
                        G_wo_entity[x][br_hub]['weight'] if br_hub in G[x] else 0
                    ) / np.sqrt(sum(G_wo_entity[x][n]['weight'] for n in G_wo_entity[x]))
                    assert not (common_neigh & hub_core)
                    assert not br_hub in common_neigh
                    assert involvement <= 1.2, (x, involvement)

                    cluster[x] = pr[x] * involvement * left_nodes[x]
                    left_nodes[x] -= involvement
                    if left_nodes[x] < 0:
                        left_nodes[x] = 0
                    inv[x] = involvement
                clusters.append(cluster)
                hubs.append((br_hub, score, 1))
                ranking = get_ranking(G_wo_entity, scoring_func)
        if score < th_hub:
            print('here')
            break

    scoring_func = lambda x: (
        (pr[x] * (entity_co[x] if x in entity_co else 0) / max_pr_co) *
        (left_nodes[x] * node_contrib(x))**2
    )
    ranking = get_ranking(G_wo_entity, scoring_func)
    while True:
        hub, score = ranking.pop(0)
        if left_nodes[hub] > 0.75:
            if score > th_hub:
                left_nodes[hub] = 0
                hub_w = pr[hub]
                cluster = {hub: hub_w}
                inv = dict()
                hub_neighbors = set(G_wo_entity.neighbors(hub))

                hub_neighbors_involvement = {
                    n:
                        sum(G_wo_entity[n][z]['weight'] for z in
                            set(G_wo_entity[n]) & hub_neighbors) /
                        sum(G[n][z]['weight'] for z in set(G_wo_entity[n]))
                    for n in hub_neighbors
                }

                for x in set(pr) - {hub} - set(broaders):
                    common_neigh = (set(G_wo_entity.neighbors(x)) &
                                    hub_neighbors) - {x}
                    involvement = np.sqrt(
                        sum(
                            G[x][n]['weight'] * hub_neighbors_involvement[n]
                            for n in common_neigh - {hub}
                        ) +
                        (
                            G[x][hub]['weight'] if hub in G[x] else 0
                        )
                    ) / np.sqrt(sum(G_wo_entity[x][n]['weight'] for n in G_wo_entity[x]))
                    if involvement > 1.1:
                        print(x, involvement)
                        print(sum(G[x][n]['weight'] for n in G[x]))
                        print(len(common_neigh))
                        print(x in common_neigh, hub in common_neigh)
                    assert involvement <= 1.1, (x, involvement)

                    cluster[x] = pr[x] * involvement * left_nodes[x]
                    left_nodes[x] -= involvement
                    if left_nodes[x] < 0:
                        left_nodes[x] = 0
                    inv[x] = involvement
                clusters.append(cluster)
                hubs.append((hub, score, 0))
                ranking = get_ranking(G_wo_entity, scoring_func)
        if score < th_hub:
            break

    return hubs, clusters
