import numpy as np

import networkx as nx


def get_hubs(G, entity_form, th_hub, pr=None):
    """
    Pick up the sense hubs from the CO graph G

    :param G: CO graph
    :param th_hub: threshold for adding new hub
    :return: List[(str, float)], List[str]
    """
    def get_ranking(G, scoring_func):
        scoring_dict = {x: scoring_func(x) for x in G}
        ranking = sorted(scoring_dict.items(), key=lambda y: y[1], reverse=True)
        return ranking

    # for term in G[entity_form]:
    #     print()
    #     print(term)
    #     print(G[entity_form][term])
    #     print(G[term][entity_form])
    assert entity_form in G
    entity_co = {
        term: G[term][entity_form]['weight']
        for term in G[entity_form]
        if term != entity_form
    }
    # G_wo_entity = G.copy()
    G_wo_entity = G
    G_wo_entity.remove_node(entity_form)

    if pr is None:
        pr = nx.pagerank_scipy(G_wo_entity, tol=1.0e-10, max_iter=300)
    pr_co = {x: pr[x] * entity_co[x] for x in entity_co}
    max_pr_co = max(pr_co.values())

    left_nodes = {x: 1 for x in pr}
    node_contrib = lambda x: (
        sum(G_wo_entity[x][n]['weight'] * left_nodes[n]
            for n in G_wo_entity.neighbors(x)) /
        sum(G_wo_entity[x][n]['weight']
            for n in G_wo_entity.neighbors(x))
    )
    hubs = []
    clusters = []
    scoring_func = lambda x: (
        (pr[x] * (entity_co[x] if x in entity_co else 0) / max_pr_co) *
        (left_nodes[x] * node_contrib(x))  # ** 2
    )
    ranking = get_ranking(G_wo_entity, scoring_func)
    while True:
        hub, score = ranking.pop(0)
        # print()
        # print(ranking[:3])
        # print(left_nodes[hub], score, th_hub)
        if left_nodes[hub] > 0.5:
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

                for x in set(pr) - {hub}:
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
                    ) / np.sqrt(sum(
                        G_wo_entity[x][n]['weight'] for n in G_wo_entity[x]))
                    # if involvement > 1.1:
                    #     print(x, involvement)
                    #     print(sum(G[x][n]['weight'] for n in G[x]))
                    #     print(len(common_neigh))
                    #     print(x in common_neigh, hub in common_neigh)
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
    # G_wo_entity = G.copy()
    G_wo_entity = G
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
