import functools
import logging
import scipy.sparse
import time

import igraph as ig

module_logger = logging.getLogger(__name__)

class EntityCoOccurrences:
    def __init__(self, cos, token2ind, ind2token, entity_str):
        """

        :param scipy.sparse.csr_matrix cos:
        :param dict[str, int] token2ind: from token to its index
        :param dict[int, str] ind2token: from index to token
        :param str entity_str:
        :rtype: EntityCoOccurrences
        """
        assert entity_str in token2ind, entity_str
        self.entity_str = entity_str
        self.token2ind = token2ind
        self.ind2token = ind2token
        self.cos_mx = cos
        self.order2_cos = self.get_order2_cos_around_entity()
        self._graph = None
        self._co_dict = None

    @property
    @functools.lru_cache()
    def pr(self, with_entity=False):
        start_pr = time.time()
        g = self.graph
        pr = dict(zip(list(g.vs['name']), g.pagerank(weights='weight')))
        if not with_entity:
            del pr[self.entity_str]
        else:
            assert set(pr) == set(g.vs['name']), set(pr) - set(g.vs['name'])
        module_logger.info(f'Pagerank done in {time.time()-start_pr:0.3f}s')
        return pr

    # @property
    # @functools.lru_cache()
    # def graph_wo_e(self):
    #     graph_wo_e = self.graph.copy()
    #     graph_wo_e.delete_vertices(self.entity_str)
    #     return graph_wo_e

    @property
    # @functools.lru_cache()
    def graph(self):
        if self._co_dict is None:
            graph, co_dict = get_co_graph_dict(self.cos_mx, self.token2ind,
                                               self.ind2token,
                                               self.order2_cos, self.entity_str)
            self._graph = graph
            self._co_dict = co_dict
        # graph, co_dict = get_co_graph_dict(self.cos_mx, self.token2ind,
        #                                    self.ind2token,
        #                                    self.order2_cos, self.entity_str)
        # assert not any(x is None for x in graph.es['weight']), \
        #     [i for i, x in enumerate(graph.es['weight']) if x is None]
        return self._graph

    def co_func(self, n1, n2):
        if n1 == n2:
            return 1
        else:
            try:
                return self.co_dict[n1][n2]
            except KeyError:
                return 0

    @property
    # @functools.lru_cache()
    def co_dict(self):
        if self._co_dict is None:
            graph, co_dict = get_co_graph_dict(self.cos_mx, self.token2ind,
                                               self.ind2token,
                                               self.order2_cos, self.entity_str)
            self._graph = graph
            self._co_dict = co_dict
        # adjlist = self.graph.get_adjlist()
        # for token_str, adj_inds in zip(self.graph.vs['name'], adjlist):
        #     ans[token_str] = {self.graph.vs['name'][ind] for ind in adj_inds}
        # ans = dict()
        # edges = self.graph.get_edgelist()
        # weights = self.graph.es['weight']
        # for e, w in zip(edges, weights):
        #     s, t = self.graph.vs['name'][e[0]], self.graph.vs['name'][e[1]]
        #     try:
        #         ans[s][t] = w
        #     except KeyError:
        #         ans[s] = {t: w}
        return self._co_dict

    def get_order2_cos_around_entity(self, neigh_divider=20):
        """
        Get 2nd order COs to entity. Filter out vertices that have little
        (less than neighbors_th) neighbors, i.e. noisy vertices.

        :rtype: list
        """
        start = time.time()
        entity_ind = self.token2ind[self.entity_str]
        # bin COs matrix to get the edges
        entity_row = self.cos_mx.getrow(entity_ind)
        entity_cos_inds = set(list(entity_row.indices) + [entity_ind])
        # we filter out badly connected nodes; threshold comes from the number
        # of entity connections
        neighbors_th = len(entity_cos_inds) / neigh_divider
        module_logger.info('Neighbors limit = ' + str(round(neighbors_th)))
        # 2nd order COs binary matrix to get 2nd order COs.
        cos_bin = scipy.sparse.csr_matrix(
            ([1]*len(self.cos_mx.data), self.cos_mx.indices, self.cos_mx.indptr),
            shape=self.cos_mx.shape
        )
        entity_row_bin = scipy.sparse.csr_matrix(
            ([1] * len(entity_row.data), entity_row.indices, entity_row.indptr),
            shape=entity_row.shape
        )
        entity_2_row_bin = entity_row_bin.dot(cos_bin)
        order_2_cos_inds = entity_cos_inds | {
            entity_2_row_bin.indices[i]
            for i in range(len(entity_2_row_bin.indices))
            if entity_2_row_bin.data[i] >= neighbors_th
        }
        order_2_cos = [self.ind2token[x] for x in order_2_cos_inds]
        module_logger.info(
            f'Number of 2nd order coocs: {len(order_2_cos)},'
            f'done in {time.time() - start:0.3f}s')
        return order_2_cos


def get_co_graph_dict(cos, token2ind, ind2token, order_2_cos, canonical_entity_form,
                      co_average_score_th=2.5):
    """

    :param scipy.sparse.csr_matrix cos:
    :param dict[str, int] token2ind:
    :param dict[int, str] ind2token:
    :param list[str] order_2_cos:
    :param str canonical_entity_form:
    :return: co-oc graph including entity
    :rtype: igraph.Graph
    """
    start = time.time()
    V = set(order_2_cos)
    V_inds = [token2ind[x] for x in V]
    cos_inds2V_inds = {x: i for i, x in enumerate(V_inds)}
    cos_V = cos[:, V_inds]
    # prepare a graph to populate
    G = ig.Graph(directed=True)
    G.add_vertices(list(V))
    co_dict = dict()
    edges = []
    weights = []
    for e, co1 in enumerate(V):
        co1_ind = token2ind[co1]
        # normalize so that the sum of all edges outcoming from a token in V to other tokens in V is 1
        denom = cos_V.getrow(co1_ind).sum()
        co1_row_denom = (cos_V.getrow(co1_ind) / denom).toarray()[0]
        co1_cos = {ind2token[V_inds[i]] for i in
                   cos_V.getrow(co1_ind).indices}
        for co2 in co1_cos:
            co_score = co1_row_denom[cos_inds2V_inds[token2ind[co2]]]
            if co2 != canonical_entity_form:
                if co_score > co_average_score_th / len(V):
                    edges.append((co1, co2))
                    weights.append(co_score)
                    try:
                        co_dict[co1][co2] = co_score
                    except KeyError:
                        co_dict[co1] = {co2: co_score}
            else:  # co2 is entity
                edges.append((co1, co2))
                co_score = co1_row_denom[cos_inds2V_inds[token2ind[co2]]]
                weights.append(co_score)
                try:
                    co_dict[co1][co2] = co_score
                except KeyError:
                    co_dict[co1] = {co2: co_score}
    G.add_edges(edges)
    G.es['weight'] = weights
    co_co_time = time.time()
    module_logger.info(f'Graph done in {co_co_time - start:0.3f}s')
    module_logger.info(f'Number of nodes: {len(G.vs)}, '
                        f'number of edges: {len(G.es)}')
    return G, co_dict


def get_hubs(entity_str, cos, token2ind, ind2token, th_hub, broader_groups=()):
    """
    Pick up the sense hubs from the entity co-occurrences

    :param scipy.sparse.csr_matrix cos: sparse matrix of token co-occurrences
    :param dict[str, int] token2ind: map from token string to its index in cos
    :param dict[int, str] ind2token: inverse of :token2ind:
    :param str entity_str: how the entity is represented in :token2ind:
    :param float th_hub: threshold for adding new hub
    :param list[list[str]] broader_groups: broaders that describe the target sense
    :rtype: (list[(str, float, list[str])], list[dict[str, float]], EntityCoOccurrences)
    """

    def get_ranking(items, scoring_func):
        scoring_dict = {x: scoring_func(x) for x in items}
        ranking = sorted(scoring_dict.items(), key=lambda y: y[1], reverse=True)
        return ranking

    def _get_hubs(left_nodes, scoring_func, broaders):
        hubs = []
        clusters = []
        candidates = (set(e_cos.co_dict[e_cos.entity_str]) | {x for x in broaders if x in e_cos.pr}) - {e_cos.entity_str}
        ranking = get_ranking(candidates, scoring_func)
        # ranking = get_ranking(e_cos.graph_wo_e.vs['name'], scoring_func)  # TODO: better pre-filtering
        while True:
            br_hub, initital_score = ranking.pop(0)
            score = scoring_func(br_hub)
            m = f'hub: {br_hub}, left_nodes[br_hub] = {left_nodes[br_hub]}, ' \
                f'score = {score}, th_hub = {th_hub}'
            module_logger.debug(m)
            # if left_nodes[br_hub] > 0.75:
            if score > th_hub:
                hub_core = {br_hub} | (set(broaders) & set(e_cos.pr))
                hub_neighbors = ({e_cos.graph.vs['name'][x] for x in e_cos.graph.predecessors(br_hub)} | {br_hub}) - {e_cos.entity_str}
                hub_involvement = {
                    n:
                        sum(e_cos.co_func(n, z) for z in set(e_cos.co_dict[n].keys()) & hub_neighbors) /
                        sum(e_cos.co_func(n, z) for z in set(e_cos.co_dict[n].keys()))
                    for n in {n2 #e_cos.graph_wo_e.vs['name'][n2]
                              for n1 in hub_neighbors
                              for n2 in e_cos.co_dict[n1]
                              if n2 != entity_str}} #e_cos.graph_wo_e.predecessors(n1)}}
                cluster = {node: e_cos.pr[node]*hub_involvement[node] for node in hub_core if node in hub_involvement}
                for n in hub_core:
                    left_nodes[n] -= hub_involvement[n]
                # cluster = {node: pr[node] for node in hub_core}
                # for n in hub_core:
                #     left_nodes[n] = 0
                for x in set(hub_involvement) - hub_core:
                    involvement = hub_involvement[x]
                    cluster[x] = e_cos.pr[x] * involvement #* left_nodes[x]
                    left_nodes[x] -= involvement
                    if left_nodes[x] < 0:
                        left_nodes[x] = 0
                clusters.append(cluster)
                hubs.append((br_hub, score, broaders))
            elif initital_score < th_hub:
                break
        return hubs, clusters, left_nodes

    # @functools.lru_cache()
    # def co_func(n1, n2):
    #     if n1 == n2:
    #         return 1
    #     else:
    #         try:
    #             return e_cos.co_dict[n1][n2]
    #         except KeyError:
    #             return 0

    # def get_sns(G, n):
    #     """
    #     Get successor names.
    #     :param igraph.Graph G:
    #     :param str n:node name
    #     :rtype: set
    #     """
    #     s_ids = G.successors(n)
    #     return {ind2token[i] for i in s_ids}

    # def to_dict(graph):
    #     ans = dict()
    #     edges = graph.get_edgelist()
    #     weights = graph.es['weight']
    #     for e, w in zip(edges, weights):
    #         s, t = ind2token[e[0]], ind2token[e[1]]
    #         try:
    #             ans[s][t] = w
    #         except KeyError:
    #             ans[s] = {t: w}
    #     return ans

    # G = G_w_entity.copy()
    # G.delete_vertices([entity_form])
    # token2ind = {name: i for i, name in enumerate(G_w_entity.vs['name'])}
    # ind2token = {i: name for name, i in token2ind.items()}
    # co_dict = to_dict(G_w_entity)

    e_cos = EntityCoOccurrences(entity_str=entity_str, cos=cos,
                                token2ind=token2ind, ind2token=ind2token)

    # prepare initial values for nodes
    left_nodes = {x: 1 for x in e_cos.pr}
    # how much node neighbors are still present in the graph
    node_presence = lambda x: (
            left_nodes[x] *
            sum(e_cos.co_func(x, n) * left_nodes[n] for n in e_cos.co_dict[x] if n != e_cos.entity_str) /
            sum(e_cos.co_func(x, n) for n in e_cos.co_dict[x] if n != e_cos.entity_str))
    pr_co_scoring = lambda x: e_cos.pr[x] * e_cos.co_func(x, e_cos.entity_str)
    max_pr_co = max(pr_co_scoring(x) for x in e_cos.co_dict[e_cos.entity_str]
                    if x != e_cos.entity_str)

    hubs, clusters = [], []
    if broader_groups:
        if isinstance(broader_groups[0], str):
            broader_groups = [broader_groups]
        for broaders in broader_groups:
            br_G = set(broaders) & set(e_cos.token2ind) - {e_cos.entity_str}
            assert br_G, broaders
            pr_ebr_co_scoring = lambda x: e_cos.pr[x] * (e_cos.co_func(x, e_cos.entity_str) +
                                                   sum(e_cos.co_func(x, br) for br in br_G)) / len(br_G)
            max_pr_co_ebr = max(pr_ebr_co_scoring(x) for x in e_cos.co_dict[e_cos.entity_str]
                                if x != e_cos.entity_str)
            scoring_func = lambda x: pr_ebr_co_scoring(x) / max_pr_co_ebr * node_presence(x)
            br_hubs, br_clusters, left_nodes = _get_hubs(
                left_nodes, scoring_func, broaders=broaders)
            hubs += br_hubs
            clusters += br_clusters

    scoring_func = lambda x: pr_co_scoring(x) / max_pr_co * node_presence(x)
    other_hubs, other_clusters, left_nodes = _get_hubs(left_nodes, scoring_func, broaders=[])
    hubs += other_hubs
    clusters += other_clusters

    return hubs, clusters, e_cos
