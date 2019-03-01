import functools
import logging
from collections import defaultdict
from typing import List, Set, Callable, Dict, Union, Tuple

import scipy.sparse
import numpy as np
import time

import igraph as ig

module_logger = logging.getLogger(__name__)


class EntityCoOccurrences:
    def __init__(self,
                 cos: scipy.sparse.csr_matrix,
                 token2ind: Dict[str, int],
                 ind2token: Dict[int, str],
                 entity_str: str):
        assert entity_str in token2ind, entity_str
        self.entity_str: str = entity_str
        self.token2ind: Dict[str, int] = token2ind
        self.ind2token: Dict[int, str] = ind2token
        self.cos_mx: scipy.sparse.csr_matrix = cos
        self.important_cos: List[str] = self.get_cos_around_entity()
        self._graph: ig.Graph = None
        self._co_dict: Dict[str, Dict[str, float]] = None

    @property
    @functools.lru_cache()
    def pr(self, with_entity=False) -> Dict[str, float]:
        start_pr = time.time()
        g = self.graph
        pr = dict(zip(list(g.vs['name']), g.pagerank(weights='weight')))
        if not with_entity:
            del pr[self.entity_str]
            assert set(g.vs['name']) - set(pr) == {self.entity_str}, \
                set(g.vs['name']) - set(pr)
        else:
            assert set(pr) == set(g.vs['name']), set(pr) - set(g.vs['name'])
        module_logger.info(f'Pagerank done in {time.time()-start_pr:0.3f}s')
        return pr

    @property
    def graph(self) -> ig.Graph:
        self._check_graph_or_codict()
        return self._graph

    @property
    def co_dict(self) -> Dict[str, Dict[str, float]]:
        self._check_graph_or_codict()
        return self._co_dict

    def _check_graph_or_codict(self):
        if self._co_dict is None:
            graph, co_dict = self.get_co_graph_dict()
            self._graph = graph
            self._co_dict = co_dict

    def co_func(self, n1: str, n2: str) -> float:
        if n1 == n2:
            return 1
        else:
            try:
                return self.co_dict[n1][n2]
            except KeyError:
                return 0

    def get_cos_around_entity(self, neigh_divider: int = 20) -> List[str]:
        start = time.time()
        entity_ind = self.token2ind[self.entity_str]
        # bin COs matrix to get the edges
        entity_row = self.cos_mx.getrow(entity_ind)
        entity_cos_inds = set(list(entity_row.indices) + [entity_ind])

        module_logger.info(f'Number of 1st order coocs: {len(entity_row.data)}')
        if len(entity_row.data) >= 7500:
            important_cos_inds = entity_cos_inds
        else:
            # we filter out badly connected nodes; threshold comes from the
            # number of entity connections
            neighbors_th = len(entity_cos_inds) / neigh_divider
            module_logger.info('Neighbors limit = ' + str(round(neighbors_th)))
            # 2nd order COs binary matrix to get 2nd order COs.
            entity_row_bin = scipy.sparse.csr_matrix(
                ([1] * len(entity_row.data), entity_row.indices,
                 entity_row.indptr),
                shape=entity_row.shape
            )
            cos_bin = scipy.sparse.csr_matrix(
                ([1]*len(self.cos_mx.data), self.cos_mx.indices,
                 self.cos_mx.indptr),
                shape=self.cos_mx.shape
            )
            entity_2_row_bin = entity_row_bin.dot(cos_bin)
            important_cos_inds = entity_cos_inds | {
                entity_2_row_bin.indices[i]
                for i in range(len(entity_2_row_bin.indices))
                if entity_2_row_bin.data[i] >= neighbors_th
            }
        important_cos = [self.ind2token[x] for x in important_cos_inds]
        assert self.entity_str in important_cos

        module_logger.info(
            f'Number of important coocs: {len(important_cos)},'
            f'done in {time.time() - start:0.3f}s')
        return important_cos

    def get_co_graph_dict(
            self,
            # cos, token2ind, ind2token, important_cos,
            co_average_score_th: float = .2) -> (ig.Graph,
                                                 Dict[str, Dict[str, float]]):
        def normalize_vals(mx):
            denom_v = mx.sum(axis=0).A1
            ans_mx = scipy.sparse.csr_matrix(mx / denom_v).T
            return ans_mx

        start = time.time()
        vs_tokens = list(self.important_cos)
        vs_inds = [self.token2ind[x] for x in vs_tokens]
        vs_tokens_set = set(vs_tokens)

        cos_vv = scipy.sparse.csr_matrix(self.cos_mx[:, vs_inds][vs_inds, :])

        denom_v = cos_vv.sum() / cos_vv.count_nonzero()
        co_score_th = denom_v * co_average_score_th
        cos_vv = cos_vv.multiply(cos_vv > co_score_th)

        cos_vv = normalize_vals(cos_vv)
        cos_vv.eliminate_zeros()
        cos_vv = cos_vv.tocoo()
        module_logger.info(f'CO matrix done, size = {len(cos_vv.data)}')

        edgelist = list(zip(cos_vv.row.tolist(), cos_vv.col.tolist()))
        module_logger.info(f'edgelist done, size = {len(edgelist)}')
        weights = cos_vv.data.tolist()
        graph = ig.Graph(edgelist,
                         edge_attrs={'weight': weights},
                         vertex_attrs={'name': vs_tokens},
                         directed=True)
        co_dict = defaultdict(dict)
        for e, w in zip(edgelist, weights):
            source = vs_tokens[e[0]]
            target = vs_tokens[e[1]]
            co_dict[source][target] = w

        co_co_time = time.time()
        module_logger.info(f'Graph done in {co_co_time - start:0.3f}s')
        module_logger.info(f'Number of nodes: {len(graph.vs)}, '
                           f'number of edges: {len(graph.es)}')
        return graph, co_dict


def get_hubs(entity_str: str,
             cos: scipy.sparse.csr_matrix,
             token2ind: Dict[str, int],
             ind2token: Dict[int, str],
             th_hub: float,
             broader_groups: List[List[str]] = ()) -> (
                (List[Tuple[str, float, List[str]]],
                 List[Dict[str, float]],
                 EntityCoOccurrences)):
    """
    Pick up the sense hubs from the entity co-occurrences
    """

    def get_ranking(items, scoring_func):
        scoring_dict = {x: scoring_func(x) for x in items}
        ranking = sorted(scoring_dict.items(), key=lambda y: y[1], reverse=True)
        return ranking

    def _get_hubs(scoring_func, broaders, th_hub=th_hub):
        hubs = []
        clusters = []
        candidates = (set(e_cos.co_dict[e_cos.entity_str]) |
                      {x for x in broaders if x in e_cos.pr}) - \
                     {e_cos.entity_str}
        assert candidates <= set(e_cos.pr), candidates - set(e_cos.pr)
        ranking = get_ranking(candidates, scoring_func)
        while True:
            br_hub, initital_score = ranking.pop(0)
            score = scoring_func(br_hub)
            #
            m = f'hub: {br_hub}, broaders = {broaders}, ' \
                f'left_nodes[{br_hub}] = {left_nodes[br_hub]:0.3f}, ' \
                f'score = {score:0.4f}, th_hub = {th_hub:0.4f}'
            if score > th_hub: m += '\nNEW HUB!!'
            module_logger.debug(m)
            #
            if score > th_hub and left_nodes[br_hub] > 0.5:
                # hub_core = {br_hub} | (set(broaders) & set(e_cos.pr))
                hub_neighbors = ({e_cos.graph.vs['name'][x]
                                  for x in e_cos.graph.predecessors(br_hub)} |
                                 {br_hub}) - {e_cos.entity_str}
                hub_involvement = {
                    n: sum(e_cos.co_func(n, z) for z in hub_neighbors - {n})
                        for n in e_cos.pr
                        if n != entity_str}
                cluster = dict()
                for x in set(hub_involvement):
                    involvement = hub_involvement[x]
                    assert involvement <= 1, (x, involvement)
                    cluster[x] = e_cos.pr[x] * involvement #* left_nodes[x]
                    left_nodes[x] -= involvement
                    if left_nodes[x] < 0:
                        left_nodes[x] = 0
                clusters.append(cluster)
                hubs.append((br_hub, score, broaders)) #score
                if broaders:
                    th_hub *= 2
            elif initital_score < th_hub:
                break
        return hubs, clusters

    def get_hyperlex_scoring_func(br_g:Set[str]) -> Callable[[str], float]:
        def node_presence(x: str) -> float:
            """
            how much node neighbors are still present in the graph
            """
            val = left_nodes[x]
            if val != 0:
                val *= sum(e_cos.co_func(x, n) * left_nodes[n]
                           for n in e_cos.co_dict[x]
                           if n != e_cos.entity_str)
            if val != 0:
                val /= sum(e_cos.co_func(x, n)
                           for n in e_cos.co_dict[x]
                           if n != e_cos.entity_str)
            return val

        def pr_co_score(x: str, br_g: Set[str]) -> float:
            br_g_e = set(br_g) | {e_cos.entity_str}
            val = e_cos.pr[x]
            val *= sum(e_cos.co_func(x, br) for br in br_g_e)
            val /= len(br_g_e)
            return val

        def scoring_func(x:str) -> float:
            val = pr_co_score(x, br_g=br_g) / normalizer
            val *= node_presence(x)
            return val

        normalizer = max(pr_co_score(x, br_g=br_g)
                         for x in e_cos.co_dict[e_cos.entity_str]
                         if x != e_cos.entity_str)
        return scoring_func

    e_cos = EntityCoOccurrences(entity_str=entity_str, cos=cos,
                                token2ind=token2ind, ind2token=ind2token)
    # prepare initial values for nodes
    left_nodes = {x: 1 for x in e_cos.pr}

    hubs, clusters = [], []
    if broader_groups:
        if isinstance(broader_groups[0], str):
            broader_groups = [broader_groups]
        for broaders in broader_groups:
            br_G = set(broaders) & set(e_cos.token2ind) - {e_cos.entity_str}
            assert br_G, broaders
            scoring_func = get_hyperlex_scoring_func(br_g=br_G)
            br_hubs, br_clusters = _get_hubs(
                scoring_func, broaders=broaders)
            hubs += br_hubs
            clusters += br_clusters

    scoring_func = get_hyperlex_scoring_func(br_g=set())
    other_hubs, other_clusters = _get_hubs(
        scoring_func, broaders=[])
    hubs += other_hubs
    clusters += other_clusters

    return hubs, clusters, e_cos

