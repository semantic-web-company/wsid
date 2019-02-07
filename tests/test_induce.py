# __author__ = 'artreven'
# Use with Nosetests
import cProfile
import os

dirname = os.path.dirname(os.path.realpath(__file__))
datapath = dirname + '/data/'

import wsid.cooccurrence as cooc
import wsid.induce as induce
from wsid.induce.utils import InducedSense, InducedModel


class TestInduce:
    def setUp(self):
        x3 = cooc.load_cos(cos_path=os.path.join(datapath, 'cos.npz'),
                           t2i_path=os.path.join(datapath, 't2i_cos.pkl'),
                           i2t_path=os.path.join(datapath, 'i2t_cos.pkl'))
        self.cos, self.t2i, self.i2t = x3

    def test_get_graph(self):
        G, e_es = induce.get_co_graph(
            self.cos, self.t2i, self.i2t,
            order_2_cos=list(self.t2i.keys()),
            canonical_entity_form='add')
        assert G

    def test_induce(self):
        hubs, clusters, _, _, _ = induce.induce(
            canonical_entity_form='add',
            cos=self.cos, ind2token=self.i2t, token2ind=self.t2i)
        assert hubs
