import sys
from experiments.nodeclass import Node_class
from experiments.link_prediction.link import Link
from experiments.network_visualization.visual import Net_visual

class Exp:
    def __init__(self, dataset, method):
        self.dataset = dataset
        self.method = method

    def nc(self):
        # node classification
        the_nc = Node_class()
        the_nc.test(self.dataset, self.method)

    def lp(self):
        # link prediction
        the_lp = Link()
        the_lp.test(self.dataset, self.method)

    def nv(self):
        # network visualization
        the_nv = Net_visual()
        the_nv.test()
