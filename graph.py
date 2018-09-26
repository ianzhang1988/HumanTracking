#coding=utf-8

from itertools import product
import numpy as np
import ransac

class Node():
    def __init__(self):
        self.edges = []
        self.data = None
        self.index = None # for cluster idx

    def set_data(self, data):
        self.data = data

    def _add_edge(self,edge):
        self.edges.append(edge)

    def connect(self, other_node):
        e = Edge()
        e.connect(self, other_node)
        return e

    def is_connected(self, other_node):
        for edge in self.edges:
            if edge.is_connected(self, other_node):
                return True
        return False

    def delete_edge(self, edge):
        if edge in self.edges:
            self.edges.remove(edge)

    def __del__(self):
        for edge in self.edges:
            edge.__del__()

class HypoNode(Node):
    def __init__(self):
        super(HypoNode, self).__init__()
        self.x = 0
        self.y = 0

class Edge():
    global_calc_func = None
    def __init__(self):
        self.node1=None
        self.node2=None
        self.weight=0
        self.calc_func = None

    def set_weight(self, weight):
        self.weight=weight

    def connect(self,node1, node2):
        node1._add_edge(self)
        node2._add_edge(self)
        self.node1 = node1
        self.node2 = node2

    def __del__(self):
        if self.node1:
            self.node1.delete_edge(self)
        if self.node2:
            self.node2.delete_edge(self)

    def set_calc_func(self, func):
        self.calc_func = func

    def is_connected(self, node1, node2):
        return (node1, node2) == (self.node1, self.node2) or (node2, node1) == (self.node1, self.node2)

    @classmethod
    def set_global_clac_func(cls, func):
        cls.global_calc_func = func

    def calc_weight(self):
        if self.calc_func:
            self.weight = self.calc_func(self.node1, self.node2)
        else:
            self.weight = Edge.global_calc_func(self.node1, self.node2)

class Graph():
    def __init__(self):
        self.nodes = []
        self.edges = []

    def make_node(self, num):
        self.nodes = [Node() for n in range(num)]
        return self.nodes

    def add_node(self, num = 1):
        new_node = [Node() for n in range(num)]
        self.nodes.extend(new_node)
        return new_node

    def connect(self,idx1, idx2):
        edge = Edge()
        node1 = self.nodes[idx1]
        node2 = self.nodes[idx2]
        edge.connect(node1, node2)
        self.edges.append(edge)
        return edge

class ClasterGraph():
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.cluster = []
        self.motion_factor = 2

    def none_zero_cluster_num(self):
        return len([c for c in self.cluster if len(c) > 0])

    def set_cluster_num(self,num):
        for i in range(num):
            self.cluster.append([])

    def add_node_to_cluster(self, idx, node):
        self.cluster[idx].append(node)
        node.index = idx
        node.cluster = self.cluster[idx]
        self.nodes.append(node)

    def connect_one_to_other_cluster(self):
        for c in self.cluster:
            for n in c:
                for other in self.nodes:
                    if other in c:
                        continue
                    if n.is_connected(other):
                        continue
                    e = n.connect(other)
                    self.edges.append(e)

    def calc_weight(self):
        for e in self.edges:
            e.calc_weight()

    def remove_nodes(self,nodes):
        for n in nodes:
            n.cluster.remove(n)
            self.nodes.remove(n)

    def calc_sub_hist_weight(self, sub):
        sum_weight = 0
        all_edges = set()
        for n in sub:
            for e in n.edges:
                if e.node1 not in sub or e.node2 not in sub:
                    continue
                all_edges.add(e)

        for e in all_edges:
            sum_weight += e.weight
        return sum_weight

    def pos_vec(self, node):
        return np.array((node.data['x'] + node.data['w']/2, node.data['y']+ node.data['h']/2 ),dtype=np.float32)

    def calc_sub_motion_weight(self, sub):
        node_speed=[]
        for i in range(len(sub) - 1 ):
            vec1 = self.pos_vec(sub[i+1])
            vec2 = self.pos_vec(sub[i])
            speed = vec1 - vec2
            node_speed.append(speed)

        sum = 0

        # for i in range(len(sub) - 1): # i is prediction
        for i in range(len(sub)):  # i is prediction
            for j in range(len(sub) - 1):  # other node
                if i == j:
                    continue
                vecX = self.pos_vec(sub[i])
                vecY = self.pos_vec(sub[j])
                sum += 1 - 1 / ( 1+ np.linalg.norm(vecX - (vecY + node_speed[j]*(i - j))) )

        # seams no need
        # the last node
        # i = len(sub) - 1
        # for j in range(len(sub) - 1):
        #     vecX = self.pos_vec(sub[-1])
        #     vecY = self.pos_vec(sub[j])
        #     sum += 1 - 1 / ( 1+ np.linalg.norm(vecX - (vecY + node_speed[j] * (i - j))) )

        return sum

    def add_index(self, sub):
        for idx, node in enumerate(sub):
            node.index = idx
            return sub

    def get_largest_sub_graph(self, motion = False, hypothetical = False):
        hist_weight = 0
        motion_weight = 0
        counter = 0


        nodes = []
        cluster = [c for c in self.cluster if len(c) > 0]
        sub_graphs = product(*cluster)

        least = -1

        for sub in sub_graphs:
            sub = self.add_index(sub)

            sum_weight = 0
            sum_weight += self.calc_sub_hist_weight(sub)

            hist_weight += sum_weight

            if motion:
                motion_value = self.motion_factor * self.calc_sub_motion_weight(sub)
                sum_weight += motion_value

                motion_weight += motion_value

            # if hypothetical: # only len(sub) > 3 make sense
            #     hypothetical_value = self.motion_factor * self.calc_sub_motion_weight(sub)
            #     sum_weight = hypothetical_value

            if least < 0 or sum_weight < least:
                least = sum_weight
                nodes = sub

            counter+=1

        print('least', least)
        print( '-----', len(sub), counter, hist_weight)
        if counter > 0 and hist_weight > 0:
            print('hist_weight', hist_weight/counter)
            print('motion_weight', motion_weight /counter)
            print('motion_weight/sum_weight', motion_weight/ hist_weight)

        return nodes

class ClasterHypoGraph( ClasterGraph ):
    def __init__(self):
        super(ClasterHypoGraph, self).__init__()
        self.full_hypo = True
        self.appearance_penalty = 1.2
        self.Hypothetical_threshold = 35

        self.model = ransac.HypoNodeModel()


    def hypothetical_node(self, sub):
        hypo_nodes = []
        addition_weight = 0

        data = np.vstack([ self.pos_vec(n) for n in sub ])
        d = (len(sub) - 2)//2-1# alsoinliers > d
        if d < 0:
            d = 0
        result = ransac.ransac( data, self.model, 2, len(sub)*2, self.Hypothetical_threshold,
                                       d, return_all=True, by_count = True )
        if result is None:
            return None

        (a1, a0), inliner = result
        ideal_count = len(sub)
        if self.full_hypo:
            ideal_count = len(self.cluster)
        inliner = set(inliner)

        ideal = set(range(ideal_count))
        predict_idx = sorted(list(ideal.difference( inliner )))

        for idx in predict_idx:
            hypo_node = HypoNode()
            hypo_node.x, hypo_node.y = (a1*idx + a0).tolist()
            hypo_node.index = idx
            hypo_nodes.append( hypo_node )

        return hypo_nodes

    def gen_hype_nodes_data(self, hype_nodes, old_nodes):
        if len(hype_nodes) < 1 or len(old_nodes) < 1:
            return

        avg_with = np.mean( [n.data['w'] for n in old_nodes] )
        avg_height = np.mean([n.data['h'] for n in old_nodes])
        frame_count = old_nodes[0].data['frame_count']

        for n in hype_nodes:

            data = {
                'x': n.x - avg_with / 2,
                'y': n.y - avg_height / 2,
                'w': avg_with,
                'h': avg_height,
                'frame_count': frame_count,
                'hist_feature':[],
            }
            n.data = data

    def put_nodes_together(self, hype_nodes, old_nodes ):
        hybrid_nodes = []

        # print( old_nodes )
        length = len(old_nodes)
        if self.full_hypo:
            length = len(self.cluster)

        remainder_nodes = []
        for n in range(length):
            h_node = next((i for i in hype_nodes if i.index == n), None)
            o_node = next((i for i in old_nodes if i.index == n), None)
            if h_node:
                hybrid_nodes.append(h_node)
            else:
                hybrid_nodes.append(o_node)
                remainder_nodes.append(o_node)

        weight = self.calc_sub_hist_weight(old_nodes)
        avg_weight = weight / len(old_nodes)
        hypo_appearance_weight = self.appearance_penalty * avg_weight

        self.connect_hybrid_nodes(hype_nodes, remainder_nodes, hypo_appearance_weight)
        self.add_hypo_node_to_graph(hype_nodes)

        return hybrid_nodes

    def connect_hybrid_nodes(self, hypo_nodes, remainder_nodes, appearance_weight):
        for h in hypo_nodes:
            for r in remainder_nodes:
                edge = h.connect(r)
                edge.weight = appearance_weight

    def add_hypo_node_to_graph(self, hypo_nodes):
        for n in hypo_nodes:
            self.add_node_to_cluster(n.index, n)

    def assembly_hybrid_node(self, hype_nodes, old_nodes):
        self.gen_hype_nodes_data(hype_nodes, old_nodes)
        hybrid_nodes = self.put_nodes_together( hype_nodes, old_nodes )
        return hybrid_nodes


    def get_largest_sub_graph(self, motion = False, hypothetical = False):
        hist_weight = 0
        motion_weight = 0
        counter = 0

        nodes = []
        cluster = [c for c in self.cluster if len(c) > 0]
        sub_graphs = product(*cluster)

        least = -1

        hypo_counter = 0

        for sub in sub_graphs:
            # sub = self.add_index(sub)

            sum_weight = 0
            sum_weight += self.calc_sub_hist_weight(sub)
            hist_weight += sum_weight

            motion_value = self.motion_factor * self.calc_sub_motion_weight(sub)
            motion_weight += motion_value
            sum_weight += motion_value

            if least < 0 or sum_weight < least:
                least = sum_weight
                nodes = sub

            counter+=1

        print('least', least)
        print( '-----', len(sub), counter, hist_weight)
        if counter > 0 and hist_weight > 0:
            print('hist_weight', hist_weight/counter)
            print('motion_weight', motion_weight /counter)
            print('motion_weight/sum_weight', motion_weight/ hist_weight)

        hypo_node = self.hypothetical_node(nodes)
        if hypo_node is None:
            print('Hypo None')
            return nodes

        if len(hypo_node) < 1:
            print('Hypo no need')
            return nodes

        new_sub = self.assembly_hybrid_node(hypo_node, nodes)
        hypo_sum_weight = self.calc_sub_hist_weight(new_sub) + self.motion_factor * self.calc_sub_motion_weight(new_sub)


        print('=======',sum_weight / len(nodes), hypo_sum_weight / len(new_sub))
        if sum_weight / len(nodes) > hypo_sum_weight / len(new_sub):
            nodes = new_sub
            hypo_counter +=1

        print('hypo num',hypo_counter)

        return nodes