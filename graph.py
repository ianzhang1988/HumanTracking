#coding=utf-8

from itertools import product
import numpy as np
import ransac

class Node():
    def __init__(self):
        self.edges = []
        self.data = None

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

        self.Hypothetical_appearance_penalty = 0.8
        self.Hypothetical_threshold = 100

        # [0], [1] inputdata.shape = (n,2), 对线性关系来说，[:,1] 是x [:,2] 是
        self.model = ransac.LinearLeastSquaresModel([0],[1],debug=False)


    def none_zero_cluster_num(self):
        return len([c for c in self.cluster if len(c) > 0])

    def set_cluster_num(self,num):
        for i in range(num):
            self.cluster.append([])

    def add_node_to_cluster(self, idx, node):
        self.cluster[idx].append(node)
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
            #spead = (float(sub[i+1].data['x']) - float(sub[i].data['x']))**2 + (float(sub[i+1].data['y']) - float(sub[i].data['y']))**2
            vec1 = self.pos_vec(sub[i+1])
            vec2 = self.pos_vec(sub[i])
            speed = vec1 - vec2
            node_speed.append(speed)

        sum = 0

        for i in range(len(sub) - 1): # i is prediction
            for j in range(len(sub) - 1):  # other node
                if i == j:
                    continue
                vecX = self.pos_vec(sub[i])
                vecY = self.pos_vec(sub[j])
                sum += 1 - 1 / ( 1+ np.linalg.norm(vecX - (vecY + node_speed[j]*(i - j))) )

        # the last node
        i = len(sub) - 1
        for j in range(len(sub) - 1):
            vecX = self.pos_vec(sub[-1])
            vecY = self.pos_vec(sub[j])
            sum += 1 - 1 / ( 1+ np.linalg.norm(vecX - (vecY + node_speed[j] * (i - j))) )

        return sum

    def add_hyopthetical_node(self, sub):
        new_sub = []
        addition_weight = 0

        data = np.vstack([ self.pos_vec(n) for n in sub ])
        result = ransac.ransac( data, self.model, 2, len(sub)*2, self.Hypothetical_threshold,
                                        (len(sub)+1)//2, return_all=True, by_count = True )
        if result is None:
            return sub

        model, inliner = result
        ideal_count = len(sub)
        inliner = set(inliner)
        ideal = set(range(ideal_count))




        return new_sub

    def get_largest_sub_graph(self, motion = False, hypothetical = False):
        hist_weight = 0
        motion_weight = 0
        counter = 0


        nodes = []
        cluster = [c for c in self.cluster if len(c) > 0]
        sub_graphs = product(*cluster)

        least = -1

        for sub in sub_graphs:
            sum_weight = 0
            sum_weight += self.calc_sub_hist_weight(sub)
            hist_weight += sum_weight

            if motion:
                motion_value = self.motion_factor * self.calc_sub_motion_weight(sub)
                motion_weight += motion_value
                sum_weight += motion_value

            if hypothetical: # only len(sub) > 3 make sense



                hypothetical_value = self.motion_factor * self.calc_sub_motion_weight(sub)
                sum_weight = hypothetical_value

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
