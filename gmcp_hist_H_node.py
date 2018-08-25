# coding = utf-8

from graph import ClasterGraph, Node, Edge
import numpy as np
import json

def calc_edge_weight_euclidean( node1, node2 ):
    hist1 = np.array(node1.data['hist_feature'])
    hist2 = np.array(node2.data['hist_feature'])
    dist = 1 - 1 / ( 1+np.linalg.norm(hist1 - hist2) )
    return dist


def read_data():
    file_name = 'pos_with_hist.json'
    data = None
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

data = read_data()

Edge.set_global_clac_func(calc_edge_weight_euclidean)

tracklet_list = []

cluster_num = len(data) // 5

print('cluster_num',cluster_num)

for n in range(cluster_num):
    g = ClasterGraph()
    g.set_cluster_num(5)
    for idx, nodes in enumerate( data[n*5:n*5+5] ):
        for i in nodes:
            node = Node()
            node.set_data(i)

            g.add_node_to_cluster(idx, node)

    g.connect_one_to_other_cluster()
    print("node num", len(g.nodes))

    g.calc_weight()

    cluster_data=[]
    while g.none_zero_cluster_num() > 3:
        sub = g.get_largest_sub_graph(True)
        g.remove_nodes(sub)
        node_data = [ i.data for i in sub]
        # print(data)
        data_without_hist = [ d for d in node_data if (d.pop('hist_feature'), True)]
        cluster_data.append(data_without_hist)

    tracklet_list.append(cluster_data)

with open('tracklet.json','w') as f:
    json.dump(tracklet_list, f)










