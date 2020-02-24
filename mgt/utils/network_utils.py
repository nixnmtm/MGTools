from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import inspect


class MGNetwork(object):
    """
    - Description   : Utility for crating network graphs from the MGT results.
    - Author        : Nixon
    """

    def __init__(self):
        self.gx = nx.Graph()

    def graph_mtrx(self, mat):
        """
        Returns the Adjacency, Degree, Laplacian Matrices using edge_weights

        :param mat: the symmetric K matrix
        :type mat: pd.DataFrame
        :return: adjacency matrix, Degree matrix, laplacian matrix and normalized laplacian matrix
        :rtype: numpy arrayedg
        """
        # for test look into apo_ressep2_graph_theory
        G = self.gx
        Npos = len(G.nodes())
        G.add_weighted_edges_from(_edges(mat))  # adding edges and edge_weights
        G.add_nodes_from(np.arange(1, Npos))  # adding nodes

        # adjacency matrix
        A_mat = nx.adjacency_matrix(G, G.nodes(), weight='weight').todense()

        # degree matrix
        D_mat = G.degree(G.nodes(), weight='weight')  # gives the degree based on the weights d(v,v)

        # Laplacian Matrix
        Lap_mat = nx.laplacian_matrix(G, G.nodes(), weight="weight").todense()

        # Normalized Laplacian Matrix
        norm_Lap = nx.normalized_laplacian_matrix(G, G.nodes(), weight="weight").todense()

        return A_mat, D_mat, Lap_mat, norm_Lap

    def get_interacting_edges(self, G, nodes, kbcut, intra=False):
        """
        Get edgelists for a set of nodes, if interactions are present

        :param G: graph object
        :param nodes: list of nodes to get combinations
        :param kbcut: kb cut off
        :param intra: if True: get only intra interacting edges for given nodes
                        False: get all interacting edges of the given nodes

        """
        edgelist = list()
        if intra:
            for x in combinations(nodes, 2):
                x = (str(x[0]), str(x[1]))
                if x in G.edges():
                    if G.get_edge_data(x[0], x[1])["weight"] > kbcut:
                        edgelist.append(x)
        else:
            for i in nodes:
                for j in G.nodes():
                    x = (str(i), str(j))
                    if x in G.edges():
                        if G.get_edge_data(x[0], x[1])["weight"] > kbcut:
                            edgelist.append(x)
        return edgelist

    def draw_protein_graph(self, G, kbcut, imp_res=None):
        """
        Draw 2D protein graph using NetworkX, shows interactions of all residues > kbcut.

        :param G: graph object with all nodes and edges with necessary attributes
        :type G: object
        :param kbcut: draw interaction greater than coupling strength cutoff
        :type kbcut: int
        :param imp_res: resids to highlight
        :param type: list of strings

        """

        nx.draw_networkx_nodes(G, pos=xypos(G), node_size=500, edgecolors='k', alpha=0.5)
        if imp_res is not None:
            nx.draw_networkx_nodes(G, pos=xypos(G), nodelist=imp_res, node_size=500,
                                   node_color='red', edgecolors='k', alpha=0.6)
        # seperate edgelists
        backbone = [(u, v) for (u, v, d) in G.edges(data=True) if d['kind'] == 'backbone']
        edg_gtkb = [(u, v) for (u, v, d) in G.edges(data=True) if float(d['weight']) > kbcut]
        edg_ltkb = [(u, v) for (u, v, d) in G.edges(data=True) if float(d['weight']) < kbcut]
        # get weights for colouring
        weights = [float(G.get_edge_data(n1, n2)["weight"]) for (n1, n2) in edg_gtkb]

        # plotting
        nx.draw_networkx_edges(G, pos=xypos(G), edgelist=backbone, width=2, edge_color="lightgray")
        edges = nx.draw_networkx_edges(G, pos=xypos(G), edgelist=edg_gtkb, edge_color=weights,
                                       width=4, edge_cmap=plt.cm.cool)
        nx.draw_networkx_labels(G, pos=xypos(G), font_size=12)
        plt.colorbar(edges)
        plt.title("MGT Network Visualization")
        plt.show()

    def draw_sub_protein_graph(self, G, resids, col, res_overlap=None, kbcut=0., intra=False, prefix=None,
                               node_size=750, fontweight="normal", fontsize=24):
        """
        Draw 2D protein graph using NetworkX, only interactions between given resids > kbcut.

        :param G: graph object with all nodes and edges with necessary attributes
        :type G: object

        :param resids: resids whose interactions are plotted
        :param type: list of strings

        :param col: color for resids
        :type col: string

        :param kbcut: draw interaction greater than coupling strength cutoff
        :type kbcut: int

        :param intra: if True, get edgeslists of intra interactions of given resids
                         False, get all interacting edgelists of given resids (default)
        :type intra: Boolean

        :param prefix: prefix for title
        :type prefix: string

        """
        nx.draw_networkx_nodes(G, pos=xypos(G), node_size=node_size, edgecolors='k', alpha=0.5)
        nx.draw_networkx_nodes(G, pos=xypos(G), nodelist=resids, node_size=node_size,
                               node_color=col, edgecolors='k', alpha=1)
        if not res_overlap is None:
            nx.draw_networkx_nodes(G, pos=xypos(G), nodelist=res_overlap, node_size=node_size*2.667,
                                   node_color=col, edgecolors='k', alpha=1)  # 2.667*750=~2000
        # seperate edgelists
        backbone = [(u, v) for (u, v, d) in G.edges(data=True) if d['kind'] == 'backbone']
        drawedges = self.get_interacting_edges(G, resids, kbcut, intra=intra)

        # get weights for colouring
        weights = [float(G.get_edge_data(n1, n2)["weight"]) for (n1, n2) in drawedges]

        # plotting
        nx.draw_networkx_edges(G, pos=xypos(G), edgelist=backbone, width=2, edge_color="lightgray")
        edges = nx.draw_networkx_edges(G, pos=xypos(G), edgelist=drawedges, edge_color=weights,
                                       width=4, edge_cmap=plt.cm.cool, edge_vmin=1.0)

        # here also we can change the node label using labels argument
        # (check networkx documentation for draw_networkx_labels)
        if res_overlap is not None:
            norm_label = {}
            overlap_label = {}
            for node in G.nodes():
                if node not in res_overlap:
                    norm_label[node] = node
                else:
                    overlap_label[node] = node
            overlap_node_edges = self.get_interacting_edges(G, overlap_label, kbcut, intra=intra)
            overlap_weights = [float(G.get_edge_data(n1, n2)["weight"]) for (n1, n2) in overlap_node_edges]
            nx.draw_networkx_edges(G, pos=xypos(G),
                                   edgelist=overlap_node_edges, edge_color=overlap_weights,
                                   width=7, style='dotted',
                                   edge_cmap=plt.cm.cool, edge_vmin=1.0, edge_vmax=np.asarray(weights).max())
            nx.draw_networkx_labels(G, pos=xypos(G), labels=norm_label, font_size=fontsize, font_weight=fontweight)
            nx.draw_networkx_labels(G, pos=xypos(G), labels=overlap_label, font_size=fontsize+2, font_weight=fontweight)
        else:
            nx.draw_networkx_labels(G, pos=xypos(G), font_size=fontsize, font_weight=fontweight)

        cb = plt.colorbar(edges)
        cb.set_label(label=r'Coupling Strength''\n''$kcal/mol/A^2$', weight=fontweight)
        ax = cb.ax
        cblabel = ax.yaxis.label
        cb.ax.tick_params(axis='y', labelsize=30)
        font = matplotlib.font_manager.FontProperties(family='arial', size=30)
        cblabel.set_font_properties(font)
        if prefix is None:
            pass
        else:
            plt.title("{} Network Visualization".format(prefix), fontsize=30)
        plt.show()

    def add_backbone(self, data, chain_id=None, node_name="resid"):
        """
        Add all nodes and their backbone edges.

        :param data: dataframe with basic datas of a protein
        :param chain_id: segment id in original pdb file
        :param node_name: node name to follow, if resid, the resids are taken
                                               if resno, the residue numbers are taken
        :return: Graph object with backbone edges added
        """

        G = self.gx
        if chain_id is not None:
            data = data[data["chain_id"] == chain_id]
        for g, d in data.groupby(["node_id", "chain_id", "resid", "resi_name", "resno"]):
            node_id, chain_id, resid, resi_name, resno = g
            x = d[d["atom"] == "CA"]["x"].values[0]
            y = d[d["atom"] == "CA"]["y"].values[0]
            z = d[d["atom"] == "CA"]["z"].values[0]
            if node_name == "resid":
                G.add_node(
                    str(resid),
                    chain_id=chain_id,
                    node_id=node_id,
                    resi_name=resi_name,
                    resid=resid,
                    resno=resno,
                    x=x,
                    y=y,
                    z=z,
                    features=None,
                )
            if node_name == "resno":
                G.add_node(
                    str(resno),
                    chain_id=chain_id,
                    node_id=node_id,
                    resi_name=resi_name,
                    resno=resno,
                    resid=resid,
                    x=x,
                    y=y,
                    z=z,
                    features=None,
                )

        # Add in edges for amino acids that are adjacent in the linear amino
        # acid sequence.
        for n, d in G.nodes(data=True):
            prev_node = str(int(n) - 1)
            next_node = str(int(n) + 1)

            # Find the previous node in the graph.
            prev_node = [n for n in G.nodes() if (prev_node in n) and (int(prev_node) == int(n))]
            next_node = [n for n in G.nodes() if (next_node in n) and (int(next_node) == int(n))]

            if len(prev_node) == 1:
                if not G.has_edge(n, prev_node[0]):
                    G.add_edge(n, prev_node[0], kind='backbone', weight=1.0)
            if len(next_node) == 1:
                if not G.has_edge(n, next_node[0]):
                    G.add_edge(n, next_node[0], kind='backbone', weight=1.0)
        return G


def to_commaseparray(inp):
    """
    Converts many int in a string to array of ints

    :param inp: string with int separated by spaces
    :return: array of int

    """
    all_res = list()
    for i in inp:
        i = i.split(" ")
        for j in i:
            all_res.append(int(j))
    return np.asarray(all_res)


def xypos(G):
    """
    Get x and y coordinates of given graph object

    :param G: graph object to add x, y coordinates
    :return: graph object with attributes x and y added
    """
    pos = dict()
    for i, d in G.nodes(data=True):
        pos[i] = (float(d["x"]), float(d["y"]))
    return pos


def retrieve_name(var):
    """
    Get function or variable name as string

    :param var: the variable or function as input
    :return: string of var name
    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def _edges(mat, inc_mat=False, resids=None):
    """
    Create weighted edgelists from Kb values

    :param mat: kmat
    :param inc_mat: create incidence matrix
    :type inc_mat: bool
    :return: weighted edgelists
    :rtype: list
    """

    if resids is not None:
        mat = mat.reindex(resids).T.reindex(resids).replace(np.nan, 0.0)
    amat = mat.unstack()
    Npos = mat.shape[0]
    count = 0
    edge_wgt = []
    for i in resids:
        for j in resids:
            #             if not amat.index.isin([(i, j)]).any():
            #                 continue
            if amat[i, j] > 0. and i != j:
                if inc_mat:
                    edge_wgt.append([i, j, np.sqrt(amat[i, j])])  # weight = sqrt of coupling strength
                else:
                    edge_wgt.append([str(i), str(j), amat[i, j]])  # actual weight (exact coupling strength)
                count += 1
    # print "Number of edges is {}".format(count/2)
    return edge_wgt
