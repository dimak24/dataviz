import argparse
from collections import defaultdict
from graphviz import Digraph
from pygraphml import GraphMLParser
from pygraphml import Graph
from typing import List, Dict
from enum import Enum
import uuid
import scipy.optimize as spopt
import numpy as np
import json

class Node:
    def __init__(self, id: str, dummy=False):
        self.dummy = dummy
        self.id = uuid.uuid4().hex.upper()[0:6] if self.dummy else id

    def __eq__(self, another):
        return self.id == another.id

    def __hash__(self):
        return hash(self.id)

class Graph:
    adj: Dict[Node, List[Node]]
    
    def __init__(self):
        self.adj = defaultdict(list)

    @staticmethod
    def from_graphml(fname: str) -> Graph:
        parser = GraphMLParser()
        gml = parser.parse(fname)
        g = Graph()
        for node in gml._nodes:
            g.adj[Node(id=node.id)]
        for edge in gml._edges:
            g.adj[Node(id=edge.node1.id)].append(Node(id=edge.node2.id))
        return g

def topsort(graph: Graph) -> (bool, List[Node]):
    class Status:
        FREE = 0
        MARK = 1
        DONE = 2

    status = {node: Status.FREE for node in graph.adj}
    result = []
    def _dfs(v):
        if status[v] == Status.DONE:
            return True
        if status[v] == Status.MARK:
            return False
        status[v] = Status.MARK
        for u in graph.adj[v]:
            if not _dfs(u):
                return False
        status[v] = Status.DONE
        result.append(v)
        return True
    
    while len(result) < len(status):
        for node, st in status.items():
            if st == Status.FREE:
                if not _dfs(node):
                    return False, []
                break
    return True, result

class RootedTree(Graph):
    root: Node
    
    def __init__(self, g: Graph):
        self.adj = g.adj
        for node in self.adj:
            is_root = True
            for nbrs in self.adj.values():
                if node in nbrs:
                    is_root = False
                    break
            if is_root:
                self.root = node
                break

class DAG(Graph):
    def __init__(self, g: Graph):
        self.adj = g.adj
        is_dag, self.toporder = topsort(g)
        if not is_dag:
            raise RuntimeError('not a DAG')

class Drawer:
    def __init__(self, graph: Graph):
        self.coords = dict()
        self.graph = graph

    def dot_draw(self, output):
        g = Digraph('G', filename=output, format='png', engine="neato")
        for node in self.graph.adj:
            g.node(node.id,
                label=node.id if not node.dummy else '',
                pos=','.join(map(str, self.coords[node])) + '!',
                shape='point' if node.dummy else 'circle',
                width='.0' if node.dummy else '.47', fixedsize='true')
            for nbr in self.graph.adj[node]:
                g.edge(node.id, nbr.id,
                       arrowhead='none' if nbr.dummy else 'normal')
        g.render()

class LayeredTreeDrawer(Drawer):
    def _dfs(self, node, x=0, y=0):
        if not self.graph.adj[node]:
            return {node: (x, y)}, [x], [x]
        x += 1
        coords, left, right = None, None, None
        x0 = 0
        for child in self.graph.adj[node]:
            coords_, left_, right_ = self._dfs(child, x, y - 1)
            x += len(coords_)
            if coords is None:
                coords = coords_
                right = right_
                left = left_
            else:
                c = 1 + max(map(lambda rl: rl[0] - rl[1], zip(right, left_)))
                right[:len(right_)] = [r + c for r in right_]
                left += [l + c for l in left_[len(left):]]
                for node_, (x_, y_) in coords_.items():
                    coords[node_] = (x_ + c, y_)
            x0 += coords[child][0]
        coords[node] = (x0 / len(self.graph.adj[node]), y)
        right = [coords[node][0]] + right
        left = [coords[node][0]] + left
        return coords, left, right

    def _process(self):
        self.coords, _, _ = self._dfs(self.graph.root)

    def __init__(self, tree: RootedTree):
        super().__init__(tree)
        self._process()

class CoffmanGrahamDrawer(Drawer):
    def _get_layers_fixed_width(self) -> (List[List[Node]], Dict[Node, int]):
        layers = []
        node_layer = defaultdict(int)
        for node in self.graph.toporder:
            if not layers or len(layers[-1]) == self.W \
                or any(node_layer[u] >= len(layers) for u in self.graph.adj[node]):
                layers.append([])
            layers[-1].append(node)
            node_layer[node] = len(layers)
        for node in node_layer:
            node_layer[node] -= len(layers)
            node_layer[node] *= -1

        return layers[::-1], node_layer

    def _get_layers_min_dummy(self) -> List[List[Node]]:
        c = [0 for node in self.graph.adj]
        A_ub = []
        indices = {node: i for i, node in enumerate(self.graph.adj)}
        for u in self.graph.adj:
            for v in self.graph.adj[u]:
                c[indices[v]] -= 1
                c[indices[u]] += 1
                A_ub.append(np.zeros(shape=(len(indices),)))
                A_ub[-1][indices[v]] = 1
                A_ub[-1][indices[u]] = -1
        A_ub = np.array(A_ub)
        res = spopt.linprog(c, bounds=(0, np.inf), A_ub=A_ub, b_ub=-np.ones(shape=(1, A_ub.shape[0])))
        res.x -= max(res.x)
        res.x *= -1
        x = [int(round(xi)) for xi in res.x]

        layers = [[] for _ in range(max(x) + 1)]
        node_layer = dict()
        for node, i in indices.items():
            node_layer[node] = x[i]
            layers[x[i]].append(node)

        return layers, node_layer

    def _add_dummy(self, layers, node_layer):
        for i, layer in enumerate(layers):
            for v in layer:
                for u in self.graph.adj[v]:
                    if node_layer[u] > 1 + i:
                        self.graph.adj[v].remove(u)
                        dummy = Node('', dummy=True)
                        self.graph.adj[v].append(dummy)
                        self.graph.adj[dummy].append(u)
                        layers[i + 1].append(dummy)
                        node_layer[dummy] = 1 + i
        return layers

    def _process(self):
        # split nodes into layers, get y coords
        layers = self._add_dummy(
            *(self._get_layers_fixed_width()
              if self.W else self._get_layers_min_dummy())
        )
        # get x coords, reducing crossings with the barycenter method
        for y, layer in enumerate(layers[::-1]):
            xs = np.zeros(len(layer))
            for i, node in enumerate(layer):
                if y == 0:
                    xs[i] = i - (len(layer) - 1) / 2
                else:
                    xs[i] = (sum(self.coords[u][0] for u in self.graph.adj[node])
                        / max(1, len(self.graph.adj[node])))
            indices = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = sorted(xs)
            for i, x in enumerate(xs):
                if i > 0 and x < xs[i - 1] + 1:
                    xs[i:] += (1 - x + xs[i - 1])
            for idx, x in zip(indices, xs):
                self.coords[layer[idx]] = (x, y)

    def __init__(self, graph: DAG, W: int=0):
        super().__init__(graph)
        self.W = W
        self._process()    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A small graph vizualization tool')
    parser.add_argument('--input',
                        help='input file (currently works only with graphml format)',
                        required=True)
    parser.add_argument(
        '--output',
        help='result (given FNAME, creates files '
              '"FNAME" with graph representation in dot format and '
              '"FNAME.png" with a picture',
        default='graph')
    parser.add_argument(
        '--alg',
        help='vizualization algorithm',
        required=True)
    parser.add_argument(
        '--params',
        help='algo parameters',
        default='{}')
    args = parser.parse_args()
    
    algo_params  = json.loads(args.params)
    graph = Graph.from_graphml(args.input)
    if args.alg == 'layered-tree':
        drawer = LayeredTreeDrawer(RootedTree(graph))
    elif args.alg == 'coffman-graham':
        drawer = CoffmanGrahamDrawer(
            DAG(graph),
            W=0 if 'W' not in algo_params else algo_params['W'],
        )
    drawer.dot_draw(args.output)
    
    

