import argparse
from collections import defaultdict
from graphviz import Digraph
from pygraphml import GraphMLParser
from pygraphml import Graph
from typing import List, Dict

class Node:
    def __init__(self, id: str, data=None):
        self.id = id
        self.data = data

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

class Drawer:
    def __init__(self, graph: Graph):
        self.coords = dict()
        self.graph = graph

    def dot_draw(self, output):
        g = Digraph('G', filename=output, format='png', engine="neato")
        for node in self.graph.adj:
            g.node(node.id,
                label=node.id,
                pos=','.join(map(str, self.coords[node])) + '!',
                width='0.47', fixedsize='true')
            for nbr in self.graph.adj[node]:
                g.edge(node.id, nbr.id)
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
    args = parser.parse_args()
    
    g = RootedTree(Graph.from_graphml(args.input))
    drawer = LayeredTreeDrawer(g)
    drawer.dot_draw(args.output)
