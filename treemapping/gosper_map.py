from typing import Dict, List
import numpy as np
from PIL import Image, ImageColor, ImageDraw
from collections import defaultdict
from itertools import chain

from ConcaveHull import ConcaveHull

class Node:
    def __init__(self, id):
        self.id = id

    def __eq__(self, another):
        return self.id == another.id

    def __hash__(self):
        return hash(self.id)

class DirectedGraph:
    adj: Dict[Node, List[Node]]
    
    def __init__(self):
        self.adj = defaultdict(list)
        
    def add_node(self, v):
        _ = self.adj[v]
        
    def add_edge(self, u, v):
        self.adj[u].append(v)        

class RootedTree(DirectedGraph):
    root: Node

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.add_node(root)

SCALE = 100

def gosper_curve(length):
    v = [
        (1, 0),
        (.5, np.sqrt(3) / 2),
        (-.5, np.sqrt(3) / 2),
        (-1, 0),
        (-.5, -np.sqrt(3) / 2),
        (.5, -np.sqrt(3) / 2),
    ]
    x, y = 0, 0
    v_i = 0
    def _step(size, is_A=True):
        nonlocal x, y, v_i
        if size == 0:
            yield x, y
            x += SCALE * v[v_i][0]
            y += SCALE * v[v_i][1]
        else:
            for op in 'A-B--B+A++AA+B-' if is_A else '+A-BB--B-A++A+B':
                if op == 'A':
                    yield from _step(size - 1, True)
                elif op == 'B':
                    yield from _step(size - 1, False)
                elif op == '+':
                    v_i = (v_i + 1) % 6
                elif op == '-':
                    v_i = (v_i - 1) % 6

    yield from _step(np.ceil(np.log(length) / np.log(7)))

class HexogonalMap:
    def __init__(self):
        self.nodes = defaultdict(int)
        
    def add(self, hexogon):
        for node in hexogon:
            node = (round(node[0]), round(node[1]))
            self.nodes[node] += 1
            if self.nodes[node] == 3:
                del(self.nodes[node])
                
    def get_nodes(self):
        nodes = list(self.nodes.keys())
        if len(nodes) < 4:
            return nodes
        ch = ConcaveHull()
        ch.loadpoints(nodes)
        ch.calculatehull()
        return [(x, y) for x, y in np.vstack(ch.boundary.exterior.coords.xy).T]

class GosperMap:
    def __init__(self, g):
        self.g = g
        self.leaves = []
        self.hexagons = dict()
        self.h = defaultdict(int)
        self.descendants = defaultdict(list)

    def _visit(self, v):
        for u in self.g.adj[v]:
            if self.g.adj[u]:
                self._visit(u)
                self.descendants[v] += self.descendants[u]
            else:
                self.leaves.append(u)
                self.descendants[v].append(u)
            self.h[v] = max(self.h[v], self.h[u] + 1)

    def _make_hexagons(self):
        def _make_hexagon(x, y):
            return (
                (x, y + SCALE * np.sqrt(3) / 3),
                (x + SCALE * .5, y + SCALE * np.sqrt(3) / 6),
                (x + SCALE * .5, y - SCALE * np.sqrt(3) / 6),
                (x, y - SCALE * np.sqrt(3) / 3),
                (x - SCALE * .5 , y - SCALE * np.sqrt(3) / 6),
                (x - SCALE * .5, y + SCALE * np.sqrt(3) / 6)
            )
    
        for v, (x, y) in zip(self.leaves, gosper_curve(len(self.leaves))):
            self.hexagons[v] = _make_hexagon(x, y)

    def _next_color(self):
        return (*np.random.randint(256, size=3), 50)
            
    def _draw(self, draw):
        def _draw(v):
            color = self._next_color()
            ch = HexogonalMap()
            for u in self.descendants[v]:
                ch.add(self.hexagons[u])
                if self.h[u] == 0:
                    draw.polygon(self.hexagons[u], outline=None, fill=color)
            for u in self.g.adj[v]:
                _draw(u)
            
            ch = ch.get_nodes()
            if ch and v != self.g.root:
                outline = (255 if self.h[v] == 3 else 0, 0, 0, self.h[v] * 90)
                draw.polygon(ch, outline=outline, fill=None)

        _draw(self.g.root)
        
    def run(self, output='graph.png'):
        self._visit(self.g.root)
        self._make_hexagons()
        
        image = Image.new(mode='RGBA', size=(3000, 3000))
        for u in self.hexagons:
            self.hexagons[u] = tuple(
                (x + image.size[0] / 2, y + image.size[1] / 2)
                for x, y in self.hexagons[u])
        draw = ImageDraw.Draw(image)
        self._draw(draw)
        image.save(output, 'png')

if __name__ == '__main__':
    g = RootedTree(Node(0))
    for i in range(1, 8):
        g.add_edge(Node(0), Node(i))

    id_ = 9
    def _complete_tree(g, level, size, root):
        global id_
        if level:
            for i in range(size):
                g.add_edge(root, Node(id_))
                id_ += 1
                _complete_tree(g, level - 1, size, Node(id_ - 1))
    for root in range(1, 8):
        _complete_tree(g, 3, 3, Node(root))

    GosperMap(g).run()
