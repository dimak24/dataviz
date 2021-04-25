from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Optional, Tuple, Set
import sys
from graphviz import Digraph
import uuid

class Graph:
    adj: Dict['Node', Set['Node']]
    
    def __init__(self):
        self.adj = defaultdict(set)
        
    def add_node(self, v):
        self.adj[v]
        
    def add_edge(self, u, v):
        self.add_node(u)
        self.add_node(v)
        self.adj[u].add(v)
        
    def inverse(self) -> 'Graph':
        g = Graph()
        for u in self.adj:
            g.add_node(u)
            for v in self.adj[u]:
                g.add_edge(v, u)
        return g

def dfs(g, v0, visited=set()):
    if v0 in visited:
        return []
    visited.add(v0)
    result = []
    for u in g.adj[v0]:
        tmp = dfs(g, u, visited)
        result = [*result, *tmp]
        visited.update(tmp)
    result.append(v0)
    return result
        
def scc_kossaraju(graph):
    visited = set()
    ans = []
    node_to_scc = dict()
    scc_graph = Graph()
    g_inv = graph.inverse()
    for node in graph.adj:
        if node not in visited:
            dfs_order = dfs(g_inv, node, visited)[::-1]
            subset = set(graph.adj.keys()) - set(dfs_order)
            for v in dfs_order:
                if v not in subset:
                    scc = dfs(graph, v, subset)
                    subset.update(scc)
                    visited.update(scc)
                    for u in scc:
                        node_to_scc[u] = len(ans)
                    scc_graph.add_node(len(ans))
                    ans.append(scc)
    for i, scc in enumerate(ans):
        for u in scc:
            for v in graph.adj[u]:
                if i != node_to_scc[v]:
                    scc_graph.add_edge(i, node_to_scc[v])
    return ans, scc_graph

def solve_2SAT(
    vars: List[int],
    clauses: List[Tuple[int, int]]
) -> Optional[List[int]]:
    g = Graph()
    for v in vars:
        g.add_node(v)
        g.add_node(-v)
    for u, v in clauses:
        g.add_edge(-u, v)
        g.add_edge(-v, u)
    ans = {}
    def update(scc):
        v0 = next(filter(lambda v: v in ans, scc), scc[0])
        if v0 not in ans:
            ans[v0], ans[-v0] = 1, 0
        for v in scc:
            if v not in ans:
                ans[v], ans[-v] = ans[v0], ans[-v0]
            elif ans[v] != ans[v0]:
                return False
        return True                

    sccs, scc_graph = scc_kossaraju(g)
    visited = set()
    for v in scc_graph.adj:
        if v not in visited:
            order = dfs(scc_graph, v, visited)
            visited.update(order)
            if not all(update(sccs[i]) for i in order):
                return None
    return [ans[v] for v in vars]

def label_placement(labels):
    def rect_intersect(r1, r2):
        x1 = max(r1[0][0], r2[0][0])
        y1 = max(r1[0][1], r2[0][1])
        x2 = min(r1[1][0], r2[1][0])
        y2 = min(r1[1][1], r2[1][1])
        return x1 < x2 and y1 < y2
        
    vars = list(range(1, len(labels) + 1))
    clauses = []
    for (i, li), (j, lj) in combinations(list(enumerate(labels)), 2):
        (xi, yi), (wi, hi), di = li
        (xj, yj), (wj, hj), dj = lj
        for ti in range(2):
            for tj in range(2):
                dxi, dyi = di[ti]
                dxj, dyj = dj[tj]
                ri = ((xi - dxi, yi - dyi), (xi + wi - dxi, yi + hi - dyi))
                rj = ((xj - dxj, yj - dyj), (xj + wj - dxj, yj + hj - dyj))
                if rect_intersect(ri, rj):
                    clauses.append(((2 * ti - 1) * (i + 1), (2 * tj - 1) * (j + 1)))
    return [(xy, wh, d[1 - i]) for (xy, wh, d), i in zip(labels, solve_2SAT(vars, clauses))]

def draw_labels(labels, output='labels'):
    def gen_id():
        return uuid.uuid4().hex.upper()[0:6]
    
    g = Digraph('G', filename=output, format='png', engine="neato")
    for (x, y), (w, h), (dx, dy) in labels:
        g.node(gen_id(), label='',
               shape='point', width='.1',
               pos=f'{x},{y}!')
        g.node(gen_id(), label='',
               shape='box', penwidth='.6', width=str(w), height=str(h),
               pos=f'{x - dx + w / 2},{y - dy + h / 2}!')
    g.render()

labels = []
filename = sys.argv[1]
with open(filename, 'r') as file:
    for line in file:
        xy, wh, d = line.split('\t')
        parse_coords = lambda xy: tuple(map(lambda x: float(x) / 10, xy.split(',')))
        labels.append((parse_coords(xy), parse_coords(wh), list(map(parse_coords, d.split()))))
result = label_placement(labels)
if result:
    draw_labels(result)
else:
    print('there is no appropriate placement :(')
