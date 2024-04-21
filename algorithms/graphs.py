# -*- coding: utf-8 -*-
"""
Graph algorithms.

* Connected components of a graph
    - <connected_components> : Find the connected components in an undirected
      graph.
    - <strongly_connected_components> : Find the strongly connected components
      in a directed graph.
* Cycles of a graph
    - <elementary_cycles_directed> : Find the elementary cycles in a directed
      graph.
    - <elementary_cycles_undirected> : Find the elementary cycles in an
      undirected graph.
    - <find_cycle_undirected> : Find a cycle in an undirected graph.
* Shortest paths
    - <shortest_path_dijkstra> : Dijkstra's algorithm to find single-source
      shortest paths in a positive-weighted undirected graph.
    - <shortest_path_faster> : Shortest path faster algorithm to find
      single-source shortest paths in a weighted graph.
    - <get_path> : Shortest path to target from the list of predecessors
      established by `shortest_path` algorithms.
    - <uniform_cost_search> : Uniform cost search algorithm to find
      single-source shortest paths in a positive-weighted undirected graph
      not known a priori.
"""

from typing import Callable

import numpy as np



# =============================================================================
# Connected components
# =============================================================================

def connected_components(graph: np.ndarray)-> list[set]:
    """
    Find the connected components in an undirected graph.
    
    graph : 2D np.ndarray[bool], symmetric
        graph[i, j] = graph[j, i] = 1 if i <-> j 
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(6789)
    >>> G = rng.binomial(1, 0.07, size=(16, 16))
    >>> connected_components(G | G.transpose())
    [{0, 12, 13, 15}, {1, 3, 4, 6}, {2, 5}, {7}, {8}, {9, 10}, {11}, {14}]
    """
    ccomp = []
    preorders = np.full((len(graph),), -1, dtype=int)
    C = 0
    
    while C < len(graph):
        v = np.argmin(preorders)
        preorders[v] = C
        C += 1
        curr_comp = {v}
        explore_stack = [w for w, in np.argwhere(graph[v])]
        while explore_stack:
            v = explore_stack.pop(-1)
            
            if preorders[v] == -1:
                preorders[v] = C
                C += 1
                curr_comp.add(v)
                explore_stack += [w for w, in np.argwhere(graph[v])]
        
        ccomp.append(curr_comp)
    return ccomp


def strongly_connected_components(graph: np.ndarray)-> list[set]:
    """
    Find the strongly connected components of a directed graph with a
    path-based approach.
    See https://en.wikipedia.org/wiki/Path-based_strong_component_algorithm
    
    graph : 2D np.ndarray[bool]
        graph[i, j] = 1 if i -> j 
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(9876)
    >>> G = rng.binomial(1, 0.09, size=(16, 16))
    >>> strongly_connected_components(G)
    [{13}, {0, 5, 6}, {10}, {15}, {12}, {7}, {11}, {1, 8, 9}, {2}, {4}, {3}, {14}]
    """
    ccomp = []
    not_distinct, unassigned = [], []
    preorders = np.full((len(graph),), -1, dtype=int)
    C = 0
    
    def explore(vertex):
        nonlocal C
        preorders[vertex] = C
        C += 1
        not_distinct.append(vertex)
        unassigned.append(vertex)
    
    explore_stack = []
    while np.any(preorders == -1):
        v = np.argmin(preorders)
        explore(v)
        explore_stack.append([v] + [w for w, in np.argwhere(graph[v])])
        cur_stack = explore_stack[-1]
        while cur_stack:
            v = cur_stack.pop(-1)
            
            if preorders[v] == -1:
                explore(v)
                cur_stack = [v] + [w for w, in np.argwhere(graph[v])]
                explore_stack.append(cur_stack)
            
            elif cur_stack and v in unassigned:
                while preorders[not_distinct[-1]] > preorders[v]:
                    not_distinct.pop(-1)
            
            elif not cur_stack:
                if v == not_distinct[-1]: # found a connected component
                    comp = set()
                    while True:
                        s = unassigned.pop(-1)
                        comp.add(s)
                        if s == v:
                            not_distinct.pop(-1)
                            break
                    ccomp.append(comp)
                # switch to next exploration stack
                explore_stack.pop(-1)
                if explore_stack: # if there is one, othervitse break the loop
                    cur_stack = explore_stack[-1]
    
    return ccomp


# =============================================================================
# Cycles finding algorithms
# =============================================================================

def elementary_cycles_directed(graph: np.ndarray):
    """
    Find the elementary cycles in a directed graph. Donald Johnson algorithm.
    From:
    * "Finding all the Elementary Circuits of a Directed Graph" - Donald B.
      Johnson. https://www.cs.tufts.edu/comp/150GA/homeworks/hw1/Johnson%2075.PDF
    
    graph : 2D np.ndarray[bool]
        graph[i, j] = 1 if i -> j.
        The graph is modified in-place by the algorithm.
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(9876)
    >>> G = rng.binomial(1, 0.09, size=(16, 16))
    >>> elementary_cycles_directed(G)
    [(13,), (0, 5, 6), (1, 9, 8)]
    >>> rng = np.random.default_rng(9876)
    >>> G = rng.binomial(1, 0.1, size=(16, 16))
    >>> elementary_cycles_directed(G)
    [(13,), (0, 13, 3), (0, 13, 3, 4, 8, 1), (0, 13, 3, 4, 8, 1, 9, 5, 6),
     (0, 13, 3, 4, 8, 1, 9, 11, 12), (0, 5, 6), (0, 5, 6, 13, 3),
     (0, 5, 6, 13, 3, 4, 8, 1), (0, 5, 6, 13, 3, 4, 8, 1, 9, 11, 12),
     (1, 9, 5, 6, 13, 3, 4, 8), (1, 9, 8), (1, 9, 13, 3, 4, 8)]
    """
    blocked = np.zeros(len(graph), dtype=bool)
    
    # Get self cycles
    cycles = [tuple(x) for x in np.argwhere(np.diag(graph))]
    np.fill_diagonal(graph, 0)
    
    # Get non-trivial cycles
    s = 0
    while s < len(graph):
        cycle = []
        stack = []
        scc = [c for c in strongly_connected_components(graph[s:, s:])
               if len(c) > 1]
        if scc:
            scc = sorted(v + s for v in min(scc, key=lambda x: min(x)))
            G = graph[np.ix_(scc, scc)]
            s = scc[0]
            ## find cycles
            blocked[:] = False
            cycle.append(scc[0])
            stack += [(v, 1) for v, in np.argwhere(G[0])]
            while stack:
                v, i = stack.pop(-1)
                blocked[cycle[i:]] = False
                cycle = cycle[:i]
                
                if scc[v] == s:
                    cycles.append(tuple(cycle))
                
                elif not blocked[scc[v]]:
                    blocked[scc[v]] = True
                    cycle.append(scc[v])
                    stack += [(w, i+1) for w, in np.argwhere(G[v])][::-1]
            s += 1
        else:
            break
    return cycles


def elementary_cycles_undirected(graph: np.ndarray):
    """
    Find the elementary cycles in an undirected graph.
    The idea is to build a spanning tree over the graph, then each additional
    edge makes up a cycle.
    From:
    * "An algorithm for finding a fundamental set of cycles of a graph" - Keith
      Paton. https://dl.acm.org/doi/pdf/10.1145/363219.363232
    
    graph : 2D np.ndarray[bool], symmetric
        graph[i, j] = graph[j, i] = 1 if i <-> j
        The graph is modified in-place by the algorithm.
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> G = rng.binomial(1, 0.11, size=(16, 16))
    >>> elementary_cycles_undirected(G | G.transpose())
    [[2, 0, 15], [2, 0, 11], [1, 7, 3, 2, 0, 14], [1, 7, 6], [8, 3, 2, 0, 10],
     [5, 2, 0, 9], [9, 0, 15], [9, 0, 10], [10, 0, 11], [10], [11, 0, 14], [12]]
    """
    cycles = []
    prev = np.full((len(graph),), -1, dtype=int)
    preorders = np.full((len(graph),), -1, dtype=int)
    C = 0
    
    def path_to(node, leaf):
        path = [leaf]
        v = leaf
        while (v := prev[v]) != -1:
            path.append(v)
            if v == node:
                return path
        raise ValueError(f"no ascending path from {leaf} to {node}")
    
    vstack = []
    while C < len(graph):
        v = np.argmin(preorders)
        preorders[v] = C
        C += 1
        stack = [(w, v) for w, in np.argwhere(graph[v])]
        graph[v] = graph[:, v] = 0
        while stack: # visit all descendants of a node
            v, parent = stack.pop(-1)
            
            if v == parent: # self loop
                cycles.append([v])

            elif preorders[v] == -1: # never been there before
                preorders[v] = C
                C += 1
                prev[v] = parent
                vstack.append(v)
            
            else: # been there before -> found cycle
                path = path_to(prev[v], parent)
                path.append(v)
                cycles.append(path)
            
            while not stack: # prepare stack of descendants of v to visit
                try:
                    v = vstack.pop(-1)
                except IndexError:
                    break
                else:
                    if (edges := np.argwhere(graph[v])).size > 0:
                        stack = [(w, v) for w, in edges]
                        graph[v] = graph[:, v] = 0
    return cycles


def find_cycle_undirected(graph: np.ndarray)-> list:
    """
    Find a cycle in an undirected graph.
    
    graph : 2D np.ndarray[bool], symmetric
        graph[i, j] = graph[j, i] = 1 if i <-> j
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> G = rng.binomial(1, 0.11, size=(16, 16))
    >>> np.fill_diagonal(G, 0) # remove self cycles
    >>> find_cycle_undirected(G | G.transpose())
    [11, 0, 14]
    """
    if np.any(np.diag(graph)): # vertex connected to itself
        return [np.argmax(np.diag(graph))]
    
    preorders = np.full((len(graph),), -1, dtype=int)
    prev = np.full((len(graph),), -1, dtype=int) # predecessor (tree structure)
    C = 0
    
    def path_to_root(vertex):
        path = [vertex]
        while (vertex := prev[vertex]) != -1:
            path.append(vertex)
        return path
    
    explore_stack = []
    while C < len(graph):
        v = np.argmin(preorders)
        preorders[v] = C
        C += 1
        parent = v
        curr_stack = [v] + [v+1+w for w, in np.argwhere(graph[v, v+1:])]
        explore_stack.append(curr_stack)
        while curr_stack:
            v = curr_stack.pop(-1)
            
            if preorders[v] == -1: # never been there before -> explore recursively
                preorders[v] = C
                C += 1
                prev[v] = parent
                curr_stack = [v] + [v+1+w for w, in np.argwhere(graph[v, v+1:])]
                explore_stack.append(curr_stack)
                parent = v
            
            elif curr_stack and preorders[v] != -1: # been there before -> found cycle
                path1, path2 = path_to_root(parent), path_to_root(v)
                i1, i2 = 0, 0
                while True:
                    if path1[i1] == path2[i2]:
                        break
                    i1 += 1
                    if path1[i1] == path2[i2]:
                        break
                    i2 += 1
                return path1[:i1] + path2[-i2::-1]
            
            elif not curr_stack:
                explore_stack.pop(-1)
                if explore_stack: # if there is one, otherwise break the loop
                    curr_stack = explore_stack[-1]
                    parent = prev[v]
    return []


# =============================================================================
# Paths finding algorithms
# =============================================================================

def shortest_path_dijkstra(graph: np.ndarray, source: int)-> tuple[list, list]:
    """
    Single-source shortest paths in a positive-weighted undirected graph using
    Dijkstra's algorithm.
    See https://en.wikipedia.org/wiki/Dijkstra's_algorithm
    
    Parameters
    ----------
    graph : 2D np.ndarray[float >= 0], symmetric
        Undirected and positive weighted graph.
    source : int
        The starting vertex index.
    
    Returns
    -------
    dist : 1D np.ndarray[float]
        Minimal distances from source vertex to graph vertices.
        dist[source] = 0.; dist[i] = inf if i is not connected to source.
    prev : 1D np.ndarray[int]
        prev[i] is the predecessor of vertex i in the shortest path from
        source to i.
        prev[source] = -1; prev[i] = -2 if vertex i is not connected to source.
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> G = rng.uniform(0, 100., size=(16, 16))
    >>> G *= rng.binomial(1, 0.15, size=(16, 16))
    >>> G = G + G.transpose()
    >>> shortest_path_dijkstra(G, 0)
    (array([  0.        ,  19.72029737,  82.32110845,  52.67949657,
            114.36239429,  27.7776856 , 111.17179708,  55.51697574,
            118.08315773,  74.47238287,  24.82457146,   6.77338975,
             27.15626118,  53.83790907,  36.31347828,  46.77646046]),
     array([-1, 11, 15, 12,  3, 11,  9, 15, 14,  3,  0,  0, 11, 11, 10,  5]))
    
    >>> rng = np.random.default_rng(12345)
    >>> G = rng.uniform(0, 100., size=(16, 16))
    >>> G *= rng.binomial(1, 0.08, size=(16, 16))
    >>> G = G + G.transpose()
    >>> shortest_path_dijkstra(G, 0)
    (array([  0.        , 111.32808597, 174.50208259,          inf,
            327.81236481, 119.38547419,          inf, 147.12476434,
            252.75923162,          inf, 148.72875656,  98.38117835,
             77.99830692,  73.54331873, 293.69326828, 138.38424905]),
     array([-1, 11,  7, -2,  8, 11, -2, 15,  2, -2, 11, 12, 13,  0,  1,  5]))
    """
    dist = np.full((len(graph),), np.inf, dtype=graph.dtype) # shortest distance from source
    dist[source] = 0.
    prev = np.full((len(graph),), -2, dtype=int) # predecessors
    prev[source] = -1
    explore_set = {v: dist[v] for v in range(len(graph))}
    
    while explore_set:
        v, _ = min(explore_set.items(), key=lambda x: x[1])
        del explore_set[v]
        
        for w, in np.argwhere(graph[v]):
            if w in explore_set:
                alt = dist[v] + graph[v, w]
                if alt < dist[w]:
                    dist[w] = alt
                    prev[w] = v
                    explore_set[w] = alt
    return dist, prev


def shortest_path_faster(graph: np.ndarray, source: int):
    """
    TODO implement negative weight version
    
    Single-source shortest paths in a positive-weighted graph using the
    shortest path faster algorithm. Works on both directed and undirected
    graphs.
    See https://en.wikipedia.org/wiki/Shortest_path_faster_algorithm
    
    Parameters
    ----------
    graph : 2D np.ndarray[float]
        Weighted directed graph.
    source : int
        The starting vertex index.
    
    Returns
    -------
    dist : 1D np.ndarray[float]
        Minimal distances from source vertex to graph vertices.
        dist[source] = 0.; dist[i] = inf if i is not connected to source.
    prev : 1D np.ndarray[int]
        prev[i] is the predecessor of vertex i in the shortest path from
        source to i.
        prev[source] = -1; prev[i] = -2 if vertex i is not connected to source.
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> G = rng.uniform(0, 100, size=(16, 16)).astype(np.float16)
    >>> G *= rng.binomial(1, 0.15, size=(16, 16))
    >>> shortest_path_faster(G, 0)
    (array([  0.   , 133.2  ,  58.88 ,     inf,     inf, 113.4  ,     inf,
             86.25 , 118.06 , 185.6  ,  24.83 ,  56.66 ,  49.56 ,   9.586,
             36.3  ,  94.4  ], dtype=float16),
     array([-1, 14, 11, -2, -2, 15, -2,  2, 14,  8,  0, 13, 13,  0, 10,  2]))
    
    >>> rng = np.random.default_rng(12345)
    >>> G = rng.uniform(0, 100., size=(16, 16))
    >>> G *= rng.binomial(1, 0.15, size=(16, 16))
    >>> G = G + G.transpose()
    >>> shortest_path_faster(G, 0) # same result as shortest_path_dijkstra(G, 0)
    (array([  0.        ,  19.72029737,  82.32110845,  52.67949657,
            114.36239429,  27.7776856 , 111.17179708,  55.51697574,
            118.08315773,  74.47238287,  24.82457146,   6.77338975,
             27.15626118,  53.83790907,  36.31347828,  46.77646046]),
     array([-1, 11, 15, 12,  3, 11,  9, 15, 14,  3,  0,  0, 11, 11, 10,  5]))
    """
    dist = np.full((len(graph),), np.inf, dtype=graph.dtype)
    dist[source] = 0
    prev = np.full((len(graph),), -2, dtype=int) # predecessors
    prev[source] = -1
    frontier = {v: dist[v] for v in range(len(graph))}
    
    while frontier:
        node, _ = min(frontier.items(), key=lambda x: x[1])
        del frontier[node]
        
        for v, in np.argwhere(graph[node]):
            alt = dist[node] + graph[node, v]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = node
                frontier[v] = alt
    
    return dist, prev


def get_path(prev: list[int], target: int)-> list[int]:
    """
    From the list of predecessors, determine the shortest path from source
    vertex (prev[source] = -1) to target vertex.
    
    Raises ValueError if the target is not connected (prev[target] = -2).
    
    Examples
    --------
    >>> prev = [-1, 11, 15, 12,  3, 11,  9, 15, 14,  3,  0,  0, 11, 11, 10,  5]
    >>> get_path(prev, 6) # path from 0 to 6
    [0, 11, 12, 3, 9, 6]
    >>> prev = [-1, 11,  7, -2,  8, 11, -2, 15,  2, -2, 11, 12, 13,  0,  1,  5]
    >>> get_path(prev, 4)
    [0, 13, 12, 11, 5, 15, 7, 2, 8, 4]
    >>> get_path(prev, 6)
    ValueError: target 6 not connected to source
    """
    if prev[target] == -2:
        raise ValueError(f"target {target} not connected to source")
    path = [target]
    v = target
    while prev[v] != -1:
        v = prev[v]
        path.append(v)
    return path[::-1]


def uniform_cost_search(graph_gen: Callable, source, target):
    """
    Single-source shortest path in a positive-weighted undirected graph using
    uniform cost search algorithm.
    The graph is unknown a priori and reveals the vertices connected to a node
    when queried.
    See https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
    
    !!! NOT TESTED
    
    graph_gen : Callable
        The graph structure.
        Calling graph(node) returns a dict {connected vertices: cost}.
    source : 
        The source vertex.
    target : 
        The target vertex.
    
    Examples
    --------
    """
    node = source
    prev, dist = {source: None}, {source: 0}
    frontier = {source: 0}
    
    while frontier:
        node, _ = min(frontier.items(), key=lambda x: x[1])
        del frontier[node]
        
        if node == target:
            return dist, prev
        
        for v, cost in graph_gen(node).items(): # dict
            alt = dist[node] + cost
            if v not in prev:
                prev[v] = node
                dist[v] = alt
                frontier[v] = alt
            elif v in frontier and frontier[v] > alt:
                frontier[v] = alt

    return dist, prev