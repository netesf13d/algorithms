# -*- coding: utf-8 -*-
"""
Unfinished algorithms.
"""

from math import ceil

import numpy as np


# =============================================================================
# 
# =============================================================================

def find_cycle_directed(graph: np.ndarray)-> list:
    """
    TODO doesn't work
    Find a cycle in a directed graph.
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(9876)
    >>> G = rng.binomial(1, 0.09, size=(16, 16))
    >>> find_cycle_directed(G)
    [13]
    >>> np.fill_diagonal(G, 0)
    >>> find_cycle_directed(G)
    
    """
    path = []
    preorders = np.full((len(graph),), -1, dtype=int)
    C = 0
    
    def explore(vertex):
        nonlocal C
        preorders[vertex] = C
        C += 1
        path.append(vertex)
    
    explore_stack = []
    while C < len(graph):
        v = np.argmin(preorders)
        explore(v)
        explore_stack.append([v] + [w for w, in np.argwhere(graph[v])])
        curr_stack = explore_stack[-1]
        while curr_stack:
            v = curr_stack.pop(-1)
            
            if preorders[v] == -1: # never been there before -> explore recursively
                explore(v)
                curr_stack = [v] + [w for w, in np.argwhere(graph[v])]
                explore_stack.append(curr_stack)
            
            elif curr_stack and (v in path): # been there before -> found cycle
                return path[path.index(v):]
            
            elif not curr_stack:
                del path[path.index(v)+1:]
                explore_stack.pop(-1)
                if explore_stack: # if there is one, othervitse break the loop
                    curr_stack = explore_stack[-1]
                else:
                    path.pop(-1)
    return []


# =============================================================================
# 
# =============================================================================

def interval_scheduling(vectors: list, weights: int):
    """
    TODO
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    
    """


def complete_kamarkar_karp(S: set, k: int):
    """
    TODO
    Largest differencing method for multi-way partitioning of a set.
    The complete Karmarkar-Karp algorithm finds the optimal solution.
    
    Examples
    --------
    >>> import numpy as np
    """


def bin_packing(S: set, C: int):
    """
    TODO
    Improved Bin Completion for Optimal Bin Packing and Number Partitioning.
    Ethan L. Schreiber and Richard E. Korf
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> S = {8,7,6,5,4}
    >>> bin_packing(S, 12)
    ([[5, 4, 7], [6, 8]], 2)
    >>> S = rng.integers(0, 100, size=(40,))
    >>> bin_packing(S, 3)
    """
    S = np.array(sorted(S))
    if S[-1] > C:
        print("largest element exceeds capacity")
        return
    
    # lower bound from wasted space
    lower = sum(S)
    temp = S
    temp_sum = 0
    while temp.size > 0:
        r = C - temp[-1]
        less_than_r = [s for s in temp[:-1] if s <= r]
        temp_sum += sum(less_than_r)
        if temp_sum <= r:
            lower += r - temp_sum
            temp_sum = 0
        else:
            temp_sum -= r
        temp = temp[len(less_than_r):-1]
    lower = ceil(lower / C)
    
    # upper bound and initial solution from Best Fit Decreasing
    bins = []
    for s in S[::-1]:
        packed = False
        for b in bins:
            if s + sum(b) <= C:
                b.append(s)
                packed = True
                break
        if not packed:
            bins.append([s])
        bins.sort(reverse=True, key=lambda x: sum(x))
    upper = len(bins)
    if upper == lower: # BFD is an optimal solution
        return bins
    
    # Refine solution
    # Improvement : variable ordering
    def sort(x):
        """
        Sort successively by:
            - Largest sum
            - Lowest cardinality
            - Largest smallest unique element
        """
        if x[-1] != x[-2]:
            return (sum(x), -len(x), x[-1])
        else:
            for i in range(2, len(x)-1):
                if x[-i] != x[-i+1] and x[-i] != x[-i-1]:
                    return (sum(x), -len(x), x[-i])
            return (sum(x), -len(x), x[0])
    
    
    
    print("suboptimal", upper, lower)
    best = upper
    
    
    
    
    
    
    
# rng = np.random.default_rng(12345)
# S = [7,7,7]
# print(bin_packing(S, 13))

# S = rng.integers(300, 1000, size=(200,))
# print(bin_packing(S, 1000))