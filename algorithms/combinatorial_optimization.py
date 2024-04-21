# -*- coding: utf-8 -*-
"""
Combinatorial optimization algorithms.

* Knapsack problem
    - <knapsack_dict> : Solve knapsack problem using a `dict` data struture.
    - <knapsack_array> : Solve knapsack problem using an `array` data struture.
* Partitioning
    - <subset_sum> : Find a subset which elements sum precisely to a given S.
    - <largest_differencing_method> : Multi-way partitioning of a set using the
      largest differencing method.
* Subsequences
    - <maximum_subarray> : Kadane's algorithm to solve the maximum subarray
      problem.
    - <longest_increasing_subsequence> : Solve the longest increasing
      subsequence problem.
    - <longest_common_subsequence> : Find the longest subsequence common to two
      sequences.
"""

from copy import copy
from itertools import combinations, product

import numpy as np


# =============================================================================
# Knapsack problem
# =============================================================================

def knapsack_dict(weights: list[float],
                  max_weight: float)-> tuple[float, list[int]]:
    """
    Solve knapsack problem using a hash table data structure.
    See https://en.wikipedia.org/wiki/Knapsack_problem
    
    Returns
    -------
    best_weight : float
        The highest weight below max_weight that can be obtained by
        combining the items.
    best_combination : set[int]
        Indices of a combination of items that sums to the best weight.
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> wt = rng.integers(0, 100, size=(40,))
    >>> max_wt = 600
    >>> knapsack_dict(wt, max_wt)
    (600, {18, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39})
    
    >>> wt = [349, 113, 394, 158, 102, 398, 321, 338, 494, 195]
    >>> knapsack_dict(wt, max_wt)
    (598, {3, 4, 7})
    """
    contents = {0: set()}
    for i, w in enumerate(weights):
        temp = copy(contents)
        for W, subset in contents.items():
            new_weight = w + W
            if new_weight <= max_weight:
                temp[new_weight] = subset | {i}
        contents = temp
    return max(contents), contents[max(contents)]
    

def knapsack_array(weights: list[int], max_weight: int):
    """
    Solve knapsack problem using an array data structure.
    See https://en.wikipedia.org/wiki/Knapsack_problem
    
    Returns
    -------
    best_weight : float
        The highest weight below max_weight that can be obtained by
        combining the items.
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> wt = rng.integers(0, 100, size=(40,))
    >>> max_wt = 600
    >>> knapsack_array(wt, max_wt)
    600
    
    >>> wt = [349, 113, 394, 158, 102, 398, 321, 338, 494, 195]
    >>> knapsack_array(wt, max_wt)
    598
    """
    # values[i, j] = max weight < j using the first i items
    values = np.full((len(weights)+1, max_weight+1), -1, dtype=int)
    values[0], values[:, 0] = 0, 0
    
    def compute_max_weight(i, j):
        if i == 0 or j == 0:
            return
        w = weights[i-1]
        if values[i-1, j] == -1:
            compute_max_weight(i-1, j)
        if w > j:
            values[i, j] = values[i-1, j]
        else:
            if values[i-1, j - w] == -1:
                compute_max_weight(i-1, j - w)
            values[i, j] = max(values[i-1, j - w] + w, values[i-1, j])
    compute_max_weight(len(weights), max_weight)
    
    return values[len(weights), max_weight]


# =============================================================================
# Partitioning
# =============================================================================

def subset_sum(a: list[float], S: int)-> set[int] | None:
    """
    Solve the subset sum problem using a dict data structure.
    See https://en.wikipedia.org/wiki/Subset_sum_problem
    
    Returns
    -------
    subset_sum_items : set[int] or None
        Indices of a combination of items that sums to S, or None is no
        combination is found.
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> a = rng.integers(0, 1000, size=(20,))
    >>> subset_sum(a, 1787)
    {0, 1, 3, 11, 14}
    """
    sums = {0: set()}
    for i, n in enumerate(a):
        temp = copy(sums)
        for s, subset in sums.items():
            temp[n + s] = subset | {i}
        sums = temp
        if S in sums and sums[S]:
            return sums[S]
    return None


def largest_differencing_method(S: set, k: int)-> tuple[list[set[int]], float]:
    """
    Multi-way partitioning of a set using the largest differencing method.
    See https://en.wikipedia.org/wiki/Largest_differencing_method
    
    Returns
    -------
    partition : list[set[int]]
        Partition of the set S in k subsets with small difference between the
        subsets. The partitions are ordered by decreasing sum.
    max_diff : float
        Maximum difference between the sum of two partitions.
    
    Examples
    --------
    >>> S = {8,7,6,5,4}
    >>> largest_differencing_method(S, 2)
    ([{4, 5, 7}, {8, 6}], 2)
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> S = rng.integers(0, 100, size=(40,))
    >>> largest_differencing_method(S, 3)
    ([{32, 64, 69, 70, 11, 44, 78, 20, 22, 88, 26, 61, 94},
      {67, 69, 7, 73, 9, 79, 47, 18, 83, 22, 56, 24, 94, 31},
      {33, 98, 66, 67, 39, 73, 59, 77, 13, 21, 22, 91}],
     20)
    """
    # initialize possible partitions
    partitions = []
    diffs = np.zeros((len(S),), dtype=float)
    for i, s in enumerate(S):
        partitions.append([{s}] + [set() for _ in range(k-1)])
        diffs[i] = s
    
    for p in range(len(S) - 1):
        i0, j0, *_ = np.argsort(diffs)[-p-2:]
        part = partitions[j0]
        for i, si in enumerate(reversed(partitions[i0])):
            part[i] |= si
        part.sort(key=lambda x: sum(x), reverse=True)
        temp = [sum(si) - sum(sj) for si, sj in combinations(part, 2)]
        diffs[i0], diffs[j0] = np.nan, (max(temp) - min(temp))
    
    partition = sorted(partitions[np.nanargmin(diffs)],
                       key=lambda x: sum(x), reverse=True)
    max_diff = abs(sum(part[0]) - sum(part[-1]))
    return partition, max_diff


# =============================================================================
# Subsequences
# =============================================================================

def maximum_subarray(a: list)-> tuple[list, float]:
    """
    Kadane's algorithm to solve the maximum subarray problem.
    See https://en.wikipedia.org/wiki/Maximum_subarray_problem
    
    Returns
    -------
    subarray : list
        The subarray with maximum sum.
    max_sum : float
        Sum of the subarray with largest sum.
    
    Examples
    --------
    >>> [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    >>> maximum_subarray(a)
    ([4, -1, 2, 1], 6)
    >>> a = [-97, 36, 18, -90, 81, -56, -49, -64, -34, -65, -31, 62, -10, 84]
    >>> maximum_subarray(a)
    ([62, -10, 84], 136)
    >>> a = [-1, -2, -3]
    >>> maximum_subarray(a)
    ([], 0)
    """
    max_sum = 0
    max_idx = (0, 0)
    
    i0 = 0
    cur_sum = 0
    for i, val in enumerate(a):
        if val >= cur_sum + val:
            i0 = i
            cur_sum = val
        else:
            cur_sum += val
        
        if cur_sum > max_sum:
            max_sum = cur_sum
            max_idx = (i0, i+1)
    
    return a[max_idx[0]:max_idx[1]], max_sum


def longest_increasing_subsequence(a: list)-> list:
    """
    Solve the longest increasing subsequence problem using a variant of
    patience sorting.
    See https://en.wikipedia.org/wiki/Patience_sorting
    
    Examples
    --------
    >>> a = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    >>> longest_increasing_subsequence(a)
    [0, 2, 6, 9, 11, 15]
    >>> a = [ 95  95  97 -24 -66  84 -80 -48 -73 -37   7 -77  59 -52  57]
    >>> longest_increasing_subsequence(a)
    [-80, -73, -37, 7, 57]
    """
    piles = []
    back_ptr = []
    for elt in a:
        stacked = False
        for i, pile in enumerate(piles):
            if not stacked and elt <= pile[-1]:
                pile.append(elt)
                if i > 0:
                    back_ptr[i-1].append(len(piles[i-1]) - 1)
                stacked = True
        if not stacked:
            if piles:
                back_ptr.append([len(piles[-1]) - 1])
            piles.append([elt])
    i = -1
    longest = [piles[-1][i]]
    for bp, pile in zip(reversed(back_ptr), reversed(piles[:-1])):
        i = bp[i]
        longest.append(pile[i])
    return longest[::-1]


def longest_common_subsequence(s1: list, s2: list)-> list:
    """
    Find the longest subsequence common to two sequences.
    https://en.wikipedia.org/wiki/Longest_common_subsequence  
    
    Examples
    --------
    >>> s1 = [6, 2, 7, 3, 2, 7, 6, 6, 9, 3, 8, 3, 5, 5, 2]
    >>> s2 = [1, 2, 6, 6, 9, 7, 2, 9, 9, 7, 6, 1, 0]
    >>> longest_common_subsequence(s1, s2)
    [2, 6, 6, 9, 2]
    >>> s1 = list("XMJYAUZ")
    >>> s2 = list("MZJAWXU")
    >>> longest_common_subsequence(s1, s2)
    ['M', 'J', 'A', 'U']
    """
    lcs_len = np.zeros((len(s1)+1, len(s2)+1), dtype=int)
    paths = np.zeros((len(s1)+1, len(s2)+1), dtype=int)
    
    def find_lcs(i0, j0):
        lcs = []
        i, j = i0, j0
        while lcs_len[i, j] > 0:
            if paths[i, j] & 0x01 == 1: # go left
                j -= 1
            elif paths[i, j] & 0x02 == 2: # go up
                i -= 1
            elif paths[i, j] & 0x04 == 4: # go up left
                i -= 1
                j -= 1
                lcs.append(s1[i])
        return lcs[::-1]
    
    for i, j in product(range(len(s1)), range(len(s2))):
        left, up = lcs_len[i+1, j], lcs_len[i, j+1]
        lcs = max(left, up)
        val = 0
        if s1[i] == s2[j]:
            diag = lcs_len[i, j] + 1
            lcs = max(diag, lcs)
            if diag == lcs:
                val += 4
        if left == lcs:
            val += 1
        if up == lcs:
            val += 2
        lcs_len[i+1, j+1] = lcs
        paths[i+1, j+1] = val
    
    return find_lcs(len(s1), len(s2))