# -*- coding: utf-8 -*-
"""
Sorting algorithms.

* Sorting algorithms
    - <heapsort> : Heapsort selection sort algorithm.
    - <mergesort> : Merge sort algorithm.
    - <quicksort> : Quicksort exchange sorting algorithm.
* Selection algorithms
    - <heapselect> : Heapsort-based selection algorithm.
    - <medians_of_medians> : Medians of medians selection algorithm.
    - <quickselect> : Quicksort-based selection algorithm.
"""


# =============================================================================
# Sorting algorithms
# =============================================================================

def heapsort(a: list)-> list:
    """
    Heapsort algorithm.
    See https://en.wikipedia.org/wiki/Heapsort
    
    Examples
    --------
    >>> a = [39, -55, 57, -37, -60, 59, 35, 97, -22, 67, -34, 13, 19, -58, -63]
    >>> heapsort(a)
    [-63, -60, -58, -55, -37, -34, -22, 13, 19, 35, 39, 57, 59, 67, 97]
    """
    iparent = lambda i: (i - 1) // 2
    ichildL = lambda i: 2*i + 1
    ichildR = lambda i: 2*i + 2
    
    def swap(i, j):
        a[i], a[j] = a[j], a[i]
    
    def find_leaf(node: int, end: int):
        i = node
        while (iR := ichildR(i)) < end:
            iL = ichildL(i)
            i = iL if a[iL] >= a[iR] else iR
        if (iL := ichildL(i)) < end:
            i = iL
        return i
    
    def sift_down(root, leaf):
        j = leaf
        while a[j] < a[root]:
            j = iparent(j)
        while j > root:
            swap(root, j)
            j = iparent(j)
    
    # Heapify
    i = iparent(len(a) - 1)
    while i >= 0:
        leaf = find_leaf(i, len(a))
        sift_down(i, leaf)
        i -= 1
    
    # sort
    end = len(a) - 1
    while end > 0:
        swap(0, end)
        leaf = find_leaf(0, end)
        sift_down(0, leaf)
        end -= 1
    
    return a


def mergesort(a: list)-> list:
    """
    Merge sort algorithm.
    See https://en.wikipedia.org/wiki/Merge_sort
    
    Examples
    --------
    >>> a = [39, -55, 57, -37, -60, 59, 35, 97, -22, 67, -34, 13, 19, -58, -63]
    >>> mergesort(a)
    [-63, -60, -58, -55, -37, -34, -22, 13, 19, 35, 39, 57, 59, 67, 97]
    """
    if len(a) <= 1:
        return a
    
    def merge(list1, list2): # list1 and list2 are sorted
        merged = []
        while list1 and list2:
            if list2[0] >= list1[0]:
                merged.append(list1.pop(0))
            else:
                merged.append(list2.pop(0))
        if list1:
            merged += list1
        if list2:
            merged += list2
        return merged
    
    merged = [[a[i]] for i in range(len(a))]
    while len(merged) != 1:
        temp = [merge(*merged[2*i:2*(i+1)]) for i in range(len(merged)//2)]
        temp += merged[2*(len(merged)//2):]
        merged = temp
    return merged[0]


def quicksort(a: list)-> list:
    """
    Quicksort algorithm.
    See https://en.wikipedia.org/wiki/Quicksort
    
    Examples
    --------
    >>> a = [39, -55, 57, -37, -60, 59, 35, 97, -22, 67, -34, 13, 19, -58, -63]
    >>> quicksort(a)
    [-63, -60, -58, -55, -37, -34, -22, 13, 19, 35, 39, 57, 59, 67, 97]
    """
    def swap(i, j):
        a[i], a[j] = a[j], a[i]
    
    def get_pivot(start, stop):
        mid = (stop + start) // 2
        if a[start] > a[mid]:
            swap(start, mid)
        if a[mid] > a[stop]:
            swap(mid, stop)
        return a[mid]
    
    def partition(start, stop):
        pivot = get_pivot(start, stop)
        i, j = start, stop
        while True:
            while i <= stop and a[i] <= pivot:
                i += 1
            while j >= start and a[j] >= pivot:
                j -= 1
            if j <= i:
                return j, i # low, high
            swap(i, j)
    
    partitions = [(0, len(a) - 1)]
    while partitions:
        start, stop = partitions.pop(-1)
        i, j = partition(start, stop)
        if j < stop:
            partitions.append((j, stop))
        if i > start:
            partitions.append((start, i))
    return a


# =============================================================================
# Selection algorithms
# =============================================================================

def heapselect(a: list, k: int):
    """
    k-th element selection based on heap-sorting.
    
    Examples
    --------
    >>> a = [39, -55, 57, -37, -60, 59, 35, 97, -22, 67, -34, 13, 19, -58, -63]
    >>> heapselect(a, 3)
    -55
    >>> heapselect(a, 18)
    19
    """
    if k >= len(a):
        raise ValueError("k must be < len(a)")
    
    iparent = lambda i: (i - 1) // 2
    ichildL = lambda i: 2*i + 1
    ichildR = lambda i: 2*i + 2
    
    def swap(i, j):
        a[i], a[j] = a[j], a[i]
    
    def find_leaf(node, end):
        i = node
        while (iR := ichildR(i)) <= end:
            iL = ichildL(i)
            i = iR if a[iR] <= a[iL] else iL
        if (iL := ichildL(i)) <= end:
            return iL
        return i
    
    def sift_down(node, end):
        i = find_leaf(node, end)
        while a[node] < a[i]:
            i = iparent(i)
        while i > node:
            swap(node, i)
            i = iparent(i)
    
    end = len(a) - 1
    # heapify
    i = iparent(end)
    while i >= 0:
        sift_down(i, end)
        i -= 1
    # sort first k elements
    for _ in range(k+1):
        swap(0, end)
        end -= 1
        sift_down(0, end)
    
    return a[-k-1]


def median_of_medians(a: list, k: int):
    """
    k-th element selection using medians of medians.
    
    Examples
    --------
    >>> a = [39, -55, 57, -37, -60, 59, 35, 97, -22, 67, -34, 13, 19, -58, -63]
    >>> median_of_medians(a, 3)
    -55
    >>> median_of_medians(a, 18)
    19
    """
    if k >= len(a):
        raise ValueError("k must be < len(a)")
    
    def swap(i, j):
        a[i], a[j] = a[j], a[i]
    
    def partition5(start, stop):
        for i in range(start+1, stop+1):
            j = i
            while j > start and a[j-1] > a[j]:
                swap(j-1, j)
        return (start + stop) // 2
    
    def medians_of(start, stop):
        if stop - start < 5:
            idx = partition5(start, stop)
            return (idx, idx)
        idx = start - 1
        for i in range(start, stop, 5):
            idx += 1
            med = partition5(i, min(i+4, stop))
            swap(idx, med)
        return (start, idx)  
    
    def partition(start, stop, pivot, n):
        idx = start
        for i in range(start, stop+1):
            if a[i] < pivot:
                swap(i, idx)
                idx += 1
        start_eq = idx
        for i in range(start_eq, stop+1):
            if a[i] == pivot:
                swap(i, idx)
                idx += 1
        return (start_eq, idx)
    
    start, stop = 0, len(a) - 1
    while True:
        imed, jmed = medians_of(start, stop)
        while imed != jmed:
            imed, jmed = medians_of(imed, jmed)
        pivot = a[imed]
        start_, stop_ = partition(start, stop, pivot, k)
        if k >= start_ and k < stop_:
            return a[k]
        if k < start_:
            start, stop = start, start_ - 1
        if k >= stop_:
            start, stop = stop_, stop


def quickselect(a: list, k: int):
    """
    k-th element selection based on quicksort.
    
    Examples
    --------
    >>> a = [39, -55, 57, -37, -60, 59, 35, 97, -22, 67, -34, 13, 19, -58, -63]
    >>> quickselect(a, 3)
    -55
    >>> quickselect(a, 18)
    19
    """
    if k >= len(a):
        raise ValueError("k must be < len(a)")
    
    def swap(i, j):
        a[i], a[j] = a[j], a[i]
    
    def get_pivot(start, stop):
        mid = (start + stop) // 2
        if a[mid] < a[start]:
            swap(start, mid)
        if a[mid] > a[stop]:
            swap(stop, mid)
        return a[mid]
    
    def partition(start, stop):
        pivot = get_pivot(start, stop)
        i, j = start, stop
        while True:
            while i <= stop and a[i] <= pivot:
                i += 1
            while j >= start and a[j] >= pivot:
                j -= 1
            if j <= i:
                return j, i # low, high
            swap(i, j)
    
    part = (0, len(a) - 1)
    while True:
        start, stop = part
        i, j = partition(start, stop)
        if j <= k and j < stop:
            part = (j, stop)
        elif i >= k and i > start:
            part = (start, i)
        else:
            return a[k]