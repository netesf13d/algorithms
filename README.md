# algorithms
Python implementation of various algorithms. The kind of algorithms asked for in quant interviews...

### Sorting algorithms
* Sorting algorithms
  - `heapsort` : Heapsort selection sort algorithm.
  - `mergesort` : Merge sort algorithm.
  - `quicksort` : Quicksort exchange sorting algorithm.
* Selection algorithms
  - `heapselect` : Heapsort-based selection algorithm.
  - `medians_of_medians` : Medians of medians selection algorithm.
  - `quickselect` : Quicksort-based selection algorithm.

### Graph traversal algorithms
* Connected components of a graph
  - `connected_components` : Find the connected components in an undirected graph.
  - `strongly_connected_components` : Find the strongly connected components in a directed graph.
* Cycles of a graph
  - `elementary_cycles_directed` : Find the elementary cycles in a directed graph.
  - `elementary_cycles_undirected` : Find the elementary cycles in an undirected graph.
  - `find_cycle_undirected` : Find a cycle in an undirected graph.
* Shortest paths
  - `shortest_path_dijkstra` : Dijkstra's algorithm to find single-source shortest paths in a positive-weighted undirected graph.
  - `shortest_path_faster` : Shortest path faster algorithm to find single-source shortest paths in a weighted graph.
  - `get_path` : Shortest path to target from the list of predecessors established by `shortest_path` algorithms.
  - `uniform_cost_search` : Uniform cost search algorithm to find single-source shortest paths in a positive-weighted undirected graph not known a priori.
    
### Tree traversal algorithms
* Binary tree traversal
  - `breadth_first_search` : Breadth-first search in a binary tree.
  - `depth_first_search` : Depth-first search in a binary tree.

### Combinatorial optimization algorithms
* Knapsack problem
  - `knapsack_dict` : Solve knapsack problem using a `dict` data struture.
  - `knapsack_array` : Solve knapsack problem using an `array` data struture.
* Partitioning
  - `subset_sum` : Find a subset which elements sum precisely to a given S.
  - `largest_differencing_method` : Multi-way partitioning of a set using the largest differencing method.
* Subsequences
  - `maximum_subarray` : Kadane's algorithm to solve the maximum subarray problem.
  - `longest_increasing_subsequence` : Solve the longest increasing subsequence problem.
  - `longest_common_subsequence` : Find the longest subsequence common to two sequences.

### Miscellaneous algorithms
These are not "standard" algorithms but can typically be asked for in quant interviews.

* Miscellaneous algorithms
  - `sqrt` : compute square root of a float using Newton-Raphson method.
* Dynamic programming algorithms
  - `part_array_permutation` : solution to a dynamic programming problem.
  - `largest_square` : solution to a dynamic programming problem.
