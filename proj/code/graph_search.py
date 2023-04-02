from collections import defaultdict
from heapq import heappush, heappop  # Recommended.
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap  # Recommended.


def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    priority_queue = []
    visited = set()  # keep the visited vertex
    queue_set = set()  # keep track of the indices already in the priority queue
    vertex_costs = defaultdict(lambda: float('inf'))  # keep the cost for each vertex
    parent = {}  # keep track of the parent of each vertex
    vertex_costs[start_index] = 0  # initialize the cost of start vertex to be 0
    heappush(priority_queue, (0, start_index))
    queue_set.add(start_index)

    flag = False  # check if there is a path

    # while priority_queue and min(vertex_costs.values()) < float('inf'):  # this is super slow, min() is O(n), dict.values() is O(n)
    # while priority_queue and priority_queue[0][0] < float('inf'):
    while priority_queue:
        _, vertex = heappop(priority_queue)
        queue_set.remove(vertex)
        visited.add(vertex)
        if vertex == goal_index:
            flag = True  # there is a path
            break

        # for adjVertex, edgeCost in graph.adj[vertex].items():
        for adjVertex, edgeCost in find_adjacent_costs(vertex, occ_map).items():
            if adjVertex in visited:
                continue
            d = vertex_costs[vertex] + edgeCost
            if d < vertex_costs[adjVertex]:
                vertex_costs[adjVertex] = d
                parent[adjVertex] = vertex
                if adjVertex not in queue_set:
                    heappush(priority_queue, (d + heuristic(adjVertex, goal_index) * astar, adjVertex))
                    queue_set.add(adjVertex)

    path = []
    if not flag:
        # there is no path
        return None, 0
    else:
        # there is a path
        v = goal_index

        # link the goal point if its metric is not at the center of the voxel
        if not np.array_equal(goal, occ_map.index_to_metric_center(v)):
            path.append(goal)

        path.append(occ_map.index_to_metric_center(v))
        while v in parent.keys():
            v = parent.get(v)
            path.append(occ_map.index_to_metric_center(v))

        # link the start point if its metric is not at the center of the voxel
        if not np.array_equal(start, occ_map.index_to_metric_center(v)):
            path.append(start)

        path.reverse()  # the order is from start point to goal point
        path = np.array(path)

        # Return a tuple (path, nodes_expanded)
        return path, len(path)


def heuristic(vertex1, vertex2):
    """
    The heuristic function with respect to vertex1 and vertex2
    Parameters:
        vertex1, tuple of vertex indices
        vertex2, tuple of vertex indices
    Output:
        return the value of heuristic function
    """
    return np.linalg.norm(np.array(vertex1) - np.array(vertex2))


def cost(vertex1, vertex2):
    """
    The function to calculate edge cost with respect to vertex1 and vertex2
    Parameters:
        vertex1, tuple of vertex indices
        vertex2, tuple of vertex indices
    Output:
        return the value of cost function
    """
    return np.linalg.norm(np.array(vertex1) - np.array(vertex2))


def find_adjacent_costs(vertex, occupancyMap):
    """
    Find the adjacent voxel of given index of a voxel
    Parameter:
        vertex, the index of given vertex in tuple
    Output:
        adj, the dict of all its adjacent vertex and costs
            key: adjacent voxel index
            value: the edge cost
    """
    adj = {}
    for i_prime in range(vertex[0] - 1, vertex[0] + 2):
        for j_prime in range(vertex[1] - 1, vertex[1] + 2):
            for k_prime in range(vertex[2] - 1, vertex[2] + 2):
                if i_prime == vertex[0] and j_prime == vertex[1] and k_prime == vertex[2]:
                    continue
                elif occupancyMap.is_occupied_index((i_prime, j_prime, k_prime)):
                    continue
                else:
                    # set the cost to each adjacent vertex
                    adj[(i_prime, j_prime, k_prime)] = cost(vertex, (i_prime, j_prime, k_prime))
    return adj
