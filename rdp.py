"""
The Ramer-Douglas-Peucker algorithm roughly ported from the pseudo-code provided
by http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
"""
from math import sqrt


def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_line_distance(point, start, end):
    if start == end:
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) -
            (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        return n / d


def rdp(points, epsilon):
    """Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]

    return results


def rdp_with_index(points, indices, epsilon):
    """rdp with returned point indices
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon:
        first_points, first_indices = rdp_with_index(points[:index+1], indices[:index+1], epsilon)
        second_points, second_indices = rdp_with_index(points[index:], indices[index:], epsilon)
        results = first_points[:-1] + second_points
        results_indices = first_indices[:-1] + second_indices
    else:
        results, results_indices = [points[0], points[-1]], [indices[0], indices[-1]]
    return results, results_indices


if __name__ == "__main__":
    points = [[0, 0], [100, 100], [230, 200], [220, 210], [350, 350], [450, 450]]
    pts = rdp(points, epsilon=20)
    import numpy as np
    print(np.array(pts))
    pts, indices = rdp_with_index(points, list(range(len(points))), epsilon=20)
    print(pts)
    print(indices)

