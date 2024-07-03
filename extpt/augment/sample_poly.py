import random

import numpy as np 
import matplotlib.path as pltPath
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull 


def get_conv_hull(poly):
    poly = np.array(poly)
    hull = ConvexHull(poly)
    return poly[hull.vertices,:]


def draw_poly(poly, points):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(poly, facecolor='orange', lw=2)
    ax.add_patch(patch)
    if points:
        pX, pY = [list(t) for t in zip(*points)]
        ax.scatter(pX, pY)
    ax.set_xlim(211.4, 211.6)
    ax.set_ylim(0.06, 0.09)
    plt.show()


def get_bounding_rect(polygon: [(float, float)]) -> [(float, float)]:
    polygon.sort(key=lambda p: p[0])  # sorted by x value of the point
    leftmost_x = polygon[0][0]
    rightmost_x = polygon[-1][0]

    polygon.sort(key=lambda p: p[1])   # sorted by y value of the point
    bottommost_y =  polygon[0][1]
    topmost_y =   polygon[-1][1]

    return [(leftmost_x, bottommost_y), (rightmost_x, topmost_y)]


def sample_rect(rect: [(float, float)]) -> (float, float):
    bot_left, top_right = rect
    rx = random.uniform(bot_left[0], top_right[0])
    ry = random.uniform(bot_left[1], top_right[1])
    return (rx, ry)


def point_in_poly(poly, x, y):
    poly = get_conv_hull(poly)
    path = pltPath.Path(poly)
    return path.contains_points([[x, y]])[0]


def sample_poly(polygon: [(float, float)], n_samples: int) -> [(float, float)]:
    bounding_rect = get_bounding_rect(polygon)
    sampled_points = []
    while len(sampled_points) < n_samples:
        px, py = sample_rect(bounding_rect)
        if point_in_poly(polygon, px, py):
            sampled_points.append((px, py))
    return sampled_points


if __name__ == "__main__":
    poly = [(211.52272660027916, 0.0706306), (211.4235507562188, 0.072757825), (211.57564411779444, 0.081660256)] 
    points = sample_poly(poly, 3)
    poly = get_conv_hull(poly)
    path = pltPath.Path(poly)
    draw_poly(path, points)