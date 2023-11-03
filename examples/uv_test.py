#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Plotting the Duck Example from GLScene with NURBS-Python

    GLScene: http://wiki.freepascal.org/GLScene
    GLScene on GitHub: https://github.com/cutec-chris/GLScene

    The "Ducky" example contains 3 surfaces. The surface data is accessible via the "GLScene on GitHub" link above in
    the directory "Demos/media/" with the file names "duck1.nurbs", "duck2.nurbs" and "duck3.nurbs". The contents of
    these files are extracted into NURBS-Python format.

    The main difference between the surface formats is the row order. Most of the time it affects the evaluation result.
    It is easy to understand the issues caused by the row order: Using a 1-dimensional array of control points in the
    wrong order may cause a "hole" or leave some regions unevaluated. For the 2-dimensional array of control points,
    most of the time you will get an IndexError exception.

    NURBS-Python uses v-row order but GLScene uses u-row order (please see docs for more details on the row order).
    This means that u- and v-directions of the *.nurbs files must be transposed (or flipped) in order to get the
    correct shapes. This example fixes the issues caused by the row order difference.
"""

import os
from geomdl import NURBS
from geomdl import multi
from geomdl import exchange
from geomdl import utilities
from geomdl import compatibility
from geomdl.visualization import VisMPL
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
import torch

def plot_points3d(P, colors=[]):
    # given a point cloud P plot the points connected with a wireframe
    # create a figure
    fig = plt.figure()
    # add a subplot
    ax = fig.add_subplot(111, projection='3d')
    # plot the points
    #if empty colors then plot the points in red
    if len(colors) == 0:
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')
    else:
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=colors, marker='o')
    # set the limits based on points P
    ax.set_xlim3d(np.min(P[:, 0]), np.max(P[:, 0]))
    ax.set_ylim3d(np.min(P[:, 1]), np.max(P[:, 1]))
    ax.set_zlim3d(np.min(P[:, 2]), np.max(P[:, 2]))

    # show the plot
    plt.show()
def read_weights(filename, sep=","):
    try:
        with open(filename, "r") as fp:
            content = fp.read()
            content_arr = [float(w) for w in (''.join(content.split())).split(sep)]
            return content_arr
    except IOError as e:
        print("An error occurred: {}".format(e.args[-1]))
        raise e

# Fix file path
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# duck1.nurbs
# Process control points and weights
d2_ctrlpts = exchange.import_txt("Ducky/duck1.ctrlpts", separator=" ")
# d1_weights = read_weights("duck1.weights")
# d1_ctrlptsw = compatibility.combine_ctrlpts_weights(d2_ctrlpts, d1_weights)

# Create a NURBS surface
duck1 = NURBS.Surface()
duck1.name = "body"
duck1.order_u = 4
duck1.order_v = 4
duck1.ctrlpts_size_u = 14
duck1.ctrlpts_size_v = 13
duck1.ctrlpts = d2_ctrlpts
duck1.knotvector_u = [-1.5708, -1.5708, -1.5708, -1.5708, -1.0472, -0.523599, 0, 0.523599, 0.808217,
                      1.04015, 1.0472, 1.24824, 1.29714, 1.46148, 1.5708, 1.5708, 1.5708, 1.5708]
duck1.knotvector_v = [-3.14159, -3.14159, -3.14159, -3.14159, -2.61799, -2.0944, -1.0472, -0.523599,
                      6.66134e-016, 0.523599, 1.0472, 2.0944, 2.61799, 3.14159, 3.14159, 3.14159, 3.14159]

duck1.sample_size = 30
duck1.evaluate()
print(len(duck1.evalpts))
print(duck1.evalpts[0])
print(duck1.evaluate_single([0.5, 0.5]))


max_range = 0.5
delta = 0.05
duck1.delta = 0.05
P = []
#iterate with a range with a delta of 0.05
# create a uv grid of points

#create a spaced list
duck1.delta = 0.05
U = np.arange(max_range, 1, delta)
V = np.arange(max_range, 1, delta)
#iterate a uv map
for u in U:
    for v in V:
        P.append(duck1.evaluate_single([u, v]))

plot_points3d(np.asarray(P))


# Visualization configuration
duck1.vis = VisMPL.VisSurface(ctrlpts=False, legend=False)

points = np.asarray(duck1.vertices).tolist()
# transform points to a list of shape (N, 3)
points = np.asarray(points).reshape((-1, 3))
plot_points3d(points)
# Render the ducky
# duck1.render()


