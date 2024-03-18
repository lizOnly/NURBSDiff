#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Examples for the NURBS-Python Package
    Released under MIT License
    Developed by Onur Rauf Bingol (c) 2016-2017

    This example is contributed by John-Eric Dufour (@jedufour)
"""

import os
from geomdl import NURBS
from geomdl import exchange, utilities
# from geomdl.visualization import VisMPL as vis
from geomdl import operations
import numpy as np
import itertools

def reconstructed_mesh(object_name, filename, num_ctrl_pts1, num_ctrl_pts2, u, v):
    # Create a NURBS surface instance
    surf = NURBS.Surface()

    # Set degrees
    surf.degree_u = 3
    surf.degree_v = 3

    surf.ctrlpts_size_u = num_ctrl_pts1
    surf.ctrlpts_size_v = num_ctrl_pts2

    surf.ctrlpts = exchange.import_txt(filename, separator=" ")

    # surf.ctrlpts = exchange.import_txt("../objects/duck1.ctrlpts", separator=" ")

    # Set knot vectors
    surf.knotvector_u = list(u)
    surf.knotvector_v = list(v)

    surf.sample_size = 100
    # surf.vis = vis.VisSurface()
    # operations.refine_knotvector(surf, [0, 1])
    
    surf.render()

    uv_vals = list(itertools.product(surf.knotvector_u, surf.knotvector_v))
    surftans = [[] for _ in range(len(uv_vals))]
    surfnorms = [[] for _ in range(len(uv_vals))]
    idx = 0

    for u,v in list(itertools.product(surf.knotvector_u, surf.knotvector_v)):
        try:
            surftans[idx] = operations.tangent(surf, [u,v], normalize=True)
            surfnorms[idx] = operations.normal(surf, [u,v], normalize=True)
        except:
            print(u, v)
            surftans[idx] = operations.tangent(surf, [u,v], normalize=False)
            surfnorms[idx] = operations.normal(surf, [u,v], normalize=False)
        idx += 1

    # Prepare points for plotting
    surfpts = np.array(surf.evalpts)
    tangent_vectors = np.array(surftans)
    normal_vectors = np.array(surfnorms)

    # save the object
    exchange.export_obj(surf, filename + "_mesh.obj")

    return surfpts, tangent_vectors, normal_vectors

def reconstructed_mesh_with_weights(object_name, filename, u, v):
    surf = NURBS.Surface()

    # Set degrees
    surf.degree_u = 3
    surf.degree_v = 3

    surf.set_ctrlpts(*exchange.import_txt(filename, two_dimensional=True))

    # surf.ctrlpts = exchange.import_txt("../objects/duck1.ctrlpts", separator=" ")

    # Set knot vectors
    surf.knotvector_u = list(u)
    surf.knotvector_v = list(v)

    surf.sample_size = 40
    # surf.vis = vis.VisSurface()
    # operations.refine_knotvector(surf, [0, 1])
    
    surf.render()

    # save the object
    exchange.export_obj(surf, f"generated/{object_name}/mesh_weighted.obj")

    pass
    

def reconstructed_mesh_test(object_name, filename, num_ctrl_pts1, num_ctrl_pts2):
    # Create a NURBS surface instance
    surf = NURBS.Surface()

    # Set degrees
    surf.degree_u = 3
    surf.degree_v = 3

    surf.ctrlpts_size_u = num_ctrl_pts1
    surf.ctrlpts_size_v = num_ctrl_pts2

    surf.ctrlpts = exchange.import_txt(filename, separator=" ")
    # surf.ctrlpts = exchange.import_txt("../objects/duck1.ctrlpts", separator=" ")

    # Set knot vectors
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)

    surf.sample_size = 30
    # surf.vis = vis.VisSurface()
    
    surf.render()

    # save the object
    exchange.export_obj(surf, f"generated/{object_name}/test_mesh.obj")

    pass
# reconstructed_mesh_test('cat_dynamic', 'generated/cat_dynamic/origin_ctrlpts_z.ctrlpts', 30, 25)
# evalpts = surf.evalpts
# with open("sampled_point_cloud.off", "w") as f:
#     f.write("OFF\n")
#     f.write("900 0 0\n")
#     for i in range(900):
#         f.write(str(evalpts[i][0]) + " " + str(evalpts[i][1]) + " " + str(evalpts[i][2]) + "\n")
# Set evaluation delta
# total = 0
# for i in range(100):
#     surf.sample_size = 30
#     surf.evaluate()

#     evt = evalpts = torch.reshape(torch.tensor(surf.evalpts), (30, 30, 3))
    
#     evt = torch.reshape(evt, (1, 30, 30, 3))
#     total += chamfer_distance(evalpts, target) +  hausdorff_distance(evalpts, target) + 0.1 * laplacian_loss_unsupervised(evt)
# print(total/100)

