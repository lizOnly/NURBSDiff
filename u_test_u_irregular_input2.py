import math
import time
import torch
import numpy as np
import os
from geomdl import NURBS
from geomdl import operations
from geomdl import exchange, utilities

#from DuckyFittingOriginal import read_weights
# from examples.splinenet import DGCNNControlPoints, get_graph_feature
from examples.test.mesh_reconstruction import reconstructed_mesh
# from examples.test.test_dgcnn import DGCNN, DGCNN_without_grad
torch.manual_seed(120)
from tensorboard_logger import configure, log_value
from tqdm import tqdm
# from pytorch3d.loss import chamfer_distance
from NURBSDiff.nurbs_eval import SurfEval
from NURBSDiff.surf_eval import SurfEval as SurfEvalBS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import itertools
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from geomdl import exchange, utilities
# from geomdl.visualization import VisMPL
from geomdl import compatibility
from torch.autograd.variable import Variable
import torch.nn.functional as F
# from scipy.spatial.distance import directed_hausdorff
# import offset_eval as off
import random
# from read_config import Config

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20


def init_plt():
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Times') 
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def generate_gradient(start_color, end_color, steps):
    # Convert the start and end colors to RGB tuples
    rgb_start = tuple(int(start_color[i:i+2], 16) for i in (1, 3, 5))
    rgb_end = tuple(int(end_color[i:i+2], 16) for i in (1, 3, 5))
    # Calculate the step size for each RGB component
    step_size = tuple((rgb_end[i] - rgb_start[i]) / (steps - 1) for i in range(3))
    # Generate the gradient colors
    gradient = []
    for i in range(int(steps)):
        # Calculate the RGB values for the current step
        r = int(rgb_start[0] + i * step_size[0])
        g = int(rgb_start[1] + i * step_size[1])
        b = int(rgb_start[2] + i * step_size[2])
        
        # Convert the RGB values to a hexadecimal string
        hex_color = '#' + format(r, '02x') + format(g, '02x') + format(b, '02x')
        
        # Add the hexadecimal color string to the gradient list
        gradient.append(hex_color)
    return gradient

def generate_sheet(point_cloud, ctrl_pts_u, ctrl_pts_v, axis='z', object_name=None):
    point_cloud = np.array(point_cloud).reshape(-1, 3)
    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])
    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])
    
    sheet_ctrlpts = []
    if axis == 'z':
        factor_x = (max_x - min_x) / (ctrl_pts_u - 1)
        factor_y = (max_y - min_y) / (ctrl_pts_v - 1)
        for i in range(ctrl_pts_u):
            for j in range(ctrl_pts_v):
                sheet_ctrlpts.append([min_x + i * factor_x, min_y + j * factor_y, max_z + 0.5])

    elif axis == 'x':
        factor_y = (max_y - min_y) / (ctrl_pts_u - 1)
        factor_z = (max_z - min_z) / (ctrl_pts_v - 1)
        for i in range(ctrl_pts_u):
            for j in range(ctrl_pts_v):
                sheet_ctrlpts.append([max_x + 0.5, min_y + i * factor_y, min_z + j * factor_z])
    elif axis == 'y':
        factor_x = (max_x - min_x) / (ctrl_pts_u - 1)
        factor_z = (max_z - min_z) / (ctrl_pts_v - 1)
        for i in range(ctrl_pts_u):
            for j in range(ctrl_pts_v):
                sheet_ctrlpts.append([min_x + i * factor_x, max_y + 0.5, min_z + j * factor_z])
    
    sheet_ctrlpts = np.array(sheet_ctrlpts).reshape(ctrl_pts_u, ctrl_pts_v, 3)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
    ax1.plot_wireframe(sheet_ctrlpts[:, :, 0], sheet_ctrlpts[:, :, 1], sheet_ctrlpts[:, :, 2])
    fig.show()
    with open(f'generated/{object_name}/origin_ctrlpts_{axis}.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(ctrl_pts_u * ctrl_pts_v) + ' ' + '0 0\n')
        for i in range(ctrl_pts_u):
            for j in range(ctrl_pts_v):
                line = str(sheet_ctrlpts[i, j, 0]) + ' ' + str(sheet_ctrlpts[i, j, 1]) + ' ' + str(sheet_ctrlpts[i, j, 2]) + '\n'
                f.write(line)
    
    return sheet_ctrlpts
    
        
    
def generate_cylinder(point_cloud, ctrl_pts_u, ctrl_pts_v, closed=True, axis='z', object_name=None, ratio_1=1.1, ratio_2=1.1):
    
    point_cloud = np.array(point_cloud).reshape(-1, 3)
    # Calculate the center of the cylinder
    cylinder_center = np.mean(point_cloud, axis=0)

    # Calculate the radius of the cylinder
    

    # Calculate the height of the cylinder
    if axis == 'z':
        min_z = np.min(point_cloud[:, 2])
        max_z = np.max(point_cloud[:, 2])
        cylinder_height = (max_z - min_z) * ratio_1
        distances = np.linalg.norm(point_cloud[:, :2] - cylinder_center[:2], axis=1)
    elif axis == 'y':
        min_y = np.min(point_cloud[:, 1])
        max_y = np.max(point_cloud[:, 1])
        cylinder_height = (max_y - min_y) * ratio_1
        distances = np.linalg.norm(point_cloud[:, [0, 2]] - [cylinder_center[0], cylinder_center[2]], axis=1)
    elif axis == 'x':
        min_x = np.min(point_cloud[:, 0])
        max_x = np.max(point_cloud[:, 0])
        cylinder_height = (max_x - min_x) * ratio_1
        distances = np.linalg.norm(point_cloud[:, 1:] - cylinder_center[1:], axis=1)

    cylinder_radius = np.max(distances) * ratio_2
    
    cylinder_ctrlpts = []
    factor = cylinder_height / (ctrl_pts_u - 3)
    
    base_x = cylinder_center[0]
    base_y = cylinder_center[1]
    base_z = cylinder_center[2]
    
    if closed:

        if axis == 'z':
            for i in range(ctrl_pts_v):
                cylinder_ctrlpts.append([base_x, base_y, min_z])
        
            for j in range(ctrl_pts_u - 2):
                for i in range(ctrl_pts_v - 1):
                    x = i / (ctrl_pts_v - 2) * 2 * math.pi
                    cylinder_ctrlpts.append([math.sin(x) * cylinder_radius + base_x, math.cos(x) * cylinder_radius + base_y, min_z + j * factor])
                # make each row closed
                cylinder_ctrlpts.append([base_x, cylinder_radius + base_y, min_z + j * factor])
                
            for i in range(ctrl_pts_v):
                cylinder_ctrlpts.append([base_x, base_y, max_z])
        elif axis == 'y':
            for i in range(ctrl_pts_v):
                cylinder_ctrlpts.append([base_x, min_y, base_z])
        
            for j in range(ctrl_pts_u - 2):
                for i in range(ctrl_pts_v - 1):
                    x = i / (ctrl_pts_v - 2) * 2 * math.pi
                    cylinder_ctrlpts.append([math.sin(x) * cylinder_radius + base_x, min_y + j * factor, math.cos(x) * cylinder_radius + base_z])
                # make each row closed
                cylinder_ctrlpts.append([base_x, min_y + j * factor, cylinder_radius + base_z])
                
            for i in range(ctrl_pts_v):
                cylinder_ctrlpts.append([base_x, max_y, base_z])
        elif axis == 'x':
            for i in range(ctrl_pts_v):
                cylinder_ctrlpts.append([min_x, base_y, base_z])
        
            for j in range(ctrl_pts_u - 2):
                for i in range(ctrl_pts_v - 1):
                    x = i / (ctrl_pts_v - 2) * 2 * math.pi
                    cylinder_ctrlpts.append([min_x + j * factor, math.sin(x) * cylinder_radius + base_y, math.cos(x) * cylinder_radius + base_z])
                # make each row closed
                cylinder_ctrlpts.append([min_x + j * factor, base_y, cylinder_radius + base_z])
                
            for i in range(ctrl_pts_v):
                cylinder_ctrlpts.append([max_x, base_y, base_z])
    else:
        # TODO: implement open case
        pass
    
    cylinder_ctrlpts = np.array(cylinder_ctrlpts).reshape(ctrl_pts_u, ctrl_pts_v, 3)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho', aspect='equal')
    ax1._axis3don = False
    ax1.plot_wireframe(cylinder_ctrlpts[:, :, 0], cylinder_ctrlpts[:, :, 1], cylinder_ctrlpts[:, :, 2])
    fig.show()
    with open(f'generated/{object_name}/origin_ctrlpts_{axis}.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(ctrl_pts_u * ctrl_pts_v) + ' ' + '0 0\n')
        for i in range(ctrl_pts_u):
            for j in range(ctrl_pts_v):
                line = str(cylinder_ctrlpts[i, j, 0]) + ' ' + str(cylinder_ctrlpts[i, j, 1]) + ' ' + str(cylinder_ctrlpts[i, j, 2]) + '\n'
                f.write(line)
    return cylinder_ctrlpts

def plot_subfigure_no_uv(surface_points, ax, color, label):
    ax.plot_wireframe(surface_points[:, :, 0],
                                surface_points[:, :, 1],
                                surface_points[:, :, 2],
                                    color=color, label=label[0])
    
    ax.azim = 45
    ax.dist = 6.5
    ax.elev = 30
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False

def adjust_plot(ax):
    ax.azim = 45
    ax.dist = 6.5
    ax.elev = 30

    # ax.set_xlim([-1, 1])
    # ax.set_xlim([-1, 1])
    # ax.set_xlim([-1, 1])
    # ax.set_xlim([-1, 1])
    # ax.set_xlim([-1, 1])
    # ax.set_xlim([-1, 1])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False

def plot_subfigure(num_ctrl_pts1, num_ctrl_pts2, uspan_uv, vspan_uv, surface_points, ax, colors, ctrlpts, color, label, uvlabel=False):
    u_index = 0
    for i in range(num_ctrl_pts1 - 3):
        u_index += uspan_uv[i + 3]
        v_index = 0
        for j in range(num_ctrl_pts2 - 3):
            if u_index == 512 or v_index == 512 or u_index - uspan_uv[i + 3] == 0 or u_index - uspan_uv[i + 3] == 256:
                if u_index == 512: 
                    ax.plot_wireframe(surface_points[u_index - uspan_uv[i + 3]:-1, v_index:v_index + vspan_uv[j + 3], 0],
                                surface_points[u_index - uspan_uv[i + 3]:-1, v_index:v_index + vspan_uv[j + 3], 1],
                                surface_points[u_index - uspan_uv[i + 3]:-1, v_index:v_index + vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j], label = 'u = 1, v = 1' if(uvlabel and v_index + vspan_uv[j + 3] == 512) else None)
                elif v_index == 512:
                    ax.plot_wireframe(surface_points[u_index - uspan_uv[i + 3]:-1, v_index:-1, 0],
                                surface_points[u_index - uspan_uv[i + 3]:-1, v_index:-1, 1],
                                surface_points[u_index - uspan_uv[i + 3]:-1, v_index:-1, 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j])
                elif u_index - uspan_uv[i + 3] == 0:
                    ax.plot_wireframe(surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 0],
                                    surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 1],
                                    surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j], label = 'u = 0, v = 0' if(uvlabel and v_index == 0) else None)
                elif u_index - uspan_uv[i + 3] == 256:
                    ax.plot_wireframe(surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 0],
                                surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 1],
                                surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j], label = 'u = 0.5, v = 0.5'.format(v_index + vspan_uv[j + 3]) if(uvlabel and v_index + vspan_uv[j + 3] == 256) else None)
            else:
                ax.plot_wireframe(surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 0],
                                surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 1],
                                surface_points[u_index - uspan_uv[i + 3]:u_index, v_index:v_index + vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j])
             
            v_index += vspan_uv[j + 3]

    # ax.plot_wireframe(ctrlpts[:, :, 0], ctrlpts[:, :, 1], ctrlpts[:, :, 2], linestyle='dashdot', color=color,label=label[0])

    ax.azim = 45
    ax.dist = 6.5
    ax.elev = 30
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False

def plot_diff_subfigure(surface_points, ax):

    ax.plot_wireframe(surface_points[:, :, 0], surface_points[:, :, 1], surface_points[:, :, 2]
                        ,color='#ffc38a', label = 'diff(target-predict)')
    
    ax.azim = 45
    ax.dist = 6.5
    ax.elev = 30
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax._axis3don = False

def chamfer_distance(pred, gt, sqrt=False):
    """
    Computes average chamfer distance prediction and groundtruth
    :param pred: Prediction: M x N x 3
    :param gt: ground truth: M x N x 3
    :return:
    """
    # print(pred.shape)
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).cuda()

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).cuda()

    pred = torch.unsqueeze(pred, 1)
    gt = torch.unsqueeze(gt, 2)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 3)
    if sqrt:
        diff = torch.sqrt(diff)

    cd = torch.mean(torch.min(diff, 1)[0], 1) + torch.mean(torch.min(diff, 2)[0], 1)
    cd = torch.mean(cd) / 2.0
    return cd

def chamfer_distance_each_row(pred, gt, sqrt=False):
    """
    Computes average chamfer distance prediction and groundtruth
    :param pred: Prediction: M x N x 3
    :param gt: ground truth: M x N x 3
    :return:
    """
    # print(pred.shape)
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).cuda()

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).cuda()

    # pred = torch.unsqueeze(pred, 1)
    # gt = torch.unsqueeze(gt, 2)
    row, col = pred.shape[0], pred.shape[1]
    total_cd = 0
    for i in range(row):
        gt_row = gt[i]
        pred_row = pred[i]
        
        diff = pred_row.unsqueeze(0) - gt_row.unsqueeze(1)
        dist = torch.sum(diff ** 2, dim=2)
        if sqrt:
            dist = torch.sqrt(dist)
        dist1, _ = torch.min(dist, dim=1)
        dist2, _ = torch.min(dist, dim=0)
        total_cd += torch.mean(dist1, 0) + torch.mean(dist2, 0)
    return total_cd / 2.0 / row
 
def hausdorff_distance_each_row(pred, gt):
    """
    Computes the Hausdorff Distance between two point clouds
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return: Hausdorff Distance
    """
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).cuda()

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).cuda()

    # pred = torch.unsqueeze(pred, 1)
    # gt = torch.unsqueeze(gt, 2)
    row, col = pred.shape[0], pred.shape[1]
    total_cd = 0
    for i in range(row):
        gt_row = gt[i]
        pred_row = pred[i]

        diff = pred_row.unsqueeze(0) - gt_row.unsqueeze(1)

        row_max = torch.max(torch.min(diff, dim=1)[0])
        col_max = torch.max(torch.min(diff, dim=0)[0])
        total_cd += torch.max(row_max, col_max)
    return total_cd / row / 2.0
       
def sinkhorn_loss(pred, gt, epsilon=0.1, n_iters=5):
    """
    Computes the Sinkhorn distance between prediction and groundtruth point clouds.
    :param pred: Prediction: B x N x 3
    :param gt: Ground truth: B x M x 3
    :param epsilon: Regularization strength for Sinkhorn-Knopp algorithm
    :param n_iters: Number of iterations for Sinkhorn-Knopp algorithm
    :return: Sinkhorn distance between prediction and groundtruth point clouds
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred.astype(np.float32)).cuda()

    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt.astype(np.float32)).cuda()

    batch_size, n_points, _ = pred.shape
    _, m_points, _ = gt.shape

    # Compute pairwise distances
    dist = torch.sum((pred.unsqueeze(2) - gt.unsqueeze(1)) ** 2, dim=-1)

    # Compute Sinkhorn-Knopp distance
    K = torch.exp(-dist / epsilon)
    u = torch.ones(batch_size, n_points).cuda() / n_points
    v = torch.ones(batch_size, m_points).cuda() / m_points

    for i in range(n_iters):
        u = 1.0 / torch.matmul(K, v.unsqueeze(-1)).squeeze(-1)
        v = 1.0 / torch.matmul(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)

    T = torch.matmul(torch.matmul(torch.diag(u), K), torch.diag(v))
    sinkhorn_dist = torch.sum(T * dist) / batch_size

    return sinkhorn_dist

def laplacian_loss_splinenet(output, gt, dist_type="l2"):
    batch_size, grid_size, grid_size, input_channels = gt.shape
    filter = ([[[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    filter = np.stack([filter, np.roll(filter, 1, 0), np.roll(filter, 2, 0)])

    filter = -np.array(filter, dtype=np.float32)
    filter = Variable(torch.from_numpy(filter)).cuda()

    laplacian_output = F.conv2d(output.permute(0, 3, 1, 2), filter, padding=1)
    laplacian_input = F.conv2d(gt.permute(0, 3, 1, 2), filter, padding=1)
    if dist_type == "l2":
        dist = (laplacian_output - laplacian_input) ** 2
    elif dist_type == "l1":
        dist = torch.abs(laplacian_output - laplacian_input)
    dist = torch.sum(dist, 1)
    dist = torch.mean(dist)
    return dist

def laplacian_loss_unsupervised(output, dist_type="l2"):
    filter = ([[[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    filter = np.stack([filter, np.roll(filter, 1, 0), np.roll(filter, 2, 0)])

    filter = -np.array(filter, dtype=np.float32)
    filter = Variable(torch.from_numpy(filter)).cuda()
    # print(output.shape)
    laplacian_output = F.conv2d(output.permute(0, 3, 1, 2), filter, padding=1)
    # print(laplacian_output.shape)

    if dist_type == "l2":
        dist = torch.sum((laplacian_output) ** 2, 1) 

        # dist = torch.sum((laplacian_output) ** 2, (1,2,3)) + torch.sum((laplacian_input)**2,(1,2,3))
    elif dist_type == "l1":
        dist = torch.abs(torch.sum(laplacian_output.mean(),1))
    dist = torch.mean(dist)
    # num_points = output.shape[1] * output.shape[2] * output.shape[3]
    return dist

def laplacian_loss_unsupervised_sober(output, dist_type="l2"):
    sobel_filter = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float32)
    sobel_filter = sobel_filter.repeat(output.shape[1], 1, 1, 1)  # Repeat the filter for each channel

    laplacian_output = F.conv2d(output.permute(0, 3, 1, 2), sobel_filter, padding=1)
    if dist_type == "l2":
        dist = torch.sum((laplacian_output) ** 2, (1, 2, 3))
    elif dist_type == "l1":
        dist = torch.abs(torch.sum(laplacian_output.mean(), 1))
    dist = torch.mean(dist)

    return dist

def laplacian_loss(output, gt, dist_type="l2"):
    """
    Computes the laplacian of the input and output grid and defines
    regression loss.
    :param output: predicted control points grid. Makes sure the orientation/
    permutation of this output grid matches with the ground truth orientation.
    This is done by finding the least cost orientation during training.
    :param gt: gt control points grid.
    """

    filter = ([[[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    filter = np.stack([filter, np.roll(filter, 1, 0), np.roll(filter, 2, 0)])

    filter = -np.array(filter, dtype=np.float32)
    filter = Variable(torch.from_numpy(filter)).cuda()

    laplacian_output = F.conv2d(output.permute(0, 3, 1, 2), filter, padding=1)
    laplacian_input = F.conv2d(gt.permute(0, 3, 1, 2), filter, padding=1)
    if dist_type == "l2":
        dist = torch.sum((laplacian_output) ** 2, (1,2,3)) - torch.sum((laplacian_input)**2,(1,2,3)) 
        # dist = torch.sum((laplacian_output) ** 2, (1,2,3)) + torch.sum((laplacian_input)**2,(1,2,3))
    elif dist_type == "l1":
        dist = torch.abs(torch.sum(laplacian_output.mean(),1) - torch.sum(laplacian_input.mean(),1))
    dist = torch.mean(dist)
    return abs(dist)


def hausdorff_distance(pred, gt):
    """
    Computes the Hausdorff Distance between two point clouds
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return: Hausdorff Distance
    """
    batch_size = pred.shape[0]
    pred = torch.unsqueeze(pred, 1)  # B x 1 x N x 3
    gt = torch.unsqueeze(gt, 2)  # B x M x 1 x 3
    # print(pred.shape, gt.shape)
    dist_matrix = torch.sqrt(torch.sum((pred - gt) ** 2, dim=3))  # B x M x N

    row_max, _ = torch.max(torch.min(dist_matrix, dim=2)[0], dim=1)
    col_max, _ = torch.max(torch.min(dist_matrix, dim=1)[0], dim=1)

    hd = torch.max(row_max, col_max)
    hd = torch.mean(hd)
    return hd
def point_pointcloud_distance(point, point_cloud):
    """
    Computes the minimum distance between a point and a point cloud.
    
    Args:
        point (torch.Tensor): The point for which we want to compute the minimum distance.
        point_cloud (torch.Tensor): The point cloud, represented as a 2D tensor where each row corresponds to a point.
        
    Returns:
        The minimum distance between the point and the point cloud.
    """
    dists = torch.norm(point_cloud - point, dim=1)
    return torch.min(dists)

def pointcloud_pointcloud_distance(target_pc, predicted_pc):
    for i in range(target_pc.shape[0]):
        dists = point_pointcloud_distance(target_pc[i], predicted_pc)
        if i == 0:
            dist = torch.min(dists)
        else:
            dist += torch.min(dists)
    return dist/target_pc.shape[0]

def all_permutations_half(array):
    """
    This method is used to generate permutation of control points grid.
    This is specifically used for closed b-spline surfaces. Note that
    In the pre-processing step, all closed splines are made to close in u
    direction only, thereby reducing the possible permutations to half. This
    is done to speedup the training and also to facilitate learning for neural
    network.
    """
    permutations = []
    permutations.append(array)
    permutations.append(torch.flip(array, (1,)))
    permutations.append(torch.flip(array, (2,)))
    permutations.append(torch.flip(array, (1, 2)))
    permutations = torch.stack(permutations, 0)
    permutations = permutations.permute(1, 0, 2, 3, 4)
    return permutations


def roll(x: torch.Tensor, shift: int, dim: int = -1, fill_pad=None):
    """
    Rolls the tensor by certain shifts along certain dimension.
    """
    if 0 == shift:
        return x
    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift))
        return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim))), gap], dim=dim)
    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, torch.arange(shift, x.size(dim)).cuda())
        return torch.cat([gap, x.index_select(dim, torch.arange(shift).cuda())], dim=dim)


def control_points_permute_closed_reg_loss(output, grid_size_x, grid_size_y):
    """
    control points prediction with permutation invariant loss
    :param output: output of the network
    :param control_points: N x grid_size x grid_size x 3
    :param grid_size_x: size of the control points in u direction
    :param grid_size_y: size of the control points in v direction
    """
    batch_size = output.shape[0]
    output = output.view(batch_size, grid_size_x, grid_size_y, 3)
    # output = torch.unsqueeze(output, 1)

    # N x 8 x grid_size x grid_size x 3
    rhos = []
    for i in range(grid_size_y):
        new_control_points = roll(output, i, 1)
        rhos.append(all_permutations_half(new_control_points))
    control_points = torch.cat(rhos, 1)
    print(control_points.shape)
    diff = (output - control_points) ** 2
    diff = torch.sum(diff, (2, 3, 4))

    loss, index = torch.min(diff, 1)
    loss = torch.mean(loss) / (grid_size_x * grid_size_y * 3)

    return loss, control_points[np.arange(batch_size), index]

def extended_parameter(inp_ctrl_pts, weights):
    first_layer_int = inp_ctrl_pts[:, 0, :, :]
    first_layer_weights = weights[:, 0, :, :]
    # Replicate the first layer to match the desired size
    extended_inp_ctrl_pts = torch.cat((first_layer_int.unsqueeze(1), inp_ctrl_pts), dim=1)
    first_column_int = extended_inp_ctrl_pts[:, :, 0, :]
    extended_inp_ctrl_pts = torch.cat((first_column_int.unsqueeze(2), extended_inp_ctrl_pts), dim=2)
    extended_weights = torch.cat((first_layer_weights.unsqueeze(1), weights), dim=1)
    first_column_weights = extended_weights[:, :, 0, :]
    extended_weights = torch.cat((first_column_weights.unsqueeze(2), extended_weights), dim=2)

def compute_edge_lengths(points, u, v):
    points = points.reshape(u, v, 3)

    # Compute the differences between consecutive points
    point_diff = points[:, :-1, :] - points[:, 1:, :]
    last_to_first_diff = points[:, -1, :] - points[:, 0, :]

    # Compute the L2 norm of differences along each dimension
    point_diff_norm = torch.norm(point_diff, dim=2)
    last_to_first_diff_norm = torch.norm(last_to_first_diff, dim=1)
    
    # Compute the total distance
    total_distance = torch.sum(point_diff_norm) + torch.sum(last_to_first_diff_norm)
    
    # Compute the differences between consecutive points
    point_diff = points[-1:, :, :] - points[1:, :, :]
    last_to_first_diff = points[-1, :, :] - points[0, :, :]

    # Compute the L2 norm of differences along each dimension
    point_diff_norm = torch.norm(point_diff, dim=2)
    last_to_first_diff_norm = torch.norm(last_to_first_diff, dim=1)

    # Compute the total distance
    total_distance = torch.sum(point_diff_norm) + torch.sum(last_to_first_diff_norm)

    # Compute the total distance
    total_distance = torch.sum(point_diff_norm) + torch.sum(last_to_first_diff_norm)
    
    # Compute the average distance
    average_distance = total_distance / (points.shape[0] * points.shape[1]) / 2

    return average_distance

def read_irregular_file(path):

    input_point_list = []
    target_list = []

    with open(path, 'r') as f:
        # with open('../../meshes/cube_cluster0_geodesic.txt', 'r') as f:
        # with open('ex_ducky.off', 'r') as f:

        lines = f.readlines()

        # skip the first line

        # lines = random.sample(lines, k=resolution * resolution)
        # extract vertex positions

        resolution_u = 0
        vertex_positions = []

        for line in lines:
            if line.startswith('#'):
                resolution_u += 1
                if len(vertex_positions) > 0:
                    target = torch.tensor(vertex_positions).float().cuda()
                    target_list.append(target)
                    vertex_positions = []

            else:
                x, y, z = map(float, line.split()[:3])
                # min_coord = min(min_coord, x, y, z)
                # max_coord = max(max_coord, x, y, z)
                vertex_positions.append((x, y, z))
                input_point_list.append((x, y, z))

    return input_point_list, target_list, vertex_positions, resolution_u
def get_normals(weights, inp_ctrl_pts, knot_int_u, knot_int_v, num_ctrl_pts1, num_ctrl_pts2, layer):
    predictedweights = weights.detach().cpu().numpy().squeeze(0)
    predictedctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()

    predictedknotu = knot_int_u.detach().cpu().numpy().squeeze().tolist()
    predictedknotu = [0., 0., 0., 0., 0.] + predictedknotu + [1., 1., 1., 1.]
    predictedknotv = knot_int_v.detach().cpu().numpy().squeeze().tolist()
    predictedknotv = [0., 0., 0., 0., 0.] + predictedknotv + [1., 1., 1., 1.]

    surf = NURBS.Surface()
    # Set degrees
    surf.degree_u = 3
    surf.degree_v = 3

    surf.ctrlpts_size_u = num_ctrl_pts1
    surf.ctrlpts_size_v = num_ctrl_pts2

    # reshape predictedctrlpts to a list
    predictedctrlpts = predictedctrlpts.reshape(num_ctrl_pts1 * num_ctrl_pts2, 3)
    surf.ctrlpts = predictedctrlpts
    predictedweights = predictedweights.reshape(num_ctrl_pts1 * num_ctrl_pts2, 1)
    # surf.weights = predictedweights

    U, V = layer.getrealUV()
    U = U.detach().cpu().numpy().reshape(-1, 1)
    V = V.detach().cpu().numpy().reshape(-1, 1)

    # Set knot vectors
    surf.knotvector_u = list(U)
    surf.knotvector_v = list(V)

    uv_vals = list(itertools.product(surf.knotvector_u, surf.knotvector_v))
    surfnorms = [[] for _ in range(len(uv_vals))]
    idx = 0

    for u, v in list(itertools.product(surf.knotvector_u, surf.knotvector_v)):
        surfnorms[idx] = operations.normal(surf, [u, v], normalize=False)
        idx += 1

    surfpts = np.array(surf.evalpts)
    normal_vectors = np.array(surfnorms)
def main():

    gt_path = os.path.dirname(os.path.realpath(__file__))
    gt_path = gt_path.split("/")[0:-1]
    gt_path = "/".join(gt_path)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # gt_path = "/home/lizeth/Documents/Repositories/pygeodesics/data/brain.obj"
    # cm_path = '/home/lizeth/Documents/Repositories/NURBSDiff/data/cm_brain.txt'
    # ctr_pts_path = '/home/lizeth/Documents/Repositories/NURBSDiff/data/cm_brain_ctrpts.txt'

    # gt_path = "/mnt/Chest/Repositories/NURBSDiff/data/luigi.obj"
    # cm_path = '/mnt/Chest/Repositories/NURBSDiff/data/cm_luigi_0.025_20.txt'
    # ctr_pts_path = '/mnt/Chest/Repositories/NURBSDiff/data/cm_luigi_uniform_warp_offset_0.003_20.txt'

    # gt_path = "/home/lizeth/Documents/Repositories/pygeodesics/data/duck_clean.obj"
    # cm_path = '/home/lizeth/Documents/Repositories/NURBSDiff/data/cm_ducky_0.003_50.txt'
    # ctr_pts_path = '/home/lizeth/Documents/Repositories/NURBSDiff/data/cm_ducky_0.003_20.txt'

    gt_path = gt_path + "/pygeodesics/data/sphere.obj"
    cm_path = dir_path + '/data/cm_sphere_uniform_pts_0.003_50.txt'
    ctr_pts_path = dir_path + '/data/cm_sphere_off_2_0.003_20.txt'

    # ctr_pts = 40
    # resolution_u = 64
    # resolution_v = 64
    p = q = 3
    
    object_name = gt_path.split("/")[-1].split(".")[0]
    
    num_epochs = 10
    loss_type = "chamfer"
    ignore_uv = True
    axis = "y"
    
    def get_current_time():
        return time.strftime("%m%d%H%M%S", time.localtime())
    current_time = get_current_time()

    configure("logs/tensorboard/{}".format(f'{object_name}_irregular_input/{current_time}'), flush_secs=2)

    #ducky parameters
    
    out_dim_u = 250
    out_dim_v = 250
    ctr_pts_u = 15
    ctr_pts_v = 15
    resolution_v = 50

    w_lap = 0.8
    mod_iter = 5
    cglobal = 1
    average = 1
    learning_rate = 0.5

    #best
    learning_rate = 0.05
    
    # load point cloud
    max_coord = min_coord = 0




    input_point_list, target_list, vertex_positions, resolution_u = read_irregular_file(cm_path)

    print("#input points " + str(len(input_point_list)))

    target = torch.tensor(vertex_positions).float().cuda()
    print(target.shape)
    target_list.append(target)

    sample_size_u = resolution_u
    sample_size_v = resolution_v

    

    num_eval_pts_u = resolution_u

    cp_input_point_list, cp_target_list, cp_vertex_positions, cp_resolution_u = read_irregular_file(ctr_pts_path)

    print("#input control points " + str(len(cp_input_point_list)))

    # reshape cp_input_point_list to (1, cp_resolution_u, cp_resolution_u, 3)
    inp_ctrl_pts = torch.tensor(cp_input_point_list).float().cuda().reshape(1, cp_resolution_u, cp_resolution_u, 3)

    ctr_pts_u = cp_resolution_u
    ctr_pts_v = cp_resolution_u

    num_ctrl_pts1 = ctr_pts_u
    num_ctrl_pts2 = ctr_pts_v

    inp_ctrl_pts.requires_grad = True
    n_knots_u = num_ctrl_pts1+p+1-2*p-1
    n_knots_v = num_ctrl_pts2+q+1-2*q-1

    knots_u = (torch.range(0.01, n_knots_u)/n_knots_u)
    knots_u[-1] = 1
    knots_v = (torch.range(0.01, n_knots_u)/n_knots_v)
    knots_v[-1] = 1

    # inp_ctrl_pts = torch.rand((1, num_ctrl_pts1, num_ctrl_pts2, 3), requires_grad=False).float().cuda()
    knot_int_u = torch.nn.Parameter(torch.ones(num_ctrl_pts1 - p).unsqueeze(0).cuda(), requires_grad=True)
    # knot_int_u = torch.nn.Parameter(knots_u.unsqueeze(0).cuda(), requires_grad=True)
    # knot_int_u = torch.nn.Parameter(torch.cat((torch.ones(num_ctrl_pts1 - p - 3), torch.zeros(3)), dim=0).unsqueeze(0).cuda(), requires_grad=True)
    knot_int_v = torch.nn.Parameter(torch.ones(num_ctrl_pts2 - q).unsqueeze(0).cuda(), requires_grad=True)
    # knot_int_v = torch.nn.Parameter(knots_v.unsqueeze(0).cuda(), requires_grad=True)
    # knot_int_v = torch.nn.Parameter(torch.cat((torch.ones(num_ctrl_pts2 - q - 1), torch.zeros(1)), dim=0).unsqueeze(0).cuda(), requires_grad=True)
    # knot_int_u = torch.nn.Parameter(torch.ones(num_ctrl_pts1+p+1-2*p-1).unsqueeze(0).cuda(), requires_grad=True)
    # knot_int_v = torch.nn.Parameter(torch.ones(num_ctrl_pts2+q+1-2*q-1).unsqueeze(0).cuda(), requires_grad=True)
    if inp_ctrl_pts.requires_grad:
        print("inp_ctrl_pts is trainable")
    else:
        print("inp_ctrl_pts is not trainable")
    weights = torch.nn.Parameter(torch.ones(1,num_ctrl_pts1,num_ctrl_pts2,1).float().cuda(), requires_grad=True)

    # print(target.shape)
    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=sample_size_u, out_dim_v=sample_size_v, method='tc', dvc='cuda').cuda()
    
    opt1 = torch.optim.Adam(iter([inp_ctrl_pts, weights]), lr=learning_rate)
    opt2 = torch.optim.Adam(iter([knot_int_u, knot_int_v]), lr=1e-2)
    lr_schedule1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=10, factor=0.1, verbose=True, min_lr=1e-5,
                                                              eps=1e-08, threshold=1e-4, threshold_mode='rel', cooldown=0,
                                                              )
    lr_schedule2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=5, factor=0.1, verbose=True, min_lr=1e-5, 
                                                              eps=1e-08, threshold=1e-4, threshold_mode='rel', cooldown=0,)
    pbar = tqdm(range(num_epochs))
    # fig = plt.figure(figsize=(15, 9))
    time1 = time.time()
    
    
    knot_rep_p_0 = torch.zeros(1,p+1).cuda()
    knot_rep_p_1 = torch.zeros(1,p).cuda()
    knot_rep_q_0 = torch.zeros(1,q+1).cuda()
    knot_rep_q_1 = torch.zeros(1,q).cuda()
    beforeTrained = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))[0].detach().cpu().numpy().squeeze()
    
    with open(f'generated/{object_name}/ctrpts_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}_before_trained.OFF', 'w') as f:
       # Loop over the array rows
        f.write('OFF\n')
        f.write(str(sample_size_u * sample_size_v) + ' ' + '0 0\n')
        for i in range(sample_size_u):
            for j in range(sample_size_v):
                # print(predicted_target[i, j, :])
                line = str(beforeTrained[i, j, 0]) + ' ' + str(beforeTrained[i, j, 1]) + ' ' + str(beforeTrained[i, j, 2]) + '\n'
                f.write(line)
    
    for i in pbar:
        # torch.cuda.empty_cache()
        knot_rep_p_0 = torch.zeros(1,p+1).cuda()
        knot_rep_p_1 = torch.zeros(1,p).cuda()
        knot_rep_q_0 = torch.zeros(1,q+1).cuda()
        knot_rep_q_1 = torch.zeros(1,q).cuda()

        with torch.no_grad():
            # #rows
            if average > 0:
                inp_ctrl_pts[:, 0, :, :] = inp_ctrl_pts[:, 0, :, :].mean(1)
                inp_ctrl_pts[:, -1, :, :] = inp_ctrl_pts[:, -1, :, :].mean(1)
                inp_ctrl_pts[:, :, 0, :] = inp_ctrl_pts[:, :, -3, :] = (inp_ctrl_pts[:, :, 0, :] + inp_ctrl_pts[:, :, -3, :]) / 2
                inp_ctrl_pts[:, :, 1, :] = inp_ctrl_pts[:, :, -2, :] = (inp_ctrl_pts[:, :, 1, :] + inp_ctrl_pts[:, :, -2, :]) / 2
                inp_ctrl_pts[:, :, 2, :] = inp_ctrl_pts[:, :, -1, :] = (inp_ctrl_pts[:, :, 2, :] + inp_ctrl_pts[:, :, -1, :]) / 2

            pass
        
        def closure():
            # if i % 100 < 30:
            #     opt1.zero_grad()
            # else:
            #     opt2.zero_grad()

            opt1.zero_grad()

            out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))

            loss = 0

            # get the normals


            if ignore_uv:
                lap = laplacian_loss_unsupervised(inp_ctrl_pts)
                out = out.reshape(sample_size_u, sample_size_v, 3)




                if loss_type == 'chamfer':
                    # if global loss

                    if cglobal == True:
                        # loss = chamfer_distance(out_knn, input_ctrl_pts_knn)
                        tgt = torch.stack(target_list)
                        tgt = tgt.reshape(-1, 3).unsqueeze(0)
                        out = out.reshape(1, sample_size_u*sample_size_v, 3)

                        if (i + 1) % mod_iter == 0:
                            #copy tgt to host
                            tgt_cpu = tgt.detach().cpu().numpy().squeeze()
                            out_cpu = out.detach().cpu().numpy().squeeze()
                            #visualize tgt and out
                            fig = plt.figure()
                            ax = fig.add_subplot( projection='3d')
                            ax.scatter(tgt_cpu[:, 0], tgt_cpu[:, 1], tgt_cpu[:, 2], c='r', marker='o')
                            ax.scatter(out_cpu[:, 0], out_cpu[:, 1], out_cpu[:, 2], c='b', marker='o')
                            plt.show()

                        loss = (1-w_lap) * chamfer_distance(out, tgt) + w_lap * lap

                    #decrease w_lap according to the epoch
                    # if i < 600:
                    #     w_lap = 0.1
                    # else:
                    #     w_lap = 0.1 * (1 - (i - 600)/600)
                    else:
                        loss = (1-w_lap) * chamfer_distance_each_row(out, target_list) + w_lap * lap

                    log_value('chamfer_distance', loss, i)
                    # log_value('laplacian_loss', lap * 10, i)
                    # log_value('close_loss_column', close_loss_column, i)

            loss.sum().backward(retain_graph=True)
            return loss


        if i % 100 < 30:
            loss = opt1.step(closure)
            lr_schedule1.step(loss)
        else:
            loss = opt2.step(closure)
            lr_schedule2.step(loss)

        out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))


        if (i + 1) % mod_iter == 0:
            fig = plt.figure()
            predicted = out.detach().cpu().numpy().squeeze()
            # ctrlpts = inp_ctrl_pts.reshape(num_ctrl_pts1, num_ctrl_pts2, 3)
            predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
            # predctrlptsctrlpts = predctrlpts.reshape(num_ctrl_pts1, num_ctrl_pts2, 3)

            ax2 = fig.add_subplot( projection='3d')
            surf2 = ax2.plot_wireframe(predicted[:, :, 0], predicted[:, :, 1], predicted[:, :, 2], color='green',
                                       label='Predicted Surface')
            surf2 = ax2.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2],
                                       linestyle='dashed', color='orange', label='Predicted Control Points')
            ax2.azim = 45
            ax2.dist = 6.5
            ax2.elev = 30
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_zticks([])
            ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax2._axis3don = False


            plt.show()

        if loss.item() < 1e-6:
            print((time.time() - time1)/ (i + 1)) 
            break
        
        pbar.set_description("Loss %s: %s" % (i+1, loss.item()))

    print((time.time() - time1)/ (num_epochs+1))

    U, V = layer.getrealUV()       
    U = U.detach().cpu().numpy().reshape(-1, 1)
    V = V.detach().cpu().numpy().reshape(-1, 1)
    target_mpl = target.cpu().numpy().squeeze()

    predicted = out.detach().cpu().numpy().squeeze()

    predictedweights = weights.detach().cpu().numpy().squeeze(0)
    predictedctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
    # print(predictedweights.shape)
    # print(predictedctrlpts.shape)
    predictedknotu = knot_int_u.detach().cpu().numpy().squeeze().tolist()
    predictedknotu = [0., 0., 0., 0., 0.] + predictedknotu + [1., 1., 1., 1.]
    predictedknotv = knot_int_v.detach().cpu().numpy().squeeze().tolist()
    predictedknotv = [0., 0., 0., 0., 0.] + predictedknotv + [1., 1., 1., 1.]

    # Open the file in write mode
    with open('generated/u_test.ctrlpts', 'w') as f:
        # Loop over the array rows
        x = predictedctrlpts
        x = x.reshape(ctr_pts_u, ctr_pts_v, 3)
        
        for i in range(ctr_pts_u):
            for j in range(ctr_pts_v):
                # print(predicted_target[i, j, :])
                line = str(x[i, j, 0]) + ' ' + str(x[i, j, 1]) + ' ' + str(x[i, j, 2])
                f.write(line)
                # if (j == ctr_pts - 1):
                f.write('\n')
                # else:
                #     f.write(';')

    with open('generated/u_test.weights', 'w') as f:
        # Loop over the array rows
        x = predictedweights

        for row in x:
            # Flatten the row to a 1D array
            row_flat = row.reshape(-1)
            # Write the row values to the file as a string separated by spaces
            f.write(','.join([str(x) for x in row_flat]) + '\n')

    with open('generated/u_test.knotu', 'w') as f:
        # Loop over the array rows
        x = predictedknotu

        for row in x:
            # Flatten the row to a 1D array
    
            # Write the row values to the file as a string separated by spaces
            f.write(','.join([str(row)]) + '\n')

    with open('generated/u_test.knotv', 'w') as f:
        # Loop over the array rows
        x = predictedknotv

        for row in x:
            # Flatten the row to a 1D array
   
            # Write the row values to the file as a string separated by spaces
            f.write(','.join([str(row)]) + '\n')


    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=out_dim_u, out_dim_v=out_dim_v, method='tc', dvc='cuda').cuda()
    knot_rep_p_0 = torch.zeros(1,p+1).cuda()
    knot_rep_p_1 = torch.zeros(1,p).cuda()
    knot_rep_q_0 = torch.zeros(1,q+1).cuda()
    knot_rep_q_1 = torch.zeros(1,q).cuda()

    out2 = layer((torch.cat((inp_ctrl_pts, weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
    out2 = out2.detach().cpu().numpy().squeeze(0).reshape(out_dim_u, out_dim_v, 3)
    # ax4.plot_wireframe(out2[:, :, 0], out2[:, :, 1], out2[:, :, 2], color='cyan', label='Reconstructed Surface')
    # adjust_plot(ax4)

    # target_mpl = target_mpl.reshape(resolution_u, resolution_v, 3)
    predicted = predicted.reshape(sample_size_u, sample_size_v, 3)
    

    with open(f'generated/{object_name}/points_{ctr_pts_u}_{out_dim_u}x{out_dim_v}_{resolution_v}.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(out_dim_u * out_dim_v) + ' ' + '0 0\n')
        for i in range(out_dim_u):
            for j in range(out_dim_v):
                # print(predicted_target[i, j, :])
                line = str(out2[i, j, 0]) + ' ' + str(out2[i, j, 1]) + ' ' + str(out2[i, j, 2]) + '\n'
                f.write(line)
                

    with open(f'generated/{object_name}/ctrpts_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}.OFF', 'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(num_ctrl_pts1 * num_ctrl_pts2) + ' ' + '0 0\n')
        for i in range(num_ctrl_pts1):
            for j in range(num_ctrl_pts2):
                line = str(predictedctrlpts[i, j, 0]) + ' ' + str(predictedctrlpts[i, j, 1]) + ' ' + str(predictedctrlpts[i, j, 2]) + '\n'
                f.write(line)
                
    with open(f'generated/{object_name}/ctrpts_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}', 'w') as f:
        # Loop over the array rows
        for i in range(num_ctrl_pts1):
            for j in range(num_ctrl_pts2):
                line = str(predictedctrlpts[i, j, 0]) + ' ' + str(predictedctrlpts[i, j, 1]) + ' ' + str(predictedctrlpts[i, j, 2]) + '\n'
                f.write(line)

    #write U and V in a file
    np.save(f'generated/{object_name}/U_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}', U)
    np.save(f'generated/{object_name}/V_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}', V)


                
    filename_ctrpts = f'generated/{object_name}/ctrpts_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}'

    U = np.load(f'generated/{object_name}/U_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}.npy')
    V = np.load(f'generated/{object_name}/V_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}.npy')

    # force the last 4 elements of and V to be 1
    U[-4:] = 1
    V[-4:] = 1
    surfpts, tangent_vectors, normal_vectors = reconstructed_mesh(object_name, filename_ctrpts, num_ctrl_pts1, num_ctrl_pts2, U, V)

    # Start plotting of the surface and the control points grid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)

    # Plot surface points
    #plot points 3d
    ax.scatter(surfpts[:, 0], surfpts[:, 1], surfpts[:, 2], color='xkcd:gold', marker='^')
    # ax.plot_trisurf(surfpts[:, 0], surfpts[:, 1], surfpts[:, 2], color='xkcd:gold', alpha=0.5)

    # Plot tangent vectors (u-dir)
    ax.quiver(tangent_vectors[:, 0, 0], tangent_vectors[:, 0, 1], tangent_vectors[:, 0, 2],
              tangent_vectors[:, 1, 0], tangent_vectors[:, 1, 1], tangent_vectors[:, 1, 2],
              color='xkcd:bright blue', length=1)

    # Plot tangent vectors (v-dir)
    ax.quiver(tangent_vectors[:, 0, 0], tangent_vectors[:, 0, 1], tangent_vectors[:, 0, 2],
              tangent_vectors[:, 2, 0], tangent_vectors[:, 2, 1], tangent_vectors[:, 2, 2],
              color='xkcd:neon green', length=1)

    # Plot normal vectors
    ax.quiver(normal_vectors[:, 0, 0], normal_vectors[:, 0, 1], normal_vectors[:, 0, 2],
              normal_vectors[:, 1, 0], normal_vectors[:, 1, 1], normal_vectors[:, 1, 2],
              color='xkcd:bright red', length=2)

    # Add legend to 3D plot, @ref: https://stackoverflow.com/a/20505720
    surface_prx = matplotlib.lines.Line2D([0], [0], linestyle='none', color='xkcd:gold', marker='^')
    tanu_prx = matplotlib.lines.Line2D([0], [0], linestyle='none', color='xkcd:bright blue', marker='>')
    tanv_prx = matplotlib.lines.Line2D([0], [0], linestyle='none', color='xkcd:neon green', marker='>')
    normal_prx = matplotlib.lines.Line2D([0], [0], linestyle='none', color='xkcd:bright red', marker='>')
    ax.legend([surface_prx, tanu_prx, tanv_prx, normal_prx],
              ['Surface Plot', 'Tangent Vectors (u-dir)', 'Tangent Vectors (v-dir)', 'Normal Vectors'],
              numpoints=1)
    # # Rotate the axes and update the plot
    # for angle in range(0, 360, 10):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)

    plt.show()
    
    pass

if __name__ == '__main__':
    


    init_plt()

    main()


