
import time
import torch
import numpy as np
import os
from geomdl import NURBS
from geomdl import operations

from examples.test.mesh_reconstruction import reconstructed_mesh

torch.manual_seed(120)
from tensorboard_logger import configure, log_value
from tqdm import tqdm
import itertools
from pytorch3d.structures import Pointclouds

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
# git clone https://github.com/facebookresearch/pytorch3d.git
# cd pytorch3d && pip install -e .
from NURBSDiff.nurbs_eval import SurfEval

import matplotlib.pyplot as plt
import matplotlib

from torch.autograd.variable import Variable
import torch.nn.functional as F



SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")


def chamfer_distance_corr(pred, gt, sqrt=False):
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

def laplacian_loss_unsupervised(output, dist_type="l2"):
    filter = ([[[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    filter = np.stack([filter, np.roll(filter, 1, 0), np.roll(filter, 2, 0)])

    filter = -np.array(filter, dtype=np.float32)
    if torch.cuda.is_available():
        filter = Variable(torch.from_numpy(filter)).cuda()
    else:
        filter = Variable(torch.from_numpy(filter))
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
                    if torch.cuda.is_available():
                        target = torch.tensor(vertex_positions).float().cuda()
                    else:
                        target = torch.tensor(vertex_positions).float()
                    target_list.append(target)
                    vertex_positions = []

            else:
                x, y, z = map(float, line.split()[:3])
                # min_coord = min(min_coord, x, y, z)
                # max_coord = max(max_coord, x, y, z)
                vertex_positions.append((x, y, z))
                input_point_list.append((x, y, z))

    return input_point_list, target_list, vertex_positions, resolution_u

def plot_pointcloud(points, title=""):
    # Sample points uniformly from the surface of the mesh.
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()
def plot_tangent_normals(surfpts, tangent_vectors, normal_vectors):
    # Start plotting of the surface and the control points grid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)

    # Plot surface points
    # plot points 3d
    ax.scatter(surfpts[:, 0], surfpts[:, 1], surfpts[:, 2], color='xkcd:gold', marker='^')
    # ax.plot_trisurf(surfpts[:, 0], surfpts[:, 1], surfpts[:, 2], color='xkcd:gold', alpha=0.5)

    # Plot tangent vectors (u-dir)
    ax.quiver(tangent_vectors[:, 0, 0], tangent_vectors[:, 0, 1], tangent_vectors[:, 0, 2],
              tangent_vectors[:, 1, 0], tangent_vectors[:, 1, 1], tangent_vectors[:, 1, 2],
              color='xkcd:bright blue', length=0.25)

    # Plot tangent vectors (v-dir)
    ax.quiver(tangent_vectors[:, 0, 0], tangent_vectors[:, 0, 1], tangent_vectors[:, 0, 2],
              tangent_vectors[:, 2, 0], tangent_vectors[:, 2, 1], tangent_vectors[:, 2, 2],
              color='xkcd:neon green', length=0.25)

    # Plot normal vectors
    ax.quiver(normal_vectors[:, 0, 0], normal_vectors[:, 0, 1], normal_vectors[:, 0, 2],
              normal_vectors[:, 1, 0], normal_vectors[:, 1, 1], normal_vectors[:, 1, 2],
              color='xkcd:bright red', length=0.35)

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

def create_grid_points(res_u, res_v):
    cp_resolution_u = res_u
    cp_resolution_v = res_v

    x = np.array([np.linspace(-1, 1, num=cp_resolution_u) for _ in range(cp_resolution_v)])
    y = np.array([np.linspace(-1, 1, num=cp_resolution_v) for _ in range(cp_resolution_u)]).T

    zs = np.ones(x.shape) * -2
    z = zs.reshape(x.shape)

    return x,y,z

def create_mesh_from_grid(grid_points):
    grid_points
    verts = grid_points.reshape(-1, 3)
    faces_idx = []
    for i in range(0, grid_points.shape[1]-1):
        for j in range(0, grid_points.shape[2]-1):
            faces_idx.append([i * grid_points.shape[2] + j, i * grid_points.shape[2] + j + 1,
                                (i + 1) * grid_points.shape[2] + j + 1])
            faces_idx.append([i * grid_points.shape[2] + j, (i + 1) * grid_points.shape[2] + j + 1,
                                (i + 1) * grid_points.shape[2] + j])

    #create a torch in32 tensor for  the faces
    faces_idx = torch.tensor(faces_idx, dtype=torch.int32).cuda()
    trg_mesh = Meshes(verts=[verts], faces=[faces_idx])
    return trg_mesh

def get_normals(weights, inp_ctrl_pts, num_ctrl_pts1, num_ctrl_pts2, layer):
    predictedweights = weights.detach().cpu().numpy().squeeze(0)
    predictedctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()

    # predictedknotu = knot_int_u.detach().cpu().numpy().squeeze().tolist()
    # predictedknotu = [0., 0., 0., 0., 0.] + predictedknotu + [1., 1., 1., 1.]
    # predictedknotv = knot_int_v.detach().cpu().numpy().squeeze().tolist()
    # predictedknotv = [0., 0., 0., 0., 0.] + predictedknotv + [1., 1., 1., 1.]

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

    u_span, v_span = layer.get_spanned_uv()
    u_span = u_span.detach().cpu().numpy().astype(float)
    v_span = v_span.detach().cpu().numpy().astype(float)
    # uv_vals = list(itertools.product(u, v))
    surfnorms = [[] for _ in range(len(u_span) * len(v_span))]
    idx = 0

    cross = list(itertools.product(u_span, v_span))


    for u, v in list(itertools.product(u_span, v_span)):
        surfnorms[idx] = operations.normal(surf, [u, v], normalize=True)
        idx += 1

    surfpts = np.array(surf.evalpts)
    normal_vectors = np.array(surfnorms)

    return  surfpts, normal_vectors


def get_grid_init_points_patches(ctrlpts, k, s):
    #get control poitns shape as u and v
    u = ctrlpts.shape[1]
    v = ctrlpts.shape[2]
    # get the list of indices (i,j) for the convlution using k as kernel size and s as stride
    indices = [(i, j) for i in range(0, u - k + 1, s) for j in range(0, v - k + 1, s)]

    return indices

class Mask:
    count = 0
    idx_patch = -1
    def init(self, masks, n_iter_patch):
        self.mask = masks
        self.n_iter_patch = n_iter_patch

    def update(self, indices, k):
        if self.idx_patch == -1:
            ii, j = indices[self.idx_patch]
            self.mask[:, ii:ii + k, j:j + k, :] = 0
            self.idx_patch += 1

        self.count += 1
        if self.count == self.n_iter_patch:
            self.count = 0
            ii, j = indices[self.idx_patch]
            if self.idx_patch != -1:
                self.mask[:, ii:ii + k, j:j + k, :] = 1
            self.idx_patch += 1
            ii, j = indices[self.idx_patch]
            self.mask[:, ii:ii + k, j:j + k, :] = 0
        return self.mask

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

    # gt_path = gt_path + "/pygeodesics/data/duck_clean.obj"
    # cm_path =  dir_path + '/data/cm_ducky_0.003_50.txt'
    # ctr_pts_path =  dir_path + '/data/cm_ducky_0.003_20.txt'

    # gt_path = gt_path + "/pygeodesics/data/sphere.obj"
    # cm_path = dir_path + '/data/cm_sphere_uniform_pts_0.003_50.txt'
    # ctr_pts_path = dir_path + '/data/cm_sphere_off_2_0.003_20.txt'

    gt_path = gt_path + "/pygeodesics/data/sphere_normals.obj"
    cm_path = dir_path + '/data/cm_sphere_half_500.003_50.txt'
    ctr_pts_path = dir_path + '/data/cm_sphere_half0.003_20.txt'


    # ctr_pts = 40
    # resolution_u = 64
    # resolution_v = 64
    p = q = 3

    object_name = gt_path.split("/")[-1].split(".")[0]

    num_epochs = 2000
    loss_type = "chamfer"

    axis = "y"

    def get_current_time():
        return time.strftime("%m%d%H%M%S", time.localtime())

    current_time = get_current_time()

    configure("logs/tensorboard/{}".format(f'{object_name}_irregular_input/{current_time}'), flush_secs=2)

    out_dim_u = 250
    out_dim_v = 250


    w_lap = 0.1
    w_chamfer = 1
    w_normals = 0

    target_from_path = False


    mod_iter = 2000
    cglobal = 1
    average = 0
    use_grid = True
    show_normals = True

    n_ctrpts = 10
    resolution_u = 30  # samples in the v directions columns per curve points
    resolution_v = 30  # samples in the u direction rows per curve points

    k = 6 # kernel size
    s = 2 # stride
    n_iter_patch = 20
    using_mask = False

    # best
    learning_rate = 0.05

    chamfer_losses = []
    laplacian_losses = []
    normal_losses = []

    input_point_list, target_list, vertex_positions, resolution_uu = read_irregular_file(cm_path)

    print("#input points " + str(len(input_point_list)))

    # if torch.cuda.is_available():
    #     target = torch.tensor(vertex_positions).float().cuda()
    # else:
    #     target = torch.tensor(vertex_positions).float()
    #
    # print(target.shape)
    # target_list.append(target)


    if target_from_path == True:
        verts, faces, properties = load_obj(gt_path)

        target_vert = torch.tensor(verts).float().cuda()
        gt_normals = properties.normals
        gt_normals = torch.tensor(gt_normals).float().cuda().unsqueeze(0)
    else:
    # create a torch tensor from input_point_list
        target_vert = torch.tensor(input_point_list).float().cuda()
        point_cloud = Pointclouds(points=[target_vert])
        gt_normals = point_cloud.estimate_normals(16)



    sample_size_u = resolution_u
    sample_size_v = resolution_v

    if use_grid == False:
        cp_input_point_list, cp_target_list, cp_vertex_positions, cp_resolution_u = read_irregular_file(ctr_pts_path)
        cp_resolution_v = cp_target_list[0].shape[0]
    else:
        cp_resolution_u = n_ctrpts
        cp_resolution_v = n_ctrpts



    if torch.cuda.is_available():
        if use_grid:
            create_grid_points(cp_resolution_u, cp_resolution_v)
            x , y, z = create_grid_points(cp_resolution_u, cp_resolution_v)
            inp_ctrl_pts = torch.from_numpy(np.array([x, y, z])).permute(1, 2, 0).unsqueeze(0).contiguous().float().cuda()
            # if use_mesh_losses:
            #     new_ctrl_mesh = create_mesh_from_grid(inp_ctrl_pts)
        else:
            inp_ctrl_pts = torch.tensor(cp_input_point_list).float().cuda().reshape(1, cp_resolution_u, cp_resolution_v, 3).cuda()

    else:
        inp_ctrl_pts = torch.tensor(cp_input_point_list).float().reshape(1, cp_resolution_u, cp_resolution_u, 3)

    ctr_pts_u = cp_resolution_u
    ctr_pts_v = cp_resolution_v

    num_ctrl_pts1 = ctr_pts_u
    num_ctrl_pts2 = ctr_pts_v

    inp_ctrl_pts.requires_grad = True
    #create a int tensor idx_patch as int
    idx_patch = torch.tensor(0).int().cuda()


    #used only when masking out the control points
    indices = get_grid_init_points_patches(inp_ctrl_pts, k, s)




    if torch.cuda.is_available():
        knot_int_u = torch.nn.Parameter(torch.ones(num_ctrl_pts1 - p).unsqueeze(0).cuda(), requires_grad=True)
        knot_int_v = torch.nn.Parameter(torch.ones(num_ctrl_pts2 - q).unsqueeze(0).cuda(), requires_grad=True)
        weights = torch.nn.Parameter(torch.ones(1, num_ctrl_pts1, num_ctrl_pts2, 1).float().cuda(), requires_grad=True)
        layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=sample_size_u,
                         out_dim_v=sample_size_v, method='tc', dvc='cuda').cuda()

    else:
        knot_int_u = torch.nn.Parameter(torch.ones(num_ctrl_pts1 - p).unsqueeze(0), requires_grad=True)
        knot_int_v = torch.nn.Parameter(torch.ones(num_ctrl_pts2 - q).unsqueeze(0), requires_grad=True)
        # knot_int_v = torch.nn.Parameter(knots_v.unsqueeze(0).cuda(), requires_grad=True)
        print(sample_size_u, sample_size_v)

        weights = torch.nn.Parameter(torch.ones(1, num_ctrl_pts1, num_ctrl_pts2, 1).float(), requires_grad=True)
        layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=sample_size_u,
                         out_dim_v=sample_size_v, method='tc')

    if using_mask:
        mask = torch.ones(inp_ctrl_pts.shape).cuda()
        mask_obj = Mask()
        mask_obj.init( mask, n_iter_patch)

        mask_weights = torch.ones(weights.shape).cuda()
        mask_weights_obj = Mask()
        mask_weights_obj.init(mask_weights, n_iter_patch)

    opt1 = torch.optim.Adam(iter([inp_ctrl_pts, weights]), lr=learning_rate)
    opt2 = torch.optim.Adam(iter([knot_int_u, knot_int_v]), lr=1e-2)
    lr_schedule1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=10, factor=0.1, verbose=True, min_lr=1e-5,
                                                              eps=1e-08, threshold=1e-4, threshold_mode='rel',
                                                              cooldown=0,
                                                              )
    lr_schedule2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=5, factor=0.1, verbose=True, min_lr=1e-5,
                                                              eps=1e-08, threshold=1e-4, threshold_mode='rel',
                                                              cooldown=0, )
    pbar = tqdm(range(num_epochs))
    # fig = plt.figure(figsize=(15, 9))
    time1 = time.time()

    if torch.cuda.is_available():
        knot_rep_p_0 = torch.zeros(1, p + 1).cuda()
        knot_rep_p_1 = torch.zeros(1, p).cuda()
        knot_rep_q_0 = torch.zeros(1, q + 1).cuda()
        knot_rep_q_1 = torch.zeros(1, q).cuda()
    else:
        knot_rep_p_0 = torch.zeros(1, p + 1)
        knot_rep_p_1 = torch.zeros(1, p)
        knot_rep_q_0 = torch.zeros(1, q + 1)
        knot_rep_q_1 = torch.zeros(1, q)
    beforeTrained = layer((torch.cat((inp_ctrl_pts, weights), -1),
                           torch.cat((knot_rep_p_0, knot_int_u, knot_rep_p_1), -1),
                           torch.cat((knot_rep_q_0, knot_int_v, knot_rep_q_1), -1)))[0].detach().cpu().numpy().squeeze()

    with open(
            f'generated/{object_name}/ctrpts_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}_before_trained.OFF',
            'w') as f:
        # Loop over the array rows
        f.write('OFF\n')
        f.write(str(sample_size_u * sample_size_v) + ' ' + '0 0\n')
        for i in range(sample_size_u):
            for j in range(sample_size_v):
                # print(predicted_target[i, j, :])
                line = str(beforeTrained[i, j, 0]) + ' ' + str(beforeTrained[i, j, 1]) + ' ' + str(
                    beforeTrained[i, j, 2]) + '\n'
                f.write(line)

    for i in pbar:
        # torch.cuda.empty_cache()

        if torch.cuda.is_available():
            knot_rep_p_0 = torch.zeros(1, p + 1).cuda()
            knot_rep_p_1 = torch.zeros(1, p).cuda()
            knot_rep_q_0 = torch.zeros(1, q + 1).cuda()
            knot_rep_q_1 = torch.zeros(1, q).cuda()
        else:
            knot_rep_p_0 = torch.zeros(1, p + 1)
            knot_rep_p_1 = torch.zeros(1, p)
            knot_rep_q_0 = torch.zeros(1, q + 1)
            knot_rep_q_1 = torch.zeros(1, q)

        with torch.no_grad():
            # #rows
            if average > 0:
                inp_ctrl_pts[:, 0, :, :] = inp_ctrl_pts[:, 0, :, :].mean(1)
                inp_ctrl_pts[:, -1, :, :] = inp_ctrl_pts[:, -1, :, :].mean(1)
                inp_ctrl_pts[:, :, 0, :] = inp_ctrl_pts[:, :, -3, :] = (inp_ctrl_pts[:, :, 0, :] + inp_ctrl_pts[:, :,
                                                                                                   -3, :]) / 2
                inp_ctrl_pts[:, :, 1, :] = inp_ctrl_pts[:, :, -2, :] = (inp_ctrl_pts[:, :, 1, :] + inp_ctrl_pts[:, :,
                                                                                                   -2, :]) / 2
                inp_ctrl_pts[:, :, 2, :] = inp_ctrl_pts[:, :, -1, :] = (inp_ctrl_pts[:, :, 2, :] + inp_ctrl_pts[:, :,
                                                                                                   -1, :]) / 2
            pass

        def closure():
            # if i % 100 < 30:
            #     opt1.zero_grad()
            # else:
            #     opt2.zero_grad()

            opt1.zero_grad()

            out = layer((
                        torch.cat((inp_ctrl_pts, weights), -1), torch.cat((knot_rep_p_0, knot_int_u, knot_rep_p_1), -1),
                        torch.cat((knot_rep_q_0, knot_int_v, knot_rep_q_1), -1)))
            loss = 0

            # get the normals

            if w_normals > 0:
                surfpts, out_normals_points = get_normals(weights, inp_ctrl_pts, num_ctrl_pts1, num_ctrl_pts2, layer)
                out_normals = out_normals_points[:, 1]
                #create a tensor with out_normals
                out_normals = torch.tensor(out_normals).float().cuda().unsqueeze(0)


            loss_laplacian = laplacian_loss_unsupervised(inp_ctrl_pts)
            out = out.reshape(sample_size_u, sample_size_v, 3)

            if loss_type == 'chamfer':
                # if global loss

                if cglobal == True:

                    # tgt = torch.stack(target_list)
                    # tgt = tgt.reshape(-1, 3).unsqueeze(0)
                    tgt = torch.tensor(target_vert).float().cuda().unsqueeze(0)
                    out = out.reshape(1, sample_size_u * sample_size_v, 3)


                    if (i + 1) % mod_iter == 0:
                        # copy tgt to host
                        tgt_cpu = target_vert.detach().cpu().numpy().squeeze()
                        out_cpu = out.detach().cpu().numpy().squeeze()

                        gt_normals_cpu = gt_normals.detach().cpu().numpy().squeeze()
                        gt_normals_cpu = np.stack((tgt_cpu, gt_normals_cpu), axis=1)

                        if w_normals > 0:
                            out_normals_cpu = out_normals.detach().cpu().numpy().squeeze()
                            out_normals_cpu = np.stack((out_cpu, out_normals_cpu), axis=1)

                        # visualize tgt and out
                        fig = plt.figure()
                        ax = fig.add_subplot(projection='3d')
                        # a = 102
                        # b = 153
                        a = 0
                        b = -1
                        ax.scatter(tgt_cpu[a:b, 0], tgt_cpu[a:b, 1], tgt_cpu[a:b, 2], c='r', marker='o')
                        ax.scatter(out_cpu[a:b, 0], out_cpu[a:b, 1], out_cpu[a:b, 2], c='b', marker='o')

                        if show_normals == True:
                            ax.quiver(gt_normals_cpu[:, 0, 0], gt_normals_cpu[:, 0, 1], gt_normals_cpu[:, 0, 2],
                                      gt_normals_cpu[:, 1, 0], gt_normals_cpu[:, 1, 1], gt_normals_cpu[:, 1, 2],
                                      color='green', length=0.15)
                            if w_normals > 0:
                                ax.quiver(out_normals_cpu[:, 0, 0], out_normals_cpu[:, 0, 1], out_normals_cpu[:, 0, 2],
                                          out_normals_cpu[:, 1, 0], out_normals_cpu[:, 1, 1], out_normals_cpu[:, 1, 2],
                                          color='black', length=0.15)


                        plt.show()
                    if w_normals > 0:
                        loss_chamfer, loss_normals = chamfer_distance(out, tgt, x_normals=out_normals, y_normals=gt_normals)
                        loss = w_chamfer * loss_chamfer + w_lap * loss_laplacian + w_normals * loss_normals

                        # Save the losses for plotting
                        chamfer_losses.append(w_chamfer * float(loss_chamfer.detach().cpu()))
                        normal_losses.append(w_normals * float(loss_normals.detach().cpu()))
                        laplacian_losses.append(w_lap * float(loss_laplacian.detach().cpu()))

                    else:
                        loss_chamfer, _ = chamfer_distance(out, tgt)
                        loss = w_chamfer * loss_chamfer + w_lap * loss_laplacian

                # decrease w_lap according to the epoch
                # if i < 600:
                #     w_lap = 0.1
                # else:
                #     w_lap = 0.1 * (1 - (i - 600)/600)
                else:
                    loss = (1 - w_lap) * chamfer_distance_each_row(out, target_list) + w_lap * lap

                log_value('chamfer_distance', loss, i)
                # log_value('laplacian_loss', lap * 10, i)
                # log_value('close_loss_column', close_loss_column, i)

            loss.sum().backward(retain_graph=True)

            grads_ctrpts = inp_ctrl_pts.grad
            grads_weights = weights.grad
            grads_knot_u = knot_int_u.grad
            grads_knot_v = knot_int_v.grad

            # # avoiding the sphere to open up
            # close_sphere == True:
            # inp_ctrl_pts.grad[:, 0, :, :] = 0
            # inp_ctrl_pts.grad[:, -1, :, :] = 0
            # inp_ctrl_pts.grad[:, :, 0, :] = inp_ctrl_pts.grad[:, :, -1, :] = 0

            # disabling knots
            knot_int_u.grad = torch.zeros(knot_int_u.grad.shape).cuda()
            knot_int_v.grad = torch.zeros(knot_int_v.grad.shape).cuda()

            if using_mask:
                masked = mask_obj.update(indices, k)
                masked_weights = mask_weights_obj.update(indices, k)
                inp_ctrl_pts.grad = torch.masked_fill(inp_ctrl_pts.grad, masked.bool(), 0)
                weights.grad = torch.masked_fill(weights.grad, masked_weights.bool(), 0)


            return loss

        if i % 100 < 30:
            loss = opt1.step(closure)
            lr_schedule1.step(loss)
        else:
            loss = opt2.step(closure)
            lr_schedule2.step(loss)

        loss = opt1.step(closure)
        lr_schedule1.step(loss)

        out = layer((torch.cat((inp_ctrl_pts, weights), -1), torch.cat((knot_rep_p_0, knot_int_u, knot_rep_p_1), -1),
                     torch.cat((knot_rep_q_0, knot_int_v, knot_rep_q_1), -1)))

        if (i + 1) % mod_iter == 0:
            fig = plt.figure()
            predicted = out.detach().cpu().numpy().squeeze()
            # ctrlpts = inp_ctrl_pts.reshape(num_ctrl_pts1, num_ctrl_pts2, 3)
            predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
            # predctrlptsctrlpts = predctrlpts.reshape(num_ctrl_pts1, num_ctrl_pts2, 3)

            ax2 = fig.add_subplot(projection='3d')
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
            if w_normals > 0 :
                show_losses(chamfer_losses, normal_losses, laplacian_losses)

        if loss.item() < 1e-6:
            print((time.time() - time1) / (i + 1))
            break

        pbar.set_description("Loss %s: %s" % (i + 1, loss.item()))

    print((time.time() - time1) / (num_epochs + 1))

    U, V = layer.getrealUV()
    U = U.detach().cpu().numpy().reshape(-1, 1)
    V = V.detach().cpu().numpy().reshape(-1, 1)

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

    if torch.cuda.is_available():
        layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=out_dim_u, out_dim_v=out_dim_v,
                         method='tc', dvc='cuda').cuda()
        knot_rep_p_0 = torch.zeros(1, p + 1).cuda()
        knot_rep_p_1 = torch.zeros(1, p).cuda()
        knot_rep_q_0 = torch.zeros(1, q + 1).cuda()
        knot_rep_q_1 = torch.zeros(1, q).cuda()
    else:
        layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=p, q=q, out_dim_u=out_dim_u, out_dim_v=out_dim_v,
                         method='tc')
        knot_rep_p_0 = torch.zeros(1, p + 1)
        knot_rep_p_1 = torch.zeros(1, p)
        knot_rep_q_0 = torch.zeros(1, q + 1)
        knot_rep_q_1 = torch.zeros(1, q)

    knots_u = torch.cat((knot_rep_p_0, knot_int_u, knot_rep_p_1), -1)
    knots_v = torch.cat((knot_rep_q_0, knot_int_v, knot_rep_q_1), -1)

    out2 = layer((torch.cat((inp_ctrl_pts, weights), -1), knots_u, knots_v))
    out2 = out2.detach().cpu().numpy().squeeze(0).reshape(out_dim_u, out_dim_v, 3)

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
                line = str(predictedctrlpts[i, j, 0]) + ' ' + str(predictedctrlpts[i, j, 1]) + ' ' + str(
                    predictedctrlpts[i, j, 2]) + '\n'
                f.write(line)

    with open(f'generated/{object_name}/ctrpts_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}', 'w') as f:
        # Loop over the array rows
        for i in range(num_ctrl_pts1):
            for j in range(num_ctrl_pts2):
                line = str(predictedctrlpts[i, j, 0]) + ' ' + str(predictedctrlpts[i, j, 1]) + ' ' + str(
                    predictedctrlpts[i, j, 2]) + '\n'
                f.write(line)

    # write U and V in a file
    np.save(f'generated/{object_name}/U_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}', U)
    np.save(f'generated/{object_name}/V_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}', V)

    filename_ctrpts = f'generated/{object_name}/ctrpts_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}'

    U = np.load(f'generated/{object_name}/U_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}.npy')
    V = np.load(f'generated/{object_name}/V_{ctr_pts_u}_dim_{out_dim_u}x{out_dim_v}_{resolution_v}.npy')

    # force the last 4 elements of and V to be 1
    U[-4:] = 1
    V[-4:] = 1
    # surfpts, tangent_vectors, normal_vectors = reconstructed_mesh(object_name, filename_ctrpts, num_ctrl_pts1, num_ctrl_pts2, U, V)
    surfpts, tangent_vectors, normal_vectors = reconstructed_mesh(object_name, filename_ctrpts, num_ctrl_pts1,
                                                                  num_ctrl_pts2, U, V)
    plot_tangent_normals(surfpts, tangent_vectors, normal_vectors)

    pass
def show_losses(chamfer_losses, normal_losses, laplacian_losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    ax.plot(chamfer_losses, label="chamfer loss")
    ax.plot(normal_losses, label="normal loss")
    ax.plot(laplacian_losses, label="laplacian loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")
    plt.show()
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

if __name__ == '__main__':
    init_plt()

    main()