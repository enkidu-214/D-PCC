import torch
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from models.autoencoder import AutoEncoder
import time
import argparse
import pickle as pkl
from args.semantickitti_args import parse_sonardata_args
from models.utils import save_pcd, AverageMeter, str2bool
from dataset.dataset import CompressDataset
from metrics.PSNR import get_psnr
from metrics.density import get_density_metric
from metrics.F1Score import get_f1_score
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()

def make_dirs(save_dir):
    gt_patch_dir = os.path.join(save_dir, 'patch/gt')
    if not os.path.exists(gt_patch_dir):
        os.makedirs(gt_patch_dir)
    pred_patch_dir = os.path.join(save_dir, 'patch/pred')
    if not os.path.exists(pred_patch_dir):
        os.makedirs(pred_patch_dir)
    gt_merge_dir = os.path.join(save_dir, 'merge/gt')
    if not os.path.exists(gt_merge_dir):
        os.makedirs(gt_merge_dir)
    pred_merge_dir = os.path.join(save_dir, 'merge/pred')
    if not os.path.exists(pred_merge_dir):
        os.makedirs(pred_merge_dir)

    return gt_patch_dir, pred_patch_dir, gt_merge_dir, pred_merge_dir

def load_model(args, model_path):
    # load model
    model = AutoEncoder(args).cuda()
    model.load_state_dict(torch.load(model_path))
    # update entropy bottleneck
    model.feats_eblock.update(force=True)
    if args.quantize_latent_xyzs == True:
        model.xyzs_eblock.update(force=True)
    model.eval()

    return model

def compress(args, model, xyzs, feats):
    # input: (b, c, n)

    encode_start = time.time()
    # raise dimension 对输入升维，中间也激活和归一化过
    feats = model.pre_conv(feats)

    # encoder forward xyzs是原始的，feats是升维的
    # 调用了Encoder的forward函数
    gt_xyzs, gt_dnums, gt_mdis, latent_xyzs, latent_feats = model.encoder(xyzs, feats)
    # decompress size
    feats_size = latent_feats.size()[2:]

    # compress latent feats
    latent_feats_str = model.feats_eblock.compress(latent_feats)

    # compress  latent xyzs
    if args.quantize_latent_xyzs == True:
        analyzed_latent_xyzs = model.latent_xyzs_analysis(latent_xyzs)
        # decompress size
        xyzs_size = analyzed_latent_xyzs.size()[2:]
        latent_xyzs_str = model.xyzs_eblock.compress(analyzed_latent_xyzs)
    else:
        # half float representation
        latent_xyzs_str = latent_xyzs.half()
        xyzs_size = None

    encode_time = time.time() - encode_start

    # bpp calculation
    points_num = xyzs.shape[0] * xyzs.shape[2]
    feats_bpp = (sum(len(s) for s in latent_feats_str) * 8.0) / points_num
    if args.quantize_latent_xyzs == True:
        xyzs_bpp = (sum(len(s) for s in latent_xyzs_str) * 8.0) / points_num
    else:
        xyzs_bpp = (latent_xyzs.shape[0] * latent_xyzs.shape[2] * 16 * 3) / points_num
    actual_bpp = feats_bpp + xyzs_bpp

    return latent_xyzs_str, xyzs_size, latent_feats_str, feats_size, encode_time, actual_bpp

def test_xyzs(args):
    # load data
    test_dataset = CompressDataset(data_path=args.test_data_path, cube_size=args.test_cube_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size)
    # indicate the last patch number of each full point cloud
    pcd_last_patch_num = test_dataset.pcd_last_patch_num

    # set up folders for saving point clouds
    model_path = args.model_path
    experiment_id = model_path.split('/')[-3]
    save_dir = os.path.join(args.output_path, experiment_id, 'pcd')
    gt_patch_dir, pred_patch_dir, gt_merge_dir, pred_merge_dir = make_dirs(save_dir)

    # load model
    model = load_model(args, model_path)

    # metrics
    patch_bpp = AverageMeter()
    patch_encode_time = AverageMeter()
    pcd_bpp = AverageMeter()

    # test
    with torch.no_grad():
        for i, input_dict in enumerate(test_loader):
            # input: (b, n, c)
            input = input_dict['xyzs'].cuda()
            # normals : (b, n, c)
            gt_normals = input_dict['normals'].cuda()
            # (b, c, n)
            input = input.permute(0, 2, 1).contiguous()
            # input就是二维的，无法理解加这个的意义
            xyzs = input[:, :3, :].contiguous()
            gt_patches = xyzs
            feats = input

            # 有用的是latent_xyzs_str, xyzs_size, latent_feats_str, feats_size
            # 格式分别为
            latent_xyzs_str, xyzs_size, latent_feats_str, feats_size, encode_time, \
            actual_bpp = compress(args, model, xyzs, feats)

            # update metrics
            patch_encode_time.update(encode_time)
            patch_bpp.update(actual_bpp)
            pcd_bpp.update(actual_bpp)
            # 将数据保存为二进制文件
            with open('compressed_data.pkl', 'wb') as f:
                pkl.dump({
                    'latent_xyzs_str': latent_xyzs_str,
                    'xyzs_size': xyzs_size,
                    'latent_feats_str': latent_feats_str,
                    'feats_size': feats_size
                }, f)

def test_normals(args):
    # load data
    test_dataset = CompressDataset(data_path=args.test_data_path, cube_size=args.test_cube_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size)
    # indicate the last patch number of each full point cloud
    pcd_last_patch_num = test_dataset.pcd_last_patch_num

    # set up folders for saving point clouds
    model_path = args.model_path
    experiment_id = model_path.split('/')[-3]
    save_dir = os.path.join(args.output_path, experiment_id, 'pcd')
    gt_patch_dir, pred_patch_dir, gt_merge_dir, pred_merge_dir = make_dirs(save_dir)

    # load model
    args.in_fdim = 6
    model = load_model(args, model_path)

    # metrics
    patch_bpp = AverageMeter()
    patch_encode_time = AverageMeter()
    pcd_bpp = AverageMeter()

    # merge xyzs and normals
    pcd_gt_patches = []
    pcd_pred_patches = []
    pcd_gt_normals = []
    pcd_pred_normals = []

    # test
    with torch.no_grad():
        for i, input_dict in enumerate(test_loader):
            # input: (b, n, c)
            input = input_dict['xyzs'].cuda()
            # normals : (b, n, c)
            gt_normals = input_dict['normals'].cuda()
            # (b, c, n)
            input = input.permute(0, 2, 1).contiguous()
            # concat normals
            input = torch.cat((input, gt_normals.permute(0, 2, 1).contiguous()), dim=1)
            xyzs = input[:, :3, :].contiguous()
            gt_patches = xyzs
            feats = input

            # compress
            latent_xyzs_str, xyzs_size, latent_feats_str, feats_size, encode_time, \
            actual_bpp = compress(args, model, xyzs, feats)
            
            # update metrics
            patch_encode_time.update(encode_time)
            patch_bpp.update(actual_bpp)
            pcd_bpp.update(actual_bpp)
            # 将数据保存为二进制文件
            with open('compressed_data.pkl', 'wb') as f:
                pkl.dump({
                    'latent_xyzs_str': latent_xyzs_str,
                    'xyzs_size': xyzs_size,
                    'latent_feats_str': latent_feats_str,
                    'feats_size': feats_size
                }, f)


def reset_model_args(train_args, model_args):
    for arg in vars(train_args):
        setattr(model_args, arg, getattr(train_args, arg))

def parse_test_args():
    parser = argparse.ArgumentParser(description='Test Arguments')

    # dataset
    parser.add_argument('--model_path', default='path to ckpt', type=str, help='path to ckpt')
    parser.add_argument('--batch_size', default=1, type=int, help='the test batch_size must be 1')
    parser.add_argument('--downsample_rate', default=[1/3, 1/3, 1/3], nargs='+', type=float, help='downsample rate')
    parser.add_argument('--max_upsample_num', default=[8, 8, 8], nargs='+', type=int, help='max upsample number, reversely symmetric with downsample_rate')
    parser.add_argument('--bpp_lambda', default=1e-3, type=float, help='bpp loss coefficient')
    # normal compression
    parser.add_argument('--compress_normal', default=True, type=str2bool, help='whether compress normals')
    # compress latent xyzs
    parser.add_argument('--quantize_latent_xyzs', default=True, type=str2bool, help='whether compress latent xyzs')
    parser.add_argument('--latent_xyzs_conv_mode', default='mlp', type=str, help='latent xyzs conv mode, mlp or edge_conv')
    # sub_point_conv mode
    parser.add_argument('--sub_point_conv_mode', default='mlp', type=str, help='sub-point conv mode, mlp or edge_conv')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    test_args = parse_test_args()
    assert test_args.dataset in ['shapenet', 'sonardata']
    # the test batch_size must be 1
    assert test_args.batch_size == 1

    model_args = parse_sonardata_args()
    reset_model_args(test_args, model_args)

    if model_args.compress_normal == False:
        test_xyzs(model_args)
    else:
        test_normals(model_args)