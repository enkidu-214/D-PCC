import onnx.checker
import torch
import torchvision.models as models
import argparse
import numpy as np
from args.sonardata_args import parse_sonardata_args
from models.autoencoder import AutoEncoder
from models.utils import save_pcd, AverageMeter, str2bool

def read_bin_to_matrix(file_path):
    # 使用 np.fromfile 直接读取二进制文件中的 32 位浮点数数据
    np_data = np.fromfile(file_path, dtype=np.float32)

    # 确保数据长度可以被4整除
    assert np_data.size % 4 == 0, "The total number of float32 values is not divisible by 4."

    # 将数据调整为 (n, 4) 的形状
    num_rows = np_data.size // 4
    np_data = np_data.reshape(num_rows, 4)

    # 现在我们将调整为 (1, 4, n) 的形状
    np_data = np_data.transpose(1, 0)  # 调整维度为 (4, n)
    np_data = np_data[np.newaxis, :, :]  # 添加新轴，使其维度变为 (1, 4, n)
    # 将 NumPy 数组转换为 PyTorch tensor，并移动到 GPU
    tensor_data = torch.from_numpy(np_data).to('cuda')
    return tensor_data

def parse_test_args():
    parser = argparse.ArgumentParser(description='Test Arguments')

    # dataset
    parser.add_argument('--dataset', default='sonardata', type=str, help='shapenet or sonardata')
    parser.add_argument('--model_path', default='path to ckpt', type=str, help='path to ckpt')
    parser.add_argument('--batch_size', default=1, type=int, help='the test batch_size must be 1')
    parser.add_argument('--downsample_rate', default=[1/2, 1/3, 1/3], nargs='+', type=float, help='downsample rate')
    parser.add_argument('--max_upsample_num', default=[8, 8, 8], nargs='+', type=int, help='max upsmaple number, reversely symmetric with downsample_rate')
    parser.add_argument('--bpp_lambda', default=1e-4, type=float, help='bpp loss coefficient')
    # normal compression
    parser.add_argument('--compress_normal', default=True, type=str2bool, help='whether compress normals')
    # compress latent xyzs
    parser.add_argument('--quantize_latent_xyzs', default=False, type=str2bool, help='whether compress latent xyzs')
    parser.add_argument('--latent_xyzs_conv_mode', default='mlp', type=str, help='latent xyzs conv mode, mlp or edge_conv')
    # sub_point_conv mode
    parser.add_argument('--sub_point_conv_mode', default='mlp', type=str, help='sub-point conv mode, mlp or edge_conv')

    args = parser.parse_args()
    return args

def reset_model_args(train_args, model_args):
    for arg in vars(train_args):
        setattr(model_args, arg, getattr(train_args, arg))

def test():
    # 创建模型实例
    test_args = parse_test_args()  # 你的参数
    model_args = parse_sonardata_args()
    reset_model_args(test_args, model_args)
    if model_args.compress_normal == True:
        model_args.in_fdim = 4
    model = AutoEncoder(model_args)
    # model = models.resnet50(pretrained=True)
    model.load_state_dict(torch.load("/home/data6T/pxy/D-PCC/output/2024-09-20T14:50:05.963269/ckpt/ckpt-epoch-120.pth"))
    model.eval()
    model.to("cuda")  
    input = read_bin_to_matrix("dataset/train_data_bin/016546_sampled.bin")
    latent_xyzs, latent_feats = model(input)
    print(latent_xyzs.shape)
    print(latent_feats.shape)
    print(latent_xyzs)
    print(latent_feats)


if __name__ == '__main__':
    test()