import torch
import torchvision.models as models
import argparse
from args.sonardata_args import parse_sonardata_args
from models.autoencoder import AutoEncoder
from models.utils import save_pcd, AverageMeter, str2bool
import onnx , onnxsim
import sys

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

def exportDownSample():
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

    onnx_model_path = "DPCC.onnx"
    # 定义动态轴
    dynamic_axes = {
        'input': {2: 'N'},       # N 是输入的动态维度
        'output_0': {2: 'M'},    # M 是第一个输出的动态维度
        'output_1': {2: 'M'},    # M 是第二个输出的动态维度
    }
    model.to("cuda")
    # model(torch.randn(1, 4, 8314).cuda())
    # exit(0)
    torch.onnx.export(model, torch.randn(1, 4, 8314).cuda(),
                        onnx_model_path,
                        input_names=["input"],
                        output_names=["output_0","output_1"],
                        dynamic_axes=dynamic_axes,
                        opset_version=13,
                        do_constant_folding=False)

    print(f"Model has been converted to ONNX and saved at {onnx_model_path}")

    with open(onnx_model_path, "rb") as f:
        onnx_model = onnx.load(f)

    model_simp, check = onnxsim.simplify(onnx_model, perform_optimization=False)

    assert check, "Simplified ONNX model could not be validated"

    simplified_model_path = "DPCC_sim.onnx"
    
    onnx.save(model_simp, simplified_model_path)


if __name__ == '__main__':

    exportDownSample()
