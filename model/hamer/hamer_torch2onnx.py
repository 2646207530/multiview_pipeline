import os
import random
import numpy as np
from rich.progress import track
import torch
import onnx
import onnxruntime as ort
from rich import print

import sys

from onnx_utils import check_onnx_model
sys.path.append('../')
sys.path.append('../../')

from infer import hamer_inference

# hamer pytorch: dict_output1, dict_output2 = model(dict_input)
# hamer onnx: tuples_output = model(tuples_input)

def hamer_input_dict2tuple(input):
    return input['img'], input['box_center'], input['box_size'], \
            input['img_size'], input['inv_trans'], input['do_flip'],

def hamer_input_tuple2dict(input):
    return {'img': input[0], 'box_center': input[1], 'box_size': input[2], 
            'img_size': input[3], 'inv_trans': input[4], 'do_flip': input[5],}

def hamer_input_tuple2onnx_dict(input_tuple, input_names):
    """将tuple格式的输入转换为ONNX需要的字典格式（numpy数组）"""
    onnx_input = {}
    for i, name in enumerate(input_names):
        if isinstance(input_tuple[i], torch.Tensor):
            onnx_input[name] = input_tuple[i].cpu().numpy()
        else:
            onnx_input[name] = input_tuple[i]
    return onnx_input

def hamer_output_tuple_of_dict2tuple(output):
    # dict1: dict_keys(['pred_cam', 'pred_mano_params', 'pred_cam_t', 'focal_length', 'pred_keypoints_3d', 'pred_vertices', 'pred_keypoints_2d'])
    # dict2: dict_keys(['global_orient', 'hand_pose', 'betas', 'trans'])
    return output[0]['pred_cam'], output[0]['pred_mano_params'], output[0]['pred_cam_t'], output[0]['focal_length'], \
        output[0]['pred_keypoints_3d'], output[0]['pred_vertices'], output[0]['pred_keypoints_2d'], \
        output[1]['global_orient'], output[1]['hand_pose'], output[1]['betas'], output[1]['trans'],

def hamer_output_tuple2tuple_of_dict(output):
    return {
        'pred_cam': output[0], 'pred_mano_params': output[1], 'pred_cam_t': output[2], 'focal_length': output[3],
        'pred_keypoints_3d': output[4], 'pred_vertices': output[5], 'pred_keypoints_2d': output[6],
    }, {
        'global_orient': output[7], 'hand_pose': output[8], 'betas': output[9], 'trans': output[10],
    }


# 定义forward hook函数，将tuple输入转换为dict输入
def hamer_forward_pre_hook(module, input):
    dict_input = hamer_input_tuple2dict(input)
    return dict_input

# 定义forward hook函数，将dict输出转换为tuple输出
def hamer_forward_hook(module, input, output):
    """
    在模型前向传播后，将tuple of dict格式的输出转换为tuple格式
    """
    return hamer_output_tuple_of_dict2tuple(output)


def compare_pytorch_with_onnx(pytorch_model, ort_session, device, pytorch_input, ort_input, iter_times=1, rtol=1e-3, atol=1e-3):
    # 获取输出名称
    output_names = [output.name for output in ort_session.get_outputs()]
    input_names = [input.name for input in ort_session.get_inputs()]
    print(f"output_names: {output_names}")
    print(f"input_names: {input_names}")
    
    # 将tuple格式的输入转换为ONNX需要的字典格式
    if isinstance(ort_input, tuple):
        ort_input_dict = hamer_input_tuple2onnx_dict(ort_input, input_names)
    else:
        ort_input_dict = ort_input

    pytorch_model= pytorch_model.to(device).eval()

    for _ in track(range(iter_times)):

        # pytorch
        with torch.no_grad():
            pytorch_output_tuple = pytorch_model(*pytorch_input)

        # onnx
        # ort_output_tuple = ort_session.run(None, {'input_img': ort_input})
        ort_output_tuple = ort_session.run(output_names, ort_input_dict)

        assert len(pytorch_output_tuple) == len(ort_output_tuple)
        trouble_outputs, ok_outputs = [], []
        for i, (pytorch_output, ort_output) in enumerate(zip(pytorch_output_tuple, ort_output_tuple)):
            if not isinstance(pytorch_output, torch.Tensor):
                print(f"Output {output_names[i]} is not a tensor, but {type(pytorch_output)}. Skipping...")
                continue

            pytorch_output = pytorch_output.cpu().numpy()
            assert pytorch_output.shape == ort_output.shape
            try:
                np.testing.assert_allclose(pytorch_output, ort_output, rtol=rtol, atol=atol)
                ok_outputs.append(output_names[i])
            except AssertionError as e:
                trouble_outputs.append(output_names[i])
                print(f"Output {output_names[i]} failed the test: {e}")
            
            # print pytorch_output and ort_output
            print(f"Output {output_names[i]}:")
            print(f"pytorch_output: {pytorch_output.shape}, ort_output: {ort_output.shape}")
            print(f"pytorch_output: {pytorch_output}")
            print(f"ort_output: {ort_output}")
            print(f"\n\n\n")
    
    # 只输出最后一次的结果
    if len(trouble_outputs) > 0:
        print('Trouble outputs:')
        print(trouble_outputs)
    
    if len(ok_outputs) > 0:
        print('OK outputs:')
        print(ok_outputs)


def torch2onnx(model_type: str, out_dir: str, validate: bool, to_simplify: bool, shape_infer: bool, verbose: bool):
    dynamic = False # fail with dynamic=True
    do_constant_folding = True 
    onnx_path = os.path.join(out_dir, f'{model_type}.onnx')
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    hamer_infer = hamer_inference()

    batch_size = 1
    device = 'cuda'
    opset_version = 16

    # input_names = ['img', 'box_center', 'box_size', 'img_size', 'inv_trans', 'do_flip']
    input_names = ['img']
    output_names = ['pred_cam', 'pred_mano_params', 'pred_cam_t', 'focal_length',
                    'pred_keypoints_3d', 'pred_vertices', 'pred_keypoints_2d',
                    'global_orient', 'hand_pose', 'betas', 'trans']
    # NOTE: 如果input是tuple且以dict结尾，那么结尾的dict会被当作name argument, 参考https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export
    dummy_input_for_pytorch = {
        'img': torch.randn(1, 3, hamer_infer.cfg.MODEL.IMAGE_SIZE, hamer_infer.cfg.MODEL.IMAGE_SIZE, device=device),
        # 'box_center': torch.zeros(1, 2, dtype=torch.float32, device=device),  # 假设中心为0
        # 'box_size': torch.tensor([[200.0]], dtype=torch.float32, device=device), # 假设box_size
        # 'img_size': torch.tensor([[hamer_infer.cfg.MODEL.IMAGE_SIZE, hamer_infer.cfg.MODEL.IMAGE_SIZE]], dtype=torch.float32, device=device),
        # 'inv_trans': torch.zeros(1, 6, dtype=torch.float32, device=device),
        # 'do_flip': torch.zeros(1, dtype=torch.float32, device=device),
    }
    dummy_input_for_onnx = hamer_input_dict2tuple(dummy_input_for_pytorch)

    with torch.no_grad():
        pytorch_model = hamer_infer.model.to(device)
        pytorch_model.eval()

        # 注册forward hook来处理输入输出格式转换
        pre_hook_handle = pytorch_model.register_forward_pre_hook(hamer_forward_pre_hook)
        post_hook_handle = pytorch_model.register_forward_hook(hamer_forward_hook)
        
        # 测试用tuple输入
        test_tuple_output = pytorch_model(*dummy_input_for_onnx) # test forward with tuple input
        print("Model forward with tuple input has been successful!")
        print(f"Tuple output shape: {[t.shape if hasattr(t, 'shape') else type(t) for t in test_tuple_output]}")

        # TEMP
        # if dynamic:
        #     # dynamic_axes = {input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}}
        #     # torch.onnx.export(pytorch_model, dummy_input, onnx_path, verbose=verbose, input_names=input_names,
        #     #                 output_names=output_names, opset_version=opset_version, dynamic_axes=dynamic_axes,
        #     #                 do_constant_folding=do_constant_folding)
        #     raise NotImplementedError("onnx export with dynamic axes is not implemented yet!")
        # else:
        #     print(f"Exporting ONNX model with opset_version={opset_version}")
        #     torch.onnx.export(
        #         pytorch_model, 
        #         dummy_input_for_onnx, 
        #         onnx_path, 
        #         verbose=verbose, 
        #         input_names=input_names, 
        #         output_names=output_names, 
        #         opset_version=opset_version, 
        #         do_constant_folding=do_constant_folding,
        #     )
        #     print(f"ONNX export completed: {onnx_path}")
        
        # # 导出完成后移除hook
        # pre_hook_handle.remove()
        # post_hook_handle.remove()
        # print("Hooks removed after ONNX export!")
        
        if validate:
            check_onnx_model(onnx_path)

        # if to_simplify:
        #     print("Simplifying onnx model ...")
        #     onnx_model = onnx.load(onnx_path)
        #     # simplifying dynamic model
        #     simplified_model, is_success = simplify(onnx_model, overwrite_input_shapes={input_names[0]: [batch_size, 3, imheight, imwidth]})
        #     assert is_success, "Failed to simplify"
        #     onnx.save(simplified_model, onnx_path)
        #     check_onnx_model(onnx_path)

        if shape_infer:
            # print("Using shape inference ...")
            # from onnx import shape_inference
            # onnx_model = onnx.load(onnx_path)
            # inferred_model = shape_inference.infer_shapes(onnx_model)
            # onnx.save(inferred_model, onnx_path)
            # check_onnx_model(onnx_path)
            raise NotImplementedError("onnx shape inference is not implemented yet!")

    print("Exporting .pth model to onnx model has been successful!")

    sess_options = ort.SessionOptions()
    if device == 'cpu':
        ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    else:
        ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CUDAExecutionProvider'])
    compare_pytorch_with_onnx(
        pytorch_model,
        ort_session,
        device,
        pytorch_input=dummy_input_for_onnx,
        ort_input=dummy_input_for_onnx,
        iter_times=10,
        rtol=1e-4,
        atol=1e-4,
    )

    print(f"[Precision check] Comparing .pth model with onnx model {onnx_path} is successful!")


def set_seed(seed=42):
    random.seed(seed)  # Python的随机数生成器
    np.random.seed(seed)  # NumPy的随机数生成器
    torch.manual_seed(seed)  # PyTorch的随机数生成器

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)  # Python的哈希算法的随机种子


class TorchReprModifier:
    """from https://stackoverflow.com/questions/70704619/is-it-possible-to-show-variable-shapes-and-lengths-in-vscode-python-debugger-jus"""

    def __init__(self):
        self.original_torch_repr = torch.Tensor.__repr__

    def enable_custom_repr(self):
        # 定义自定义的PyTorch张量表示方法
        def custom_torch_repr(tensor):
            # return f'Tensor.shape:{tuple(tensor.shape)} {self.original_torch_repr(tensor)}'
            return f"{tuple(tensor.shape)} {tensor.device} {tensor.dtype} {self.original_torch_repr(tensor)}"

        torch.Tensor.__repr__ = custom_torch_repr

    def restore_original_repr(self):
        torch.Tensor.__repr__ = self.original_torch_repr


if __name__ == '__main__':
    # usage
    # python my_smplerx_torch2onnx.py \
    #    --model_type $MODEL_TYPE \
    #    --out_dir $OUT_DIR \
    #    --validate \
    #    --simplify \
    #    --shape_infer \
    #    --verbose > $OUT_DIR/torch2onnx.log

    import argparse
    parser = argparse.ArgumentParser(description='PyTorch to ONNX')
    parser.add_argument('--model_type', type=str, default='hamer', help='model type')
    parser.add_argument('--out_dir', type=str, default='/home/pt/vGesture/software/hamer/_DATA/hamer_ckpts/onnx/', help='output directory')
    parser.add_argument('--validate', action='store_true', help='validate onnx model', default=True)
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model', default=False)
    parser.add_argument('--shape_infer', action='store_true', help='use shape inference', default=False)
    parser.add_argument('--verbose', action='store_true', help='verbose', default=False)
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--vsdebug', action='store_true', help='for vsdebug')
    args = parser.parse_args()

    TorchReprModifier().enable_custom_repr()
    if args.vsdebug:
        import debugpy
        debugpy.connect(("localhost", 5678))

    set_seed(args.seed)
    torch2onnx(args.model_type, args.out_dir, args.validate, args.simplify, args.shape_infer, args.verbose)
