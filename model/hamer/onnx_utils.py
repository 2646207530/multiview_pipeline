
import onnx
import onnxruntime as ort


def check_onnx_model(onnx_path):
    print(f"checking onnx model: {onnx_path} .......")
    # check output names and shapes
    onnx_model = onnx.load(onnx_path)
    # 打印输出信息
    print("[Model outputs]:")
    for output in onnx_model.graph.output:
        # 尝试打印输出形状（如果有的话）注意：某些情况下输出形状可能是动态的，因此可能不会显示具体的维度
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f"Name: {output.name}, Shape: {shape}")

    # check the onnx model
    try:
        onnx.checker.check_model(onnx_path)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')

    # Running ORT check 增加ORT验证
    try:
        sess = ort.InferenceSession(onnx_path)
        print(f"ORT Loaded {onnx_path} !")
        for _ in sess.get_inputs(): print(f"Input: {_}")
        for _ in sess.get_outputs(): print(f"Output: {_}")
        print("ORT Check Done !")
    except Exception as e:
        print(f"ORT validation failed: {e}")
        # 尝试查看模型的详细信息
        print("Trying to load model with onnx.load_model()...")
        try:
            model_proto = onnx.load(onnx_path)
            print(f"Model IR version: {model_proto.ir_version}")
            print(f"Model opset: {[opset.version for opset in model_proto.opset_import]}")
        except Exception as load_error:
            print(f"Failed to load model: {load_error}")
        raise e
