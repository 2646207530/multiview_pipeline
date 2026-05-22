import torch
import torch.nn as nn

# 修改后的模型定义，明确接收两个输入
class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 5)
        self.linear2 = nn.Linear(3, 5)
    
    def forward(self, feature1, feature2):  # 现在明确接收两个参数
        # 处理两个输入
        output1 = self.linear1(feature1)
        output2 = self.linear2(feature2)
        
        # 输出是两个字典组成的元组
        return (
            {'output1': output1, 'output2': output2},
            {'logits': output1 + output2, 'attention': torch.sigmoid(output1 * output2)}
        )

# 创建模型实例
model = ExampleModel()
model.eval()

# 准备示例输入（现在是两个独立的张量）
example_feature1 = torch.randn(1, 3)
example_feature2 = torch.randn(1, 3)

# 定义输入和输出的名称
input_names = ['feature1', 'feature2']
output_names = [
    'output1', 'output2',  # 第一个字典的输出
    'logits', 'attention'  # 第二个字典的输出
]

# 导出ONNX模型
torch.onnx.export(
    model,
    (example_feature1, example_feature2),  # 直接传递两个输入张量
    "tmp/model_with_dict_io.onnx",
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        'feature1': {0: 'batch_size'},
        'feature2': {0: 'batch_size'},
        'output1': {0: 'batch_size'},
        'output2': {0: 'batch_size'},
        'logits': {0: 'batch_size'},
        'attention': {0: 'batch_size'}
    },
    opset_version=13,
    verbose=True
)

print("ONNX模型导出成功!")


import numpy as np
import onnxruntime as ort

# 加载ONNX模型
onnx_model_path = "tmp/model_with_dict_io.onnx"
ort_session = ort.InferenceSession(
    onnx_model_path,
    providers=['CUDAExecutionProvider']  # 使用GPU加速
)

# 打印输入输出信息
print("输入信息:")
for i, input_info in enumerate(ort_session.get_inputs()):
    print(f"  Input {i}: name={input_info.name}, shape={input_info.shape}, type={input_info.type}")

print("\n输出信息:")
for i, output_info in enumerate(ort_session.get_outputs()):
    print(f"  Output {i}: name={output_info.name}, shape={output_info.shape}, type={output_info.type}")

# 准备输入数据 (与导出时的input_names顺序一致)
batch_size = 2  # 可以使用不同的batch size测试动态轴
input_data = {
    'feature1': np.random.randn(batch_size, 3).astype(np.float32),
    'feature2': np.random.randn(batch_size, 3).astype(np.float32)
}

# 进行推理
# 注意: ONNX Runtime的输入需要按照input_names的顺序提供
outputs = ort_session.run(
    None,  # 表示我们要获取所有输出
    {
        'feature1': input_data['feature1'],
        'feature2': input_data['feature2']
    }
)

# 输出结果是展平后的列表，我们需要按照导出时的结构重新组织
# 根据导出时的output_names顺序:
# ['output1', 'output2', 'logits', 'attention']
output_dict1 = {
    'output1': outputs[0],
    'output2': outputs[1]
}

output_dict2 = {
    'logits': outputs[2],
    'attention': outputs[3]
}

# 打印结果
print("\n推理结果:")
print("第一个字典输出:")
for k, v in output_dict1.items():
    print(f"  {k}: shape={v.shape}, \n{v}")

print("\n第二个字典输出:")
for k, v in output_dict2.items():
    print(f"  {k}: shape={v.shape}, \n{v}")