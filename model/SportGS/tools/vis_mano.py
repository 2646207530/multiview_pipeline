import numpy as np
import pickle
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. 加载 MANO 官方模型数据
# ---------------------------------------------------------
# 请确保 MANO_RIGHT.pkl 文件在当前目录下
try:
    with open('/mnt/sda2/lxy/arctic/unpack/body_models/mano/MANO_RIGHT.pkl', 'rb') as f:
        # 使用 latin1 编码读取 Python 2 的 pickle 文件
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        mano_data = u.load()
except FileNotFoundError:
    raise FileNotFoundError("未找到 'MANO_RIGHT.pkl' 文件。请从 MANO 官网下载并放置在该脚本目录下。")

# ---------------------------------------------------------
# 2. 提取并计算“展平手”的顶点坐标
# ---------------------------------------------------------
# 'v_template' 是 MANO 初始姿势的 778 个顶点
# 它的形状是 (778, 3)
flat_vertices = np.array(mano_data['v_template'])

# 【可选但推荐】将手部重心移动到原点 (0,0,0)，方便观察
centroid = np.mean(flat_vertices, axis=0)
flat_vertices_centered = flat_vertices - centroid

print(f"成功加载 MANO 右手模型，共有 {flat_vertices_centered.shape[0]} 个顶点。")

# ---------------------------------------------------------
# 3. 创建 3D 交互式可视化 (Plotly)
# ---------------------------------------------------------
num_verts = len(flat_vertices_centered)
indices = np.arange(num_verts)
hover_texts = [f"Vertex ID: {i}" for i in indices] # 鼠标悬停显示的文本

fig = go.Figure(data=[go.Scatter3d(
    x=flat_vertices_centered[:, 0], # X 轴数据
    y=flat_vertices_centered[:, 1], # Y 轴数据
    z=flat_vertices_centered[:, 2], # Z 轴数据
    mode='markers',
    marker=dict(
        size=3,                  # 点的大小
        color=indices,           # 颜色根据 Index 渐变，一眼看出点序
        colorscale='Viridis',    # 使用好看的渐变色带
        opacity=1.0
    ),
    text=hover_texts,            # 绑定悬停文本
    hoverinfo='text'             # 鼠标指上去只显示 ID
)])

# ---------------------------------------------------------
# 4. 优化 3D 布局
# ---------------------------------------------------------
fig.update_layout(
    title="MANO 右手官方模板顶点索引 (0-777)",
    scene=dict(
        xaxis_title='X (左右)',
        yaxis_title='Y (上下)',
        zaxis_title='Z (前后)',
        aspectmode='data', # 1:1:1 真实比例，不让模型变形
        camera=dict(
            # 设置一个默认视角，正对着手背
            eye=dict(x=1.5, y=-1.5, z=0.8) 
        )
    ),
    margin=dict(l=0, r=0, b=0, t=50)
)

# ---------------------------------------------------------
# 5. 显示
# ---------------------------------------------------------
#fig.show() # 在默认浏览器中打开
fig.write_html("/home/cyc/pycharm/lxy/3DGS/SportGS/output/mano_verts.html")
print("HTML 文件已生成：mano_verts.html")