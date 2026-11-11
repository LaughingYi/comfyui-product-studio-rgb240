# 商品精修 · RGB240 最终交接版

只交付最后的纯底 **RGB(240,240,240)** 工作流。包、肩带、链条、五金、防尘袋等沿用原照片内容；排除盒子、相框、说明书，并尝试去除外部悬挂鱼线。原照投影用于局部落地阴影，没有全背景渐变或按轮廓描边的旧合成器。

**交接状态：本地 macOS CPU 参考实现；尚未在 Linux / ECS / NVIDIA GPU 上验收。** 当前 SAM、YOLO 和 ViTMatte 使用 CPU，工作流中 BiRefNet 也选 CPU。买了 GPU 并不会自动让这几个步骤加速。开发需先复现参考结果，再按部署文档做 GPU 移植和验收；本仓库没有另附未验证的 GPU 算法版本。

## 给开发人员

1. 拉取本仓库的 `v0.1.0-rgb240` 标签，按 [部署说明](docs/DEPLOY.md) 从零准备 ComfyUI。
2. 按 [模型清单](models/manifest.json) 下载或传入五个固定权重，约 0.97 GB；校验 SHA256。权重、原照片、旧版工作流不在 Git 里。
3. 先运行 CPU 参考流程，复核主体、配件、透明边缘与局部投影，再在实际服务器完成 GPU 迁移和样图验收。

## 仓库内容

| 路径 | 用途 |
| --- | --- |
| `custom_nodes/product_studio_rgb240/` | 最终三个处理节点；保留已有节点 ID，便于原流程衔接 |
| `workflows/rgb240_ui.json` | ComfyUI 界面版，选择输入图后运行 |
| `workflows/rgb240_api.json` | 标准服务器 API 版，输入图片名由客户端替换 |
| `scripts/` | 节点安装、模型下载及校验、启动、预检查、串行批处理 |
| `SOURCE_LOCK.json` | ComfyUI / BiRefNet / CLIP 的固定提交 |
| `requirements.node-reference.txt` | 本地参考依赖；Linux CUDA PyTorch 另按硬件匹配 |
| `acceptance/` | 不含照片的本地核验记录；不等于所有款式已通过 |

服务器工作流用标准 `LoadImage` / `SaveImage` 替换桌面选图、目录和命名保存节点，不需要桌面端。图像处理参数保留最终 RGB240 版本。没有打包旧渐变底、轮廓阴影或其他问题版本。

## 使用边界

- 底色恒定为 240；商品与局部投影处的像素自然不等于 240。输出默认 2048 × 2048 PNG，保持原画面构图比例，不额外裁切或重新摆放配件。
- 不使用扩散模型重画商品；边缘有透明度估计、去背景色污染和轻微明暗调整，因此不能承诺所有像素与原照逐点相同。
- 自动分割仍可能漏配件或误删薄边。贴在手柄上的透明鱼线结、复杂遮挡、与桌面近色的布袋需要人工放大复核；不能把三张测试结果当成任意商品都能自动通过的保证。
- 投影来自原照可见地面的亮度估计；原照投影弱、地面被遮挡或材质复杂时可能不够自然。节点 `report` 中的 `review_needed` 是复核提示，不是自动质量验收。
- 本次交接不修改现有桌面端安装，不含云服务器账号、密钥或访问令牌。

## 本地轻量检查

在已装好节点依赖的 Python 环境运行：

```bash
python -m unittest discover -s tests -v
python -m compileall -q custom_nodes scripts tests
```

第三方来源见 [依赖说明](docs/SOURCES.md)。云端发布前请固定最终 Linux 环境和镜像，保留这次参考标签用于对照。

本次打包核验：6 项轻量检查通过；独立 ComfyUI 加载了精简后的节点包，三张已有测试图复用之前的 BiRefNet 初始蒙版重跑后续处理，输出与最终 RGB240 PNG 的 RGB 像素差为 0。记录见 [clean_package_replay.json](acceptance/clean_package_replay.json)。这次没有重跑 BiRefNet 全推理，也没有验证新的商品款式或云端环境。
