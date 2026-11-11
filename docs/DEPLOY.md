# ECS 从零部署与验收

## 1. 确认服务器信息

开发先记录 Linux 发行版、CPU / 内存、GPU 型号、显存、`nvidia-smi` 输出和可用磁盘。用户尚未确认 GPU 型号，因此本交接不预设 CUDA 镜像，也不承诺吞吐、显存需求或 GPU 加速倍数。

建议交付顺序：固定代码和权重 → CPU 参考复现 → GPU 移植 → 样图验收 → 固定 Linux 依赖和容器镜像。不能只上传工作流 JSON。

## 2. 拉取固定代码

以下命令面向新目录；已有安装不要直接覆盖。私有仓库需要 GitHub 账号访问权限。系统先装好 Git、curl、适用版本的 Python 和 venv，以及 OpenCV 所需系统库。

```bash
git clone --branch v0.1.0-rgb240 https://github.com/LaughingYi/comfyui-product-studio-rgb240.git
cd comfyui-product-studio-rgb240
export RGB240_REPO="$PWD"
export COMFY_ROOT="$(dirname "$RGB240_REPO")/ComfyUI-rgb240"

git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFY_ROOT"
git -C "$COMFY_ROOT" checkout c76e64fe5f9108af6946ac3a39ff4280142d6dcd
git clone https://github.com/lldacing/ComfyUI_BiRefNet_ll.git \
  "$COMFY_ROOT/custom_nodes/ComfyUI_BiRefNet_ll"
git -C "$COMFY_ROOT/custom_nodes/ComfyUI_BiRefNet_ll" \
  checkout 5443a2aa16cfbd98bb2f7dcc8bdcb70439e08529
```

## 3. Python 环境

本地参考使用 Python 3.13.12，完整版本记录在 `environment_reference_macos.json`。这是来源记录，不是 Linux 环境验证报告。不要上传或复用 macOS 的 `.venv`。

```bash
python3 -m venv "$COMFY_ROOT/.venv"
source "$COMFY_ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
```

先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 和实际驱动安装兼容的 Linux `torch` / `torchvision`。CPU 复现也需要这两个包。本地版本为 torch 2.12.1 / torchvision 0.27.1；如果没有对应 Linux/CUDA 发行包，不要复制 Mac wheel，应记录实际匹配版本并重新验收。

安装 ComfyUI 与节点依赖，再检查冲突：

```bash
python -m pip install -r "$COMFY_ROOT/requirements.txt"
python -m pip install -r "$RGB240_REPO/requirements.node-reference.txt"
python -m pip check
```

节点依赖文件锁定的是参考版本，并非在新 Linux 上已经求解好的完整 lockfile。遇到版本不可用或冲突时请停止、记录问题，定向兼容后重新验收，不要无差别升级全部依赖。`ComfyUI_BiRefNet_ll` 需要的 numpy、OpenCV、timm 已列入节点依赖；不需要再安装不相关节点集合或扩散模型。

## 4. 权重安装与节点注册

```bash
python "$RGB240_REPO/scripts/models.py" --model-root "$COMFY_ROOT/models" --download
python "$RGB240_REPO/scripts/install_nodes.py" --comfy-root "$COMFY_ROOT"
python "$RGB240_REPO/scripts/preflight.py" --comfy-root "$COMFY_ROOT"
```

下载脚本只取清单中的五个文件并核验字节数、SHA256；已有但不匹配的文件会报错，不自动覆盖。网络不畅可由开发从可信下载环境传到相同路径，重新执行不带 `--download` 的校验命令。不要替换成同名的其他模型。Linux 文件名大小写敏感，必须是 `models/BiRefNet/General-HR.safetensors`。

特别注意 CLIP 是 YOLO-World 的隐含依赖，目标为 `models/detection/clip/ViT-B-32.pt`。安装脚本通过独立的 `YOLO_CONFIG_DIR=$COMFY_ROOT/.rgb240-ultralytics` 配置 `weights_dir`。后续启动必须使用相同配置，否则可能找错模型路径或试图临时联网下载。

## 5. 启动与标准 API

```bash
bash "$RGB240_REPO/scripts/start_reference.sh"
```

此脚本先校验模型，按 CPU 参考模式运行，只加载本项目与 BiRefNet 两个自定义节点包，监听 `127.0.0.1:8188`。用 SSH 隧道或已有后端访问，不要求把 ComfyUI 直接开放到公网。需要其他端口时设置 `COMFY_PORT`。

界面导入 `workflows/rgb240_ui.json`，选择输入图后运行；安装脚本也会写入 `user/default/workflows/RGB240.json`。服务器 API 模板里的 `__UPLOAD_IMAGE__` 必须替换为上传接口返回的图片名。

串行处理（适合先做少量样图复核）：

```bash
python "$RGB240_REPO/scripts/run_batch.py" \
  --url http://127.0.0.1:8188 \
  --input-dir /path/to/approved-inputs \
  --output-dir /path/to/results \
  --timeout 3600
```

客户端按 `/upload/image` → `/prompt` → `/history/{prompt_id}` → `/view` 上传、排队、取图。输出为 PNG；同名已有结果跳过。超时不会取消服务器任务，异常中的 prompt_id 可用于查询，避免重复提交。输入会留在 ComfyUI 输入目录，服务端输出也会留存；生产系统由开发按业务保留策略管理。

## 6. GPU 移植工作量

此标签保留最终图像算法，没有夹带第二套未验证的 GPU 版本。仅运行 CUDA 版 PyTorch不足以切换以下写死的设备：

| 位置 | 当前参考行为 | GPU 移植点 |
| --- | --- | --- |
| 工作流节点 2 | BiRefNet `device=CPU`, `dtype=float32` | NVIDIA 环境可先试 `AUTO` + float32，核对实际设备和结果 |
| `group_mask.py` | SAM / YOLO 推理 `device='cpu'` | 显式统一 CUDA 设备，确认 CLIP 也在兼容设备 |
| `__init__.py` 与 `edge_matting.py` | ViTMatte 模型和处理器 tensor 在 CPU | 模型和所有输入 tensor 转同一 CUDA 设备，输出回 CPU 做原有数组合成 |
| 原照投影 / 边缘合成 | OpenCV / NumPy CPU | 可先保留；不要改变算法去追求全 GPU |

先逐步骤以 float32 对比，不要同时更换精度、分辨率、模型和分割参数。通过后再评估半精度、模型复用和并发。记录首张 / 稳态耗时、峰值显存、吞吐和失败率；不要据 CPU 跑通就宣称 GPU 性能已验收。

## 7. 视觉与部署验收

使用用户批准的原始照片另行传输，仓库不存商品照片。至少包含：手提包+织带+布袋+锁、链条包+薄配件+布袋、独立肩带的包，以及后续新增复杂款式。

逐项检查：

- 与本地最终 RGB240 结果对照，包型、Logo、皮纹、五金、配件数量和位置不变，没有大块漏抠或透明洞。
- 100% / 200% 查看手柄、链条孔洞、布袋薄边、扣件，不能有明显锯齿、白边、模糊描边或悬浮感。
- 远离商品与局部投影的背景为 RGB(240,240,240)，无全局渐变；阴影方向和强弱合理，不能画整圈边框代替投影。
- 盒子、相框、说明书去除；外部鱼线检查到手柄连接处。无法自动清除的细线结需单独处理，不可冒险抹掉手柄细节。
- 记录自动处理失败或需人工修复的图片，不把 `review_needed=false` 当作视觉通过。
- 同时记录代码提交、模型 SHA、Linux 全依赖、GPU / 驱动 / CUDA、输入输出尺寸和耗时。GPU 结果允许微小数值差异，但不允许可见结构或边缘损失。

通过上述样图后，再制作生产 Docker 镜像：固定 NVIDIA CUDA/PyTorch 基础镜像及 digest，固定本仓库标签与上游提交，挂载外部模型和输入输出目录，在同一 GPU 上做容器内复测。GPU 型号未知阶段不提供一个声称已验证的通用 Docker 镜像。

第三方运行、权重和商业使用条件由开发部署时核对上游许可证；来源列在 [SOURCES.md](SOURCES.md)。
