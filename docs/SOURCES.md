# 依赖与来源

本仓库不打包第三方权重、第三方节点源码或完整 ComfyUI。固定提交见 `SOURCE_LOCK.json`，固定权重来源与 SHA256 见 `models/manifest.json`。

| 依赖 | 上游 |
| --- | --- |
| ComfyUI | https://github.com/Comfy-Org/ComfyUI |
| BiRefNet 节点 | https://github.com/lldacing/ComfyUI_BiRefNet_ll |
| BiRefNet HR 权重 | https://huggingface.co/ZhengPeng7/BiRefNet_HR |
| MobileSAM / YOLO-World 发行权重 | https://github.com/ultralytics/assets/releases/tag/v8.3.0 |
| Ultralytics | https://github.com/ultralytics/ultralytics |
| CLIP 代码与 ViT-B/32 权重入口 | https://github.com/openai/CLIP |
| ViTMatte-S 权重与配置 | https://huggingface.co/hustvl/vitmatte-small-composition-1k |

节点目录中的 `vitmatte_config.json` 和 `vitmatte_processor.json` 对应上表 ViTMatte-S 模型配置。安装和商业部署时保留并遵循各上游适用许可证；本仓库的私有访问权限不替代第三方授权。

部署参考文档：

- [ComfyUI 服务端路由](https://docs.comfy.org/development/comfyui-server/comms_routes)
- [ComfyUI 模型目录](https://docs.comfy.org/basic-concepts/models)
- [NVIDIA Container Toolkit 安装](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [GitHub 大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)：本项目通过清单单独获取权重，不向普通 Git 提交大模型。
