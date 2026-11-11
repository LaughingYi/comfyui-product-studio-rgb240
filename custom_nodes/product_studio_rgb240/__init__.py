import json
from pathlib import Path
import cv2
import numpy as np
import torch
import folder_paths
from comfy_api.latest import ComfyExtension, IO
from safetensors.torch import load_file
from transformers import VitMatteConfig, VitMatteForImageMatting, VitMatteImageProcessor
from .edge_matting import refine_edges
from .group_mask import run_local_repair
from .studio_source import compose_source

GroundObjects = IO.Custom("PRODUCT_GROUND_OBJECTS")

def image_mask(batch, index, shape):
    mask = batch[min(index, batch.shape[0] - 1)].cpu().numpy().astype(np.float32)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask, 0, 1)

class ProductGroupV3(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id='ProductGroupV3', display_name='商品与配件 · RGB240', category='商品精修/RGB240',
            description='沿用本地 MobileSAM 和 YOLO-World，补全配件并排除包装、说明书。自动识别仍需复核。',
            inputs=[IO.Image.Input('images'), IO.Mask.Input('mask'),
                IO.Boolean.Input('remove_boxes', default=True),
                IO.Boolean.Input('remove_lines', default=True),
                IO.Boolean.Input('remove_documents', default=True)],
            outputs=[IO.Mask.Output('group_mask'), IO.Mask.Output('excluded_mask'), IO.String.Output('report'), GroundObjects.Output('ground_objects')])

    @classmethod
    def execute(cls, images, mask, remove_boxes=True, remove_lines=True, remove_documents=True):
        groups, exclusions, reports, objects = [], [], [], []
        for i in range(images.shape[0]):
            rgb = images[i].cpu().numpy()[..., :3]
            raw = image_mask(mask, i, rgb.shape[:2])
            group, excluded, report, parts = run_local_repair(rgb, raw, Path(folder_paths.models_dir) / 'detection',
                remove_boxes, remove_lines, remove_documents)
            groups.append(torch.from_numpy(group))
            exclusions.append(torch.from_numpy(excluded))
            reports.append(report)
            objects.append(parts)
        return IO.NodeOutput(torch.stack(groups), torch.stack(exclusions), '\n'.join(reports), objects)

class ProductEdgeMatteV3(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id='ProductEdgeMatteV3', display_name='原分辨率边缘抠像 · ViTMatte-S', category='商品精修/RGB240',
            description='只估计边缘透明度；保留确定的商品内部，不生成或重画 RGB。仅读取本地模型。',
            inputs=[IO.Image.Input('images'), IO.Mask.Input('mask'), IO.Mask.Input('excluded_mask'),
                IO.Int.Input('edge_band', default=12, min=4, max=32, step=1)],
            outputs=[IO.Mask.Output('fine_mask'), IO.String.Output('report')])

    @classmethod
    def execute(cls, images, mask, excluded_mask, edge_band=12):
        weights = Path(folder_paths.models_dir) / 'matting/vitmatte-small-composition-1k.safetensors'
        if not weights.is_file():
            raise FileNotFoundError(f'本地 ViTMatte-S 模型缺失：{weights}；未自动下载。')
        package = Path(__file__).parent
        config = VitMatteConfig.from_dict(json.loads((package / 'vitmatte_config.json').read_text()))
        model = VitMatteForImageMatting(config)
        model.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
        model.eval()
        processor = VitMatteImageProcessor.from_dict(json.loads((package / 'vitmatte_processor.json').read_text()))
        masks, reports = [], []
        for i in range(images.shape[0]):
            rgb = images[i].cpu().numpy()[..., :3]
            raw = image_mask(mask, i, rgb.shape[:2])
            excluded = image_mask(excluded_mask, i, rgb.shape[:2])
            alpha, report = refine_edges(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                raw, model, processor, excluded, edge_band)
            masks.append(torch.from_numpy(alpha))
            reports.append(report)
        return IO.NodeOutput(torch.stack(masks), '\n'.join(reports))

class ProductStudioSourceShadow(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(node_id='ProductStudioSourceShadow', display_name='纯底 RGB240 · 原照投影', category='商品精修/纯底240',
            description='固定纯色背景 RGB(240,240,240)，只从原照地面估计局部投影亮度；排除商品与道具后平滑，不按商品轮廓描边。',
            inputs=[IO.Image.Input('images'), IO.Mask.Input('mask'), GroundObjects.Input('ground_objects'), IO.Mask.Input('excluded_mask'),
                IO.Int.Input('out_size', default=2048, min=256, max=4096, step=64),
                IO.Float.Input('shadow_strength', default=1, min=0, max=1.5, step=.05),
                IO.Float.Input('light_strength', default=1, min=0, max=1, step=.05)],
            outputs=[IO.Image.Output('image'), IO.String.Output('report')])

    @classmethod
    def execute(cls, images, mask, ground_objects, excluded_mask, out_size=2048, shadow_strength=1, light_strength=1):
        style=json.loads((Path(__file__).parent/'rgb240_style.json').read_text())
        results,reports=[],[]
        for i in range(images.shape[0]):
            rgb=images[i].cpu().numpy()[..., :3]
            alpha=image_mask(mask,i,rgb.shape[:2])
            excluded=image_mask(excluded_mask,i,rgb.shape[:2])
            parts=ground_objects[min(i,len(ground_objects)-1)]
            out,shadow,report=compose_source(rgb,alpha,excluded,parts,style,out_size,shadow_strength,light_strength)
            results.append(torch.from_numpy(out))
            reports.append(json.dumps(report,ensure_ascii=False))
        return IO.NodeOutput(torch.stack(results),'\n'.join(reports))

class ProductStudioRGB240Extension(ComfyExtension):
    async def get_node_list(self):
        return [ProductGroupV3, ProductEdgeMatteV3, ProductStudioSourceShadow]

async def comfy_entrypoint():
    return ProductStudioRGB240Extension()
