"""
Inference images: Extract matting on images.

Example:

python inference_images.py \
  --images-src 20251127_ZCAM/people/photo/2025-11-27-112900/ \
  --images-bgr 20251127_ZCAM/people/photo/2025-11-27-112742/ \
  --output-dir output \
  -y

"""

import argparse
import torch
import os
import shutil

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from threading import Thread
from tqdm import tqdm

from dataset import ImagesDataset, ZipDataset
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from inference_utils import HomographicAlignment


# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Inference images')

parser.add_argument('--model-type', type=str, default='mattingrefine', choices=['mattingbase', 'mattingrefine'])
parser.add_argument('--model-backbone', type=str, default='resnet101', choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, default='checkpoint/pytorch_resnet101.pth')
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.9)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)

parser.add_argument('--images-src', type=str, required=True)
parser.add_argument('--images-bgr', type=str, required=True)

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'])
parser.add_argument('--num-workers', type=int, default=0, 
    help='number of worker threads used in DataLoader. Note that Windows need to use single thread (0).')
parser.add_argument('--preprocess-alignment', action='store_true')

parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--output-types', type=str, nargs='+', default=['com','pha','fgr'], choices=['com', 'pha', 'fgr', 'err', 'ref'])
parser.add_argument('-y', action='store_true')

args = parser.parse_args()


assert 'err' not in args.output_types or args.model_type in ['mattingbase', 'mattingrefine'], \
    'Only mattingbase and mattingrefine support err output'
assert 'ref' not in args.output_types or args.model_type in ['mattingrefine'], \
    'Only mattingrefine support ref output'


# --------------- Main ---------------


# 自动选择设备：优先 CUDA 可用，否则回退 CPU；允许用户通过 --device 覆盖
if args.device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(args.device)

# Load model
if args.model_type == 'mattingbase':
    model = MattingBase(args.model_backbone)
if args.model_type == 'mattingrefine':
    model = MattingRefine(
        args.model_backbone,
        args.model_backbone_scale,
        args.model_refine_mode,
        args.model_refine_sample_pixels,
        args.model_refine_threshold,
        args.model_refine_kernel_size)

model = model.to(device).eval()
if not os.path.exists(args.model_checkpoint):
    raise FileNotFoundError(
        f"Checkpoint not found: {args.model_checkpoint}. You can provide one via --model-checkpoint or place it under 'checkpoint/'."
    )
model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)


# Load images
dataset = ZipDataset([
    ImagesDataset(args.images_src),
    ImagesDataset(args.images_bgr),
], assert_equal_length=True, transforms=A.PairCompose([
    HomographicAlignment() if args.preprocess_alignment else A.PairApply(nn.Identity()),
    A.PairApply(T.ToTensor())
]))
dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)

# 如果数据集为空，给出明确提示并退出，避免出现 0it 进度条引起困惑
if len(dataset) == 0:
    raise ValueError(f"No images found in --images-src '{args.images_src}'. Provide a directory containing .jpg/.png files or a valid image file path.")


# Create output directory
if os.path.exists(args.output_dir):
    if args.y or input(f'Directory {args.output_dir} already exists. Override? [Y/N]: ').lower() == 'y':
        shutil.rmtree(args.output_dir)
    else:
        exit()

for output_type in args.output_types:
    os.makedirs(os.path.join(args.output_dir, output_type))
    

# Worker function
def writer(img, path):
    try:
        # 确保父目录存在（支持层级子目录）
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        img = to_pil_image(img[0].cpu())
        img.save(path)
    except Exception as e:
        print(f"[writer] Failed to save image to '{path}': {e}")
    
    
# Conversion loop
with torch.no_grad():
    threads = []
    for i, (src, bgr) in enumerate(tqdm(dataloader)):
        src = src.to(device, non_blocking=True)
        bgr = bgr.to(device, non_blocking=True)

        if args.model_type == 'mattingbase':
            pha, fgr, err, _ = model(src, bgr)
        elif args.model_type == 'mattingrefine':
            pha, fgr, _, _, err, ref = model(src, bgr)

        pathname = dataset.datasets[0].filenames[i]
        # 当 --images-src 是文件而不是目录时，使用 basename 直接作为文件名
        if os.path.isfile(args.images_src):
            pathname = os.path.basename(pathname)
        else:
            pathname = os.path.relpath(pathname, args.images_src)
        pathname = os.path.splitext(pathname)[0]
        if pathname in ('', '.'):
            # 回退到原始文件名或生成一个唯一名称，避免 PIL 解析空扩展名问题
            pathname = os.path.splitext(os.path.basename(dataset.datasets[0].filenames[i]))[0] or f'image_{i}'

        if 'com' in args.output_types:
            com = torch.cat([fgr * pha.ne(0), pha], dim=1)
            t = Thread(target=writer, args=(com, os.path.join(args.output_dir, 'com', pathname + '.png')))
            threads.append(t); t.start()
        if 'pha' in args.output_types:
            t = Thread(target=writer, args=(pha, os.path.join(args.output_dir, 'pha', pathname + '.jpg')))
            threads.append(t); t.start()
        if 'fgr' in args.output_types:
            t = Thread(target=writer, args=(fgr, os.path.join(args.output_dir, 'fgr', pathname + '.jpg')))
            threads.append(t); t.start()
        if 'err' in args.output_types:
            err = F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False)
            t = Thread(target=writer, args=(err, os.path.join(args.output_dir, 'err', pathname + '.jpg')))
            threads.append(t); t.start()
        if 'ref' in args.output_types:
            ref = F.interpolate(ref, src.shape[2:], mode='nearest')
            t = Thread(target=writer, args=(ref, os.path.join(args.output_dir, 'ref', pathname + '.jpg')))
            threads.append(t); t.start()

    # 等待所有写线程结束，保证程序结束前文件全部落盘
    for t in threads:
        t.join()
