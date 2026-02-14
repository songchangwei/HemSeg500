import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import cv2

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
    
    
from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

import json

def read_jsonl(file_path):


    # 打开文件并逐行读取
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉行末的换行符
            line = line.strip()
        
            # 尝试解析 JSON 对象
            try:
                data = json.loads(line)
            
                # 访问数据中的各个字段
                output_directory = data.get("output_directory")
                file_name = data.get("file")
                boxes = data.get("boxes", [])
                points = data.get("points", [])
                
                name_without_extension = file_name.split('.')[0]
                number = int(name_without_extension)

            
                # 在这里处理你的数据，例如：打印出来
                print(f"Output Directory: {output_directory}")
                print(f"File Number: {number}")
                print(f"Boxes: {boxes}")
                print(f"Points: {points}")
                
                video_dir = "/home/user512-001/songcw/data/naochuxue/test/image_struct_jpg/"+output_directory
                ann_frame_idx = number  # the frame index we interact with
                ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
                output_directory = 'output_masks/boxs_struct/'+output_directory
                n = len(boxes)
                boxes = np.array(boxes, dtype=np.float32)
                labels = np.array([1]* n, np.int32)


                
                video_segments = get_video_segments(video_dir,ann_frame_idx,ann_obj_id,boxes[0],labels)
                save_mask(output_directory,video_segments)
                print("----------")
            
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")



def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
def save_mask(output_directory,video_segments):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for frame_idx, obj_masks in video_segments.items():
        for obj_id, mask in obj_masks.items():
            # 检查并调整掩码格式
            mask_image = (mask * 255).astype(np.uint8)

            # 如果掩码有多余维度，去掉它
            if mask_image.ndim > 2:
                mask_image = np.squeeze(mask_image)

            # 检查掩码是否为二维
                assert mask_image.ndim == 2, "Expected a 2D mask image"

            # 创建输出文件名
            output_filename = f"{frame_idx:04}.jpg"
            output_filepath = os.path.join(output_directory, output_filename)

            # 保存为JPG图像
            cv2.imwrite(output_filepath, mask_image)

            print(f"Saved mask for frame {frame_idx}, object {obj_id} to {output_filepath}")
            
            
def get_video_segments(video_dir,ann_frame_idx,ann_obj_id,boxes,labels):
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)


    #ann_frame_idx = 14  # the frame index we interact with
    #ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
    # sending all clicks (and their labels) to `add_new_points_or_box`
    #points = np.array([[324, 294], [320, 300]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=boxes,
    )
    
    

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments


def main():

    read_jsonl('/home/user512-001/songcw/code/naochuxue/nii2jpg/output_data.jsonl')


if __name__=='__main__':
    main()