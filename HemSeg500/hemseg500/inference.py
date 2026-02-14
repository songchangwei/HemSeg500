import torch,random,os
import numpy as np
import monai
from models import unet,attentionunet,vnet,segresnet,unetr,swinunetr
from monai.inferers import sliding_window_inference
from datasets import data_preprocessing_for_inference,post_trans
from monai.data import decollate_batch
import json
from monai.transforms import SaveImage
from monai.metrics import DiceMetric,MeanIoU,HausdorffDistanceMetric




def make_deterministic(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
make_deterministic(42)

def save_with_original_name(save_dir, file_name,ext):
    # Creating the full path for saving the file
    return os.path.join(save_dir, f"{file_name}{ext}")

def parse_json(file_path):
    """解析JSON文件并返回每一项的值"""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            model = data.get('model')
            data_dir = data.get('datadir')
            result_dir = data.get('result_dir')
            model_save_path = data.get('model_save_path')
            return model, data_dir, result_dir, model_save_path
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print("JSON解码出错。")
    return None, None, None, None

def initialize_model_and_loss(model_name, device):
    if model_name == 'unet3d':
        model = unet().to(device)
    elif model_name == 'attentionunet':
        model = attentionunet().to(device)
    elif model_name == 'vnet':
        model = vnet().to(device)
    elif model_name == 'segresnet':
        model = segresnet().to(device)
    elif model_name == 'unetr':
        model = unetr().to(device)
    elif model_name == 'swinunetr':
        model = swinunetr().to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model



def inference_model(model, test_loader, device, post_transforms, result_dir,model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        saver = SaveImage(output_dir=result_dir, output_postfix="seg", output_ext=".nii.gz", separate_folder=False)
        
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        iou_metric =  MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
        hd_metric = HausdorffDistanceMetric(include_background=True,percentile=95, reduction="mean", get_not_nans=False)

        with torch.no_grad():
            for _, test_data in enumerate(test_loader, start=1):
                test_images, test_labels = test_data['image'].to(device), test_data['label'].to(device)
                file_name = test_data['file_name'][0]
                roi_size = (64, 64, 32)
                sw_batch_size = 16
                test_outputs = sliding_window_inference(test_images, roi_size, sw_batch_size, model)
                test_outputs = [post_transforms(i) for i in decollate_batch(test_outputs)]
                
                dice_metric(y_pred=test_outputs, y=test_labels)
                iou_metric(y_pred=test_outputs, y=test_labels)
                hd_metric(y_pred=test_outputs, y=test_labels)
                
                for test_output in test_outputs:
                    saver(test_output,filename=os.path.join(result_dir, file_name))
        print("evaluation dice metric:", dice_metric.aggregate().item())
        print("evaluation iou metric:", iou_metric.aggregate().item())
        print("evaluation hd metric:", hd_metric.aggregate().item())
        # reset the status            
        dice_metric.reset()
        iou_metric.reset()
        hd_metric.reset()



def inference(config_path='config/swinunetr.json'):
    model_name, data_dir, result_dir, model_save_path = parse_json(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_model_and_loss(model_name, device)
    post_transforms = post_trans()
    test_loader = data_preprocessing_for_inference(data_dir)
    train_check_data = monai.utils.misc.first(test_loader)
    print(f'test_check_image:{train_check_data["image"].shape}')
    
    inference_model(model,test_loader,device,post_transforms,result_dir,model_save_path)


    

if __name__ == "__main__":
    inference()
