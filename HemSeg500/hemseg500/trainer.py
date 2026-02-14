import torch,random,os
import numpy as np
import monai
from torch.optim.lr_scheduler import StepLR
from models import unet,attentionunet,vnet,unetr,segresnet,swinunetr,unetplusplus
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter
import logging
from datasets import data_preprocessing_for_train,post_trans
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from tqdm import tqdm
import json




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

def parse_json(file_path):
    """解析JSON文件并返回每一项的值"""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            model = data.get('model')
            loss = data.get('loss')
            epoch = data.get('epoch')
            datadir = data.get('datadir')
            writer_path = data.get('writer_path')
            model_save_path = data.get('model_save_path')
            batch_size = data.get('batch_size')
            logging_file = data.get('logging_file')
            val_interval = data.get('val_interval')
            return model, loss, epoch, datadir, writer_path, model_save_path, batch_size,logging_file,val_interval
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print("JSON解码出错。")
    return None, None, None, None, None, None, None



def initialize_model_and_loss(model_name, loss_name, device):
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
    elif model_name == 'unetplusplus':
        model = unetplusplus().to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if loss_name == 'diceloss':
        loss_function = monai.losses.DiceLoss(sigmoid=True)
    else:
        raise ValueError(f"Unsupported loss name: {loss_name}")

    return model, loss_function



def setup_logging(log_file='training_log.txt'):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

def train_one_epoch(epoch, model, train_loader, device, loss_function, optimizer, scheduler, writer):
    #print("-" * 10)
    model.train()
    epoch_loss = 0
    epoch_len = len(train_loader)

    with tqdm(total=epoch_len, desc=f"Epoch {epoch+1}/200") as pbar:
        for step, batch_data in enumerate(train_loader, start=1):
            inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            pbar.set_postfix(train_loss=loss.item())
            pbar.update(1)


            # 记录损失
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

    avg_epoch_loss = epoch_loss / epoch_len

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    return avg_epoch_loss,current_lr



def validate_model(epoch, model, val_loader, device, loss_function, post_transforms, dice_metric, writer,  best_metric, best_metric_epoch, metric_values, model_save_path):
    
        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for val_step, val_data in enumerate(val_loader, start=1):
                val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                roi_size = (64, 64, 32)
                sw_batch_size = 16
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_loss = loss_function(val_outputs, val_labels)
                val_outputs = [post_transforms(i) for i in decollate_batch(val_outputs)]
                val_epoch_loss += val_loss.item()
                dice_metric(y_pred=val_outputs, y=val_labels)
        
        val_epoch_loss /= val_step
        writer.add_scalar("val_loss", val_epoch_loss, epoch + 1)
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        metric_values.append(metric)
        
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            logging.info("saved new best metric model")
        
        writer.add_scalar("val_mean_dice", metric, epoch + 1)

        return metric,best_metric, best_metric_epoch,val_epoch_loss

def train_and_evaluate(config_path='config/swinunetr.json'):
    
    monai.config.print_config()

    model_name, loss_name, epoch_num, data_dir, writer_path, model_save_path, batch_size, logging_file,val_interval = parse_json(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    setup_logging(logging_file)
    logging.info(f"Initializing model: {model_name} with loss: {loss_name} on device: {device}")

    model, loss_function = initialize_model_and_loss(model_name, loss_name, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.9)

    logging.info("Loading data...")
    train_loader, val_loader = data_preprocessing_for_train(data_dir, batch_size)
    train_check_data = monai.utils.misc.first(train_loader)
    logging.info(f'train_check_image:{train_check_data["image"].shape},     train_check_label:{train_check_data["label"].shape}')
    val_check_data = monai.utils.misc.first(val_loader)
    logging.info(f'val_check_image, {val_check_data["image"].shape},    val_check_label:{val_check_data["label"].shape}')
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_transforms = post_trans()
    writer = SummaryWriter(writer_path)

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    logging.info("Starting training...")
    for epoch in range(epoch_num):
        avg_epoch_loss,current_lr = train_one_epoch(epoch, model, train_loader, device, loss_function, optimizer, scheduler, writer)
        epoch_loss_values.append(avg_epoch_loss)
        logging.info(f"Training at Epoch [{epoch+1}/{epoch_num}], Average Loss: {avg_epoch_loss:.4f}, Current Learning Rate: {current_lr}")
        if (epoch + 1) % val_interval == 0:
            val_metric,best_metric, best_metric_epoch,val_epoch_loss = validate_model(
            epoch, model, val_loader, device, loss_function, post_transforms, dice_metric, writer,
            best_metric=best_metric, best_metric_epoch=best_metric_epoch,
            metric_values=metric_values, model_save_path=model_save_path
            )
            logging.info(f"Validation at Epoch [{epoch+1}/{epoch_num}], Current Metric: {val_metric:.4f}, Current Average Loss:{val_epoch_loss:.4f}, Best Metric: {best_metric:.4f} at Epoch: {best_metric_epoch}")


    logging.info(f"Training completed. Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

if __name__ == "__main__":
    train_and_evaluate()
