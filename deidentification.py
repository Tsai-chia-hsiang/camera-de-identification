import os
import os.path as osp
import os
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from deidutils.pathutils import all_files, build_file_tree
from deidutils.dataset import DeID_Dataset
from deidutils.model import YOLO_Human_Detector, License_Plate_Segmentor, YOLO_NAS_Human_Detector
from deidutils.imageutils import mutithread_write_RGB_imgs, mask_batch_tensor_image

def write_batches(deid_img_buf, batchsize):
    for j in range(len(deid_img_buf)):
        mutithread_write_RGB_imgs(
            imgs = deid_img_buf[j][0],
            paths = deid_img_buf[j][1],
            MAX_Thread_num = batchsize
        )

def get_ymal_config(config_file:os.PathLike)->dict:
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

def parse_arguments()-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        default=osp.join(".", "config.yml"), type=str, 
        help="your config (.yml) file path"
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    config = get_ymal_config(args.config)

    print(f"source images root : {config['srcroot']}, save to : {config['dstroot']}")
    print(f"batch size : {config['batchsize']}; bufisze : {config['bufsize']}")

    deid_img_root = build_file_tree(config["srcroot"], config["dstroot"])

    dataset = DeID_Dataset(img_paths=all_files(config["srcroot"], only_want_ftype=".jpg"))
    loader = DataLoader(dataset=dataset, batch_size=config["batchsize"])
    
    license_plate_seg = License_Plate_Segmentor(
        deID_modelpath=config["license_plate_model"]["modelpath"],
        device=config["device"], 
        thr=config["license_plate_model"]["threshold"]
    )
    human_detector = YOLO_NAS_Human_Detector(
        yolonas_type=config["yolo_nas"], 
        device=config["device"]
    )
    deid_img_buf = [None]*config["bufsize"]
    
    clear_chached_buf = False
    pbar = tqdm(loader)
    torch.set_grad_enabled(False)
    i = 0
    for i, (oriimg, norimg, pth) in enumerate(pbar):
        
        clear_chached_buf = False

        l=license_plate_seg(batch_img=norimg.to(torch.device(config["device"])))
        pbar.set_postfix(ordered_dict = {"stage":"license"})
        h = human_detector(batch_img = list(pth), verbose=False)
        pbar.set_postfix(ordered_dict = {"stage":"human"})
            
        deid_img = mask_batch_tensor_image(
            img = oriimg.to(device=torch.device(config["device"])), 
            mask = l*h, return_np_img = True
        )
        
        to_files = list(
            osp.join(deid_img_root, pth_i.removeprefix(config["srcroot"]))
            for pth_i in list(pth)
        )
        
        deid_img_buf[i%config["bufsize"]] = (deid_img, to_files)
        
        if (i+1) % config["bufsize"] == 0:
            pbar.set_postfix(ordered_dict = {"stage":f"write batches"})
            write_batches(
                deid_img_buf = deid_img_buf, 
                batchsize = config["batchsize"]
            )
            clear_chached_buf = True

    if not clear_chached_buf:
        write_batches(
            deid_img_buf = deid_img_buf[0:i%config["bufsize"]+1], 
            batchsize = config["batchsize"]
        )
        
    os.system("rm -rf ~/sg_logs/*") 
