import os
import os.path as osp
import os
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

if __name__ == "__main__":

    device="cuda:1"

    batchsize = 18
    bufsize = 10

    src_img_root = osp.join("0_1015/")
    deid_img_root = build_file_tree(src_img_root, osp.join("de_identification"))

    dataset = DeID_Dataset(img_paths=all_files(src_img_root, only_want_ftype=".jpg"))
    loader = DataLoader(dataset=dataset, batch_size=batchsize)
    
    license_plate_seg = License_Plate_Segmentor(
        deID_modelpath=osp.join("model","license_plate_seg.pth"),
        device=device, thr=5e-2
    )
    human_detector = YOLO_NAS_Human_Detector(yolonas_type="yolo_nas_l", device=device)
    deid_img_buf = [None]*bufsize
    
    clear_chached_buf = False
    pbar = tqdm(loader)
    torch.set_grad_enabled(False)
    i = 0
    for i, (oriimg, norimg, pth) in enumerate(pbar):
        
        clear_chached_buf = False

        l=license_plate_seg(batch_img=norimg.to(torch.device(device)))
        pbar.set_postfix(ordered_dict = {"stage":"license"})
        h = human_detector(batch_img = list(pth), verbose=False)
        pbar.set_postfix(ordered_dict = {"stage":"human"})
            
        deid_img = mask_batch_tensor_image(
            img = oriimg.to(device=torch.device(device)), 
            mask = l*h, return_np_img = True
        )
        
        to_files = list(
            osp.join(deid_img_root, pth_i.removeprefix(src_img_root))
            for pth_i in list(pth)
        )
        
        deid_img_buf[i%bufsize] = (deid_img, to_files)
        
        if (i+1) % bufsize == 0:
            pbar.set_postfix(ordered_dict = {"stage":f"write batch"})
            write_batches(deid_img_buf = deid_img_buf, batchsize = batchsize)
            clear_chached_buf = True

    if not clear_chached_buf:
        write_batches(
            deid_img_buf = deid_img_buf[0:i%bufsize+1], 
            batchsize = batchsize
        )
        
    os.system("rm -rf ~/sg_logs/*")   
