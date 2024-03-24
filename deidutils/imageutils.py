import os
import cv2
import torch
import numpy as np
import threading 
import math

def get_tensor_size(t:torch.Tensor, out_level:str="MB")->float:
    
    tensor_bytes = (t.numel()*t.element_size())
    MB = tensor_bytes/(1024**2)
    if out_level == "GB":
        return MB/1024
    return MB

def read_cv2_RGB(imgpath:os.PathLike)->np.ndarray:
    return cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)

def mutithread_write_RGB_imgs(imgs:list[np.ndarray], paths:list[os.PathLike], MAX_Thread_num:int=10):
    
    write_times = math.ceil(len(paths)/MAX_Thread_num)
    for i in range(write_times):
        T = list(
            threading.Thread(
                target=write_RGB_img, args=(imgs[j], paths[j])
            ) for j in range(
                MAX_Thread_num*i, 
                min(MAX_Thread_num*(i+1), len(paths))
            )
        )
    
        for tid in range(len(T)):
            T[tid].start()
        
        for tid in range(len(T)):
            T[tid].join()

def write_RGB_img(img:np.ndarray, imgpath:os.PathLike)->bool:
    return cv2.imwrite(
        imgpath, 
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
    )

def mask_batch_tensor_image(img:torch.Tensor, mask:torch.Tensor, return_np_img:bool=False)->torch.Tensor|np.ndarray:
    """
    img : A torch Tensor with shape (batchsize, C, H, W)
    """
    I = (img*mask.expand(mask.size()[0], 3, -1, -1))
    if return_np_img:
        return I.permute(0,2,3,1).cpu().numpy()
    return I