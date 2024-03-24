import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, Any
from .imageutils import read_cv2_RGB

class DeID_Dataset(Dataset):
    
    def __init__(
            self, img_paths:list, 
            img_reader:Callable[[os.PathLike],Any]=read_cv2_RGB, 
            **img_reader_kwargs
        ) -> None:
        
        super().__init__()

        self.nor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        self.fp = img_paths
        self.reader = img_reader
        self.reader_kargs = img_reader_kwargs
  

    def __getitem__(self, index)->tuple[torch.Tensor, torch.Tensor, tuple]:
        
        fpi = self.fp[index]
        origin_image = transforms.ToTensor()(self.reader(fpi, **self.reader_kargs))
        return origin_image*255, self.nor(origin_image), fpi

    def __len__(self):
        return len(self.fp)
    
if __name__ == "__main__":
    pass