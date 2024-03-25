import os
import cv2
import numpy as np
import torch
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3_ResNet101_Weights, DeepLabV3
from ultralytics.utils.ops import crop_mask
from ultralytics import YOLO
from super_gradients.training import models as sgmodel
import torch.nn.functional as F

class Based_De_Identificator():
    
    def __init__(self, deID_modelpath:os.PathLike, device:str="cup") -> None:
        self.device = self.get_device(device)
        self.model = self.load_model(deID_modelpath).to(self.device)
        
    def get_device(self, device:str):
        return device

    def load_model(self, modelpath:os.PathLike):
        raise NotImplementedError

    def post_processing(self, r, **kwargs)->torch.Tensor:
        raise NotImplementedError

    def forward(self, batch, **kwargs):
        return self.model(batch, **kwargs)
    
    def __call__(self, batch_img, **kwargs) -> torch.Tensor:
        """
        Please modify the post_processing() function
        to let it output a mask whoes value are 0
        if they belong to the agent that needed to be deidentification,
        otherwise 1 for all image in batch_img

        size : (batchsize * h * w * 3)
        """
        return self.post_processing(self.forward(batch=batch_img, **kwargs), **kwargs)

class License_Plate_Segmentor(Based_De_Identificator):
    
    def __init__(self, deID_modelpath: os.PathLike, thr:float=0.1, device: str = "cup") -> None:
        """
        Using model from https://github.com/dbpprt/pytorch-licenseplate-segmentation 
        """
        
        super().__init__(deID_modelpath, device)
        _ = self.model.eval()
        self.__thr=thr

    def get_device(self, device: str):
        return torch.device(device=device)

    def load_model(self, modelpath:os.PathLike)->DeepLabV3:
        """
        model path downloaded from https://github.com/dbpprt/pytorch-licenseplate-segmentation 
        """
        def model_arch(outputchannels=1)->DeepLabV3:
            model = models.segmentation.deeplabv3_resnet101(
                weights=DeepLabV3_ResNet101_Weights.DEFAULT,
                pretrained=True, progress=True, aux_loss= True
            )
            model.classifier = DeepLabHead(2048, outputchannels)
            return model
        print(modelpath)
        model = model_arch()
        model.load_state_dict(torch.load(modelpath, map_location='cpu')['model'])

        return model
    
    def post_processing(self, r, **kwargs)->torch.Tensor:
        return (r['out'] <= self.__thr).to(dtype=torch.float) 

class YOLO_Human_Detector(Based_De_Identificator):
    
    def __init__(self, deID_modelpath: os.PathLike, device: str = "cup", privacy_issue_class_label:list= [0,1,3]) -> None:
        
        super().__init__(deID_modelpath, device)
        self.torch_device = torch.device(self.device)
        self.privacy_issue_class_label = privacy_issue_class_label
    
    def load_model(self, modelpath: os.PathLike):
        return YOLO(modelpath)

    def privacy_issue_obj(self, class_reuslt:torch.Tensor):
        
        idx = (class_reuslt == self.privacy_issue_class_label [0])
        for i in self.privacy_issue_class_label[1:]:
                idx = idx | (class_reuslt == i)
                
        return torch.squeeze(torch.nonzero(idx), dim = 1)
            
    def _extract_info(self, r)->tuple[torch.Tensor, tuple]:

        privacy_idices = self.privacy_issue_obj(r.boxes.cls.to(dtype=torch.long))
        mask_size = (privacy_idices.size()[0],  r.boxes.orig_shape[0],  r.boxes.orig_shape[1])
        
        return r.boxes.xyxy[privacy_idices], mask_size
            
    def post_processing(self, r:list, **kwargs) -> torch.Tensor:
  
        def _aggreage_mask(mask:torch.Tensor)->torch.Tensor:
            return 1-((mask.sum(dim=0, keepdim=True))>0).to(dtype=torch.float)

        def _make_mask(ri) -> torch.Tensor:
            
            bbox, mask_size = self._extract_info(r=ri)

            return _aggreage_mask(
                crop_mask(
                    masks = torch.ones(
                        size = mask_size, device = self.torch_device
                    ),
                    boxes = bbox
                )
            )
        #cv2.imwrite("vis_mask.jpg",(m[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
        return torch.stack([_make_mask(ri=ri) for ri in r])

class YOLO_NAS_Human_Detector(YOLO_Human_Detector):
    
    def __init__(self, yolonas_type: str, device: str = "cup", privacy_issue_class_label: list = [0, 1, 3]) -> None:
        super().__init__(yolonas_type, device, privacy_issue_class_label)
    
    def load_model(self, yolonas_type: str):
        print(yolonas_type)
        return sgmodel.get(yolonas_type, pretrained_weights="coco")
    
    def forward(self, batch, **kwargs):
        if len(batch) > 1:
            return self.model.predict(batch)._images_prediction_lst
        else:
            return [self.model.predict(batch)]
        
    def _extract_info(self, r)->tuple[torch.Tensor, tuple]:
        
        privacy_idices = self.privacy_issue_obj(
            torch.tensor(r.prediction.labels, dtype=torch.long)
        )
        # _ = input(privacy_idices)

        mask_size = (privacy_idices.size()[0], r.image.shape[0], r.image.shape[1])

        return torch.tensor(
            r.prediction.bboxes_xyxy, dtype=torch.float32, device=self.torch_device
        )[privacy_idices], mask_size
            