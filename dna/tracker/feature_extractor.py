from typing import List, Optional

from pathlib import Path
import logging

import numpy as np
from scipy.stats import multivariate_normal
import torch
import torchvision

from dna import Image
from dna.detect import Detection

class FeatureExtractor:
    def __init__(self, wt_path:Path, logger:logging.Logger=None) -> None:
        #loading this encoder is slow, should be done only once.	
        self.encoder = torch.load(wt_path)			
            
        self.encoder = self.encoder.cuda()
        self.encoder = self.encoder.eval()
        logger.info(f"DeepSORT model from {wt_path}")

        self.gaussian_mask = get_gaussian_mask().cuda()
        self.transforms = torchvision.transforms.Compose([ \
                            torchvision.transforms.ToPILImage(),\
                            torchvision.transforms.Resize((128,128)),\
                            torchvision.transforms.ToTensor()])

    def extract(self, image:Image, detections: List[Detection]):
        if detections:
            tlwh_list = [det.bbox.to_tlwh() for det in detections]
            processed_crops = self.pre_process(image, tlwh_list).cuda()
            processed_crops = self.gaussian_mask * processed_crops

            features = self.encoder.forward_once(processed_crops)
            features = features.detach().cpu().numpy()
            if len(features.shape)==1:
                features = np.expand_dims(features,0)

            return features
        else:
            return []
		
    def pre_process(self, frame, tlwhs):
        crops = []
        for d in tlwhs:
            for i in range(len(d)):
                if d[i] <0:
                    d[i] = 0	

            img_h,img_w,img_ch = frame.shape

            xmin,ymin,w,h = d

            if xmin > img_w:
                xmin = img_w

            if ymin > img_h:
                ymin = img_h

            xmax = xmin + w
            ymax = ymin + h

            ymin = abs(int(ymin))
            ymax = abs(int(ymax))
            xmin = abs(int(xmin))
            xmax = abs(int(xmax))

            try:
                crop = frame[ymin:ymax,xmin:xmax,:]
                crop = self.transforms(crop)
                crops.append(crop)
            except:
                continue

        return torch.stack(crops)

def get_gaussian_mask():
	#128 is image size
	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.22,0.22])
	covariance = np.diag(sigma**2) 
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
	z = z.reshape(x.shape) 

	z = z / z.max()
	z  = z.astype(np.float32)

	mask = torch.from_numpy(z)

	return mask