from typing import List
from re import L
from dna import detect
from cv2 import determinant
import dna
from dna import Size2d
from dna.utils import draw_ds_detections, draw_ds_tracks
import nn_matching
from .tracker import Tracker 
from application_util import preprocessing as prep
from application_util import visualization
from detection import Detection

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
from scipy.stats import multivariate_normal

from dna import Box, color
import logging
LOGGER = logging.getLogger('dna.tracker.deepsort')


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

class deepsort_rbc():
	def __init__(self, domain: Box, wt_path, params):
		self.domain = domain

		#loading this encoder is slow, should be done only once.	
		self.encoder = torch.load(wt_path)			
			
		self.encoder = self.encoder.cuda()
		self.encoder = self.encoder.eval()
		LOGGER.info(f"Deep sort model loaded from path: {wt_path}")

		self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", params.metric_threshold , 100)
		self.tracker = Tracker(domain, self.metric, params=params)

		self.gaussian_mask = get_gaussian_mask().cuda()
		self.transforms = torchvision.transforms.Compose([ \
							torchvision.transforms.ToPILImage(),\
							torchvision.transforms.Resize((128,128)),\
							torchvision.transforms.ToTensor()])

	def run_deep_sort(self, frame, bboxes: List[Box], scores: List[float]):
		if len(bboxes) > 0:
			tlwh_list = [b.to_tlwh() for b in bboxes]
			features = self.extract_features(frame, tlwh_list)
			dets = [Detection(bbox, score, feature)	for bbox, score, feature in zip(bboxes, scores, features)]
			outboxes = np.array(tlwh_list)
			outscores = np.array([d.confidence for d in dets])
			indices = prep.non_max_suppression(outboxes, 0.8, outscores)
			dets = [dets[i] for i in indices]
		else:
			dets = []

		##################################################################################
		# kwlee
		if dna.DEBUG_SHOW_IMAGE:
			convas = draw_ds_detections(frame.copy(), dets, color.GREEN, color.BLACK, line_thickness=1)
			cv2.imshow("dets", convas)
			cv2.waitKey(1)
		##################################################################################

		self.tracker.predict()

		##################################################################################
		# kwlee
		if dna.DEBUG_SHOW_IMAGE:
			convas = draw_ds_tracks(frame.copy(), self.tracker.tracks, color.RED, color.BLACK, 1,
									dna.DEBUG_TARGET_TRACKS)
			cv2.imshow("predictions", convas)
			cv2.waitKey(1)
		##################################################################################

		deleteds = self.tracker.update(dets)

		return self.tracker, deleteds

	def extract_features(self, frame, tlwhs):
		processed_crops = self.pre_process(frame, tlwhs).cuda()
		processed_crops = self.gaussian_mask * processed_crops

		features = self.encoder.forward_once(processed_crops)
		features = features.detach().cpu().numpy()
		if len(features.shape)==1:
			features = np.expand_dims(features,0)

		return features
		
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