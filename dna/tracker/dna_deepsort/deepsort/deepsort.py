from typing import List
from re import L
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision

import dna
from dna import Box, color, Frame, Image, plot_utils
from dna.detect import Detection
from dna.utils import draw_ds_detections, draw_ds_tracks
from dna.tracker import DNASORTParams

import nn_matching
from . import utils
from .tracker import Tracker
from application_util import preprocessing as prep
from application_util import visualization

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
	def __init__(self, domain: Box, wt_path:Path, detection_threshold:float, params:DNASORTParams):
		self.domain = domain

		#loading this encoder is slow, should be done only once.	
		self.encoder = torch.load(wt_path)			
			
		self.encoder = self.encoder.cuda()
		self.encoder = self.encoder.eval()
		LOGGER.info(f"DeepSORT model from {wt_path}")

		self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", params.metric_threshold_loose , 100)
		self.tracker = Tracker(domain, self.metric, params=params)

		self.gaussian_mask = get_gaussian_mask().cuda()
		self.transforms = torchvision.transforms.Compose([ \
							torchvision.transforms.ToPILImage(),\
							torchvision.transforms.Resize((128,128)),\
							torchvision.transforms.ToTensor()])

	def run_deep_sort(self, frame:Frame, detections: List[Detection]):
		self.tracker.predict()

		if detections:
			self.attach_features(frame.image, detections)

			outboxes = np.array([det.bbox.to_tlwh() for det in detections])
			outscores = np.array([det.score for det in detections])
			indices = prep.non_max_suppression(outboxes, 0.8, outscores)
			detections = list(utils.get_items(detections, indices))
			if dna.DEBUG_SHOW_IMAGE:
				self.draw_detections('detections', frame.image.copy(), detections)
		else:
			detections = []

		# ##################################################################################
		# # kwlee
		# if dna.DEBUG_SHOW_IMAGE:
		# 	convas = draw_ds_tracks(frame.image.copy(), self.tracker.tracks, color.RED, color.BLACK, 1,
		# 							dna.DEBUG_TARGET_TRACKS)
		# 	cv2.imshow("predictions", convas)
		# 	cv2.waitKey(1)
		# ##################################################################################

		deleteds = self.tracker.update(detections, frame)

		return self.tracker, deleteds

	def draw_detections(self, title:str, convas:Image, detections:List[Detection], line_thickness=1):
		for idx, det in enumerate(detections):
			if det.score < self.tracker.params.detection_threshold:
				convas = plot_utils.draw_label(convas, str(idx), det.bbox.br.astype(int), color.WHITE, color.RED, 1)
				convas = det.bbox.draw(convas, color.RED, line_thickness=line_thickness) 
		for idx, det in enumerate(detections):
			if det.score >= self.tracker.params.detection_threshold:
				convas = plot_utils.draw_label(convas, str(idx), det.bbox.br.astype(int), color.WHITE, color.BLUE, 1)
				convas = det.bbox.draw(convas, color.BLUE, line_thickness=line_thickness)
		cv2.imshow(title, convas)
		cv2.waitKey(1)

	def attach_features(self, image:Image, detections: List[Detection]):
		tlwh_list = [det.bbox.to_tlwh() for det in detections]
		features = self.extract_features(image, tlwh_list)
		for det, feature in zip(detections, features):
			det.feature = feature

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