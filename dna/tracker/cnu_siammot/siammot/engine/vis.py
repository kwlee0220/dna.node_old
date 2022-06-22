import cv2
import numpy as np
from matplotlib import cm

from detectron2.structures import Instances, Boxes

class VisGenerator:
    """
    Generate a video for visualization
    """
    def __init__(self, num_colors=100):
        """
        vis_height is the resolution of output frame
        """
        # self._vis_height = vis_height
        # by default, 50 colors
        self.num_colors = num_colors
        self.colors = self.get_n_colors(self.num_colors)
        # use coco class name order
        self.class_names = ['person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle']


    @staticmethod
    def get_n_colors(n, colormap="gist_ncar"):
        # Get n color samples from the colormap, derived from: https://stackoverflow.com/a/25730396/583620
        # gist_ncar is the default colormap as it appears to have the highest number of color transitions.
        # tab20 also seems like it would be a good option but it can only show a max of 20 distinct colors.
        # For more options see:
        # https://matplotlib.org/examples/color/colormaps_reference.html
        # and https://matplotlib.org/users/colormaps.html

        colors = cm.get_cmap(colormap)(np.linspace(0, 1, n))
        # Randomly shuffle the colors
        np.random.shuffle(colors)
        # Opencv expects bgr while cm returns rgb, so we swap to match the colormap (though it also works fine without)
        # Also multiply by 255 since cm returns values in the range [0, 1]
        colors = colors[:, (2, 1, 0)] * 255
        return colors

    def frame_vis_generator(self, frame, results: Instances, file_object=None, frame_id=None):
        # frame, results = self.normalize_output(frame, results)
        frame = np.array(frame)
        ids = results.get('ids')
        results = results[ids >= 0]
        bbox = results.pred_boxes.tensor.detach().cpu().numpy()
        ids = results.get('ids').tolist()
        labels = results.get('pred_classes').tolist()

        for i, entity_id in enumerate(ids):
            color = self.colors[entity_id % self.num_colors]
            class_name = str(self.class_names[labels[i]])
            text_width = len(class_name) * 20
            x1, y1, x2, y2 = (np.round(bbox[i, :])).astype(np.int)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)
            cv2.putText(frame, str(entity_id), (x1 + 5, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)
            # Draw black background rectangle for test
            cv2.rectangle(frame, (x1-5, y1-25), (x1+text_width, y1), color, -1)
            cv2.putText(frame, '{}'.format(class_name), (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
        return frame
