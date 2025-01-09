import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader
from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader

# Define the Deepfake Detection Model
class DeepfakeDetectionModel():
    def __init__(self,input_path,output_path):
        self.max_frame = 400
        self.video_path = input_path
        self.out_dir = output_path
        self.cfg_path = "i3d_ori.yaml"
        self.optimal_threshold = 0.04
        pass

    def load_weights(self, ckpt_path):
        self.ckpt_path = ckpt_path
        cfg.init_with_yaml()
        cfg.update_with_yaml(self.cfg_path)

        cfg.freeze()

        self.classifier = PluginLoader.get_classifier(cfg.classifier_type)()
        self.classifier.cuda()
        self.classifier.eval()
        self.classifier.load(self.ckpt_path)
        print(f"Successfully loaded weights from {self.ckpt_path}.")

        self.crop_align_func = FasterCropAlignXRay(cfg.imsize)

        os.makedirs(self.out_dir, exist_ok=True)
        basename = f"{os.path.splitext(os.path.basename(self.video_path))[0]}.avi"
        self.out_file = os.path.join(self.out_dir, basename)

    def predict(self, clips_for_video, data_storage, frame_boxes, frames):
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1, 1)
        # Step 1: Process the video to extract clips and frames
        clip_size = cfg.clip_size
        preds = []
        frame_res = {}
        crop_align_func = FasterCropAlignXRay(cfg.imsize)
        # Step 2: Process clips for prediction
        for clip in tqdm(clips_for_video, desc="Testing"):
            images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
            landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
            frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]
            _, images_align = crop_align_func(landmarks, images)

            for i in range(clip_size):
                img1 = cv2.resize(images[i], (cfg.imsize, cfg.imsize))
                img = np.concatenate((img1, images_align[i]), axis=1)

            images = torch.as_tensor(images_align, dtype=torch.float32).cuda().permute(3, 0, 1, 2)
            images = images.unsqueeze(0).sub(mean).div(std)

            # Step 3: Model inference (prediction)
            with torch.no_grad():
                output = self.classifier(images)
            pred = float(F.sigmoid(output["final_output"]))

            for f_id in frame_ids:
                if f_id not in frame_res:
                    frame_res[f_id] = []
                frame_res[f_id].append(pred)
            preds.append(pred)

        print(f"Average prediction: {np.mean(preds)}")

        boxes = []
        scores = []

        # Step 4: Generate the final results
        for frame_idx in range(len(frames)):
            if frame_idx in frame_res:
                pred_prob = np.mean(frame_res[frame_idx])
                rect = frame_boxes[frame_idx]
            else:
                pred_prob = None
                rect = None
            scores.append(pred_prob)
            boxes.append(rect)

        # Assuming you have a SupplyWriter for saving results
        SupplyWriter(self.video_path, self.out_file, self.optimal_threshold).run(frames, scores, boxes)

        # Return probabilities for real and fake
        real_prob = round(np.mean(preds), 3)
        fake_prob = round(1 - real_prob, 3)

        return real_prob, fake_prob
