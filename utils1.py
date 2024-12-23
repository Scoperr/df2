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

def process_video(video_path):
    max_frame = 400
    clip_size = cfg.clip_size
    # Load or detect faces from video frames and cache the results
    cache_file = f"{video_path}_{max_frame}.pth"

    if os.path.exists(cache_file):
        detect_res, all_lm68 = torch.load(cache_file)
        frames = grab_all_frames(video_path, max_size=max_frame, cvt=True)
        print("Detection result loaded from cache.")
    else:
        print("Detecting faces in video frames...")
        detect_res, all_lm68, frames = detect_all(video_path, return_frames=True, max_size=max_frame)
        torch.save((detect_res, all_lm68), cache_file)
        print("Detection finished.")

    print(f"Number of frames: {len(frames)}")
    shape = frames[0].shape[:2]

    # Process face detection results
    all_detect_res = []
    assert len(all_lm68) == len(detect_res)
    
    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_faces.append((box, lm5, face_lm68, score))
        all_detect_res.append(new_faces)

    detect_res = all_detect_res

    print("Splitting into super clips...")
    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)

    print(f"Number of full tracks: {len(tracks)}")

    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)

    data_storage = {}
    frame_boxes = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        print(f"Processing track {track_i} from frame {start} to {end}")
        assert len(detect_res[start:end]) == len(track)
        super_clips.append(len(track))

        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)

            top_left = big_box[:2][None, :]
            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]
            base_key = f"{track_i}_{j}_"
            data_storage[f"{base_key}img"] = cropped
            data_storage[f"{base_key}ldm"] = info
            data_storage[f"{base_key}idx"] = frame_idx
            frame_boxes[frame_idx] = np.rint(box).astype(int)

    print(f"Sampling clips from super clips: {super_clips}")
    clips_for_video = []
    pad_length = clip_size - 1

    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))

        if super_clip_size < clip_size:  # padding
            post_module = inner_index[1:-1][::-1] + inner_index
            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]
            pre_module = inner_index + inner_index[1:-1][::-1]
            l_pre = len(pre_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]
            inner_index = pre_module + inner_index + post_module

        super_clip_size = len(inner_index)

        frame_range = [
            inner_index[i:i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size
        ]
        for indices in frame_range:
            clip = [(super_clip_idx, t) for t in indices]
            clips_for_video.append(clip)

    return clips_for_video, data_storage, frame_boxes, frames
