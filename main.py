import sys
import os
import argparse
import json
from datetime import datetime
import time
import cv2
from config import config as cfg
from model1 import DeepfakeDetectionModel # Updated model import
from utils1 import process_video

# 2: Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Demo Inference for deepfake_detection model')
    # Input and Output Paths
    parser.add_argument('--input_path', type=str, required=True,
    help='Path to the input video/audio/image file or folder')
    parser.add_argument('--output_path', type=str, required=True,
    help='Directory where results will be saved')
    parser.add_argument('--info_path', type=str,
    default='method_info.json', help='Path to method info file')
    parser.add_argument('--device', type=str, default='cuda', help='Set to "cuda" for GPU or "cpu"')
    parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from',
    default="checkpoints/deepfake_model.hdf5")
    # argument to handle folder inputs
    parser.add_argument('--d', action="store_true", help='Use this argument when input path is a folder')
    
    # parser = argparse.ArgumentParser(description='Demo Inference for deepfake_detection model')

    # # Input and Output Paths
    # parser.add_argument('--input_path', type=str, default='data/shining.mp4', 
    #                     help='Path to the input video/audio/image file or folder (default: data/shining.mp4)')
    # parser.add_argument('--output_path', type=str, default='result', 
    #                     help='Directory where results will be saved (default: result)')
    # parser.add_argument('--info_path', type=str, default='method_info.json', help='Path to method info file')
    # parser.add_argument('--device', type=str, default='cuda', help='Set to "cuda" for GPU or "cpu"')
    # parser.add_argument('--checkpoint_path', type=str, default="checkpoints/deepfake_model.hdf5", 
    #                     help='Name of saved checkpoint to load weights from')

    # # Argument to handle folder inputs
    # parser.add_argument('--d', action="store_true", help='Use this argument when input path is a folder')

    return parser.parse_args()

def get_result_description(real_prob):
    if real_prob >= 0.99:
        return 'This sample is certainly real.'
    elif real_prob >= 0.75:
        return 'This sample is likely real.'
    elif real_prob >= 0.25:
        return 'This sample is maybe real.'
    elif real_prob >= 0.01:
        return 'This sample is unlikely real.'
    else:
        return 'There is no chance that the sample is real.'

def run_inference(input_path, output_path, info, checkpoint_path, device):
    start_time = time.time()
    # Simulate loading the model
    model = DeepfakeDetectionModel(input_path,output_path)  # Updated model name
    model.load_weights(checkpoint_path)
    
    # Simulate frame extraction (For real use, replace this with actual frame extraction code)
    print(f"Processing {input_path}")
    # Step 1: Process the video and retrieve necessary data
    clips_for_video, data_storage, frame_boxes, frames = process_video(input_path)
    real_prob, fake_prob = model.predict(clips_for_video, data_storage, frame_boxes, frames)
    # Check if advanced result exists (this could be based on the model output)
    advanced_result = None  # Or add the actual code to generate an advanced result
    
    # JSON result structuring
    json_result = {
        "Task": info['task'],
        "Input File": os.path.basename(input_path),
        "Analytic Name": info['analytic_name'],
        "Analysis Date": str(datetime.now()),
        "Original Result": {"Real Probability": real_prob, "Fake Probability": fake_prob},
        "Original Result Description": info['result_description'],
        "Result": {"Real Probability": real_prob, "Fake Probability": fake_prob},
        "Result Description": get_result_description(real_prob),
        "Analytic Description": info['analytic_description'],
        "Analysis Scope": info['analysis_scope'],
        "Reference": info['paper_reference'],
        "Code Link": info['code_reference'],
        "Error": "None",  # Error handling
        "Analysis Time in Second": round(time.time() - start_time, 2),
        "Device": device,
        "Advanced Results": "Available" if advanced_result else "Not Available"
    }

    # Save the result in the output path
    result_path = os.path.join(output_path, "result.json")
    with open(result_path, 'w') as json_file:
        json.dump(json_result, json_file, indent=4)
        print(f"Results saved to {result_path}")




# 5: Main Execution
def main():
    args = parse_args()
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    # Load method_info.json
    with open(args.info_path, 'r') as info_file:
        info = json.load(info_file)['altfreezing'] # Updated info
    # check if input is a folder and handle it
    if args.d:
        # If input is a folder, iterate over all files
        input_files = os.listdir(args.input_path)
        for file in input_files:
            file_path = os.path.join(args.input_path, file)
            if file.endswith(".mp4"): # Assuming .mp4 videos
                run_inference(file_path, args.output_path, info,
                args.checkpoint_path, args.device)
    else:
        # If input is a single file
        run_inference(args.input_path, args.output_path, info,
        args.checkpoint_path, args.device)
if __name__ == "__main__":
    main()