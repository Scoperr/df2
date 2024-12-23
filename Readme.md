To run locally:
  1. clone repo
  2. pip install -r requirements.txt
  3. store ur test-video in data folder.
  4. python main.py --input_path data/test_video.mp4 --output_path results --info_path method_info.json --device cpu --checkpoint_path checkpoints/model.pth

To run using Docker:
  1. https://hub.docker.com/r/tarunam172/altfreezing-image2
  2. Use the above link and pull image to local.
  3. To see the results folder, open a dir containing two folders (data,results) and CD to curr dir in cmd prompt.
  4. Put the .mp4 video in data folder.
  5. Run this cmd,
      docker run -v "path_to_dataFolder:/app/data" -v "path_to_resultsFolder:/app/results" tarunam172/altfreezing-image2 python main.py --input_path /app/data/your_test_video_name.mp4 --output_path /app/results --info_path /app/method_info.json --device cpu --checkpoint_path /app/checkpoints/model.pth
  6. Once it is run, result.json file will be created in results folder.


