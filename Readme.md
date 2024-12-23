USE THIS CMD -> docker run -v "C:\D Drive\RA2\demo\data:/app/data" -v "C:\D Drive\RA2\demo\results:/app/results" tarunam172/altfreezing-image2 python main.py --input_path /app/data/ex3.mp4 --output_path /app/results --info_path /app/method_info.json --device cpu --checkpoint_path /app/checkpoints/model.pth

To run locally:
  1. clone repo
  2. pip install -r requirements.txt
  3. store ur test-video in data folder.
  4. python main.py --input_path data/ex3.mp4 --output_path results --info_path method_info.json --device cpu --checkpoint_path checkpoints/model.pth

To run using Docker:
  1. https://hub.docker.com/r/tarunam172/altfreezing-image2
  2. Use the above link and pull image to local.
  3. To see the results folder, open a dir containing two folders (data,results).
  4. Put the .mp4 video in data folder.
  5. Run this cmd,
      docker run -v "C:\D Drive\RA2\demo\data:/app/data" -v "C:\D Drive\RA2\demo\results:/app/results" tarunam172/altfreezing-image2 python main.py --input_path /app/data/your_test_video_name.mp4 --output_path /app/results --info_path /app/method_info.json --device cpu --checkpoint_path /app/checkpoints/model.pth


