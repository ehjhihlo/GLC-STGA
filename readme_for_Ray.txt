for online demo, please run the following command in the terminal:
    ```
    python3 demo/vis_realtime_framebyframe.py
    ```
    make sure check the webcam is installed

    if you want to save predicted 3d poses, run the following command in the terminal:
    ```
    python3 demo/vis_realtime_framebyframe.py --outputdir <your_output_directory>
    ```

for inference on an image, please run the following command in the terminal:
    ```
    python3 demo/vis_realtime_image.py --image-path <your_image_path> --outputdir <your_output_directory>
    ```

    for running with MotionGFormer:
    ```
    python3 demo/vis_realtime_image.py --image-path <your_image_path> --outputdir <your_output_directory> --usebaseline
    ```


for inference on a video (image sequence), please run the following command in the terminal:
    ```
    python3 demo/vis_image_seq.py --image-dir <your_image_path>  --outputdir <your_output_directory>
    ```

    for running with MotionGFormer:
    ```
    python3 demo/vis_image_seq.py --image-dir <your_image_path>  --outputdir <your_output_directory> --usebaseline
    ```