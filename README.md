# Face Liveness Detection

This project implements a face liveness detection system using the CelebA-Spoof dataset.

## Requirements

- Python 3
- OpenCV-Python
- NumPy
- PyTorch (>= 0.4.1)
- Pillow (PIL)
- Torchvision

You can install the required packages using pip:

```bash
pip install opencv-python numpy torch torchvision pillow
```

## Running the Liveness Detection Model

There are two ways to run the liveness detection model:

### 1. Using the Webcam

To run the liveness detection model using your webcam, simply execute:

```bash
python run_liveness.py

# For a single image:
python your_script.py /path/to/image.jpg

# For a whole folder:
python your_script.py /path/to/image_folder
```
This will open your webcam and display the liveness detection results in real-time. Press 'q' to quit.

### 2. Using a Single Image

To run the liveness detection model on a single image, provide the image path as a command-line argument:

```bash
python run_liveness.py path/to/your/image.jpg
```

## Troubleshooting

### Common Issues

1. **File Path Error**: If you encounter a file not found error for the checkpoint file, make sure the path to `ckpt_iter.pth.tar` is correct. The file should be located at:
   ```
   CelebA-Spoof/intra_dataset_code/ckpt_iter.pth.tar
   ```

2. **CUDA Issues**: If you encounter CUDA-related errors, the model will automatically fall back to CPU processing. You can verify this by checking the "Using device" message in the console output.

3. **Import Errors**: If you encounter import errors, make sure you're running the script from the correct directory. The script adds the necessary paths to the Python path.

## Model Details

The liveness detection model is based on the AENet architecture from the CelebA-Spoof dataset. It classifies faces as either "live" (real) or "spoof" (fake).

The model outputs a probability score between 0 and 1, where:
- Values closer to 1 indicate a higher probability of being a real face
- Values closer to 0 indicate a higher probability of being a fake face

## Credits

This implementation uses the model from the CelebA-Spoof dataset:

```
@inproceedings{CelebA-Spoof,
title={CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations},
author={Zhang, Yuanhan and Yin, Zhenfei and Li, Yidong and Yin, Guojun and Yan, Junjie and Shao, Jing and Liu, Ziwei},
booktitle={European Conference on Computer Vision (ECCV)},
year={2020}
}
```
