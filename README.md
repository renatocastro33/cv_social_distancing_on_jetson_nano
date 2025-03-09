# Social Distancing Detection with Bird's Eye View

This repository contains a computer vision system for real-time **social distancing monitoring** using YOLOv3 object detection and **Bird's Eye View (BEV) transformation**. The system detects people, evaluates if they respect a predefined minimum distance, and provides visual feedback with bounding boxes and a BEV plot.

## Main Features

- Real-time social distancing detection.
- Bird's Eye View visualization for better distance analysis.
- Configurable input sources: video files or live camera.
- Jetson Nano and standard computer compatibility.
- Optional video recording of detection and BEV output.
- Fully adjustable distance parameters based on real-world measurements.

---

## Repository Structure

```
.
├── Calibration_Social_Distancing.py   # Calibration process to generate homography matrix for BEV
├── Detection_Social_Distancing.py    # Real-time social distancing detection and visualization
├── utils_social_distancing.py       # Utility functions for object detection, distance calculation, and visualization
├── Models/                          # YOLOv3 model files (weights, config, and class names)
├── Images/                          # Generated calibration and result images
├── Numpy Files/                     # Saved numpy arrays (homography matrix, calibration points)
├── RecordedVideos/                  # Saved recorded videos if enabled
└── README.md                        # This file
```

---

## Setup

### 1. Environment Setup

- **Python 3.6+**
- Required packages:
  ```bash
  pip install opencv-python numpy matplotlib imutils
  ```

- **Jetson Nano users**: 
  - VNC must be set up to use the graphical interface (UI is required for calibration).
  - Recommended guide: [Jetson Nano VNC Access](https://medium.com/@bharathsudharsan023/jetson-nano-remote-vnc-access-d1e71c82492b)
  - VNC software used: **TightVNC Server**

### 2. Model Files (YOLOv3)
Place these files inside the `Models/` directory:
- `yolov3.weights`
- `yolov3.cfg`
- `coco.names`

---

## Calibration Process

**Run calibration before running the detection system to generate BEV transformation and distance parameters.**

```bash
python Calibration_Social_Distancing.py -i <video_path_or_0> -m <jetson_or_computer>
```

### Parameters:
- `-i` or `--input`: Path to video file or `0` for live camera.
- `-m` or `--machine`: Device type, options: `jetson` or `computer`.

### Process:
1. **Select 4 points for Bird's Eye View (clockwise order).**
2. **Input real-world measurements:**
   - Width (cm)
   - Height (cm)
   - Minimum social distance (cm)
3. **Select 4 points for Region of Interest (ROI), also clockwise.**
4. Calibration outputs:
   - Homography matrix (`Homolographic_Matrix.npy`)
   - Distance scaling (`min_distance_pix.npy`)
   - ROI and BEV points.

These files are automatically saved in `Numpy Files/` and `Images/`.

---

## Social Distancing Detection

Run the main detection system:

```bash
python Detection_Social_Distancing.py -i <video_path_or_0> -m <jetson_or_computer> -r <True_or_False> -d <True_or_False> -n <numberDropFrames>
```

### Parameters:
- `-i` or `--input`: Path to video file or `0` for live camera.
- `-m` or `--machine`: Device type, options: `jetson` or `computer`.
- `-r` or `--record`: **True** to enable recording of output videos.
- `-d` or `--noDrop`: **False** to allow frame skipping (recommended for performance), **True** to process all frames.
- `-n` or `--numberDropFrames`: Number of frames to skip (recommended: `7`).

### Outputs:
- **Real-time visualization** of:
  - Social distancing compliance (bounding boxes in red and green).
  - Bird's Eye View with spatial representation of distances.
- Optional video recordings:
  - `RecordedVideos/Record_Detection_<timestamp>.avi`
  - `RecordedVideos/Record_Birdeyeview_<timestamp>.avi`

---

## How It Works

1. **Calibration Phase**:
   - Select BEV and ROI points.
   - Map image coordinates to real-world dimensions.
   - Compute transformation matrix for BEV visualization.
   
2. **Detection Phase**:
   - Detect people in each frame using **YOLOv3**.
   - Transform detected points into BEV coordinates.
   - Measure pairwise distances.
   - Classify pairs respecting or violating social distancing.
   - Display annotated frame and BEV map.

---

## Visual Results

- **Green bounding box**: Person respecting social distance.
- **Red bounding box**: Person violating social distance.
- **Bird's Eye View**:
  - **Green dots**: Compliant individuals.
  - **Red dots**: Violating individuals.
  - Optional lines between individuals too close to each other.

---

## Example Usage

### Calibration (Jetson Nano live camera):
```bash
python Calibration_Social_Distancing.py -i 0 -m jetson
```

### Detection (with video file, recording enabled, frame skip enabled):
```bash
python Detection_Social_Distancing.py -i ./Videos/test.mp4 -m computer -r True -d False -n 7
```

---

## Customization

- **Detection Model**: Easily swap YOLOv3 model files under `Models/`.
- **Distance Parameters**: Adjusted during calibration.
- **Frame Skipping**: Fine-tune with `--numberDropFrames` for real-time performance.
- **API Integration**: Extend with `utils_mdp.py` for incident reporting or alerts.

---

## Notes

- **System requires UI** for calibration (Jetson Nano must use VNC or desktop access).
- For better performance on low-power devices (e.g., Jetson Nano), enable frame dropping.

---

## License

MIT License. Feel free to use, modify, and share.