A full-stack machine learning app that detects and classifies vehicles in images using three different models — from a simple CNN built from scratch to a pretrained Faster R-CNN detector. NOTE: Will add a docker file for ease of use. 

## Models

| Model | Type | Classes | Accuracy |  
| Binary CNN | Classifier | airplane vs car | 94.1% |  
| Multi-class CNN | Classifier | airplane / car / boat | 87.6% |  
| Faster R-CNN ResNet50 | Object Detector | car / airplane / boat | Pretrained on COCO |  

## Stack
- **Frontend:** React + TypeScript (Vite)  
- **Backend:** FastAPI (Python)  
- **ML Framework:** TensorFlow 2.10  
- **Base Model:** Faster R-CNN ResNet50 pretrained on COCO  

## Project Structure

CNN/
├── cnn_basics.py        # Binary CNN training script  
├── cnn_multiclass.py    # Multi-class CNN training script  
└── VehicleClassifier.py # Standalone detection script  

├── backend/
│   ├── main.py          # FastAPI endpoints  
│   ├── detector.py      # Model inference logic  
│   └── models/          # Saved Keras models (binary + multiclass)  
├── frontend/  
│   └── src/  
│       ├── App.tsx  
│       └── components/  
│           ├── ImageUpload.tsx  
│           └── DetectionResult.tsx  



## Setup

### Prerequisites
- Python 3.10
- Node.js
- Miniconda

### 1. Create Python environment

```
conda create -n jetdetect python=3.10 -y
conda activate jetdetect
pip install tensorflow==2.10.0
pip install "numpy<2"
pip install fastapi uvicorn python-multipart pillow pyyaml==5.4.1

```
2. Download the pretrained model
Download Faster R-CNN ResNet50 pretrained on COCO and extract it to the project root:
```
faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/
```
4. Train and save the CNN models
```
python cnn_basics.py
python cnn_multiclass.py
```
4. Start the backend
```
cd backend
uvicorn main:app --reload
Backend runs at http://127.0.0.1:8000
```
5. Start the frontend
```
cd frontend
npm install
npm run dev
Frontend runs at http://localhost:5173
```
Usage
Open http://localhost:5173 in your browser
Select a model from the top buttons
Drag and drop or click to upload an image containing boat, plane, and or car
View the prediction or bounding box results
Notes
The pretrained Faster R-CNN model is not included in the repo due to GitHub's 100MB file size limit — download it separately (see Setup step 2)
AMD GPU users: the DirectML plugin is incompatible with the Object Detection API — all inference runs on CPU
Binary and multi-class CNNs were trained on CIFAR-10 (32×32 images) — real-world accuracy may vary
