from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from detector import predict_binary, predict_multiclass, predict_rcnn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect/binary")
async def detect_binary(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    return predict_binary(image)

@app.post("/detect/multiclass")
async def detect_multiclass(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    return predict_multiclass(image)

@app.post("/detect/rcnn")
async def detect_rcnn(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    return predict_rcnn(image)
