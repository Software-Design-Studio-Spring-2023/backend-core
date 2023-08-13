# Create basic API that can call FaceEncoder.encode_known_faces()
# and FaceDetector.recognise_faces() with the correct parameters.

# Path: server/routes.py
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile

from server.FaceDetector import FaceDetector

router = APIRouter()



