from pathlib import Path
from flask import Flask, jsonify, request
from FaceEncoder import FaceEncoder as FaceEncoderClass
from FaceDetector import FaceDetector as FaceDetectorClass

FaceDetector = FaceDetectorClass()
FaceEncoder = FaceEncoderClass()
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"

@app.route('/encode', methods=['GET', 'POST'])
def encode():
    """
    Encodes the faces contained in ./training/{label}/img_{num}.jpeg
    and stores them in a pickle file in ./output.
    Only needs to be run when the initial training set is created or modified.
    """
    model = "hog"
    try:
        FaceEncoder.encode_known_faces()
        return jsonify({"message": "Encoding complete"})
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/recognise', methods=['POST'])
def recognise():
    """
    TODO: create request schema validation
    Load unknown files and classify them using the encoding created from
    encode_known_faces.
    """
    # get image from request
    img = request.files['image']
    # encodings_location = FaceDetector.DEFAULT_ENCODINGS_PATH
    try:
        names, image = FaceDetector.recognise_faces(image_file=img)
        return {"names": names, "image": image}
    except Exception as e:
        return {"error": str(e)}
    

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)


# def init_routes() -> APIRouter:
#     router = APIRouter()

#     @router.get("/")
#     async def root():
#         return {"message": "Hello World"}

#     @router.post("/encode")
#     async def encode(
#         model: str = "hog", encodings_location: Path = FaceDetector.DEFAULT_ENCODINGS_PATH
#     ) -> dict:
#         """
#         Encodes the faces contained in ./training/{label}/img_{num}.jpeg
#         and stores them in a pickle file in ./output.
#         Only needs to be run when the initial training set is created or modified.
#         """
#         try:
#             FaceDetector.encode_known_faces(model, encodings_location)
#             return {"message": "Encoding complete"}
#         except Exception as e:
#             return {"error": str(e)}

#     @router.post("/recognise")
#     async def recognise(
#         image_location: str,
#         model: str = "hog",
#         encodings_location: Path = FaceDetector.DEFAULT_ENCODINGS_PATH,
#     ) -> dict:
#         """
#         Load unknown files and classify them using the encoding created from
#         encode_known_faces.
#         """
#         try:
#             result = FaceDetector.recognize_faces(image_location, model, encodings_location)
#             return {"message": "Recognition complete", "result": result}
#         except Exception as e:
#             return {"error": str(e)}

#     return router
