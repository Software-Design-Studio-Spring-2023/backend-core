from typing import Union
from flask_socketio import SocketIO, emit, join_room
from flask import Flask, jsonify, request, render_template
import requests
import os
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np
import base64
from server.infrastructure.ObjectDetection import ObjectDetectionWrapper
from server.infrastructure.FaceEncoder import FaceEncoder as FaceEncoderClass
from server.infrastructure.FaceDetector import FaceDetector as FaceDetectorClass

FaceDetector = FaceDetectorClass()
FaceEncoder = FaceEncoderClass()
app = Flask(__name__)
app.config['SECRET'] = 'password'
socketio = SocketIO(app, cors_allowed_origins="*")

active_sessions = {} # students on the website
ClientInfo = dict[str, Union[ObjectDetectionWrapper, bool]]

registered_clients: dict[str, ClientInfo] = {} # students that have joined a room and are being monitored
video_writers: dict[str, VideoWriter_fourcc] = {}  # To hold VideoWriter objects for different clients/sessions


@socketio.on('connect')
def handle_connect():
    join_room(request.sid) # TODO: join the room of the client id

@socketio.on('join')
def handle_join(data):
    # register opencv VideoCapture object with the client url
    client_data = {
        "detection": ObjectDetectionWrapper(data['student_id']),
        "started": False
    }
    # os.sleep(10)
    registered_clients[data['student_id']] = client_data # not sure if we should use sid or student_id 
                                                  # (will have to be in the join or leave event if student_id)
    active_sessions[request.sid] = data['student_id']
    return {"message": "Joined room"}

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in active_sessions:
        del active_sessions[request.sid]
    return {"message": "Disconnected"}

@socketio.on('leave')
def handle_leave(data):
    if data['student_id'] in registered_clients:
        # del active_sessions[request.sid]
        del registered_clients[data['student_id']]
    
    # may want to analyse the video here and send the results to the host application
    # then delete VideoWriter object
    return {"message": "Left room"}

def ack():
    print('message was received!')

@socketio.on('frame')
def handle_frame(frame_data: dict[str, str]): # TODO: get client id and send to that client only, and the host application
     if frame_data['student_id'] in registered_clients:
        # print('frame received from client: ' + frame_data['student_id'])

        # Decode the base64 string
        img_bytes = base64.b64decode(frame_data['frame'])

        # Convert the bytes to a numpy array and read with OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        if nparr.size == 0:
            return
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect objects in the frame
        detections = registered_clients[frame_data['student_id']]['detection'].detectFrameByFrame(img)

        # Write the frame to file
        writeFrameToFile(img, frame_data)

        # for now we just publish to every client, but we should only publish to the host application and the client that sent the frame
        # emit('frame_data', {"message": "frame received", "detections": detections[1]}, callback=ack) 

        return {"message": "Frame received", "detections": detections[1]}

        
def writeFrameToFile(img: bytes, frame_data: dict[str, str]):
        # Determine the filename from the property in the request
        # For this example, I'm assuming 'filename' is a property in frame_data
        filename = frame_data['student_id'] + ".avi"  # You can change the extension based on the codec you choose.

        # Initialize VideoWriter if it's not already set for this filename
        if filename not in video_writers:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec. You can choose another one if needed.
            fps = 20.0  # Define the fps. Adjust based on your needs.
            frame_size = (img.shape[1], img.shape[0])  # Frame width, height
            video_writers[filename] = cv2.VideoWriter(filename, fourcc, fps, frame_size)

        # Write the frame to the video file
        video_writers[filename].write(img)

        # If you're done with this session and want to finalize the video:
        # (This step is based on some condition. Here I'm just providing a pseudo-code example.)
        # if some_condition:
        #     video_writers[filename].release()
        #     del video_writers[filename]

        # If you want to perform any other actions with the frame, you can continue doing so here.

@app.route("/", methods=["GET", "POST"])
def welcome():
    return "Hello World!"


@app.route("/encode", methods=["GET", "POST"])
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


@app.route("/recognise", methods=["POST"])
def recognise():
    """
    TODO: create request schema validation
    Load unknown files and classify them using the encoding created from
    encode_known_faces.
    """
    img = request.files["image"]
    try:
        names, image = FaceDetector.recognise_faces(image_file=img)
        return jsonify({"names": names, "authenticated": True})
    except Exception as e:
        return {"error": str(e)}


@app.route("/add", methods=["POST"])
def add():
    """
    Adds a new image to the training set.
    """
    label = request.form["label"]
    image = request.files["image"]
    try:
        FaceEncoder.add_student(label, image)
        return jsonify({"message": "Image added"})
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/student", methods=["GET"])
def serve_student():
    """
    Serves the student page.
    """
    return render_template("../client/index.html")

@app.route("/host", methods=["GET"])
def serve_host():
    """
    Serves the host page.
    """
    return render_template("index.html")

def downloadYolo():
    url = "https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get the root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Save the file to the root directory
    with open(os.path.join(root_dir, 'yolov3.pt'), 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
if __name__ == "__main__":
    # download the pretrained model https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt/
    # and place it in the root directory
    if not os.path.exists("yolov3.pt"):
        downloadYolo()
    
    socketio.run(app, host="127.0.0.1", port=8080, allow_unsafe_werkzeug=True )


