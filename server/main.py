from flask_socketio import SocketIO, emit, join_room, leave_room, send
from flask import Flask, Response, jsonify, request, render_template
from infrastructure.FaceEncoder import FaceEncoder as FaceEncoderClass
from infrastructure.FaceDetector import FaceDetector as FaceDetectorClass

FaceDetector = FaceDetectorClass()
FaceEncoder = FaceEncoderClass()
app = Flask(__name__)
app.config['SECRET'] = 'password'
socketio = SocketIO(app, cors_allowed_origins="*")

active_sessions = {}

@socketio.on('connect')
def handle_connect():
    active_sessions[request.sid] = None
    join_room(request.sid)  

@socketio.on('warning_event')
def handle_warning(target_sid):
    if target_sid in active_sessions:
        emit('warning_event', room=target_sid)

@socketio.on('get_active_sessions')
def send_active_sessions():
    sessions_list = list(active_sessions.keys())
    emit('active_sessions_list', {'sessions': sessions_list})

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in active_sessions:
        del active_sessions[request.sid]
    leave_room(request.sid)

def ack():
    print('message was received!')

@socketio.on('frame')
def handle_frame(frame_data): # TODO: get client id and send to that client only, and the host application
     if request.sid in active_sessions:
        print('frame received')
        # for now we just publish to every client, but we should only publish to the host application and the client that sent the frame
        emit('frame_data', frame_data, broadcast=True, callback=ack) 
        
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


if __name__ == "__main__":

    socketio.run(app, host="127.0.0.1", port=8080)
