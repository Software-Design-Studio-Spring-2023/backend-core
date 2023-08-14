from pathlib import Path
import face_recognition
import pickle



Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


class FaceEncoder:
    DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
    def encode_known_faces(
        model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
    ) -> None:
        """
        Encodes the faces contained in ./training/{label}/img_{num}.jpeg
        and storing them in a pickle file in ./output.
        Only needs to be run when the initial training set is created or modified.
        """
        names = []
        encodings = []
        for filepath in Path("training").glob("*/*"):
            name = filepath.parent.name
            image = face_recognition.load_image_file(filepath)

            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for encoding in face_encodings:
                names.append(name)
                encodings.append(encoding)

        name_encodings = {"names": names, "encodings": encodings}
        with encodings_location.open(mode="wb") as f:
            pickle.dump(name_encodings, f)