from pathlib import Path
from typing import BinaryIO
import face_recognition
import pickle
import os

cwd = Path(__file__).parent.resolve().as_posix() + "/"

Path(cwd + "training").mkdir(exist_ok=True)
Path(cwd + "output").mkdir(exist_ok=True)
Path(cwd + "validation").mkdir(exist_ok=True)


class FaceEncoder:
    DEFAULT_ENCODINGS_PATH = Path(cwd + "/output/encodings.pkl")

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
        for filepath in Path(cwd + "/training").glob("*/*"):
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

    def add_student(
        self,
        label: str,
        image: BinaryIO,
        encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    ) -> None:
        """
        Adds a new image to the training set.
        """
        Path(cwd + "training/" + label).mkdir(exist_ok=True)
        image.save(
            cwd
            + "training/"
            + label
            + "/img_"
            + str(len(os.listdir(cwd + "training/" + label)) + 1)
            + ".jpeg"
        )
        FaceEncoder.encode_known_faces()
