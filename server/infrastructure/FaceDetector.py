from io import BytesIO
import os
from pathlib import Path
import pickle
from collections import Counter
from PIL import Image, ImageDraw
import face_recognition
from werkzeug.datastructures import FileStorage

# get pwd
cwd = Path(__file__).parent.resolve().as_posix() + "/"  

class FaceDetector():
    """
    Class for detecting faces in images.
    """

    BOUNDING_BOX_COLOR = "blue"
    TEXT_COLOR = "white"
    DEFAULT_ENCODINGS_PATH = Path(cwd + "output/encodings.pkl")

    def __init__(self) -> None:
        pass

    def recognise_faces(
        self,
        image_file: FileStorage,
        model: str = "hog",
        encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    ) -> (str, BytesIO):
        """
        TODO: Pass in image as BinaryIO instead of filepath.
        Load unknown files and classifies them using the encoding created from
        encode_known_faces.
        """
        tempPath = Path(cwd + "processing/temp.jpg")
        with encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)

        image_file.save(tempPath)

        input_image = face_recognition.load_image_file(tempPath)

        input_face_locations = face_recognition.face_locations(input_image, model=model)
        input_face_encodings = face_recognition.face_encodings(
            input_image, input_face_locations
        )

        pillow_image = Image.fromarray(input_image)
        draw = ImageDraw.Draw(pillow_image)

        for bounding_box, unknown_encoding in zip(
            input_face_locations, input_face_encodings
        ):
            name = self._recognize_face(unknown_encoding, loaded_encodings)
            if not name:
                name = "Unknown"
            self._display_face(draw, bounding_box, name)
            print(name, bounding_box)
        # pillow_image.show()
        image_bytes = BytesIO()
        pillow_image.save(image_bytes, format="JPEG")
        os.remove(tempPath)
        return name, image_bytes

    def _recognize_face(self, unknown_encoding, loaded_encodings):
        """
        Compares the unknown encoding to the known encodings and returns the name.
        Returns the most highest voted name.
        """
        boolean_matches = face_recognition.compare_faces(
            loaded_encodings["encodings"], unknown_encoding
        )
        votes = Counter(
            name
            for match, name in zip(boolean_matches, loaded_encodings["names"])
            if match
        )
        if votes:
            return votes.most_common(1)[0][0]

    def validate(self, model: str = "hog"):
        """
        Validates the model by running it on the images in the validation folder.
        """
        for filepath in Path("validation").rglob("*"):
            if filepath.is_file():
                self.recognize_faces(
                    image_location=str(filepath.absolute()), model=model
                )
