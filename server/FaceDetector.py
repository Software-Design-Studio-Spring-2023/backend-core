from pathlib import Path
import pickle
from collections import Counter
from PIL import Image, ImageDraw
import face_recognition

BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")


class FaceDetector:
    """
    Class for detecting faces in images.
    """
    def recognize_faces(
        image_location: str,
        model: str = "hog",
        encodings_location: Path = DEFAULT_ENCODINGS_PATH,
        self: object = None,
    ) -> None:
        """
        Load unknown files and classifies them using the encoding created from
        encode_known_faces.
        """
        with encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)

        input_image = face_recognition.load_image_file(image_location)

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
            # Removed print(name, bounding_box)
            self._display_face(draw, bounding_box, name)
            print(name, bounding_box)
        del draw
        pillow_image.show()


    def _recognize_face(unknown_encoding, loaded_encodings):
        """
        Compares the unknown encoding to the known encodings and returns the name.
        Returns the most highest voted name.
        """
        boolean_matches = face_recognition.compare_faces(
            loaded_encodings["encodings"], unknown_encoding
        )
        votes = Counter(
            name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match
        )
        if votes:
            return votes.most_common(1)[0][0]


    def _display_face(draw, bounding_box, name):
        """
        Draws the bounding box and the name of the face.
        """
        top, right, bottom, left = bounding_box
        draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
        text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
        draw.rectangle(
            ((text_left, text_top), (text_right, text_bottom)),
            fill="blue",
            outline="blue",
        )
        draw.text(
            (text_left, text_top),
            name,
            fill="white",
        )


    def validate(model: str = "hog", self: object = None):
        """
        Validates the model by running it on the images in the validation folder.
        """
        for filepath in Path("validation").rglob("*"):
            if filepath.is_file():
                self.recognize_faces(image_location=str(filepath.absolute()), model=model)

