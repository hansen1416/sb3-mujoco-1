import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = os.path.join(os.path.dirname(
    __file__), '..', 'models', 'mediapipe', 'pose_landmarker_heavy.task')

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)


# Load the input image from an image file.
mp_image = mp.Image.create_from_file(os.path.join(os.path.dirname(
    __file__), '..', 'img', '2023-09-24-19-41-57.png'))

with PoseLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
    # ...
    # Perform pose landmarking on the provided single image.
    # The pose landmarker must be created with the image mode.
    pose_landmarker_result = landmarker.detect(mp_image)

    print(pose_landmarker_result)
