from bridge_wrapper import YOLOv7_DeepSORT
from PIL import Image
from detection_helpers import Detector
import cv2
import warnings

warnings.filterwarnings('ignore')

detector = Detector()
# pass the path to the trained weight file
detector.load_model('./weights/best.pt', trace=False)
detector.device = 0

# if len(result.shape) == 3:  # If it is image, convert it to proper image. detector will give "BGR" image
#     result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

# cv2.imshow('Horse', result)
# cv2.waitKey(0)

# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(
    reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

# output = None will not save the output video
tracker.track_video("./IO_data/input/video/dataset_line_crossing_Trim.mp4", output="./IO_data/output"
                                                                                   "/dataset_line_crossing_Trim.avi",
                    show_live=True, count_objects=True, verbose=1)


