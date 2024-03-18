import sys

import cv2
import numpy as np
import onnxruntime
import torch
import torchvision.transforms as transforms
from imutils.video import FPS
from PIL import Image

# from tensort_inference import TensorRTInference 
from models import resnet
from utils.utils import calculate_winner, setup_logger, to_numpy

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

sys.path.append("../")


class EmotionDetector:
    """
    A class for detecting and recognizing emotions in images or video frames.

    Args:
        use_cuda (bool, optional): Whether to use CUDA for faster processing if a GPU is available. Default is True.
        backend_option (int, optional): Backend option for OpenCV's DNN module. Default is 1.

    Attributes:
        EMOTIONS (dict): A dictionary mapping emotion labels to their corresponding names.
    """

    EMOTIONS = {0: "BAD", 1: "GOOD", 2: "NEUTRAL"}

    def __init__(
        self,
        model_name: str = "resnet18.onnx",
        model_option: str = "onnx",
        backend_option: int = 0 if torch.cuda.is_available() else 1,
        providers: int = 1,
        fp16=False,
        num_faces=None,
    ):
        """
        Initializes the Detector object.

        Args:
            use_cuda (bool, optional): Whether to use CUDA for faster processing if a GPU is available. Default is cuda if CUDA is available, otherwise cpu.
            backend_option (int, optional): Backend option for OpenCV's DNN module. Default is 0 if CUDA is available, otherwise 1.
        """
        self.logger = setup_logger(__name__)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_option = model_option
        self.num_faces = num_faces
        self.bbox_predictions = {
            "bbox_left": [],
            "bbox_right": [],
        }

        self.logger.info("Training Emotions")
        self.face_model = self.load_face_model(backend_option)
        self.emotion_model = self.load_trained_model(
            f"models/{model_name}",
            providers=providers if model_option == "onnx" else None,
            fp16=fp16
        )
        self.img = 0

    def load_face_model(self, backend_option: int) -> cv2.dnn_Net:
        """
        Load the face model for face detection.

        Parameters:
            backend_option (int): Backend option for OpenCV's DNN module.

        Returns:
            cv2.dnn_Net: The loaded face model for face detection.
        """
        face_model = cv2.dnn.readNetFromCaffe(
            "models/face_detector/res10_300x300_ssd_iter_140000.prototxt",
            "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
        )
        backend_target_pairs = [
            [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
            [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],
            [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
        ]

        face_model.setPreferableBackend(backend_target_pairs[backend_option][0])
        face_model.setPreferableTarget(backend_target_pairs[backend_option][1])
        return face_model

    def load_trained_model(self, model_path: str, providers=2, fp16=False):
        """
        Load a trained model.

        Args:
            model_path (str): The path to the model file or to the checkpoint file.
            model_option (str): The option for loading the model.

        Returns:
            model: The loaded model.
        """
        if self.model_option == "pytorch":
            model = resnet.ResNet18()
            model.load_state_dict(
                torch.load(model_path, map_location=self.device)["model_state_dict"]
            )
            model.to(self.device)
            model.eval()

        elif self.model_option == "onnx":
            providers_options = {
                1: ["CPUExecutionProvider"],
                2: ["CUDAExecutionProvider"],
                3: ["TensorrtExecutionProvider"],
            }
            model = onnxruntime.InferenceSession(
                model_path, providers=providers_options[providers]
            )

        # elif self.model_option == "tensorrt":
        #     ENGINE_FILE_PATH = model_path + "_b{}_{}.engine"

        #     model = TensorRTInference(
        #         model_path, ENGINE_FILE_PATH, 1 << 30, 1, fp16=fp16
        #     )

        return model

    def recognize_emotion(self, face: np.ndarray) -> str:
        try:
            transform = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.TenCrop(40),
                    transforms.Lambda(
                        lambda crops: torch.stack(
                            [transforms.ToTensor()(crop) for crop in crops]
                        )
                    ),
                    transforms.Lambda(
                        lambda tensors: torch.stack(
                            [
                                transforms.Normalize(mean=(0,), std=(255,))(t)
                                for t in tensors
                            ]
                        )
                    ),
                ]
            )

            inputs = Image.fromarray(face).resize((48, 48))
            inputs = transform(inputs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)

                if self.model_option == "pytorch":
                    outputs = self.emotion_model(inputs)

                elif self.model_option == "onnx":
                    inputs = {self.emotion_model.get_inputs()[0].name: to_numpy(inputs)}
                    outputs = self.emotion_model.run(
                        [self.emotion_model.get_outputs()[0].name], inputs
                    )
                    outputs = torch.from_numpy(outputs[0])

                elif self.model_option == "tensorrt":
                    outputs = self.emotion_model(inputs)
                    print("TensorRT Output: ", outputs)

                # combine results across the crops
                outputs = outputs.view(bs, ncrops, -1)
                outputs = torch.sum(outputs, dim=1) / ncrops
                _, preds = torch.max(outputs.data, 1)
                preds = preds.cpu().numpy()[0]

            return preds
        except cv2.error as e:
            self.logger.error("No emotion detected: ", e)

    def process_image(self, img_name: str) -> None:
        """
        Processes and displays an image with emotion recognition.

        Args:
            img_name (str): The path to the input image file.
        """
        self.img = cv2.imread(img_name)
        #self.height, self.width = self.img.shape[:2]
        self.process_frame(self.img)
        cv2.imshow("Output", self.img)
        cv2.waitKey(0)

    def process_video(self, video_path: str, display_window: bool = True) -> None:
        """
        Processes a video file, performing emotion recognition on each frame.

        Args:
            video_path (str): The path to the input video file.
                if video_path == "realsense", then the video is captured from the realsense camera.
                if video_path == 0, then the video is captured from the webcam.
                or else, the video is captured from the specified path.
            display_window (bool, optional): Whether to display the processed image using cv2.imshow.
                Defaults to True.
        """
        if video_path == "realsense":
            video_path = "v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"

        self.logger.info("Video path: %s", video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error("Error opening video stream or file")
            return

        success, self.img = cap.read()
        #self.height, self.width = self.img.shape[:2]

        fps = FPS().start()

        while success:
            try:
                self.process_frame(self.img)
                if display_window:
                    cv2.imshow("Output", self.img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    if self.num_faces == 2:
                        flag = calculate_winner(self.bbox_predictions)
                        self.logger.info("O robô deve andar para: %s", flag)
                    break
                fps.update()
                success, self.img = cap.read()
            except KeyboardInterrupt:
                break

        fps.stop()
        self.logger.info("Elapsed time: %.2f", fps.elapsed())
        self.logger.info("Approx. FPS: %.2f", fps.fps())

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, img) -> None:
        """
        Processes the current frame, detects faces, and recognizes emotions.
        """
        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )

        self.face_model.setInput(blob)
        predictions = self.face_model.forward()

        if self.num_faces == 2:
            try:
                prediction_1 = predictions[0, 0, 0, 2]
                prediction_2 = predictions[0, 0, 1, 2]

                if prediction_1 > 0.5 and prediction_2 > 0.5:
                    bbox_1 = predictions[0, 0, 0, 3:7] * np.array(
                        [width, height, width, height]
                    )
                    bbox_2 = predictions[0, 0, 1, 3:7] * np.array(
                        [width, height, width, height]
                    )
                    (x_min_1, y_min_1, x_max_1, y_max_1) = bbox_1.astype("int")
                    (x_min_2, y_min_2, x_max_2, y_max_2) = bbox_2.astype("int")
                    cv2.rectangle(
                        img, (x_min_1, y_min_1), (x_max_1, y_max_1), (0, 0, 255), 2
                    )
                    cv2.rectangle(
                        img, (x_min_2, y_min_2), (x_max_2, y_max_2), (0, 0, 255), 2
                    )

                    face_1 = img[y_min_1:y_max_1, x_min_1:x_max_1]
                    face_2 = img[y_min_2:y_max_2, x_min_2:x_max_2]

                    emotion_1 = self.recognize_emotion(face_1)
                    emotion_2 = self.recognize_emotion(face_2)

                    if x_min_1 < x_min_2:
                        self.bbox_predictions["bbox_left"].append(emotion_1)
                        self.bbox_predictions["bbox_right"].append(emotion_2)
                        cv2.putText(
                            img,
                            EmotionDetector.EMOTIONS[emotion_1],
                            (x_min_1 + 5, y_min_1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            EmotionDetector.EMOTIONS[emotion_2],
                            (x_min_2 + 5, y_min_2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    else:
                        self.bbox_predictions["bbox_left"].append(emotion_2)
                        self.bbox_predictions["bbox_right"].append(emotion_1)
                        cv2.putText(
                            img,
                            EmotionDetector.EMOTIONS[emotion_2],
                            (x_min_2 + 5, y_min_2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            EmotionDetector.EMOTIONS[emotion_1],
                            (x_min_1 + 5, y_min_1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    self.logger.error("Only one face detected.")
            except:
                self.logger.error("Only one face detected.")

        else:
            for i in range(predictions.shape[2]):
                if predictions[0, 0, i, 2] > 0.5:
                    bbox = predictions[0, 0, i, 3:7] * np.array(
                        [width, height, width, height]
                    )
                    (x_min, y_min, x_max, y_max) = bbox.astype("int")
                    cv2.rectangle(
                        img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2
                    )

                    face = img[y_min:y_max, x_min:x_max]

                    emotion = self.recognize_emotion(face)

                    cv2.putText(
                        img,
                        EmotionDetector.EMOTIONS[emotion],
                        (x_min + 5, y_min - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        return img