import argparse

from emotion_detector import EmotionDetector
# cd EmotionDetection-main, antes de rodar o programa

def parse_arguments():
    parser = argparse.ArgumentParser(description="Lucas Boscatti's Final Project Nero")
    parser.add_argument("--model_name", default="resnet18.onnx", type=str)
    parser.add_argument(
        "--backend_option",
        default=2,
        type=int,
        choices=[1, 2, 3],
        help="Choose the backend-target pair to run this demo: "
        "1: OpenCV implementation + CPU, "
        "2: CUDA + GPU (CUDA), "
        "3: CUDA + GPU (CUDA FP16)",
    )
    parser.add_argument(
        "--providers",
        default=2,       #None
        type=int,
        choices=[1, 2],      # [1,2,3], a ultima coluna que é a da jetson foi removida
        help="Choose the backend-target pair to run this demo: "
        "1: CPUExecutionProvider, "
        "2: CUDAExecutionProvider (CUDA), "
        "3: TensorrtExecutionProvider (CUDA FP16)",
    )
    parser.add_argument(
        "--file",
        default=0,
        type=str,
        help="Specify the input source: 0, 'realsense', or 'path/to/video or image'",
    )
    parser.add_argument(
        "--display_window",
        default=True,
        type=bool,
        help="Specify whether to display the window (True or False)",
    )
    parser.add_argument(
        "--option",
        default="video",
        type=str,
        choices=["video", "image"],
        help="Specify the processing option: 'video' or 'image'",
    )
    parser.add_argument(
        "--num_faces", default=None, type=int, help="Number of faces to detect"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    detector = EmotionDetector(
        backend_option=args.backend_option,
        model_name=args.model_name,
        providers=args.providers,
        num_faces=args.num_faces,
    )

    if args.option == "image":
        detector.process_image(args.file)

    detector.process_video(args.file, display_window=args.display_window)


if __name__ == "__main__":
    main()

