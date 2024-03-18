from EmotionGestureCompiler import EmotionGestureCompiler
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

def main ():
    model = EmotionGestureCompiler(
        model_name = "resnet18.onnx",
        model_option = "onnx",
        backend_option = 2, #1
        providers = 1,
        fp16 = False,
        num_faces = 1,
        train_path = 'Base_de_dados',
        k = 7
    )
    model.video(video_path = 0)

if __name__ == "__main__":
    main()