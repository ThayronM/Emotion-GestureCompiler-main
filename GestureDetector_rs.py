import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import cv2
import time

import pandas as pd
import numpy as np
import mediapipe as mp

from utils.Mconfusao import Mconfusao
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils.video import FPS
from utils.utils import setup_logger
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt


TAG_I = True
#%%
# Calculate the angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle

    return angle

# função que retorna todos os dados de um diretório
def get_allFiles (data_directory):
    return [f for f in os.listdir(data_directory) if f.endswith('.xlsx')]

# Função para extrair as features dos dados
def extract_features(data):
    mat = data.T @ data
    return mat.flatten()

def timer_UI (img, dt):
    # draw counter
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    size = 10
    textSize = cv2.getTextSize(str(dt), font, size, size)[0]
    cv2.putText(
        img,
        str(dt),
        (round((w - textSize[0])/2), round((h + textSize[1])/2)),
        font, size, (255, 255, 255), size
    )
    return img

def recording_UI (img):
    # draw red circle
    h, w = img.shape[:2]
    radius = 12
    cv2.circle(
        img,
        (w - (2*radius + 1), 2*radius + 1), radius, (0, 0, 255), 2*radius
    )
    return img

#%%
class GestureDetector:
    def __init__(
            self, gestures:list,
            train_path:str,
            k:int = 7
    ):
        self.gesture_name = gestures    # all gestures and their names

        self.status = "end"                     # current status of the recording
        self.start_time = time.time()
        self.matrix = np.zeros((1,18))          # matrix with data from many frames
        self.resp = '??'                        # current awnser
        self.logger = setup_logger(__name__)    # debug logger

        self.file_counter = {}
        self.name_order = [
                'ShoulderR_X',
                'ShoulderR_Y',
                'ShoulderR_Z',
                'ShoulderL_X',
                'ShoulderL_Y',
                'ShoulderL_Z',
                'ElbowR_X',
                'ElbowR_Y',
                'ElbowR_Z',
                'ElbowL_X',
                'ElbowL_Y',
                'ElbowL_Z',
                'WristR_X',
                'WristR_Y',
                'WristR_Z',
                'WristL_X',
                'WristL_Y',
                'WristL_Z'
            ]

        # KNN classifier with K neighbor
        self.knn_classifier = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto')

        # landmark skeleton
        self.skeleton = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.logger.info("Training Gestures")
        self.train_xlsx(train_path) # train KNN
        
        self.gesture_positions = []
        return
    
    def process_frame(self, img):
        
        # prints results
        img = self.print_data(img)

        # MAKE DETECTION
        try:
            self.detection = self.skeleton.process(img)
            img = self.print_skeleton(img)

            # simplifications
            landmark = self.detection.pose_landmarks.landmark
            m = mp.solutions.pose.PoseLandmark
            memb = [
            m.RIGHT_SHOULDER,
            m.LEFT_SHOULDER,
            m.RIGHT_ELBOW,
            m.LEFT_ELBOW,
            m.RIGHT_WRIST,
            m.LEFT_WRIST
        ]
        except:
            return img
        
        # reference
        nose = np.array([landmark[m.NOSE].x, landmark[m.NOSE].y, landmark[m.NOSE].z])
        
        # formated data of current frame
        vector = np.array([
            np.array([
                np.array([landmark[i].x, landmark[i].y, landmark[i].z]) - nose for i in memb
            ]).flatten()
        ])

        match self.status:
            case "end":
                # calculates gesture to iniciate recording
                angle = calculate_angle(vector[0,3:5], vector[0,9:11], vector[0,15:17])
                
                if angle < 70 and vector[0,16] < vector[0,4]:
                    self.status = "wait"
                    self.start_time = time.time()

            case "wait":
                # preparetion time for the recording
                timer = round(self.start_time + 3 - time.time())
                if timer > 0:
                    img = timer_UI(img, timer)  # draws timer 
                else:
                    self.status = "start"
                    self.start_time = time.time()

            case "start":
                if time.time() - self.start_time < 2:
                    # records for 2 seconds
                    self.matrix = np.concatenate((self.matrix,vector),0)
                    img = recording_UI(img)     # indicates that it's on air
                else:
                    self.status = "end"     # stop recording
                    self.matrix = np.concatenate((self.matrix[1:, :],vector),0)
                    self.resp = self.classify_video(self.matrix)
                    if self.resp == "I":
                        self.matrix = np.zeros((1,18))
                        
        self.gesture_positions.append(self.matrix[-1, :].tolist())  # Adiciona a última linha da matriz (última posição do gesto) à lista
    
        return img

    def record (self, video_path:str = 0):
        if video_path == "realsense":
            video_path = "v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"

        cap = cv2.VideoCapture(video_path)  # video inicialization
        if not cap.isOpened():
            return

        # video reading
        success, img = cap.read()
        fps = FPS().start()

        # image processing
        while success:

            image = self.process_frame(img)

            cv2.imshow('Output', image)

            # resets matrix
            if self.status == 'end':
                self.reset_pred()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            fps.update()
            success, img = cap.read()

        fps.stop()
        cap.release()
        cv2.destroyAllWindows()
        
        self.plot_gesture_positions()
        self.plot_sample()
        return
    
    
    def reset_pred (self):
        self.matrix = np.zeros((1,18))
        self.resp = "??"
        return

    def print_skeleton (self, image):
        # print parameters
        color_1 = (145, 45, 30)
        color_2 = (0, 0, 245)
        thickness = 1
        circular_radius = 2

        # RENDER DETECTIONS
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            self.detection.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(
                color=color_1,
                thickness=thickness,
                circle_radius=circular_radius),
            mp.solutions.drawing_utils.DrawingSpec(
                color=color_2,
                thickness=thickness,
                circle_radius=circular_radius)
        )
        return image

    def print_data (self, image):
        global TAG_I
        # print parameters
        font = cv2.FONT_HERSHEY_DUPLEX
        size = 0.8
        color_text = (0, 0, 255)
        thickness = 2

        # render results
        cv2.putText(
            image,
            f'Resp.: {self.resp}',
            (0, round(size*45)),
            font, size, color_text, thickness
        )
        
        if self.resp != '??':
            if self.resp == 'I' and TAG_I:
                print(f'Class: {self.resp}')
                TAG_I = False
            else:
                if self.resp != 'I':
                    print(f'Class: {self.resp}')
                    TAG_I = True

        return image
    
    def train_xlsx (self, trainData_path: str):
        subfiles = [f.path for f in os.scandir(trainData_path) if f.is_dir()]
        X = []  # train data
        Y = []  # target values

        for sub in subfiles:
            all_files = get_allFiles(sub)
            self.file_counter[str(sub.split('\\')[1])] = len(all_files)     # creates file counter

            for file in all_files:
                # colect the data
                dados = pd.read_excel(os.path.join(sub, file)).to_numpy()
                
                # saves in array-like
                X.append(extract_features(dados))
                Y.append(file.split('_')[0])    # awnser in the name of the file
        
        self.knn_classifier.fit(X, Y)
        
        previsao_knn = self.knn_classifier.predict(X)
        mat_confusion = confusion_matrix(Y, previsao_knn)
        # print(f'Database: {previsao_knn}')
        print('Accuracy score: ',accuracy_score(Y, previsao_knn)) # compara os testes Y com as previsoes
        print(mat_confusion)
        
        return
    
    def classify_video (self, matrix, threshold:float=0.9):
            # makes prediction
            prob = self.knn_classifier.predict_proba([extract_features(matrix)])[0]
            # return prediction
            return self.knn_classifier.classes_[prob.argmax()] if max(prob) >= threshold else 'I'

    def classify_xlsx(self, validation_path:str, threshold:float=0.9):
        # Função para classificar um novo arquivo xlsx

        self.gesture_counter = {i : 0 for i in self.gesture_name}

        # labels
        self.predicted_label = []
        self.real_label = []

        self.MC = Mconfusao(self.gesture_name, True)    # confusion matrix

        # getes results
        for file in get_allFiles(validation_path):
            # collect the data
            dados = pd.read_excel(os.path.join(validation_path, file)).to_numpy()

            # makes prediction
            prob = self.knn_classifier.predict_proba([extract_features(dados)])[0]

            # saves predicted and real ones
            self.real_label.append(file.split('_')[0])
            self.predicted_label.append(self.knn_classifier.classes_[prob.argmax()] if max(prob) >= threshold else 'I')
        
        # process results
        for i in range(len(self.real_label)):
            self.MC.add_one(self.real_label[i], self.predicted_label[i])

            if self.real_label[i] == self.predicted_label[i]:
                self.gesture_counter[self.real_label[i]] += 1      # counts individualy

        # shows confusion matrix
        self.MC.render()
        self.MC.great_analysis()
        return

    def saves_to_dataBase (self):
        # Salvar o arquivo 
        file_name = f"Base_de_dados/{self.resp}/{self.resp}_{self.file_counter[self.resp]+1:02d}.xlsx"
        self.file_counter[self.resp] = self.file_counter[self.resp]+1     # adds to the file counter
        
        # organizes and saves
        df = pd.DataFrame(self.matrix, columns= self.name_order)
        df.to_excel(file_name, index=False, engine='openpyxl')

        self.logger.info(f"DataSet {file_name} saved sucessefuly!")

        # resets matrix 
        self.reset_pred()
        
    def save_gesture(self, path: str):
        try:
            with open(path, 'a') as f:  # Abrir o arquivo no modo de 'append'
                f.write(f'{self.resp}\n')
        except Exception as e:
            print(f"Error: {e}")
            
            
    def gesture_ros(self):
        return self.resp
        
#%%
if __name__ == "__main__":

    data_directory = 'Base_de_dados'
    G = GestureDetector(['A', 'B', 'C', 'D', 'E'], data_directory)
    # G.classify_xlsx(data_directory[1])
    G.record()