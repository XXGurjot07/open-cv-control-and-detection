# XXGS 01/05/2021
import cv2
from mediapipe import python
import time

import mediapipe as mp


class run_detection:

    # camera index (sort of), used DroidCAM to model phone camera as webcam
    def __init__(self):
        display_hnd = input(
            "Type Y/N to Display of Hand Co-ordinates on Console - ")
        max_num_hands = int(
            input("Type Max No. of Hands You want to Detect - "))
        self.display_hnd = display_hnd
        self.my_capture = cv2.VideoCapture(1)
        self.mphands = mp.solutions.hands
        self.my_hands = mp.solutions.hands.Hands(False, max_num_hands, 1, 0.5, 0.5)  # initialize Hands solution file
        # intitialze drawing calcutions + styles
        self.mp_Draw = mp.solutions.drawing_utils
        self.mp_DrawStyles = mp.solutions.drawing_styles

        self.pTime = 0
        self.cTime = 0

        # max no. of hands is set to 1
        # while frames are coming, perform processes and draws landmarks and connection
        while True:
            # returns 2 values (bool_value, image)
            was_success, self.my_frames = self.my_capture.read()

            # MediaPipe uses RGB, Open CV uses BGR
            self.my_framesRGB = cv2.cvtColor(self.my_frames, cv2.COLOR_BGR2RGB)
            self.result = self.my_hands.process(self.my_framesRGB)

            if self.result.multi_hand_landmarks:  # if result = true i.e it found any hands, executes the for loop for each hand and draws point and line
                for hand_landmarks in self.result.multi_hand_landmarks:
                    for id, landmarks_co in enumerate(hand_landmarks.landmark):
                        height, widht, channels = self.my_frames.shape
                        x_co = int(landmarks_co.x * widht)
                        y_co = int(landmarks_co.y * height)

                        if self.display_hnd == "Y":
                            print(f"ID = {id} ----  (X,Y) = ({x_co},{y_co})")
                        else:
                            pass

                    self.mp_Draw.draw_landmarks(self.my_frames, hand_landmarks, self.mphands.HAND_CONNECTIONS,
                                                self.mp_DrawStyles.get_default_hand_landmarks_style(),
                                                self.mp_DrawStyles.get_default_hand_connections_style())

            # FPS Calculation
            self.cTime = time.time()
            self.FPS = 1 / (self.cTime - self.pTime)
            self.pTime = self.cTime
            # FPS display
            cv2.putText(self.my_frames, str(int(self.FPS)), (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (256, 155, 256), 2)

            cv2.imshow("Open CV Capture", self.my_frames)

            print("FPS = ", str(int(self.FPS)))
            # waitkey = 1ms after this loop enter next iteration
            cv2.waitKey(1)
