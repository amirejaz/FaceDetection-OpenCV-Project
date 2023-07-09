import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video/vid2.mp4")
cap.set(3, 340)
cap.set(4, 180)

mpFaceDetection = mp.solutions.face_detection
mpDraw =mp.solutions.drawing_utils
# create face detection class instance
faceDetection = mpFaceDetection.FaceDetection()


pTime = 0
while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)
    if results.detections :
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih , iw, ic = img.shape
            bbox = int((bboxC.xmin *iw)), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2 )
            # cv2.putText(img,"Person "+str(id+1),(bbox[0] + 10),(bbox[1] - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            


    cTime = time.time()
    fps = int(1/(cTime-pTime)) # calculate the frames per second of
    # our video and store it in a variable
    pTime=cTime
    resize = cv2.resize(img, (850, 600))
    cv2.putText(resize, f'FPS:{int(fps)}', (670, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2 )
    cv2.imshow("Image", resize)
    
    if cv2.waitKey(1) & 0xFF == 113:
        break
    
    