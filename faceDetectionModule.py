import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw =mp.solutions.drawing_utils
        # create face detection class instance
        self.faceDetection = self.mpFaceDetection.FaceDetection()

    def findFaces(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(results)
        bboxes = []
        if self.results.detections :
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih , iw, ic = img.shape
                bbox = int((bboxC.xmin *iw)), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2 )
        
        return img, bboxes        

    def fancyDraw(self, img, bbox, l=30, t=7, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        #Top Left x,y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        #Top Left x1,y
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        #Bottom Left x,y1
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1-l), (255, 0, 255), t)
        #Top Left x1,y1
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 255), t)

        return img 


    
    
def main():
    cap = cv2.VideoCapture("video/vid2.mp4")
    cap.set(3, 340)
    cap.set(4, 180)
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxes = detector.findFaces(img, True)

        cTime = time.time()
        fps = int(1/(cTime-pTime)) # calculate the frames per second of
        # our video and store it in a variable
        pTime=cTime
        resize = cv2.resize(img, (840, 600))
        cv2.putText(resize, f'FPS:{int(fps)}', (670, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2 )
        cv2.imshow("Image", resize)
        
        if cv2.waitKey(1) & 0xFF == 113:
            break

if __name__ == "__main__":
    main()