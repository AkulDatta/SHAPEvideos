import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise


calibration_frames = 60
bg = None

def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=25):
    global bg
    
    #capturing foreground 
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return
    else:
        segmented = max(contours, key=cv2.contourArea)
        return (thresholded, segmented)

def count(thresholded, segmented):
    chull = cv2.convexHull(segmented)

    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]
    radius = int(0.8 * maximum_distance)
    circumference = (2 * np.pi * radius)

    roi_circle = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(roi_circle, (cX, cY), radius, 255, 1)
    roi_circle = cv2.bitwise_and(thresholded, thresholded, mask=roi_circle)
    contours, _ = cv2.findContours(roi_circle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
       
        #If a finger exists, increase count 
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count

if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5
    camera = cv2.VideoCapture(0)

    # coordinates of the region of interest (ROI)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    calibrated = False
    fingers_old = 0
    same_frames = 0
    isMuted = True


    while(True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 60:
            same_frames = 0
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("Calibrating background lighting for " + str(calibration_frames) + " frames")
            elif num_frames == 29:
                print("Calibration complete")
        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand

                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                
                fingers = count(thresholded, segmented)
                if(fingers == fingers_old):
                    print(same_frames)
                    same_frames += 1
                else:
                    same_frames = 0
                if(same_frames >= 40):
                    if(fingers == 1):
                        if(isMuted):
                            print("UNMUTED") 
                            cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        else:
                            print("MUTED") 
                    if(fingers == 2):
                        print("") 
                    if(fingers == 3):
                        print("VOLUME UP") 
                    if(fingers == 4):
                        print("VOLUME DOWN") 
                    if(fingers >= 5):
                        print("RAISED HAND")
                    same_frames = 0 
                fingers_old = fingers
                cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("ROI", thresholded)

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)
        
        keypress = cv2.waitKey(1) & 0xFF
        
        #recalibrates the ROI
        if keypress == ord("r"):
            num_frames = 0 
        #quits
        if keypress == ord("q"):
            break

#closing camera instance
camera.release()
cv2.destroyAllWindows()
