import cv2

# Change 0 to the correct video device number (/dev/video0 is 0, /dev/video1 is 1)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Arducam USB Feed', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
