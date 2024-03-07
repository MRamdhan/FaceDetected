import cv2

face_ref = cv2.CascadeClassifier("face_ref.xml")

camera = cv2.VideoCapture(0)

def face_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_ref.detectMultiScale(gray_frame, scaleFactor=1.1, minSize=(100, 100))
    return faces

def draw_boxes(frame):
    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        ret, frame = camera.read()
        
        if not ret:
            print("Gagal membaca frame dari kamera")
            break

        draw_boxes(frame)
        
        cv2.imshow("Detected Face", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    close_window()

if __name__ == '__main__':
    main()