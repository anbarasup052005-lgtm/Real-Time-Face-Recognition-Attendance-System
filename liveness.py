import cv2

def is_live_face():
    cam = cv2.VideoCapture(0)
    movement_detected = 0

    prev_frame = None

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is None:
            prev_frame = gray
            continue

        diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        movement = thresh.sum()

        if movement > 500000:
            movement_detected += 1

        prev_frame = gray

        cv2.imshow("Liveness Check", frame)

        if movement_detected > 5:
            cam.release()
            cv2.destroyAllWindows()
            return True

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    return False