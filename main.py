import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_draw_ut = mp.solutions.drawing_utils
mp_draw_st = mp.solutions.drawing_styles

cam = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
    while cam.isOpened():
        ret, frame = cam.read()
        width = frame.shape[0]
        height = frame.shape[1]

        frame.flags.writeable = False

        cv2.imshow('webCam', cv2.flip(frame, 1))

        if cv2.waitKey(5) == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
