import cv2
import mediapipe as mp

import Hardcodes

mp_face_mesh = mp.solutions.face_mesh
mp_draw_ut = mp.solutions.drawing_utils
mp_draw_st = mp.solutions.drawing_styles

cam = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=3, refine_landmarks=True) as face_mesh:
    while cam.isOpened():
        ret, frame = cam.read()
        width = frame.shape[0]
        height = frame.shape[1]

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if Hardcodes.draw_mesh:
                    mp_draw_ut.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                              mp_draw_ut.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),
                                              mp_draw_st.get_default_face_mesh_tesselation_style())
                if Hardcodes.draw_contour:
                    mp_draw_ut.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                              None,
                                              mp_draw_st.get_default_face_mesh_contours_style())
                if Hardcodes.draw_iris:
                    mp_draw_ut.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_IRISES,
                                          None,
                                          mp_draw_st.get_default_face_mesh_iris_connections_style())

        if cv2.waitKey(5) == ord('q'):
            break

        cv2.imshow('Face detection', cv2.flip(frame, 1))

cam.release()
cv2.destroyAllWindows()
