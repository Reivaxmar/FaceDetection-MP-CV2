import cv2
import mediapipe as mp

import Hardcodes

# Variables for the detection and drawing of the face
mp_face_mesh = mp.solutions.face_mesh
mp_draw_ut = mp.solutions.drawing_utils
mp_draw_st = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# The camera
cam = cv2.VideoCapture(0)

# Main loop
with mp_face_mesh.FaceMesh(max_num_faces=3,
                           refine_landmarks=True) as face_mesh, mp_pose.Pose() as pose, mp_hands.Hands() as hands:
    while cam.isOpened():
        # Read the camera
        ret, frame = cam.read()
        width = frame.shape[0]
        height = frame.shape[1]

        # Process the frame. The image is in BGR, and the detector in RGB, so it changes it.
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if "Face" in Hardcodes.Mode:  # If we are drawing the face...
            # Scan it
            results = face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # If it finds something
            if results.multi_face_landmarks:
                # For each 'something' that it finds
                for face_landmarks in results.multi_face_landmarks:
                    # Draws its face mesh
                    if Hardcodes.draw_mesh:
                        mp_draw_ut.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                                  mp_draw_ut.DrawingSpec(color=(255, 0, 0), thickness=1,
                                                                         circle_radius=2),
                                                  mp_draw_st.get_default_face_mesh_tesselation_style())
                    # Face contour
                    if Hardcodes.draw_contour:
                        mp_draw_ut.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                                  None,
                                                  mp_draw_st.get_default_face_mesh_contours_style())
                    # And the iris to the screen
                    if Hardcodes.draw_iris:
                        mp_draw_ut.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_IRISES,
                                                  None,
                                                  mp_draw_st.get_default_face_mesh_iris_connections_style())
        if "Body" in Hardcodes.Mode:  # If we are drawing the body...
            # Scan it
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # If there's something on the screen
            if results.pose_landmarks:
                # For each 'something' that it finds
                for i in results.pose_landmarks.landmark:
                    # Draws its pose
                    mp_draw_ut.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_draw_ut.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=6),
                                              mp_draw_st.get_default_face_mesh_tesselation_style())
        if "Hands" in Hardcodes.Mode:  # If we are drawing the hands...
            # Scan it
            results = hands.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # If there's something on the screen
            if results.multi_hand_landmarks:
                # For each 'something' that it finds
                for i in results.multi_hand_landmarks:
                    # Draws its pose
                    mp_draw_ut.draw_landmarks(frame, i, mp_hands.HAND_CONNECTIONS,
                                              mp_draw_ut.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=6),
                                              mp_draw_st.get_default_face_mesh_tesselation_style())

        # 27 is the Escape key. Press it to exit
        if cv2.waitKey(5) == 27:
            break

        # Show everything to the screen
        cv2.imshow('Face detection', cv2.flip(frame, 1))

cam.release()
cv2.destroyAllWindows()
