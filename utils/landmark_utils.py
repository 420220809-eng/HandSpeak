import cv2
import os
import numpy as np
import pickle as pkl
import mediapipe as mp
from utils.mediapipe_utils import mediapipe_detection
import config


def landmark_to_array(mp_landmark_list):
    """Return a np array of size (nb_keypoints x 3)"""
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


def extract_landmarks(results):
    """Extract the results of both hands and convert them to a np array of size
    if a hand doesn't appear, return an array of zeros

    :param results: mediapipe object that contains the 3D position of all keypoints
    :return: Two np arrays of size (1, 21 * 3) = (1, nb_keypoints * nb_coordinates) corresponding to both hands
    """
    pose = np.zeros(99).tolist()
    if results.pose_landmarks:
        pose = landmark_to_array(results.pose_landmarks).reshape(99).tolist()

    left_hand = np.zeros(63).tolist()
    if results.left_hand_landmarks:
        left_hand = landmark_to_array(results.left_hand_landmarks).reshape(63).tolist()

    right_hand = np.zeros(63).tolist()
    if results.right_hand_landmarks:
        right_hand = (
            landmark_to_array(results.right_hand_landmarks).reshape(63).tolist()
        )
    return pose, left_hand, right_hand


def save_landmarks_from_video(video_info):
    sign_name, video_name, file_name = video_info
    landmark_list = {"pose": [], "left_hand": [], "right_hand": []}
    
    # Quality tracking
    total_frames = 0
    frames_with_hands = 0
    frames_with_left_hand = 0
    frames_with_right_hand = 0
    frames_with_pose = 0
    video_status = "OK"
    error_message = ""

    # Set the Video stream
    video_path = os.path.join(config.VIDEOS_PATH, sign_name, file_name)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {
            'sign_name': sign_name,
            'video_name': video_name,
            'status': 'ERROR',
            'error': 'Cannot open video file',
            'total_frames': 0,
            'frames_with_hands': 0,
            'frames_with_left_hand': 0,
            'frames_with_right_hand': 0,
            'frames_with_pose': 0,
            'hand_detection_rate': 0.0
        }
    
    try:
        with mp.solutions.holistic.Holistic(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE, 
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    total_frames += 1
                    
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Track detection quality
                    if results.left_hand_landmarks or results.right_hand_landmarks:
                        frames_with_hands += 1
                    if results.left_hand_landmarks:
                        frames_with_left_hand += 1
                    if results.right_hand_landmarks:
                        frames_with_right_hand += 1
                    if results.pose_landmarks:
                        frames_with_pose += 1

                    # Store results
                    pose, left_hand, right_hand = extract_landmarks(results)
                    landmark_list["pose"].append(pose)
                    landmark_list["left_hand"].append(left_hand)
                    landmark_list["right_hand"].append(right_hand)
                else:
                    break
            cap.release()
        
        # Analyze quality
        if total_frames == 0:
            video_status = "ERROR"
            error_message = "No frames extracted"
        elif frames_with_hands == 0:
            video_status = "BAD"
            error_message = "No hands detected in any frame"
        elif frames_with_hands < total_frames * 0.3:  # Less than 30% frames have hands
            video_status = "POOR"
            error_message = f"Low hand detection rate: {frames_with_hands/total_frames*100:.1f}%"
        elif frames_with_hands < total_frames * 0.7:  # Less than 70% frames have hands
            video_status = "FAIR"
            error_message = f"Moderate hand detection rate: {frames_with_hands/total_frames*100:.1f}%"
        
        # Create the folder of the sign if it doesn't exists
        path = os.path.join(config.DATASET_PATH, sign_name)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # Create the folder of the video data if it doesn't exists
        data_path = os.path.join(path, video_name)
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)

        # Saving the landmark_list in the correct folder
        save_array(
            landmark_list["pose"], os.path.join(data_path, f"pose_{video_name}.pickle")
        )
        save_array(
            landmark_list["left_hand"], os.path.join(data_path, f"lh_{video_name}.pickle")
        )
        save_array(
            landmark_list["right_hand"], os.path.join(data_path, f"rh_{video_name}.pickle")
        )
        
    except Exception as e:
        video_status = "ERROR"
        error_message = str(e)
        cap.release()
    
    # Return quality report
    hand_detection_rate = (frames_with_hands / total_frames * 100) if total_frames > 0 else 0
    
    return {
        'sign_name': sign_name,
        'video_name': video_name,
        'status': video_status,
        'error': error_message,
        'total_frames': total_frames,
        'frames_with_hands': frames_with_hands,
        'frames_with_left_hand': frames_with_left_hand,
        'frames_with_right_hand': frames_with_right_hand,
        'frames_with_pose': frames_with_pose,
        'hand_detection_rate': round(hand_detection_rate, 2)
    }


def save_array(arr, path):
    file = open(path, "wb")
    pkl.dump(arr, file)
    file.close()


def load_array(path):
    file = open(path, "rb")
    arr = pkl.load(file)
    file.close()
    return np.array(arr)
