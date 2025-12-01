import os
import cv2
import mediapipe as mp


# ==== Setting ====
VIDEO_PATH = "target.mp4"      
OUTPUT_ROOT = "data/target"   
PATCH_SIZE = 128               
MARGIN_SCALE = 1.6             


# Mediapipe Face Mesh reset
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# Landmark index
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                   291, 308, 324, 318, 402, 317, 14, 87, 178, 88]

LEFT_EYE_LANDMARKS  = [362, 382, 381, 380, 374, 373, 390, 249,
                       263, 466, 388, 387, 386, 385, 384, 398]

RIGHT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155,
                       133, 173, 157, 158, 159, 160, 161, 246]


def ensure_dirs():
    os.makedirs(os.path.join(OUTPUT_ROOT, "patch_lip"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "patch_eye_l"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "patch_eye_r"), exist_ok=True)


def landmarks_to_bbox(landmarks, indices, img_w, img_h):
    xs, ys = [], []
    for idx in indices:
        lm = landmarks[idx]
        xs.append(lm.x * img_w)
        ys.append(lm.y * img_h)
    return min(xs), min(ys), max(xs), max(ys)


def expand_and_square_bbox(x_min, y_min, x_max, y_max,
                           img_w, img_h, margin_scale):
    """
    (x_min, y_min, x_max, y_max): origin bbox
    img_w, img_h: size of image
    margin_scale: 1.0 = 100%, 1.6 = 160%
    """
    cx = (x_min + x_max) * 0.5   # center x
    cy = (y_min + y_max) * 0.5   # center y
    w = (x_max - x_min)
    h = (y_max - y_min)
    half = max(w, h) * 0.5 * margin_scale

    x_min = int(max(0, cx - half))
    x_max = int(min(img_w - 1, cx + half))
    y_min = int(max(0, cy - half))
    y_max = int(min(img_h - 1, cy + half))

    return x_min, y_min, x_max, y_max


def crop_and_resize(img, bbox, patch_size):
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (patch_size, patch_size))


def process_video():
    ensure_dirs()

    cap = cv2.VideoCapture(VIDEO_PATH)  # open current video file
    frame_idx = 0                       # index of current frame

    while True:
        ret, frame = cap.read()        # read next frame
        if not ret:                    # finish if last frame
            break

        img_h, img_w = frame.shape[:2]  # hight and weight of image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        results = face_mesh.process(rgb)              # export facial landmarks

        if not results.multi_face_landmarks:
            print(f"[warning] no detectable face → frame {frame_idx}")
            frame_idx += 1
            continue

        lm = results.multi_face_landmarks[0].landmark

        # 1) mouth/eyes bbox 
        mouth_box = landmarks_to_bbox(lm, MOUTH_LANDMARKS, img_w, img_h)
        eye_l_box = landmarks_to_bbox(lm, LEFT_EYE_LANDMARKS, img_w, img_h)
        eye_r_box = landmarks_to_bbox(lm, RIGHT_EYE_LANDMARKS, img_w, img_h)
 
        # 2) clamp bbox for squre + margin expand
        mouth_box = expand_and_square_bbox(*mouth_box, img_w, img_h, MARGIN_SCALE)
        eye_l_box = expand_and_square_bbox(*eye_l_box, img_w, img_h, MARGIN_SCALE)
        eye_r_box = expand_and_square_bbox(*eye_r_box, img_w, img_h, MARGIN_SCALE)

        # 3) crop patches from orign and resize to 128x128
        mouth_p = crop_and_resize(frame, mouth_box, PATCH_SIZE)
        left_p  = crop_and_resize(frame, eye_l_box, PATCH_SIZE)
        right_p = crop_and_resize(frame, eye_r_box, PATCH_SIZE)

    
        # Skip if failed
        if mouth_p is None or left_p is None or right_p is None:
            print(f"[경고] 패치 추출 실패 → frame {frame_idx}")
            frame_idx += 1
            continue

        # 4) Save the file
        name = f"frame_{frame_idx:06d}.png"
        cv2.imwrite(os.path.join(OUTPUT_ROOT, "patch_lip",   name), mouth_p)
        cv2.imwrite(os.path.join(OUTPUT_ROOT, "patch_eye_l", name), left_p)
        cv2.imwrite(os.path.join(OUTPUT_ROOT, "patch_eye_r", name), right_p)

        if frame_idx % 30 == 0:
            print(f"[Info] {frame_idx} frames processed.")

        frame_idx += 1

    print("Done!")
    cap.release()


if __name__ == "__main__":
    process_video()
