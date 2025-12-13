import cv2
import numpy as np
import mediapipe as mp
import sounddevice as sd
import time

# ======================
# 音声設定
# ======================
sample_rate = 44100

current_freq = 440.0       # 目標周波数（視線から）
smoothed_freq = 440.0      # 実際に鳴らす周波数（超平滑）
current_volume = 0.25
phase = 0.0

def audio_callback(outdata, frames, time_info, status):
    global phase, smoothed_freq, current_freq, current_volume

    # 周波数を超なめらかに追従（ノイズ除去の核心）
    smoothed_freq += (current_freq - smoothed_freq) * 0.01

    t = np.arange(frames) / sample_rate
    wave = np.sin(2 * np.pi * smoothed_freq * t + phase)

    phase += 2 * np.pi * smoothed_freq * frames / sample_rate
    phase %= 2 * np.pi

    outdata[:] = (wave * current_volume).reshape(-1, 1)

stream = sd.OutputStream(
    channels=1,
    callback=audio_callback,
    samplerate=sample_rate,
    blocksize=512
)
stream.start()

# ======================
# MediaPipe FaceMesh
# ======================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# ======================
# テルミン設定
# ======================
freq_min, freq_max = 220, 1320
vol_min, vol_max = 0.1, 0.6

prev_freq = 440.0
prev_vol = 0.25

def smooth(prev, new, a):
    return prev * (1 - a) + new * a

# ======================
# カメラ
# ======================
cap = cv2.VideoCapture(0)
print("🎵 視線テルミン（ノイズ低減版）起動")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0]

        def lm(id):
            p = face.landmark[id]
            return np.array([p.x * w, p.y * h])

        # ===== 左目 =====
        l_eye_l = lm(LEFT_EYE[0])
        l_eye_r = lm(LEFT_EYE[1])
        l_iris = np.mean([lm(i) for i in LEFT_IRIS], axis=0)

        left_norm = (l_iris[0] - l_eye_l[0]) / (l_eye_r[0] - l_eye_l[0] + 1e-6)

        # ===== 右目 =====
        r_eye_l = lm(RIGHT_EYE[1])
        r_eye_r = lm(RIGHT_EYE[0])
        r_iris = np.mean([lm(i) for i in RIGHT_IRIS], axis=0)

        right_norm = (r_eye_r[0] - r_iris[0]) / (r_eye_r[0] - r_eye_l[0] + 1e-6)

        # 視線（左右）0〜1
        gaze_x = np.clip((left_norm + right_norm) / 2, 0, 1)

        # テルミンっぽい非線形カーブ
        target_freq = freq_min + (gaze_x ** 2.2) * (freq_max - freq_min)

        # 音量（顔の上下で制御）
        face_y = (l_iris[1] + r_iris[1]) / 2
        target_vol = np.interp(face_y, [h * 0.3, h * 0.7], [vol_max, vol_min])
        target_vol = np.clip(target_vol, vol_min, vol_max)

        # ★ 視線側はやや速く、音量は超ゆっくり
        prev_freq = smooth(prev_freq, target_freq, a=0.08)
        prev_vol = smooth(prev_vol, target_vol, a=0.03)

        current_freq = prev_freq
        current_volume = prev_vol

        # デバッグ表示
        cv2.putText(frame, f"Freq: {int(current_freq)} Hz", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"GazeX: {gaze_x:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gaze Theremin (Low Noise)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

stream.stop()
stream.close()
cap.release()
cv2.destroyAllWindows()
