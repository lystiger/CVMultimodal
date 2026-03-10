import cv2
import torch
import numpy as np
import argparse
from collections import deque
from ultralytics import YOLO
import mediapipe as mp


# ===== LSTM MODEL =====
class HandLSTMClassifier(torch.nn.Module):
    def __init__(self, feature_dim=63, hidden=256, num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(feature_dim, hidden, num_layers=num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.ln = torch.nn.LayerNorm(hidden)
        self.classifier = torch.nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        h = self.dropout(h)
        h = self.ln(h)
        return self.classifier(h)


# ===== CONFIG =====
SEQUENCE_LENGTH = 16
FEATURE_DIM = 63
LABELS = ['Goodbye', 'Hello', 'No', 'Thank you', 'Yes']


# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ===== LANDMARK PROCESS =====
def preprocess_landmarks(landmarks):

    kp = np.array([[lm.x, lm.y] for lm in landmarks])

    wrist = kp[0]
    kp_centered = kp - wrist

    flat = []

    for p in kp_centered:
        flat.extend([p[0], p[1], 0.0])

    return np.array(flat, dtype=np.float32)


# ===== LOAD MODELS =====
def load_models(yolo_path, lstm_path):

    print("Loading YOLO...")
    yolo = YOLO(yolo_path)

    print("Loading LSTM...")
    model = HandLSTMClassifier()
    model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
    model.eval()

    return yolo, model


# ===== PROCESS VIDEO =====
def process_video(video_path, output_path, yolo, lstm):

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    buffer = deque(maxlen=SEQUENCE_LENGTH)

    gesture = "..."
    confidence = 0.0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, verbose=False)[0]

        feature = np.zeros(FEATURE_DIM, dtype=np.float32)

        if results.boxes is not None and len(results.boxes) > 0:

            box = results.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # draw YOLO box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            hand_crop = frame[y1:y2, x1:x2]

            if hand_crop.size != 0:

                rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                mp_result = hands.process(rgb)

                if mp_result.multi_hand_landmarks:

                    landmarks = mp_result.multi_hand_landmarks[0]

                    # draw landmarks
                    mp_draw.draw_landmarks(
                        hand_crop,
                        landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    feature = preprocess_landmarks(landmarks.landmark)

        buffer.append(feature)

        # ===== LSTM prediction =====
        if len(buffer) == SEQUENCE_LENGTH:

            seq = np.array(buffer, dtype=np.float32)
            seq = torch.from_numpy(seq).unsqueeze(0)

            with torch.no_grad():

                logits = lstm(seq)
                probs = torch.softmax(logits, dim=1)

            conf, idx = torch.max(probs, dim=1)

            gesture = LABELS[idx.item()]
            confidence = conf.item()

        # ===== overlay text =====
        text = f"{gesture} ({confidence:.2f})"

        cv2.putText(
            frame,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        out.write(frame)

    cap.release()
    out.release()


# ===== MAIN =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--yolo", default="best.pt")
    parser.add_argument("--lstm", default="best_model.pth")

    args = parser.parse_args()

    yolo, lstm = load_models(args.yolo, args.lstm)

    process_video(args.video, args.output, yolo, lstm)

    print("Video exported:", args.output)