import cv2
import torch
import numpy as np
import argparse
from collections import deque
from ultralytics import YOLO


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


# ===== PREPROCESS =====
def preprocess_landmarks(landmarks):

    kp = landmarks[:, :2]
    wrist = kp[0]
    kp_centered = kp - wrist

    flat = []
    for p in kp_centered:
        flat.extend([p[0], p[1], 0.0])

    return np.array(flat, dtype=np.float32)


# ===== LOAD MODELS =====
def load_models(yolo_path, lstm_path):

    print("Loading YOLO model...")
    yolo = YOLO(yolo_path)

    print("Loading LSTM model...")
    model = HandLSTMClassifier()
    model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
    model.eval()

    print("Models loaded successfully\n")

    return yolo, model


# ===== VIDEO PROCESS =====
def process_video(video_path, yolo, lstm):

    cap = cv2.VideoCapture(video_path)

    buffer = deque(maxlen=SEQUENCE_LENGTH)

    frame_id = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, verbose=False)[0]

        feature = np.zeros(FEATURE_DIM, dtype=np.float32)

        hand_detected = False

        if results.keypoints is not None and len(results.keypoints.data) > 0:

            raw = results.keypoints.data[0].cpu().numpy()

            if raw.shape == (21, 3):

                hand_detected = True
                feature = preprocess_landmarks(raw)

        buffer.append(feature)

        # ===== DEBUG INFO =====
        print(
            f"Frame {frame_id:04d} | "
            f"Hand: {'YES' if hand_detected else 'NO '} | "
            f"SeqFill: {len(buffer)}/{SEQUENCE_LENGTH}"
        )

        # ===== LSTM INFERENCE =====
        if len(buffer) == SEQUENCE_LENGTH:

            seq = np.array(buffer, dtype=np.float32)
            seq = torch.from_numpy(seq).unsqueeze(0)

            print("Input shape to LSTM:", seq.shape)

            with torch.no_grad():

                logits = lstm(seq)
                probs = torch.softmax(logits, dim=1)

            conf, idx = torch.max(probs, dim=1)

            print(
                f">>> Gesture: {LABELS[idx.item()]} | "
                f"Confidence: {conf.item():.3f}\n"
            )

        frame_id += 1

    cap.release()


# ===== MAIN =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--yolo", default="best.pt")
    parser.add_argument("--lstm", default="best_model.pth")

    args = parser.parse_args()

    yolo, lstm = load_models(args.yolo, args.lstm)

    process_video(args.video, yolo, lstm)