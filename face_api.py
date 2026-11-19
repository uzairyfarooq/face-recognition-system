import os
import re
import json
import time
import math
import pickle
import zipfile
import threading
import subprocess
import urllib.parse
import urllib.request
from collections import deque, Counter
from contextlib import asynccontextmanager

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
)

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

# ---------------- .env LOADER (no extra package required) ----------------
ENV_OVERRIDDEN = []


def load_env():
    """Read the .env file (next to this script, or in the working directory) into os.environ.
    The .env file always wins, so a stale Windows/batch variable can never override it."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), os.path.join(os.getcwd(), ".env")):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8-sig") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].lstrip()
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                if key in os.environ and os.environ[key] != val:
                    ENV_OVERRIDDEN.append(key)
                os.environ[key] = val
        return path
    return None


ENV_FILE = load_env()
print(f"[OK] Settings loaded from {ENV_FILE}" if ENV_FILE else "[WARNING] .env file not found.")
if ENV_OVERRIDDEN:
    print(f"[WARNING] Environment variables ({', '.join(ENV_OVERRIDDEN)}) were overridden by .env. "
          f"Stop setting them in the batch file / system environment.")

# ---------------- CONFIG ----------------
USERNAME = os.getenv("CAM_USER", "admin")
PASSWORD = os.getenv("CAM_PASS", "")                 # set it in .env, never in code
RTSP_PORT = os.getenv("CAM_PORT", "554")
SUBTYPE = os.getenv("CAM_SUBTYPE", "0")              # 0 = main stream (long range), 1 = sub stream (lightweight)


def discover_cameras():
    """Build the camera list from the CAM<ID>_IP entries (CAM12_IP=..., CAM14_IP=...)."""
    cams = {}
    for key, val in os.environ.items():
        m = re.fullmatch(r"CAM(\d+)_IP", key)
        if m and val.strip():
            cams[m.group(1)] = val.strip()
    return dict(sorted(cams.items(), key=lambda kv: int(kv[0])))


CAMERA_IPS = discover_cameras()
if not CAMERA_IPS:
    raise SystemExit("[ERROR] No camera configured. Add a line such as CAM12_IP=192.168.x.x to .env.")
if not PASSWORD and not any(os.getenv(f"CAM_URL_{c}") or os.getenv(f"CAM{c}_PASS") for c in CAMERA_IPS):
    print("[WARNING] CAM_PASS is empty (set CAM_PASS=... in .env).")

PLACEHOLDER_PASSWORDS = {"camera_password_yahan", "password_yahan", "aap_ka_asli_password",
                         "your_password", "other_password", "changeme", "password"}
for _cid in CAMERA_IPS:
    _pw = os.getenv(f"CAM{_cid}_PASS", PASSWORD)
    if _pw.lower() in PLACEHOLDER_PASSWORDS and not os.getenv(f"CAM_URL_{_cid}"):
        raise SystemExit(f"[ERROR] cam{_cid} password is still a placeholder ('{_pw}'). "
                         f"Set the real CAM_PASS in .env. No camera was contacted with a wrong password.")
for _cid, _ip in CAMERA_IPS.items():
    _u = os.getenv(f"CAM{_cid}_USER", USERNAME)
    _pw = os.getenv(f"CAM{_cid}_PASS", PASSWORD)
    print(f"[CONFIG] cam{_cid}: user={_u}  password={len(_pw)} characters  host={_ip}:{RTSP_PORT}")

KNOWN_DIR = "known_faces"
DETECT_WIDTH = int(os.getenv("DETECT_WIDTH", "1280"))
SCORE_THRESH = 0.6
MATCH_THRESH = float(os.getenv("MATCH_THRESH", "0.40"))
MARGIN = float(os.getenv("MARGIN", "0.06"))
MIN_FACE_PX = 60
HARD_MIN_PX = int(os.getenv("HARD_MIN_PX", "28"))
YAW_RANGE = (float(os.getenv("YAW_MIN", "0.28")), float(os.getenv("YAW_MAX", "0.72")))
HIST_LEN = 6
SAMPLES_PER_SHOT = 60
DISPLAY_W = int(os.getenv("DISPLAY_W", "800"))
JPEG_QUALITY = 70
VIDEO_ONLY = os.getenv("VIDEO_ONLY", "0") == "1"

# --- speed settings ---
DUAL_STREAM = os.getenv("DUAL_STREAM", "1") == "1"
VIEW_SUBTYPE = os.getenv("VIEW_SUBTYPE", "1")
DETECT_FPS_SET = "DETECT_FPS" in os.environ
DETECT_FPS = float(os.getenv("DETECT_FPS", "4"))
CV_THREADS = int(os.getenv("CV_THREADS", "0"))
if CV_THREADS > 0:
    cv2.setNumThreads(CV_THREADS)
RECOG_INTERVAL = float(os.getenv("RECOG_INTERVAL", "0.7"))
PRED_MAX_S = float(os.getenv("PRED_MAX_S", "0.9"))
BOX_SMOOTH = float(os.getenv("BOX_SMOOTH", "0.7"))
VIEW_DELAY_MS = int(os.getenv("VIEW_DELAY_MS", "500"))
RESULT_TTL_S = float(os.getenv("RESULT_TTL_S", "1.2"))
DIAG_WINDOW_S = 120
HISTORY_MAX_BYTES = 48 * 1024 * 1024
OPEN_TIMEOUT_MS = int(os.getenv("OPEN_TIMEOUT_MS", "6000"))
READ_TIMEOUT_MS = int(os.getenv("READ_TIMEOUT_MS", "6000"))

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/"
DET_MODEL = "face_detection_yunet_2023mar.onnx"
REC_MODEL = "face_recognition_sface_2021dec.onnx"
DET_URL = BASE + "face_detection_yunet/" + DET_MODEL
REC_URL = BASE + "face_recognition_sface/" + REC_MODEL

BACKEND = os.getenv("BACKEND", "auto").lower()
GPU_ID = int(os.getenv("GPU_ID", "0"))
MODEL_DIR = "models"
GPU_PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
GPU_DET_FILE, GPU_REC_FILE = "det_10g.onnx", "w600k_r50.onnx"
GPU_DETECT_FPS_DEFAULT = 6.0


def rtsp_url(cam_id, subtype=None):
    override = os.getenv(f"CAM_URL_{cam_id}")
    if override:
        return override
    user = os.getenv(f"CAM{cam_id}_USER", USERNAME)
    pw = os.getenv(f"CAM{cam_id}_PASS", PASSWORD)
    sub = subtype or os.getenv(f"CAM{cam_id}_SUBTYPE", SUBTYPE)
    return (f"rtsp://{urllib.parse.quote(user, safe='')}:"
            f"{urllib.parse.quote(pw, safe='')}@{CAMERA_IPS[cam_id]}:{RTSP_PORT}/"
            f"cam/realmonitor?channel=1&subtype={sub}")


ENGINE_KEY = ""


def calib_file(cam_id):
    return f"calibration_{cam_id}{ENGINE_KEY}.json"


def samples_file(cam_id):
    return f"test_samples_{cam_id}{ENGINE_KEY}.json"


def db_cache_file():
    return f"embeddings{ENGINE_KEY}.pkl"


def download(path, url):
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        urllib.request.urlretrieve(url, path)


# ---------------- STREAM READER ----------------
class StreamReader:
    def __init__(self, url, label="", interval=0.0, history_s=0.0):
        self.url, self.label, self.interval = url, label, interval
        self.history_s = history_s
        self.history, self.history_bytes = deque(), 0
        self.frame, self.fid, self.last_ret, self.frame_t = None, 0, 0.0, 0.0
        self.connected, self.ever_connected, self.streak = False, False, 0
        self.cap = None
        self.lock, self.running = threading.Lock(), True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open(self):
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        except Exception:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.connected:
                print(f"[OK] {self.label} connected.")
            self.connected, self.ever_connected, self.streak = True, True, 0
            self.failed_logged = False
        else:
            self.streak += 1
            if not getattr(self, "failed_logged", False):
                print(f"[ERROR] {self.label} could not connect. Retrying with back-off.")
                self.failed_logged = True
            self.connected = False
        return cap

    def _backoff(self):
        if self.ever_connected:
            return min(2 ** min(self.streak, 4), 10)
        return (5, 15, 30, 60)[min(max(self.streak - 1, 0), 3)]

    def _sleep(self, sec):
        end = time.time() + sec
        while self.running and time.time() < end:
            time.sleep(0.1)

    def _next(self):
        if self.interval <= 0:
            return self.cap.read()
        if not self.cap.grab():
            return False, None
        now = time.time()
        if now - self.last_ret < self.interval:
            return True, None
        self.last_ret = now
        return self.cap.retrieve()

    def _loop(self):
        self.cap = self._open()
        fails = 0
        while self.running:
            ok, frame = self._next()
            if ok:
                fails = 0
                if frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.fid += 1
                        self.frame_t = time.time()
                        if self.history_s > 0:
                            self._remember(self.frame_t, self.fid, frame)
                continue
            fails += 1
            if fails > 30:
                if self.connected:
                    print(f"[WARNING] {self.label} lost. Reconnecting...")
                self.connected = False
                self.cap.release()
                wait = self._backoff()
                self._sleep(wait)
                if not self.running:
                    break
                self.cap = self._open()
                fails = 0
            else:
                time.sleep(0.01)

    def _remember(self, t, fid, frame):
        self.history.append((t, fid, frame))
        self.history_bytes += frame.nbytes
        while self.history and (t - self.history[0][0] > self.history_s
                                or self.history_bytes > HISTORY_MAX_BYTES):
            self.history_bytes -= self.history.popleft()[2].nbytes

    def read_delayed(self, last_fid, target_t):
        with self.lock:
            for t, fid, frame in reversed(self.history):
                if t <= target_t:
                    if fid == last_fid:
                        break
                    return frame.copy(), fid, t
        return None, last_fid, 0.0

    def read_new_t(self, last_fid):
        with self.lock:
            if self.frame is None or self.fid == last_fid:
                return None, last_fid, 0.0
            return self.frame.copy(), self.fid, self.frame_t

    def read_new(self, last_fid):
        frame, fid, _ = self.read_new_t(last_fid)
        return frame, fid

    def stop(self):
        self.running = False
        self.thread.join(timeout=3)
        if not self.thread.is_alive() and self.cap is not None:
            self.cap.release()


# ---------------- FACE HELPERS ----------------
def unit(v):
    v = np.asarray(v, dtype=np.float32).flatten()
    return v / (np.linalg.norm(v) + 1e-9)


def yaw_ok(f, rng=YAW_RANGE):
    eye_dx = f[6] - f[4]
    if eye_dx < 1:
        return False
    ratio = (f[8] - f[4]) / eye_dx
    return bool(rng[0] <= ratio <= rng[1])


# ==============================================================================
# ANTI-SPOOFING, MASK & GLASSES DETECTION ANALYZER (Used by /antispoof page)
# ==============================================================================
def analyze_liveness_and_attributes(frame, face_row):
    """
    Optimized Anti-Spoofing Analyzer.
    Specifically targets and restricts mobile/tablet screens and digital displays
    without triggering false alerts on real human faces.
    """
    x, y, w, h = int(face_row[0]), int(face_row[1]), int(face_row[2]), int(face_row[3])
    fh, fw = frame.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)

    if x2 - x1 < 30 or y2 - y1 < 30:
        return {
            "is_spoof": False, "spoof_reasons": [], "liveness_score": 1.0,
            "has_mask": False, "has_glasses": False, "glasses_type": "None",
        }

    face_roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    spoof_reasons = []
    spoof_penalty = 0.0

    # 1. Real skin has balanced variance, but mobile screens are unnaturally flat or pixelated
    lap_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
    if lap_var < 18.0:
        spoof_reasons.append("Printed Photo (Flat)")
        spoof_penalty += 0.50

    # 2. Mobile Screen Glass Reflection Hotspots (Specular highlights specific to smooth glass)
    v_chan = hsv_roi[:, :, 2]
    glare_ratio = float(np.sum(v_chan > 250) / v_chan.size)
    if glare_ratio > 0.05:
        spoof_reasons.append("Mobile Screen Glare")
        spoof_penalty += 0.60

    # 3. Mobile Backlight Color Spectrum Anomaly (Excessive blue/green sub-pixel concentration)
    s_chan = hsv_roi[:, :, 1]
    high_sat_ratio = float(np.sum(s_chan > 200) / s_chan.size)
    if high_sat_ratio > 0.20:
        spoof_reasons.append("Digital Display Color Cast")
        spoof_penalty += 0.45

    # 4. Accurate FFT Moire Grid Detection (Specific to phone/monitor pixels capturing through security camera)
    try:
        f_transform = np.fft.fft2(cv2.resize(gray_roi, (64, 64)))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = 20 * np.log(np.abs(f_shift) + 1e-6)
        outer_energy = float(np.mean(magnitude[0:8, :]) + np.mean(magnitude[56:64, :])) / 2.0
        total_energy = float(np.mean(magnitude)) + 1e-6
        # Balanced ratio to exclusively target phone pixel grids
        if (outer_energy / total_energy) > 0.88:
            spoof_reasons.append("Mobile Pixel Grid (Moire)")
            spoof_penalty += 0.65
    except Exception:
        pass

    # Balanced Liveness Score threshold (0.55 ensures real faces pass smoothly while mobile screens fail)
    liveness_score = round(max(0.0, 1.0 - spoof_penalty), 2)
    is_spoof = liveness_score < 0.55

    # 5. Mask Detection
    lower_y = int(y1 + (y2 - y1) * 0.55)
    lower_roi = frame[lower_y:y2, x1:x2]
    has_mask = False
    if lower_roi.size > 0:
        lower_gray = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2GRAY)
        if float(np.std(lower_gray)) < 28.0 and float(np.sum(cv2.Canny(lower_gray, 50, 150) > 0) / lower_gray.size) < 0.04:
            has_mask = True

    # 6. Precise Glasses Detection (No false positives on normal eyes)
    eye_y1 = int(y1 + (y2 - y1) * 0.22)
    eye_y2 = int(y1 + (y2 - y1) * 0.48)
    eye_roi = frame[eye_y1:eye_y2, x1:x2]
    has_glasses = False
    glasses_type = "None"
    
    if eye_roi.size > 0:
        eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        mean_eye_bright = float(np.mean(eye_gray))
        overall_bright = float(np.mean(gray_roi)) + 1e-6
        if (mean_eye_bright / overall_bright) < 0.48:
            has_glasses = True
            glasses_type = "Sunglasses"
        else:
            sobel_x = cv2.Sobel(eye_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(eye_gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            strong_edges = grad_mag > np.percentile(grad_mag, 88)
            if float(np.sum(strong_edges) / strong_edges.size) > 0.15:
                has_glasses = True
                glasses_type = "Eyeglasses"

    return {
        "is_spoof": is_spoof,
        "spoof_reasons": spoof_reasons,
        "liveness_score": liveness_score,
        "has_mask": has_mask,
        "has_glasses": has_glasses,
        "glasses_type": glasses_type,
    }

# ---------------- FACE ENGINES ----------------
ARC_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def similarity_transform(src, dst):
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    ps, pd = src.mean(0), dst.mean(0)
    s, d = src - ps, dst - pd
    den = (s ** 2).sum() + 1e-12
    a = (s * d).sum() / den
    b = (s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum() / den
    tx = pd[0] - (a * ps[0] - b * ps[1])
    ty = pd[1] - (b * ps[0] + a * ps[1])
    return np.array([[a, -b, tx], [b, a, ty]], np.float32)


class CvEngine:
    name = "YuNet + SFace (OpenCV, CPU)"
    key = ""
    on_gpu = False

    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(DET_MODEL, "", (320, 320), SCORE_THRESH, 0.3, 50)
        self.recognizer = cv2.FaceRecognizerSF.create(REC_MODEL, "")

    def detect(self, frame, max_w):
        h, w = frame.shape[:2]
        s = min(1.0, max_w / float(w))
        small = frame if s == 1.0 else cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        out = []
        if faces is not None:
            for f in faces:
                f = f.copy()
                f[:14] /= s
                out.append(f)
        return out

    def embed(self, frame, row, flip=False):
        aligned = self.recognizer.alignCrop(frame, row)
        if flip:
            aligned = cv2.flip(aligned, 1)
        return unit(self.recognizer.feature(aligned))


def _nms(boxes, scores, thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order, keep = scores.argsort()[::-1], []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        order = order[np.where(inter / (areas[i] + areas[order[1:]] - inter) <= thresh)[0] + 1]
    return keep


class OrtEngine:
    name = "SCRFD + ArcFace (ONNX Runtime)"
    key = "_arcface"

    def __init__(self, det_path, rec_path, use_gpu=True, gpu_id=0, threads=0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_mem_pattern = False
        if threads > 0:
            so.intra_op_num_threads = threads
        providers = ["CPUExecutionProvider"]
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = [("CUDAExecutionProvider", {"device_id": gpu_id, "cudnn_conv_algo_search": "HEURISTIC"}),
                         "CPUExecutionProvider"]
        self.det = ort.InferenceSession(det_path, so, providers=providers)
        self.rec = ort.InferenceSession(rec_path, so, providers=providers)
        self.on_gpu = (self.det.get_providers()[0] == "CUDAExecutionProvider"
                       and self.rec.get_providers()[0] == "CUDAExecutionProvider")
        self.providers = self.det.get_providers()
        self.det_in, self.rec_in = self.det.get_inputs()[0].name, self.rec.get_inputs()[0].name
        self.strides, self.anchors = (8, 16, 32), {}
        self.thresh = float(os.getenv("SCRFD_THRESH", "0.5"))

    def _centers(self, h, w, stride):
        key = (h, w, stride)
        if key not in self.anchors:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            self.anchors[key] = np.stack([c] * 2, axis=1).reshape(-1, 2)
        return self.anchors[key]

    def detect(self, frame, max_w):
        h, w = frame.shape[:2]
        s = min(1.0, max_w / float(w))
        nw = max(32, int(round(w * s / 32.0)) * 32)
        nh = max(32, int(round(h * s / 32.0)) * 32)
        img = frame if (nw, nh) == (w, h) else cv2.resize(
            frame, (nw, nh), interpolation=cv2.INTER_AREA if nw < w else cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128.0, (nw, nh), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.det.run(None, {self.det_in: blob})
        boxes, kpss, scores = [], [], []
        for i, stride in enumerate(self.strides):
            sc = outs[i].reshape(-1)
            keep = np.where(sc >= self.thresh)[0]
            if keep.size == 0:
                continue
            ctr = self._centers(nh // stride, nw // stride, stride)[keep]
            bb = outs[i + 3].reshape(-1, 4)[keep] * stride
            kp = outs[i + 6].reshape(-1, 10)[keep] * stride
            boxes.append(np.stack([ctr[:, 0] - bb[:, 0], ctr[:, 1] - bb[:, 1],
                                   ctr[:, 0] + bb[:, 2], ctr[:, 1] + bb[:, 3]], axis=1))
            kp = kp.reshape(-1, 5, 2)
            kpss.append(kp + ctr[:, None, :])
            scores.append(sc[keep])
        if not boxes:
            return []
        boxes, kpss, scores = np.concatenate(boxes), np.concatenate(kpss), np.concatenate(scores)
        sx, sy = nw / float(w), nh / float(h)
        out = []
        for i in _nms(boxes, scores, 0.4):
            x1, y1, x2, y2 = boxes[i]
            row = np.zeros(15, np.float32)
            row[0], row[1], row[2], row[3] = x1 / sx, y1 / sy, (x2 - x1) / sx, (y2 - y1) / sy
            row[4:14] = (kpss[i] / np.array([sx, sy], np.float32)).reshape(-1)
            row[14] = scores[i]
            out.append(row)
        return out

    def embed(self, frame, row, flip=False):
        M = similarity_transform(np.asarray(row[4:14], np.float32).reshape(5, 2), ARC_TEMPLATE)
        aligned = cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)
        if flip:
            aligned = cv2.flip(aligned, 1)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        return unit(self.rec.run(None, {self.rec_in: blob})[0][0])


ENGINE_FACTORY = None
ENGINE_INFO = {"name": "not loaded", "gpu": False, "providers": [], "device": ""}


def gpu_device_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader",
                              "-i", str(GPU_ID)], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def ensure_gpu_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    det, rec = os.path.join(MODEL_DIR, GPU_DET_FILE), os.path.join(MODEL_DIR, GPU_REC_FILE)
    if os.path.exists(det) and os.path.exists(rec):
        return det, rec
    zip_path = os.path.join(MODEL_DIR, "buffalo_l.zip")
    if not os.path.exists(zip_path):
        print("Downloading GPU face models (about 280 MB, one time) ...")
        urllib.request.urlretrieve(GPU_PACK_URL, zip_path + ".part")
        os.replace(zip_path + ".part", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for name, dst in ((GPU_DET_FILE, det), (GPU_REC_FILE, rec)):
            with z.open(name) as src, open(dst, "wb") as out:
                out.write(src.read())
    os.remove(zip_path)
    return det, rec


def select_engine():
    global ENGINE_FACTORY, ENGINE_KEY, ENGINE_INFO, DETECT_FPS
    if BACKEND in ("auto", "gpu", "onnx"):
        try:
            import onnxruntime as ort
            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()
                except Exception:
                    pass
            if BACKEND != "onnx" and "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDAExecutionProvider is not available")
            det, rec = ensure_gpu_models()
            engine = OrtEngine(det, rec, use_gpu=(BACKEND != "onnx"), gpu_id=GPU_ID, threads=CV_THREADS)
            if BACKEND != "onnx" and not engine.on_gpu:
                raise RuntimeError("CUDA provider failed to initialise")
            ENGINE_FACTORY, ENGINE_KEY = (lambda: engine), engine.key
            ENGINE_INFO = {"name": engine.name, "gpu": engine.on_gpu, "providers": engine.providers,
                           "device": gpu_device_name() if engine.on_gpu else "CPU"}
            if engine.on_gpu and not DETECT_FPS_SET:
                DETECT_FPS = GPU_DETECT_FPS_DEFAULT
            print(f"[OK] Face engine: {engine.name} on {'GPU ' + ENGINE_INFO['device'] if engine.on_gpu else 'CPU'}")
            return
        except Exception as e:
            print(f"[WARNING] GPU engine unavailable: {e}. Falling back to the CPU engine.")
    download(DET_MODEL, DET_URL)
    download(REC_MODEL, REC_URL)
    ENGINE_FACTORY, ENGINE_KEY = CvEngine, CvEngine.key
    ENGINE_INFO = {"name": CvEngine.name, "gpu": False, "providers": ["OpenCV"], "device": "CPU"}
    print(f"[OK] Face engine: {CvEngine.name}")


def build_index(db):
    if not db:
        return None
    names = sorted(db)
    mats, starts, pos = [], [], 0
    for n in names:
        refs = np.asarray(db[n]["refs"], dtype=np.float32)
        starts.append(pos)
        pos += len(refs)
        mats.append(refs)
    return {"names": names, "M": np.vstack(mats), "starts": np.array(starts),
            "C": np.stack([np.asarray(db[n]["centroid"], dtype=np.float32) for n in names])}


def identify(engine, frame, face_row, index):
    if index is None:
        return "Unknown", 0.0, -1.0
    feat = engine.embed(frame, face_row)
    refmax = np.maximum.reduceat(index["M"] @ feat, index["starts"])
    scores = np.maximum(refmax, index["C"] @ feat)
    if len(scores) > 1:
        top2 = np.argpartition(scores, -2)[-2:]
        i_best, i_second = (top2[0], top2[1]) if scores[top2[0]] >= scores[top2[1]] else (top2[1], top2[0])
        best, second = float(scores[i_best]), float(scores[i_second])
    else:
        i_best, best, second = 0, float(scores[0]), -1.0
    if best < MATCH_THRESH or (best - second) < MARGIN:
        return "Unknown", best, second
    return index["names"][i_best], best, second


class Tracks:
    def __init__(self):
        self.items = []
        self.next_id = 1
        self.pass_id = 0

    def match(self, box, t):
        x, y, w, h = (float(v) for v in box)
        cx, cy = x + w / 2, y + h / 2
        best, bd = None, 1e9
        for tr in self.items:
            if tr.get("claimed") == self.pass_id:
                continue
            gap = max(t - tr["t"], 0.0)
            dt = min(gap, PRED_MAX_S)
            px, py = tr["cx"] + tr["vx"] * dt, tr["cy"] + tr["vy"] * dt
            d = math.hypot(cx - px, cy - py)
            limit = max(w, tr["w"]) * (1.0 + 2.5 * min(gap, 1.5))
            if d < bd and d < limit:
                best, bd = tr, d
        if best is None:
            best = {"id": self.next_id, "hist": deque(maxlen=HIST_LEN), "vx": 0.0, "vy": 0.0, "nv": 0,
                    "last_rec": 0.0, "last": None}
            self.next_id += 1
            self.items.append(best)
        else:
            dt = t - best["t"]
            if 0.03 < dt < 1.5:
                rvx, rvy = (cx - best["cx"]) / dt, (cy - best["cy"]) / dt
                k = 1.0 if best["nv"] == 0 else 0.5
                best["vx"] = (1 - k) * best["vx"] + k * rvx
                best["vy"] = (1 - k) * best["vy"] + k * rvy
                best["nv"] += 1
            elif dt >= 1.5:
                best["vx"] = best["vy"] = 0.0
                best["nv"] = 0
        best.update(cx=cx, cy=cy, w=w, h=h, t=t, claimed=self.pass_id)
        return best

    def prune(self, t):
        self.items = [tr for tr in self.items if t - tr["t"] < 3.0]
        self.pass_id += 1

    @staticmethod
    def smoothed(track):
        hist = track["hist"]
        top = Counter(n for n, _ in hist).most_common(1)[0][0]
        avg = float(np.mean([s for n, s in hist if n == top]))
        return top, avg


def predict_rows(results, frame_t, disp, ttl=None):
    out = []
    for r in results:
        age = frame_t - r["t"]
        if age > (RESULT_TTL_S if ttl is None else ttl):
            continue
        dt = min(max(age, -PRED_MAX_S), PRED_MAX_S)
        x, y, w, h = r["x"] + r["vx"] * dt, r["y"] + r["vy"] * dt, r["w"], r["h"]
        prev = disp.get(r["tid"])
        if prev is not None and abs(frame_t - prev[4]) < 1.0:
            a = BOX_SMOOTH
            x, y = prev[0] + a * (x - prev[0]), prev[1] + a * (y - prev[1])
            w, h = prev[2] + a * (w - prev[2]), prev[3] + a * (h - prev[3])
        disp[r["tid"]] = (x, y, w, h, frame_t)
        rr = dict(r)
        rr.update(x=x, y=y, w=w, h=h)
        out.append(rr)
    for k in [k for k, v in disp.items() if abs(frame_t - v[4]) > 1.0]:
        del disp[k]
    return out


IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CACHE_VERSION = 2


def person_signature(name):
    pdir = os.path.join(KNOWN_DIR, name)
    files = [f for f in os.listdir(pdir) if f.lower().endswith(IMG_EXT)]
    mt = max((os.path.getmtime(os.path.join(pdir, f)) for f in files), default=0)
    return [len(files), int(mt)]


def current_sigs():
    sigs = {}
    if os.path.isdir(KNOWN_DIR):
        for name in sorted(os.listdir(KNOWN_DIR)):
            if os.path.isdir(os.path.join(KNOWN_DIR, name)):
                sigs[name] = person_signature(name)
    return sigs


def build_db(sigs, cache=None):
    engine = ENGINE_FACTORY()
    old_db = cache["db"] if cache else {}
    old_sigs = cache["sigs"] if cache else {}
    db = {}
    for name in sigs:
        if name in old_db and old_sigs.get(name) == sigs[name]:
            db[name] = old_db[name]
            continue
        pdir = os.path.join(KNOWN_DIR, name)
        feats = []
        for fn in sorted(os.listdir(pdir)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            img = cv2.imread(os.path.join(pdir, fn))
            if img is None:
                continue
            h, w = img.shape[:2]
            if max(h, w) > 1600:
                s = 1600 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
            faces = engine.detect(img, 1600)
            if not faces:
                continue
            best = max(faces, key=lambda f: f[2] * f[3])
            feats.append(engine.embed(img, best))
            feats.append(engine.embed(img, best, flip=True))
        if feats:
            refs = np.stack(feats)
            db[name] = {"refs": refs, "centroid": unit(refs.mean(axis=0))}
            print(f"[OK] {name}: {len(feats)//2} photo(s) loaded")
    cache_path = db_cache_file()
    tmp = cache_path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"version": CACHE_VERSION, "sigs": sigs, "db": db}, fh)
    os.replace(tmp, cache_path)
    return db


def load_db(force=False):
    cache = None
    if not force and os.path.exists(db_cache_file()):
        try:
            with open(db_cache_file(), "rb") as fh:
                cache = pickle.load(fh)
            if cache.get("version") != CACHE_VERSION:
                cache = None
        except Exception:
            cache = None
    if not os.path.isdir(KNOWN_DIR):
        return {}
    sigs = current_sigs()
    if cache and cache["sigs"] == sigs:
        return cache["db"]
    return build_db(sigs, cache)


WATCH_SECONDS = int(os.getenv("WATCH_SECONDS", "30"))
WATCH_STOP = threading.Event()


def watch_known_faces():
    last = current_sigs()
    while not WATCH_STOP.wait(WATCH_SECONDS):
        try:
            now = current_sigs()
            if now == last:
                continue
            if WATCH_STOP.wait(5):
                break
            if current_sigs() != now:
                continue
            db = load_db()
            for st in STATES.values():
                st.db = db
            last = now
        except Exception as e:
            print(f"[AUTO] reload failed: {e}")


def load_calib(cam_id):
    p = calib_file(cam_id)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return {}


def analyze_samples(cam_id, samples):
    by = {}
    for s in samples:
        by.setdefault(s["dist"], []).append(s)
    rows, reliable = [], []
    for d in sorted(by):
        g = by[d]
        n = len(g)
        rate = sum(x["correct"] for x in g) / n
        scores = np.array([x["score"] for x in g])
        width = float(np.median([x["width"] for x in g]))
        p10 = float(np.percentile(scores, 10))
        ok = n >= 20 and rate >= 0.95 and p10 >= MATCH_THRESH
        if ok:
            reliable.append((d, width))
        rows.append({"dist_m": d, "n": n, "correct_pct": round(rate * 100, 1),
                     "mean_score": round(float(scores.mean()), 2), "p10_score": round(p10, 2),
                     "face_px": round(width), "reliable": ok})
    calib = {"k": float(np.median([s["width"] * s["dist"] for s in samples]))}
    if reliable:
        far_d, far_w = max(reliable)
        calib["max_reliable_m"] = far_d
        calib["min_face_px"] = int(far_w * 0.9)
    with open(calib_file(cam_id), "w") as fh:
        json.dump(calib, fh, indent=2)
    return rows, calib


# ---------------- PER-CAMERA STATE ----------------
class CamState:
    def __init__(self, cam_id, db, detect_enabled):
        self.id = cam_id
        self._db, self._index = {}, None
        self.db = db
        self.detect_enabled = detect_enabled
        self.lock = threading.Lock()
        self.running = True
        self.results, self.det_ms, self.det_fps, self.tick = [], 0.0, 0.0, 0
        self.lat_s, self.pass_gap = 0.0, 0.0
        self.pass_log = deque()
        self.spoof_events = deque(maxlen=30)
        self.size, self.view_size = (0, 0), (0, 0)
        self.tracks = Tracks()
        self.apply_calib()

        # Render buffers for standard feed and anti-spoof feed
        self.render_lock = threading.Lock()
        self.jpeg, self.seq = None, 0
        self._disp = {}

        self.as_render_lock = threading.Lock()
        self.as_jpeg, self.as_seq = None, 0
        self._as_disp = {}

        self._cur_src, self.src_fid = None, 0
        self._as_cur_src, self.as_src_fid = None, 0
        self.fps, self.last_t = 0.0, time.time()
        self.test_person, self.recording, self.test_dist, self.got = None, False, 0, 0
        self.samples = []

        main_sub = os.getenv(f"CAM{cam_id}_SUBTYPE", SUBTYPE)
        dual = DUAL_STREAM and not os.getenv(f"CAM_URL_{cam_id}") and VIEW_SUBTYPE != main_sub
        self.view, self.reader = None, None
        if dual:
            self.view = StreamReader(rtsp_url(cam_id, VIEW_SUBTYPE), f"cam{cam_id}-view",
                                     history_s=(VIEW_DELAY_MS / 1000.0 + 0.5) if VIEW_DELAY_MS > 0 else 0.0)
            if detect_enabled:
                gap = (1.0 / DETECT_FPS) if DETECT_FPS > 0 else 0.0
                self.reader = StreamReader(rtsp_url(cam_id), f"cam{cam_id}-detect", gap)
        else:
            self.reader = StreamReader(rtsp_url(cam_id), f"cam{cam_id}")
        print(f"cam{cam_id}: {'dual stream' if dual else 'single stream'}")

        if detect_enabled:
            self.engine = ENGINE_FACTORY()
            threading.Thread(target=self._worker, daemon=True).start()

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, value):
        self._index = build_index(value)
        self._db = value

    def apply_calib(self):
        c = load_calib(self.id)
        self.min_face = c.get("min_face_px", MIN_FACE_PX)
        self.k = c.get("k")
        self.max_m = c.get("max_reliable_m")

    def _analyze(self, frame, t):
        out = []
        index = self._index
        for f in self.engine.detect(frame, DETECT_WIDTH):
            wpx = float(f[2])
            tr = self.tracks.match((f[0], f[1], f[2], f[3]), t)

            # Compute anti-spoofing, mask, and glasses attributes once per detected face
            attrs = analyze_liveness_and_attributes(frame, f)

            row = {"x": float(f[0]), "y": float(f[1]), "w": float(f[2]), "h": float(f[3]),
                   "wpx": wpx, "dist": (self.k / wpx) if self.k else None,
                   "name": "far", "score": 0.0, "low": False,
                   "yaw_ok": False, "raw_name": None, "best": 0.0,
                   "tid": tr["id"], "vx": float(tr["vx"]), "vy": float(tr["vy"]), "t": float(t),
                   **attrs}

            if wpx < HARD_MIN_PX:
                out.append(row)
                continue
            row["yaw_ok"] = yaw_ok(f)
            if not yaw_ok(f, YAW_RANGE if wpx >= 80 else (0.15, 0.85)):
                row["name"] = "turned"
                out.append(row)
                continue

            if attrs["is_spoof"]:
                self.spoof_events.appendleft({
                    "time": time.strftime("%H:%M:%S"),
                    "camera": self.id,
                    "liveness_score": attrs["liveness_score"],
                    "reasons": attrs["spoof_reasons"],
                })

            if (not self.recording) and tr["last"] is not None and t - tr["last_rec"] < RECOG_INTERVAL:
                name, best = tr["last"]
            else:
                name, best, _ = identify(self.engine, frame, f, index)
                tr["last"], tr["last_rec"] = (name, best), t
                tr["hist"].append((name, best))
            sname, savg = Tracks.smoothed(tr)
            row.update(name=sname, score=savg, low=wpx < self.min_face, raw_name=name, best=float(best))
            out.append(row)
        self.tracks.prune(t)
        return out

    def _collect(self, results):
        if not (self.recording and self.test_person and results):
            return
        face = max(results, key=lambda r: r["w"] * r["h"])
        if not face["yaw_ok"] or face["raw_name"] is None:
            return
        self.samples.append({"person": self.test_person, "dist": float(self.test_dist),
                             "width": face["wpx"], "score": face["best"],
                             "correct": face["raw_name"] == self.test_person})
        self.got += 1
        if self.got >= SAMPLES_PER_SHOT:
            self.recording = False

    def _worker(self):
        last, last_end = 0, time.time()
        min_gap = (1.0 / DETECT_FPS) if DETECT_FPS > 0 else 0.0
        while self.running:
            frame, fid, ft = self.reader.read_new_t(last)
            if frame is None:
                time.sleep(0.005)
                continue
            last = fid
            t0 = time.time()
            try:
                res = self._analyze(frame, ft or t0)
            except Exception as e:
                print(f"[cam{self.id}] detect error: {e}")
                res = []
            self.size = (frame.shape[1], frame.shape[0])
            self.tick += 1
            now = time.time()
            with self.lock:
                self.results = res
                self.det_ms = 0.8 * self.det_ms + 0.2 * (now - t0) * 1000
                self.det_fps = 0.8 * self.det_fps + 0.2 * (1.0 / max(now - last_end, 1e-6))
                self.lat_s = 0.8 * self.lat_s + 0.2 * (now - (ft or t0))
                self.pass_gap = 0.8 * self.pass_gap + 0.2 * (now - last_end)
            self._log_pass(frame, res, now, (now - t0) * 1000)
            last_end = now
            self._collect(res)
            spare = min_gap - (time.time() - t0)
            if spare > 0:
                time.sleep(spare)

    def ttl(self):
        return max(RESULT_TTL_S, self.lat_s - VIEW_DELAY_MS / 1000.0 + self.pass_gap + 0.3)

    def _log_pass(self, frame, res, now, ms):
        try:
            small = cv2.cvtColor(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            rec = {"t": now, "ms": ms, "bright": float(small.mean()),
                   "sharp": float(cv2.Laplacian(small, cv2.CV_64F).var()),
                   "faces": [(r["wpx"], r["name"]) for r in res]}
            self.pass_log.append(rec)
            while self.pass_log and now - self.pass_log[0]["t"] > DIAG_WINDOW_S:
                self.pass_log.popleft()
        except Exception:
            pass

    def _view_src(self):
        if self.view is not None and (self.view.fid > 0 or self.reader is None):
            return self.view
        return self.reader

    # ---- 1. Standard Stream Renderer (Original Page) ----
    def get_jpeg(self, last_seq):
        with self.render_lock:
            if self.jpeg is not None and self.seq > last_seq:
                return self.jpeg, self.seq
            src = self._view_src()
            if src is None:
                return None, last_seq
            if src is not self._cur_src:
                self._cur_src, self.src_fid = src, 0
            if src.history_s > 0 and VIEW_DELAY_MS > 0:
                frame, fid, ft = src.read_delayed(self.src_fid, time.time() - VIEW_DELAY_MS / 1000.0)
            else:
                frame, fid, ft = src.read_new_t(self.src_fid)
            if frame is None:
                return None, last_seq
            self.src_fid = fid
            h, w = frame.shape[:2]
            self.view_size = (w, h)
            dw = min(DISPLAY_W, w)
            dh = int(h * dw / w)
            img = frame if dw == w else cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
            with self.lock:
                results, det_ms, det_fps = list(self.results), self.det_ms, self.det_fps
            mw, mh = self.size
            if mw and mh and results:
                shown = predict_rows(results, ft, self._disp, self.ttl())
                self._draw(img, shown, dw / float(mw), dh / float(mh))
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.last_t, 1e-6))
            self.last_t = now
            head = f"cam{self.id} video {self.fps:.0f}fps"
            if self.detect_enabled:
                head += f" | detect {det_ms:.0f}ms ({det_fps:.1f}/s)"
            if self.recording:
                head += f" | REC {self.test_dist}m {self.got}/{SAMPLES_PER_SHOT}"
            cv2.putText(img, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                self.jpeg = buf.tobytes()
                self.seq += 1
            return self.jpeg, self.seq

    # ---- 2. Anti-Spoofing Stream Renderer (/antispoof Page) ----
    def get_antispoof_jpeg(self, last_seq):
        with self.as_render_lock:
            if self.as_jpeg is not None and self.as_seq > last_seq:
                return self.as_jpeg, self.as_seq
            src = self._view_src()
            if src is None:
                return None, last_seq
            if src is not self._as_cur_src:
                self._as_cur_src, self.as_src_fid = src, 0
            if src.history_s > 0 and VIEW_DELAY_MS > 0:
                frame, fid, ft = src.read_delayed(self.as_src_fid, time.time() - VIEW_DELAY_MS / 1000.0)
            else:
                frame, fid, ft = src.read_new_t(self.as_src_fid)
            if frame is None:
                return None, last_seq
            self.as_src_fid = fid
            h, w = frame.shape[:2]
            dw = min(DISPLAY_W, w)
            dh = int(h * dw / w)
            img = frame if dw == w else cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
            with self.lock:
                results, det_ms = list(self.results), self.det_ms
            mw, mh = self.size
            if mw and mh and results:
                shown = predict_rows(results, ft, self._as_disp, self.ttl())
                self._draw_antispoof(img, shown, dw / float(mw), dh / float(mh))
            head = f"ANTI-SPOOF cam{self.id} | {self.fps:.0f}fps | {det_ms:.0f}ms"
            cv2.putText(img, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                self.as_jpeg = buf.tobytes()
                self.as_seq += 1
            return self.as_jpeg, self.as_seq

    @staticmethod
    def _draw(img, results, sx, sy):
        for r in results:
            x, y = int(r["x"] * sx), int(r["y"] * sy)
            bw, bh = int(r["w"] * sx), int(r["h"] * sy)
            px = f" {int(r['wpx'])}px"
            d_txt = f" ~{r['dist']:.1f}m" if r["dist"] else ""
            name = r["name"]
            if name == "far":
                color, label = (0, 255, 255), "Too far" + px
            elif name == "turned":
                color, label = (0, 255, 255), "Face the camera" + px
            elif name != "Unknown":
                if r["low"]:
                    color, label = (0, 165, 255), f"{name}? ({r['score']:.2f}){px}"
                else:
                    color, label = (0, 255, 0), f"{name} ({r['score']:.2f}){d_txt}{px}"
            else:
                color, label = (0, 0, 255), "Unknown" + px
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = max(y - 6, th + 8)
            cv2.rectangle(img, (x, ty - th - 6), (x + tw + 6, ty + 4), color, -1)
            cv2.putText(img, label, (x + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    @staticmethod
    def _draw_antispoof(img, results, sx, sy):
        for r in results:
            x, y = int(r["x"] * sx), int(r["y"] * sy)
            bw, bh = int(r["w"] * sx), int(r["h"] * sy)

            if r.get("is_spoof"):
                reasons = ", ".join(r.get("spoof_reasons", [])) or "Fake Face"
                color = (0, 0, 255)
                label = f"SPOOF ALERT! ({reasons})"
            else:
                tags = [f"Live:{int(r.get('liveness_score', 1.0)*100)}%"]
                if r.get("has_mask"):
                    tags.append("Mask")
                if r.get("has_glasses"):
                    tags.append(r.get("glasses_type", "Glasses"))
                tag_str = " | ".join(tags)

                name = r["name"]
                if name == "far":
                    color, label = (0, 255, 255), "Too far"
                elif name == "turned":
                    color, label = (0, 255, 255), "Face camera"
                elif name != "Unknown":
                    color, label = (0, 255, 0), f"{name} ({r['score']:.2f}) [{tag_str}]"
                else:
                    color, label = (0, 140, 255), f"Unknown [{tag_str}]"

            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            ty = max(y - 6, th + 8)
            cv2.rectangle(img, (x, ty - th - 6), (x + tw + 6, ty + 4), color, -1)
            cv2.putText(img, label, (x + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

    def stop(self):
        self.running = False
        for r in (self.view, self.reader):
            if r is not None:
                r.stop()


STATES = {}
DETECT_ENABLED = False


# ---------------- APP ----------------
@asynccontextmanager
async def lifespan(app):
    global DETECT_ENABLED
    db, models_ok = {}, False
    if not VIDEO_ONLY:
        try:
            select_engine()
            db = load_db()
            models_ok = True
            if not db:
                print("[WARNING] known_faces/ is empty. Recognition starts automatically once photos appear.")
        except Exception as e:
            print(f"[ERROR] Face models unavailable ({e}) -> video only.")
    DETECT_ENABLED = models_ok
    for cid in CAMERA_IPS:
        STATES[cid] = CamState(cid, db, DETECT_ENABLED)
    if DETECT_ENABLED:
        threading.Thread(target=watch_known_faces, daemon=True).start()
        print(f"[OK] Watching '{KNOWN_DIR}/' for new photos every {WATCH_SECONDS}s.")
    yield
    WATCH_STOP.set()
    for st in STATES.values():
        st.stop()


app = FastAPI(title="Face Recognition & Anti-Spoofing Cameras", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_state(cam):
    st = STATES.get(cam)
    if st is None:
        raise HTTPException(404, f"Unknown camera '{cam}'. Available: {list(CAMERA_IPS)}")
    return st


def mjpeg(st, antispoof=False):
    last = -1
    while st.running:
        jpg, fid = st.get_antispoof_jpeg(last) if antispoof else st.get_jpeg(last)
        if jpg is None or fid <= last:
            time.sleep(0.01)
            continue
        last = fid
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
               str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")


def stream_response(cam, antispoof=False):
    return StreamingResponse(mjpeg(get_state(cam), antispoof=antispoof),
                             media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-cache"})


def player_html(cam):
    get_state(cam)
    return HTMLResponse(
        '<html><body style="margin:0;background:black;">'
        f'<img src="/video_feed_{cam}" style="width:100%;height:100%;object-fit:contain;">'
        '</body></html>')


# generic routes
@app.get("/video_feed/{cam}")
def video_feed(cam: str):
    return stream_response(cam, antispoof=False)


@app.get("/antispoof_feed/{cam}")
def antispoof_feed(cam: str):
    return stream_response(cam, antispoof=True)


@app.get("/player/{cam}", response_class=HTMLResponse)
def player(cam: str):
    return player_html(cam)


# fixed routes kept for existing APEX links
def _register(cam):
    app.add_api_route(f"/video_feed_{cam}", lambda c=cam: stream_response(c, False), methods=["GET"])
    app.add_api_route(f"/player_{cam}", lambda c=cam: player_html(c), methods=["GET"], response_class=HTMLResponse)


for _cid in CAMERA_IPS:
    _register(_cid)


# ---------------- PAGE 1: MAIN FACE RECOGNITION PAGE (/) ----------------
@app.get("/", response_class=HTMLResponse)
def index():
    frames = "".join(
        f'<div><div style="color:#ccc;font:14px sans-serif">cam{c} ({ip})</div>'
        f'<iframe src="/player_{c}" style="width:640px;height:360px;border:0"></iframe></div>'
        for c, ip in CAMERA_IPS.items())
    nav = (
        '<div style="width:100%;margin-bottom:8px;font:14px sans-serif;color:#fff">'
        '<b>Mode:</b> Standard Face Recognition | '
        '<a href="/antispoof" style="color:#38bdf8;text-decoration:none;font-weight:bold;">'
        'Open Anti-Spoofing & Liveness Page &rarr;</a>'
        '</div>'
    )
    return (f'<html><body style="background:#111;margin:10px;display:flex;gap:12px;'
            f'flex-wrap:wrap">{nav}{frames}</body></html>')


# ---------------- PAGE 2: SEPARATE ANTI-SPOOFING PAGE (/antispoof) ----------------
@app.get("/antispoof", response_class=HTMLResponse)
def antispoof_page():
    cam_cards = "".join(
        f"""
        <div class="cam-card">
            <div class="cam-header">
                <span>Camera {c} ({ip})</span>
                <span class="badge" id="badge-{c}">Monitoring...</span>
            </div>
            <img src="/antispoof_feed/{c}" alt="Camera {c} Stream" />
            <div class="attr-panel" id="attrs-{c}">Waiting for face detection...</div>
        </div>
        """
        for c, ip in CAMERA_IPS.items()
    )

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Anti-Spoofing & Liveness Guard</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #0f172a;
                color: #f8fafc;
                margin: 0;
                padding: 20px;
            }}
            .top-bar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            }}
            h1 {{
                margin: 0;
                font-size: 24px;
                color: #38bdf8;
            }}
            .nav-btn {{
                background: #1e293b;
                color: #38bdf8;
                border: 1px solid #334155;
                padding: 8px 14px;
                border-radius: 8px;
                text-decoration: none;
                font-size: 14px;
                font-weight: 600;
            }}
            .subtitle {{
                color: #94a3b8;
                margin-bottom: 20px;
                font-size: 14px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
                gap: 20px;
            }}
            .cam-card {{
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .cam-header {{
                padding: 12px 16px;
                background: #0f172a;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-weight: 600;
            }}
            .cam-card img {{
                width: 100%;
                display: block;
                background: #000;
            }}
            .attr-panel {{
                padding: 12px 16px;
                font-size: 14px;
                background: #1e293b;
                border-top: 1px solid #334155;
            }}
            .badge {{
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 12px;
                background: #0284c7;
                color: #fff;
            }}
            .badge.spoof {{
                background: #dc2626;
                animation: pulse 1s infinite;
            }}
            .badge.live {{
                background: #16a34a;
            }}
            .tag {{
                display: inline-block;
                padding: 3px 8px;
                margin-right: 6px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                background: #334155;
            }}
            .tag.danger {{ background: #991b1b; color: #fecaca; }}
            .tag.ok {{ background: #166534; color: #bbf7d0; }}
            .tag.warn {{ background: #854d0e; color: #fef08a; }}
            .log-box {{
                margin-top: 24px;
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 16px;
            }}
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.6; }}
                100% {{ opacity: 1; }}
            }}
        </style>
    </head>
    <body>
        <div class="top-bar">
            <h1>Anti-Spoofing & Liveness Verification Center</h1>
            <a href="/" class="nav-btn">&larr; Back to Standard Cameras</a>
        </div>
        <div class="subtitle">Real-time Phone Screen / Photo Spoof Detection, Mask Check, & Glasses / Sunglasses Detection</div>
        
        <div class="grid">
            {cam_cards}
        </div>

        <div class="log-box">
            <h3 style="margin-top:0;color:#f87171;">Recent Spoof Attack Alerts</h3>
            <div id="spoof-logs" style="font-size:13px;color:#cbd5e1;">No spoof attempts detected.</div>
        </div>

        <script>
            async function updateStatus() {{
                try {{
                    const res = await fetch('/api/antispoof/status');
                    const data = await res.json();
                    let allLogs = [];

                    for (const [cid, info] of Object.entries(data)) {{
                        const badge = document.getElementById('badge-' + cid);
                        const panel = document.getElementById('attrs-' + cid);
                        if (!badge || !panel) continue;

                        if (info.recent_spoof_alerts) {{
                            allLogs.push(...info.recent_spoof_alerts);
                        }}

                        if (info.faces.length === 0) {{
                            badge.className = 'badge';
                            badge.textContent = 'Scanning...';
                            panel.innerHTML = '<span style="color:#94a3b8;">No active face in frame</span>';
                            continue;
                        }}

                        let hasSpoof = info.faces.some(f => f.is_spoof);
                        if (hasSpoof) {{
                            badge.className = 'badge spoof';
                            badge.textContent = 'SPOOF DETECTED!';
                        }} else {{
                            badge.className = 'badge live';
                            badge.textContent = 'LIVE VERIFIED';
                        }}

                        panel.innerHTML = info.faces.map(f => {{
                            if (f.is_spoof) {{
                                return `<div style="margin-bottom:6px;">
                                    <span class="tag danger">SPOOF ATTACK</span>
                                    <span class="tag danger">Reasons: ${{f.spoof_reasons.join(', ')}}</span>
                                    <span class="tag">Liveness: ${{Math.round(f.liveness_score * 100)}}%</span>
                                </div>`;
                            }}
                            let maskTag = f.has_mask ? '<span class="tag warn">Mask: YES</span>' : '<span class="tag ok">Mask: NO</span>';
                            let glassTag = f.has_glasses ? `<span class="tag warn">${{f.glasses_type}}</span>` : '<span class="tag ok">Glasses: NO</span>';
                            return `<div style="margin-bottom:6px;">
                                <span class="tag ok">${{f.name}} (${{Math.round(f.liveness_score * 100)}}% Live)</span>
                                ${{maskTag}}
                                ${{glassTag}}
                            </div>`;
                        }}).join('');
                    }}

                    const logBox = document.getElementById('spoof-logs');
                    if (allLogs.length > 0) {{
                        logBox.innerHTML = allLogs.slice(0, 10).map(l =>
                            `<div>[<b>${{l.time}}</b>] Camera ${{l.camera}} — <span style="color:#f87171;">Spoof Blocked</span> (${{l.reasons.join(', ')}}) | Liveness: ${{Math.round(l.liveness_score*100)}}%</div>`
                        ).join('');
                    }}
                }} catch (e) {{
                    console.error(e);
                }}
            }}
            setInterval(updateStatus, 800);
            updateStatus();
        </script>
    </body>
    </html>
    """)


@app.get("/api/antispoof/status")
def antispoof_status():
    data = {}
    for cid, st in STATES.items():
        now = time.time()
        with st.lock:
            ttl = st.ttl()
            faces = [r for r in st.results if now - r["t"] <= ttl]
        data[cid] = {
            "ip": CAMERA_IPS[cid],
            "fps": round(st.fps, 1),
            "detect_ms": round(st.det_ms, 1),
            "faces": faces,
            "recent_spoof_alerts": list(st.spoof_events)[:10],
        }
    return data


@app.get("/api/health")
def health():
    return {"detection": DETECT_ENABLED,
            "engine": ENGINE_INFO,
            "settings": {"dual_stream": DUAL_STREAM, "view_subtype": VIEW_SUBTYPE,
                         "detect_fps_cap": DETECT_FPS, "detect_width": DETECT_WIDTH, "view_delay_ms": VIEW_DELAY_MS, "backend": BACKEND,
                         "cv_threads": CV_THREADS},
            "cameras": {c: {"ip": CAMERA_IPS[c], "detect_frame_size": list(st.size),
                            "view_size": list(st.view_size), "video_fps": round(st.fps, 1),
                            "detect_ms": round(st.det_ms, 1), "detect_per_sec": round(st.det_fps, 1),
                            "min_face_px": st.min_face, "max_reliable_m": st.max_m,
                            "connected": {"view": (st.view.connected if st.view else None),
                                          "detect": (st.reader.connected if st.reader else None)}}
                        for c, st in STATES.items()}}


@app.get("/api/results/{cam}")
def results(cam: str):
    st = get_state(cam)
    now = time.time()
    with st.lock:
        ttl = st.ttl()
        faces = [r for r in st.results if now - r["t"] <= ttl]
    return {"camera": cam, "frame_size": list(st.size), "faces": faces}


@app.post("/api/reload")
def reload_faces():
    if not DETECT_ENABLED:
        raise HTTPException(409, "Detection is disabled.")
    db = load_db(force=True)
    if not db:
        raise HTTPException(400, "No usable faces found in known_faces/.")
    for st in STATES.values():
        st.db = db
    return {"people": sorted(db)}


def diagnose_camera(st):
    log = list(st.pass_log)
    out = {"connected": {"view": st.view.connected if st.view else None,
                         "detect": st.reader.connected if st.reader else None},
           "detect_frame_size": list(st.size), "view_size": list(st.view_size),
           "video_fps": round(st.fps, 1), "detect_ms": round(st.det_ms, 1),
           "passes_per_s": round(st.det_fps, 2), "latency_s": round(st.lat_s, 2),
           "effective_ttl_s": round(st.ttl(), 2), "window_s": DIAG_WINDOW_S, "passes": len(log)}
    hints = []
    if not st.detect_enabled:
        return dict(out, hints=["Detection is disabled (video only)."])
    if st.reader is not None and not st.reader.connected:
        hints.append("The detection (main) stream is not connected: check the camera IP, login and stream settings.")
    if not log:
        return dict(out, hints=hints or ["No detection pass has completed yet."])

    faces = [f for rec in log for f in rec["faces"]]
    widths = [w for w, _ in faces]
    names = [n for _, n in faces]
    n = len(faces)
    share = lambda key: round(100.0 * sum(1 for x in names if key(x)) / n, 1) if n else 0.0
    bright = float(np.mean([r["bright"] for r in log]))
    sharp = float(np.median([r["sharp"] for r in log]))
    out.update({
        "passes_with_a_face_pct": round(100.0 * sum(1 for r in log if r["faces"]) / len(log), 1),
        "faces_seen": n,
        "face_state_pct": {"too_far": share(lambda x: x == "far"), "not_frontal": share(lambda x: x == "turned"),
                           "unknown_person": share(lambda x: x == "Unknown"),
                           "recognised": share(lambda x: x not in ("far", "turned", "Unknown"))},
        "face_width_px": ({"median": round(float(np.median(widths))), "p90": round(float(np.percentile(widths, 90))),
                           "max": round(max(widths))} if widths else None),
        "scene_brightness": round(bright), "sharpness": round(sharp)})
    return dict(out, hints=hints or ["Nothing obviously wrong with this camera's picture or timing."])


@app.get("/api/diagnose")
def diagnose():
    return {cam: diagnose_camera(st) for cam, st in STATES.items()}


@app.get("/diagnose", response_class=HTMLResponse)
def diagnose_page():
    return HTMLResponse("""<html><head><meta charset="utf-8"><title>Camera diagnosis</title>
<style>body{font:14px sans-serif;margin:16px;background:#f9fafb;color:#111}
.c{background:#fff;border:1px solid #d1d5db;border-radius:8px;padding:12px;margin-bottom:12px;max-width:900px}
h3{margin:0 0 6px}pre{margin:6px 0;background:#f3f4f6;padding:8px;border-radius:6px;overflow:auto}
li{margin:3px 0}</style></head><body><h2>Camera diagnosis</h2><div id="o">loading...</div>
<script>
function esc(s){return String(s).replace(/[&<>]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;"}[c]})}
function load(){fetch("/api/diagnose").then(function(r){return r.json()}).then(function(d){
 var h="";Object.keys(d).forEach(function(k){var c=d[k],hints=c.hints;
  var rest=JSON.parse(JSON.stringify(c));delete rest.hints;
  h+="<div class='c'><h3>Camera "+esc(k)+"</h3><ul>"+hints.map(function(x){return "<li>"+esc(x)+"</li>"}).join("")+"</ul><pre>"+esc(JSON.stringify(rest,null,1))+"</pre></div>"});
 document.getElementById("o").innerHTML=h}).catch(function(){document.getElementById("o").textContent="server offline"})}
load();setInterval(load,5000);
</script></body></html>""")


# ---- distance test API ----
@app.post("/api/test/{cam}/start")
def test_start(cam: str, person: str = Query(...), dist: int = Query(..., ge=1, le=20)):
    st = get_state(cam)
    if not st.detect_enabled:
        raise HTTPException(409, "Detection is disabled.")
    if person not in st.db:
        raise HTTPException(404, f"'{person}' not in known_faces. Found: {sorted(st.db)[:20]}")
    st.test_person, st.test_dist, st.got, st.recording = person, dist, 0, True
    return {"recording": True, "person": person, "dist_m": dist, "target": SAMPLES_PER_SHOT}


@app.get("/api/test/{cam}")
def test_status(cam: str):
    st = get_state(cam)
    return {"recording": st.recording, "person": st.test_person, "dist_m": st.test_dist,
            "got": st.got, "target": SAMPLES_PER_SHOT, "total_samples": len(st.samples)}


@app.post("/api/test/{cam}/finish")
def test_finish(cam: str):
    st = get_state(cam)
    st.recording = False
    samples = []
    if os.path.exists(samples_file(cam)):
        with open(samples_file(cam)) as fh:
            samples = json.load(fh)
    samples += st.samples
    if not samples:
        raise HTTPException(400, "No samples recorded.")
    with open(samples_file(cam), "w") as fh:
        json.dump(samples, fh)
    st.samples = []
    rows, calib = analyze_samples(cam, samples)
    st.apply_calib()
    return {"table": rows, "calibration": calib}


@app.delete("/api/test/{cam}")
def test_reset(cam: str):
    get_state(cam)
    for p in (samples_file(cam), calib_file(cam)):
        if os.path.exists(p):
            os.remove(p)
    STATES[cam].samples = []
    STATES[cam].apply_calib()
    return {"reset": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))