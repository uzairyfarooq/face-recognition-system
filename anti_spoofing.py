import os
import re
import time
import math
import threading
import urllib.parse
from collections import deque, Counter
from contextlib import asynccontextmanager

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
)

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

# Import base helpers & engine from face_api.py so both files stay clean and modular
from face_api import (
    CAMERA_IPS,
    USERNAME,
    PASSWORD,
    RTSP_PORT,
    SUBTYPE,
    DETECT_WIDTH,
    HARD_MIN_PX,
    YAW_RANGE,
    DISPLAY_W,
    JPEG_QUALITY,
    DUAL_STREAM,
    VIEW_SUBTYPE,
    DETECT_FPS,
    RECOG_INTERVAL,
    VIEW_DELAY_MS,
    RESULT_TTL_S,
    rtsp_url,
    StreamReader,
    Tracks,
    predict_rows,
    yaw_ok,
    select_engine,
    load_db,
    build_index,
    identify,
    load_calib,
    MIN_FACE_PX,
    WATCH_SECONDS,
    WATCH_STOP,
    current_sigs,
)

# ==============================================================================
# ADVANCED ANTI-SPOOFING, MASK & GLASSES DETECTION MODULE
# ==============================================================================

def analyze_liveness_and_attributes(frame, face_row):
    """
    Comprehensive Anti-Spoofing & Facial Attribute Analyzer:
    1. Screen / Phone Photo Spoof Check:
       - Laplacian Texture Sharpness (detects flat/blurry printed photos or soft screen pixels)
       - Specular Glare / Reflection Hotspot Ratio (detects phone/tablet glass glare)
       - Color Frequency / Saturation Flatness (screens emit unnatural RGB/HSV distributions)
       - Moire / High-Frequency Grid Pattern Check (FFT energy ratio)
    2. Mask Detection:
       - Lower face (nose + mouth) texture & edge drop + color uniformity check
    3. Glasses / Sunglasses Detection:
       - Eye bridge & orbital region edge density + dark lens tint check
    """
    x, y, w, h = int(face_row[0]), int(face_row[1]), int(face_row[2]), int(face_row[3])
    fh, fw = frame.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)

    if x2 - x1 < 24 or y2 - y1 < 24:
        return {
            "is_spoof": False,
            "spoof_reasons": [],
            "liveness_score": 1.0,
            "has_mask": False,
            "has_glasses": False,
            "glasses_type": "None",
        }

    face_roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    spoof_reasons = []
    spoof_penalty = 0.0

    # --- CHECK 1: Texture Sharpness (Printed photo / low-res screen blur) ---
    lap_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
    if lap_var < 35.0:
        spoof_reasons.append("Low Texture/Blur")
        spoof_penalty += 0.45

    # --- CHECK 2: Specular Reflection / Phone Screen Glare ---
    v_chan = hsv_roi[:, :, 2]
    glare_ratio = float(np.sum(v_chan > 245) / v_chan.size)
    if glare_ratio > 0.075:
        spoof_reasons.append("Screen Glare")
        spoof_penalty += 0.50

    # --- CHECK 3: Screen Color / Chromaticity Anomaly (Blue-light / unnatural saturation) ---
    s_chan = hsv_roi[:, :, 1]
    high_sat_ratio = float(np.sum(s_chan > 190) / s_chan.size)
    if high_sat_ratio > 0.22:
        spoof_reasons.append("Screen Color Cast")
        spoof_penalty += 0.35

    # --- CHECK 4: Frequency Domain (FFT) Moire Pattern Check ---
    f_transform = np.fft.fft2(cv2.resize(gray_roi, (64, 64)))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = 20 * np.log(np.abs(f_shift) + 1e-6)
    center = 32
    # High-frequency outer ring vs mid-frequency energy
    outer_energy = float(np.mean(magnitude[0:10, :]) + np.mean(magnitude[54:64, :])) / 2.0
    total_energy = float(np.mean(magnitude)) + 1e-6
    hf_ratio = outer_energy / total_energy
    if hf_ratio > 0.88:
        spoof_reasons.append("Moire Grid Pattern")
        spoof_penalty += 0.40

    liveness_score = round(max(0.0, 1.0 - spoof_penalty), 2)
    is_spoof = liveness_score < 0.55

    # --- CHECK 5: Mask Detection (Lower 45% of face) ---
    lower_y = int(y1 + (y2 - y1) * 0.55)
    lower_roi = frame[lower_y:y2, x1:x2]
    has_mask = False
    if lower_roi.size > 0:
        lower_gray = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2GRAY)
        lower_std = float(np.std(lower_gray))
        lower_edges = cv2.Canny(lower_gray, 40, 120)
        lower_edge_density = float(np.sum(lower_edges > 0) / lower_edges.size)
        # Masks smooth out lip/nose shadows and create uniform fabric texture
        if lower_std < 29.0 and lower_edge_density < 0.05:
            has_mask = True

    # --- CHECK 6: Glasses / Sunglasses Detection (Eye region 20% to 50% height) ---
    eye_y1 = int(y1 + (y2 - y1) * 0.20)
    eye_y2 = int(y1 + (y2 - y1) * 0.50)
    eye_roi = frame[eye_y1:eye_y2, x1:x2]
    has_glasses = False
    glasses_type = "None"
    if eye_roi.size > 0:
        eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        eye_edges = cv2.Canny(eye_gray, 50, 150)
        edge_density = float(np.sum(eye_edges > 0) / eye_edges.size)
        mean_eye_brightness = float(np.mean(eye_gray))
        overall_brightness = float(np.mean(gray_roi)) + 1e-6

        if mean_eye_brightness / overall_brightness < 0.62:
            has_glasses = True
            glasses_type = "Sunglasses"
        elif edge_density > 0.072:
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


class AntiSpoofCamState:
    """Dedicated camera state with full Anti-Spoofing, Mask, and Glasses checks."""

    def __init__(self, cam_id, db, engine_factory):
        self.id = cam_id
        self._db, self._index = {}, None
        self.db = db
        self.engine = engine_factory()
        self.lock = threading.Lock()
        self.running = True
        self.results, self.det_ms, self.det_fps = [], 0.0, 0.0
        self.lat_s, self.pass_gap = 0.0, 0.0
        self.size, self.view_size = (0, 0), (0, 0)
        self.tracks = Tracks()
        self.spoof_events = deque(maxlen=30)  # Recent spoof alerts log

        c = load_calib(self.id)
        self.min_face = c.get("min_face_px", MIN_FACE_PX)
        self.k = c.get("k")

        self.render_lock = threading.Lock()
        self.jpeg, self.seq = None, 0
        self._disp = {}
        self._cur_src, self.src_fid = None, 0
        self.fps, self.last_t = 0.0, time.time()

        main_sub = os.getenv(f"CAM{cam_id}_SUBTYPE", SUBTYPE)
        dual = DUAL_STREAM and not os.getenv(f"CAM_URL_{cam_id}") and VIEW_SUBTYPE != main_sub
        self.view, self.reader = None, None
        if dual:
            self.view = StreamReader(
                rtsp_url(cam_id, VIEW_SUBTYPE),
                f"antispoof-cam{cam_id}-view",
                history_s=(VIEW_DELAY_MS / 1000.0 + 0.5) if VIEW_DELAY_MS > 0 else 0.0,
            )
            gap = (1.0 / DETECT_FPS) if DETECT_FPS > 0 else 0.0
            self.reader = StreamReader(rtsp_url(cam_id), f"antispoof-cam{cam_id}-detect", gap)
        else:
            self.reader = StreamReader(rtsp_url(cam_id), f"antispoof-cam{cam_id}")

        threading.Thread(target=self._worker, daemon=True).start()

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, value):
        self._index = build_index(value)
        self._db = value

    def _analyze(self, frame, t):
        out = []
        index = self._index
        for f in self.engine.detect(frame, DETECT_WIDTH):
            wpx = float(f[2])
            tr = self.tracks.match((f[0], f[1], f[2], f[3]), t)
            attrs = analyze_liveness_and_attributes(frame, f)

            row = {
                "x": float(f[0]),
                "y": float(f[1]),
                "w": float(f[2]),
                "h": float(f[3]),
                "wpx": wpx,
                "dist": (self.k / wpx) if self.k else None,
                "name": "far",
                "score": 0.0,
                "low": False,
                "yaw_ok": False,
                "tid": tr["id"],
                "vx": float(tr["vx"]),
                "vy": float(tr["vy"]),
                "t": float(t),
                **attrs,
            }

            if wpx < HARD_MIN_PX:
                out.append(row)
                continue

            row["yaw_ok"] = yaw_ok(f)
            if not yaw_ok(f, YAW_RANGE if wpx >= 80 else (0.15, 0.85)):
                row["name"] = "turned"
                out.append(row)
                continue

            # If spoof detected, flag immediately and log event
            if attrs["is_spoof"]:
                row["name"] = "SPOOF_DETECTED"
                self.spoof_events.appendleft({
                    "time": time.strftime("%H:%M:%S"),
                    "camera": self.id,
                    "liveness_score": attrs["liveness_score"],
                    "reasons": attrs["spoof_reasons"],
                })
                out.append(row)
                continue

            if tr["last"] is not None and t - tr["last_rec"] < RECOG_INTERVAL:
                name, best = tr["last"]
            else:
                name, best, _ = identify(self.engine, frame, f, index)
                tr["last"], tr["last_rec"] = (name, best), t
                tr["hist"].append((name, best))

            sname, savg = Tracks.smoothed(tr)
            row.update(name=sname, score=savg, low=wpx < self.min_face)
            out.append(row)

        self.tracks.prune(t)
        return out

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
                print(f"[antispoof-cam{self.id}] error: {e}")
                res = []
            self.size = (frame.shape[1], frame.shape[0])
            now = time.time()
            with self.lock:
                self.results = res
                self.det_ms = 0.8 * self.det_ms + 0.2 * (now - t0) * 1000
                self.det_fps = 0.8 * self.det_fps + 0.2 * (1.0 / max(now - last_end, 1e-6))
                self.lat_s = 0.8 * self.lat_s + 0.2 * (now - (ft or t0))
                self.pass_gap = 0.8 * self.pass_gap + 0.2 * (now - last_end)
            last_end = now
            spare = min_gap - (time.time() - t0)
            if spare > 0:
                time.sleep(spare)

    def ttl(self):
        return max(RESULT_TTL_S, self.lat_s - VIEW_DELAY_MS / 1000.0 + self.pass_gap + 0.3)

    def _view_src(self):
        if self.view is not None and (self.view.fid > 0 or self.reader is None):
            return self.view
        return self.reader

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
            head = f"ANTI-SPOOF cam{self.id} | {self.fps:.0f}fps | {det_ms:.0f}ms"
            cv2.putText(img, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                self.jpeg = buf.tobytes()
                self.seq += 1
            return self.jpeg, self.seq

    @staticmethod
    def _draw(img, results, sx, sy):
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


ANTISPOOF_STATES = {}


@asynccontextmanager
async def lifespan(app):
    select_engine()
    from face_api import ENGINE_FACTORY
    db = load_db()
    for cid in CAMERA_IPS:
        ANTISPOOF_STATES[cid] = AntiSpoofCamState(cid, db, ENGINE_FACTORY)
    yield
    for st in ANTISPOOF_STATES.values():
        st.stop()


antispoof_app = FastAPI(title="Anti-Spoofing & Liveness Detection Dashboard", lifespan=lifespan)
antispoof_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def mjpeg_gen(st):
    last = -1
    while st.running:
        jpg, fid = st.get_jpeg(last)
        if jpg is None or fid <= last:
            time.sleep(0.01)
            continue
        last = fid
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
               str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")


@antispoof_app.get("/antispoof_feed/{cam}")
def antispoof_feed(cam: str):
    st = ANTISPOOF_STATES.get(cam)
    if not st:
        raise HTTPException(404, f"Unknown camera '{cam}'")
    return StreamingResponse(mjpeg_gen(st), media_type="multipart/x-mixed-replace; boundary=frame")


@antispoof_app.get("/api/antispoof/status")
def antispoof_status():
    data = {}
    for cid, st in ANTISPOOF_STATES.items():
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


@antispoof_app.get("/", response_class=HTMLResponse)
def antispoof_dashboard():
    """Dedicated Anti-Spoofing, Mask & Glasses Detection Web Page."""
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
            h1 {{
                margin: 0 0 8px 0;
                font-size: 24px;
                color: #38bdf8;
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
        <h1>Anti-Spoofing & Liveness Verification Center</h1>
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


if __name__ == "__main__":
    # Runs on port 8001 so it doesn't conflict with your main face_api.py (port 8000)
    uvicorn.run(antispoof_app, host="0.0.0.0", port=int(os.getenv("ANTISPOOF_PORT", "8001")))