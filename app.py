import os
import json
import time
import threading
import collections

import cv2
import numpy as np
import torch

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory
)

from transformers import VideoMAEForVideoClassification


# ============================================================
# SENTINEL AI - INTELLIGENT CCTV SURVEILLANCE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "cctv_model")
ALERTS_DIR = os.path.join(BASE_DIR, "alerts")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(ALERTS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# MODEL SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_FRAMES = 16
IMAGE_SIZE = 224

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

# Live camera:
# Frames are still collected continuously, but expensive
# VideoMAE inference is performed only every few frames.
LIVE_INFERENCE_EVERY = 4

# Uploaded video:
# Read every 3rd frame.
VIDEO_SAMPLE_EVERY = 3

# Perform inference after this many sampled frames.
VIDEO_INFERENCE_STRIDE = 6


# ============================================================
# ALERT SETTINGS
# ============================================================

ALERT_CLASSES = {
    "gun",
    "grab",
    "hit",
    "kick",
    "struggle",
    "throw",
    "sneak",
    "fall"
}

CONFIDENCE_THRESHOLD = 0.45

CONFIRMATION_COUNT = 2

# Cooldown is mainly used for LIVE CAMERA alerts.
# Uploaded-video alerts are handled separately.
ALERT_COOLDOWN = 15


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)


# ============================================================
# LOAD CLASSES
# ============================================================

classes_file = os.path.join(
    MODEL_DIR,
    "classes.json"
)

if not os.path.exists(classes_file):
    raise FileNotFoundError(
        f"classes.json not found: {classes_file}"
    )

with open(
    classes_file,
    "r",
    encoding="utf-8"
) as f:
    classes = json.load(f)

classes = list(classes)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 65)
print("        SENTINEL AI CCTV SURVEILLANCE")
print("=" * 65)

print()
print("Loading model...")
print("Model directory:", MODEL_DIR)
print("AI device:", DEVICE)

try:

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_DIR
    )

    model = model.to(DEVICE)

    model.eval()

    print("MODEL LOADED SUCCESSFULLY")

except Exception as e:

    print()
    print("MODEL LOADING FAILED")
    print(str(e))

    raise


print()
print("MODEL CLASSES")

for i, name in enumerate(classes):
    print(f"{i:02d} -> {name}")

print()
print("Number of classes:", len(classes))
print("=" * 65)


# ============================================================
# MODEL LOCK
# ============================================================
# Prevent live camera and uploaded video from using the
# VideoMAE model simultaneously.

MODEL_LOCK = threading.Lock()


# ============================================================
# SHARED STATE
# ============================================================

state_lock = threading.Lock()

state = {

    "camera_connected": False,

    "monitoring": False,

    "current_label": "waiting",

    "confidence": 0.0,

    "alert": False,

    "fps": 0.0,

    "frames_analyzed": 0,

    "device": str(DEVICE),

    "model_loaded": True
}


# ============================================================
# ALERT STORAGE
# ============================================================

alerts_log = collections.deque(
    maxlen=200
)


# ============================================================
# CAMERA FRAME STATE
# ============================================================

latest_frame = {
    "bytes": None
}


# ============================================================
# CAMERA BUFFER
# ============================================================

frame_buffer = collections.deque(
    maxlen=60
)

buffer_lock = threading.Lock()


# ============================================================
# PREDICTION STATE
# ============================================================

prediction_history = collections.deque(
    maxlen=3
)

last_alert_time = {}

confirmation_counter = {}

live_inference_counter = 0


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_frames(frames):

    processed = []

    for frame in frames:

        frame = cv2.resize(
            frame,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        frame = frame.astype(
            np.float32
        ) / 255.0

        frame = (
            frame - MEAN
        ) / STD

        processed.append(frame)

    array = np.stack(
        processed
    )

    tensor = torch.from_numpy(
        array
    ).permute(
        0,
        3,
        1,
        2
    ).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# MODEL INFERENCE
# ============================================================

def predict_clip(frames):

    if not frames:
        return "waiting", 0.0

    # Always provide exactly 16 frames.
    indexes = np.linspace(
        0,
        len(frames) - 1,
        NUM_FRAMES
    ).astype(int)

    selected = [
        frames[i]
        for i in indexes
    ]

    tensor = preprocess_frames(
        selected
    )

    # Prevent simultaneous VideoMAE execution.
    with MODEL_LOCK:

        with torch.inference_mode():

            output = model(
                pixel_values=tensor
            )

            logits = output.logits

            probabilities = torch.softmax(
                logits,
                dim=-1
            )[0]

    confidence, index = torch.max(
        probabilities,
        dim=0
    )

    index = int(
        index.item()
    )

    confidence = float(
        confidence.item()
    )

    if index >= len(classes):

        return (
            "unknown",
            confidence
        )

    label = classes[index]

    return (
        label,
        confidence
    )


# ============================================================
# ALERT CREATION
# ============================================================

def create_alert(
    label,
    confidence,
    frame=None,
    source="Mobile Camera",
    video_time=None
):

    now = time.time()

    # --------------------------------------------------------
    # LIVE CAMERA COOLDOWN
    # --------------------------------------------------------

    if source == "Mobile Camera":

        previous = last_alert_time.get(
            label,
            0
        )

        if (
            now - previous
            < ALERT_COOLDOWN
        ):

            return False

        last_alert_time[label] = now

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------------
    # EVIDENCE IMAGE
    # --------------------------------------------------------

    image_name = ""

    if frame is not None:

        image_name = (
            time.strftime(
                "%Y%m%d_%H%M%S"
            )
            + "_"
            + label
            + "_"
            + str(
                int(
                    time.time() * 1000
                ) % 100000
            )
            + ".jpg"
        )

        image_path = os.path.join(
            ALERTS_DIR,
            image_name
        )

        cv2.imwrite(
            image_path,
            frame
        )

    # --------------------------------------------------------
    # EVENT
    # --------------------------------------------------------

    event = {

        "id": int(
            time.time() * 1000
        ),

        "time": timestamp,

        "label": label,

        "confidence": round(
            confidence * 100,
            2
        ),

        "source": source,

        "video_time": video_time,

        "image": image_name
    }

    # --------------------------------------------------------
    # SAVE ALERT
    # --------------------------------------------------------

    with state_lock:

        alerts_log.appendleft(
            event
        )

    print(
        f"ALERT: {label.upper()} "
        f"{confidence:.1%} "
        f"[{source}]"
    )

    return True


# ============================================================
# ANALYZE LIVE CAMERA FRAME
# ============================================================

def analyze_frame(frame):

    global live_inference_counter

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # ADD FRAME TO TEMPORAL BUFFER
    # --------------------------------------------------------

    with buffer_lock:

        frame_buffer.append(
            rgb
        )

        frames = list(
            frame_buffer
        )

    if len(frames) < NUM_FRAMES:

        return (
            "collecting",
            0.0,
            False
        )

    # --------------------------------------------------------
    # PERFORMANCE OPTIMIZATION
    # --------------------------------------------------------
    # Don't run expensive VideoMAE inference on every request.

    live_inference_counter += 1

    if (
        live_inference_counter
        % LIVE_INFERENCE_EVERY
        != 0
    ):

        # Return previous prediction.
        if prediction_history:

            return (
                prediction_history[-1],
                0.0,
                False
            )

        return (
            "collecting",
            0.0,
            False
        )

    # --------------------------------------------------------
    # LATEST 16 FRAMES
    # --------------------------------------------------------

    clip = frames[-NUM_FRAMES:]

    try:

        label, confidence = predict_clip(
            clip
        )

    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return (
            "error",
            0.0,
            False
        )

    prediction_history.append(
        label
    )

    # --------------------------------------------------------
    # CHECK SUSPICIOUS ACTIVITY
    # --------------------------------------------------------

    is_alert = (
        label in ALERT_CLASSES
        and
        confidence >= CONFIDENCE_THRESHOLD
    )

    if is_alert:

        confirmation_counter[label] = (
            confirmation_counter.get(
                label,
                0
            ) + 1
        )

    else:

        confirmation_counter.clear()

    # --------------------------------------------------------
    # CREATE LIVE ALERT
    # --------------------------------------------------------

    if (
        is_alert
        and
        confirmation_counter.get(
            label,
            0
        ) >= CONFIRMATION_COUNT
    ):

        create_alert(
            label=label,
            confidence=confidence,
            frame=frame,
            source="Mobile Camera"
        )

        confirmation_counter[label] = 0

    return (
        label,
        confidence,
        is_alert
    )


# ============================================================
# PROCESS CAMERA FRAME
# ============================================================

def process_camera_frame(frame):

    start = time.time()

    label, confidence, is_alert = (
        analyze_frame(frame)
    )

    elapsed = (
        time.time() - start
    )

    fps = 0

    if elapsed > 0:

        fps = 1.0 / elapsed

    with state_lock:

        state[
            "camera_connected"
        ] = True

        state[
            "monitoring"
        ] = True

        state[
            "current_label"
        ] = label

        # Only replace confidence when
        # a real inference occurred.
        if confidence > 0:

            state[
                "confidence"
            ] = round(
                confidence,
                4
            )

        state[
            "alert"
        ] = is_alert

        state[
            "fps"
        ] = round(
            fps,
            2
        )

        state[
            "frames_analyzed"
        ] += 1

    # --------------------------------------------------------
    # AI OVERLAY
    # --------------------------------------------------------

    output = frame.copy()

    if is_alert:

        color = (
            0,
            0,
            255
        )

        status = "ALERT"

    else:

        color = (
            0,
            255,
            0
        )

        status = "AI"

    display_confidence = state[
        "confidence"
    ]

    text = (
        f"{status} | "
        f"{label.upper()} | "
        f"{display_confidence:.0%}"
    )

    cv2.rectangle(
        output,
        (10, 10),
        (550, 65),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        output,
        text,
        (25, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        output,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            80
        ]
    )

    if ok:

        latest_frame[
            "bytes"
        ] = encoded.tobytes()


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        all_classes=classes,
        alert_classes=sorted(
            ALERT_CLASSES
        )
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({

        "status": "ok",

        "model_loaded": True,

        "device": str(
            DEVICE
        ),

        "classes": len(
            classes
        )
    })


# ============================================================
# STATUS
# ============================================================

@app.route("/api/status")
def api_status():

    with state_lock:

        return jsonify(
            dict(state)
        )


# ============================================================
# ALERTS
# ============================================================

@app.route("/api/alerts")
def api_alerts():

    with state_lock:

        return jsonify(
            list(alerts_log)
        )


# ============================================================
# CLEAR ALERTS
# ============================================================

@app.route(
    "/api/alerts/clear",
    methods=["POST"]
)
def clear_alerts():

    with state_lock:

        alerts_log.clear()

    return jsonify({
        "success": True
    })


# ============================================================
# CAMERA FRAME API
# ============================================================

@app.route(
    "/api/camera/frame",
    methods=["POST"]
)
def camera_frame():

    if "frame" not in request.files:

        return jsonify({
            "success": False,
            "error": "No frame received"
        }), 400

    file = request.files[
        "frame"
    ]

    data = file.read()

    np_array = np.frombuffer(
        data,
        np.uint8
    )

    frame = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return jsonify({
            "success": False,
            "error": "Invalid image"
        }), 400

    try:

        process_camera_frame(
            frame
        )

    except Exception as e:

        print(
            "Camera processing error:",
            e
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    with state_lock:

        result = {

            "success": True,

            "label": state[
                "current_label"
            ],

            "confidence": state[
                "confidence"
            ],

            "alert": state[
                "alert"
            ],

            "fps": state[
                "fps"
            ]
        }

    return jsonify(
        result
    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    def generate():

        while True:

            frame = latest_frame[
                "bytes"
            ]

            if frame is not None:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

            time.sleep(
                0.05
            )

    return Response(
        generate(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# ALERT IMAGE
# ============================================================

@app.route(
    "/alerts/<path:filename>"
)
def alert_image(filename):

    return send_from_directory(
        ALERTS_DIR,
        filename
    )


# ============================================================
# VIDEO ANALYSIS
# ============================================================

@app.route(
    "/api/analyze-video",
    methods=["POST"]
)
def analyze_video():

    if "video" not in request.files:

        return jsonify({
            "success": False,
            "error": "No video uploaded"
        }), 400

    uploaded = request.files[
        "video"
    ]

    if not uploaded.filename:

        return jsonify({
            "success": False,
            "error": "Invalid filename"
        }), 400

    filename = os.path.basename(
        uploaded.filename
    )

    # Prevent accidental path traversal
    filename = filename.replace(
        " ",
        "_"
    )

    path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    uploaded.save(
        path
    )

    cap = cv2.VideoCapture(
        path
    )

    if not cap.isOpened():

        return jsonify({
            "success": False,
            "error": "Could not open video"
        }), 400

    video_frames = []

    detected_events = []

    frame_number = 0

    frames_since_inference = 0

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:

        fps = 25

    print()
    print(
        "VIDEO ANALYSIS STARTED:",
        filename
    )

    print(
        "FPS:",
        fps
    )

    # --------------------------------------------------------
    # VIDEO LOOP
    # --------------------------------------------------------

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_number += 1

        # ----------------------------------------------------
        # SAMPLE FRAMES
        # ----------------------------------------------------

        if (
            frame_number
            % VIDEO_SAMPLE_EVERY
            != 0
        ):

            continue

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        video_frames.append(
            rgb
        )

        frames_since_inference += 1

        # Need 16 frames.
        if len(video_frames) < NUM_FRAMES:

            continue

        # ----------------------------------------------------
        # SKIP SOME INFERENCES
        # ----------------------------------------------------

        if (
            frames_since_inference
            < VIDEO_INFERENCE_STRIDE
        ):

            continue

        frames_since_inference = 0

        # ----------------------------------------------------
        # GET LATEST 16 FRAMES
        # ----------------------------------------------------

        clip = video_frames[
            -NUM_FRAMES:
        ]

        try:

            label, confidence = (
                predict_clip(
                    clip
                )
            )

            event_time = round(
                frame_number / fps,
                2
            )

            print(
                f"[{event_time:.2f}s] "
                f"{label.upper()} "
                f"{confidence:.1%}"
            )

            # ------------------------------------------------
            # SUSPICIOUS ACTIVITY
            # ------------------------------------------------

            if (
                label in ALERT_CLASSES
                and
                confidence >=
                CONFIDENCE_THRESHOLD
            ):

                event = {

                    "time": event_time,

                    "label": label,

                    "confidence": round(
                        confidence * 100,
                        2
                    )
                }

                detected_events.append(
                    event
                )

                # ------------------------------------------------
                # CREATE ALERT CENTER EVENT
                # ------------------------------------------------

                create_alert(
                    label=label,
                    confidence=confidence,
                    frame=frame,
                    source="Video Analysis",
                    video_time=event_time
                )

        except Exception as e:

            print(
                "Upload prediction error:",
                e
            )

        # ----------------------------------------------------
        # OVERLAPPING WINDOW
        # ----------------------------------------------------

        video_frames = (
            video_frames[-8:]
        )

    cap.release()

    # --------------------------------------------------------
    # REMOVE DUPLICATE CONSECUTIVE EVENTS
    # --------------------------------------------------------

    cleaned = []

    previous_label = None

    previous_time = None

    for event in detected_events:

        current_label = event[
            "label"
        ]

        current_time = event[
            "time"
        ]

        # Keep same activity again
        # if it occurs sufficiently later.
        if (
            current_label
            != previous_label
            or
            previous_time is None
            or
            current_time
            - previous_time
            >= 2.0
        ):

            cleaned.append(
                event
            )

            previous_label = (
                current_label
            )

            previous_time = (
                current_time
            )

    print(
        "VIDEO ANALYSIS COMPLETED:",
        filename
    )

    print(
        "Events:",
        len(cleaned)
    )

    return jsonify({

        "success": True,

        "filename": filename,

        "events": cleaned,

        "total_events": len(
            cleaned
        )
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 65)

    print(
        "Dashboard:"
    )

    print(
        "http://localhost:5000"
    )

    print()

    print(
        "Video feed:"
    )

    print(
        "http://localhost:5000/video_feed"
    )

    print()

    print(
        "Health:"
    )

    print(
        "http://localhost:5000/api/health"
    )

    print()

    print(
        "AI device:",
        DEVICE
    )

    print(
        "Classes:",
        len(classes)
    )

    print("=" * 65)

    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )