import streamlit as st
import cv2
import os
import math
import tempfile
import subprocess
from ultralytics import YOLO

st.set_page_config(
    page_title="MOTIONIQ",
    page_icon="🚘",
    layout="wide"
)

st.title("🚘 MOTIONIQ")
st.subheader("AI-Powered Vehicle Behaviour Analysis")
st.write("Detect, Track and Monitor Vehicle Behaviour")

uploaded_file = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file:
    if st.button("Analyse Video", type="primary"):

        with tempfile.TemporaryDirectory() as temp_dir:

            input_path = os.path.join(temp_dir, uploaded_file.name)

            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.info("Loading YOLO model...")
            model = YOLO("yolo11n.pt")

            progress = st.progress(0)
            status = st.empty()

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if not fps or fps <= 0:
                fps = 30

            positions = {}
            alerts = set()
            all_ids = set()

            results = model.track(
                source=input_path,
                conf=0.35,
                classes=[2, 3, 5, 7],
                tracker="bytetrack.yaml",
                stream=True,
                save=True,
                project=temp_dir,
                name="tracking",
                exist_ok=True,
                verbose=False
            )

            for frame_no, result in enumerate(results):

                if result.boxes is not None and result.boxes.id is not None:

                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, ids):

                        all_ids.add(track_id)

                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        current_time = frame_no / fps

                        if track_id not in positions:
                            positions[track_id] = {
                                "start": current_time,
                                "x": cx,
                                "y": cy
                            }

                        else:
                            data = positions[track_id]

                            distance = math.sqrt(
                                (cx - data["x"]) ** 2 +
                                (cy - data["y"]) ** 2
                            )

                            if distance > 20:
                                data["start"] = current_time
                                data["x"] = cx
                                data["y"] = cy

                            elif current_time - data["start"] >= 5:
                                alerts.add(track_id)

                if total_frames > 0:
                    progress.progress(
                        min((frame_no + 1) / total_frames, 1.0)
                    )

                status.write(f"Processing frame {frame_no + 1}...")

            output_dir = model.predictor.save_dir
            avi_path = os.path.join(
                str(output_dir),
                os.path.basename(input_path)
            )

            mp4_path = os.path.join(temp_dir, "MOTIONIQ_result.mp4")

            subprocess.run([
                "ffmpeg", "-y", "-i", avi_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-an", mp4_path
            ], check=True, capture_output=True)

            st.success("Analysis Completed!")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Tracked Vehicles", len(all_ids))
            col2.metric("Stationary Alerts", len(alerts))
            col3.metric("Frames Processed", frame_no + 1)

            st.subheader("Vehicle Tracking Result")
            st.video(mp4_path)

            st.subheader("Possible Stationary Vehicle Alerts")

            if alerts:
                st.warning(f"Possible stationary vehicle IDs: {sorted(alerts)}")
            else:
                st.success("No stationary vehicle alerts detected.")

            st.caption(
                "Alerts are rule-based indicators for human review, "
                "not confirmed parking violations or threat classifications."
            )
