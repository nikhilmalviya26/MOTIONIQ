import math
import os
import shutil
import subprocess
import tempfile

import cv2
import streamlit as st
from ultralytics import YOLO


st.set_page_config(
    page_title="MOTIONIQ",
    page_icon="🚘",
    layout="wide",
)

st.title("🚘 MOTIONIQ")
st.subheader("AI-Powered Vehicle Behaviour Analysis")
st.write("Detect, track and monitor vehicle behaviour from traffic video.")

uploaded_file = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov"],
)

if uploaded_file is not None and st.button("Analyse Video", type="primary"):
    # Keep all temporary files available until the result has been read.
    with tempfile.TemporaryDirectory() as temp_dir:
        safe_filename = os.path.basename(uploaded_file.name)
        input_path = os.path.join(temp_dir, safe_filename)

        with open(input_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        try:
            st.info("Loading YOLO model...")
            model = YOLO("yolo11n.pt")

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("Could not open this video. Please upload a valid video file.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if not fps or fps <= 0:
                fps = 30.0

            positions = {}
            alerts = set()
            all_ids = set()

            progress = st.progress(0.0)
            status = st.empty()

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
                verbose=False,
            )

            frame_count = 0
            for frame_no, result in enumerate(results):
                frame_count = frame_no + 1

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        all_ids.add(track_id)
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        current_time = frame_no / fps

                        if track_id not in positions:
                            positions[track_id] = {
                                "start": current_time,
                                "x": center_x,
                                "y": center_y,
                            }
                        else:
                            data = positions[track_id]
                            distance = math.hypot(
                                center_x - data["x"],
                                center_y - data["y"],
                            )

                            if distance > 20:
                                data["start"] = current_time
                                data["x"] = center_x
                                data["y"] = center_y
                            elif current_time - data["start"] >= 5:
                                alerts.add(track_id)

                if total_frames > 0:
                    progress.progress(min(frame_count / total_frames, 1.0))
                status.write(f"Processing frame {frame_count}...")

            status.empty()
            progress.progress(1.0)

            # Ultralytics saves the annotated video using the input filename.
            output_dir = str(model.predictor.save_dir)
            tracked_video_path = os.path.join(output_dir, safe_filename)

            if not os.path.isfile(tracked_video_path):
                # Fallback: locate a video file in the tracking output folder.
                possible_videos = [
                    os.path.join(output_dir, name)
                    for name in os.listdir(output_dir)
                    if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
                ] if os.path.isdir(output_dir) else []

                if possible_videos:
                    tracked_video_path = possible_videos[0]
                else:
                    st.error("Tracking finished, but the annotated output video was not created.")
                    st.stop()

            st.success("Vehicle tracking completed.")

            # Convert to browser-friendly H.264 MP4 for Streamlit playback.
            mp4_path = os.path.join(temp_dir, "MOTIONIQ_result.mp4")
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is None:
                st.error("FFmpeg is not installed on the app server, so the result cannot be converted for browser playback.")
                st.stop()

            command = [
                ffmpeg_path,
                "-y",
                "-i", tracked_video_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                mp4_path,
            ]

            try:
                conversion = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            except subprocess.CalledProcessError as error:
                st.error("FFmpeg could not convert the annotated video. Exact server message:")
                st.code(error.stderr or str(error))
                st.stop()
            except subprocess.TimeoutExpired:
                st.error("Video conversion took longer than 5 minutes and was stopped. Try a shorter video.")
                st.stop()

            if not os.path.isfile(mp4_path) or os.path.getsize(mp4_path) == 0:
                st.error("FFmpeg finished but did not create a valid MP4 output.")
                st.stop()

            with open(mp4_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.subheader("Analysis Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Unique tracking IDs", len(all_ids))
            col2.metric("Vehicles flagged for review", len(alerts))
            col3.metric("Frames processed", frame_count)

            if alerts:
                st.warning(
                    "Possible prolonged-stationary behaviour detected for tracking IDs: "
                    + ", ".join(str(track_id) for track_id in sorted(alerts))
                )
                st.caption("These are rule-based alerts for human review, not proof of wrongdoing.")
            else:
                st.info("No prolonged-stationary behaviour was flagged by the current rule.")

            st.subheader("Processed Video")
            st.video(video_bytes, format="video/mp4")
            st.download_button(
                "Download MOTIONIQ Result",
                data=video_bytes,
                file_name="MOTIONIQ_result.mp4",
                mime="video/mp4",
            )

        except Exception as error:
            st.error("MOTIONIQ could not complete the analysis.")
            st.exception(error)
