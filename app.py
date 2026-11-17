import os
        rect = (
            int(min(zx1, zx2) / 100 * width),
            int(min(zy1, zy2) / 100 * height),
            int(max(zx1, zx2) / 100 * width),
            int(max(zy1, zy2) / 100 * height),
        )
        writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        if not writer.isOpened():
            st.error("Could not create output video.")
            st.stop()

        model = load_model()
        progress = st.progress(0)
        status = st.empty()
        preview = st.empty()
        histories = defaultdict(deque)
        stop_alerted, zone_alerted, all_ids = set(), set(), set()
        alerts = []
        processed = 0

        try:
            results = model.track(
                source=input_path, conf=confidence, classes=[2, 3, 5, 7],
                tracker="bytetrack.yaml", stream=True, save=False, verbose=False
            )
            for frame_no, result in enumerate(results):
                frame = result.orig_img.copy()
                timestamp = frame_no / fps
                if zone_enabled:
                    cv2.rectangle(frame, rect[:2], rect[2:], (0, 165, 255), 2)
                    cv2.putText(frame, "RESTRICTED ZONE", (rect[0], max(24, rect[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.int().cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()
                    for box, track_id, class_id in zip(boxes, ids, classes):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        all_ids.add(track_id)
                        history = histories[track_id]
                        history.append((timestamp, cx, cy))
                        while history and timestamp - history[0][0] > stop_seconds + 2:
                            history.popleft()

                        label = result.names.get(class_id, "vehicle")
                        color = (40, 200, 40)

                        if history and timestamp - history[0][0] >= stop_seconds:
                            displacement = math.hypot(cx - history[0][1], cy - history[0][2])
                            if displacement <= movement_threshold and track_id not in stop_alerted:
                                stop_alerted.add(track_id)
                                alerts.append({"Time (video sec)": round(timestamp, 1),
                                               "Vehicle ID": track_id,
                                               "Alert": "Potential prolonged stop",
                                               "Details": f"Movement within {movement_threshold}px for about {stop_seconds}s"})
                        if zone_enabled and inside_zone(cx, cy, rect) and track_id not in zone_alerted:
                            zone_alerted.add(track_id)
                            alerts.append({"Time (video sec)": round(timestamp, 1),
                                           "Vehicle ID": track_id,
                                           "Alert": "Restricted-zone entry",
                                           "Details": "Vehicle centre entered configured zone"})
                        if track_id in stop_alerted or track_id in zone_alerted:
                            color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} ID:{track_id}", (x1, max(20, y1 - 7)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                writer.write(frame)
                processed += 1
                if total_frames > 0:
                    progress.progress(min(processed / total_frames, 1.0))
                status.write(f"Processing frame {processed} of {total_frames or 'unknown'}")
                if processed == 1 or processed % max(1, int(fps * 3)) == 0:
                    preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processing preview")

        except Exception as exc:
            writer.release()
            st.error(f"Video analysis failed: {exc}")
            st.stop()
        writer.release()
        progress.progress(1.0)
        status.success("Analysis complete.")

        if not os.path.exists(avi_path) or os.path.getsize(avi_path) == 0:
            st.error("Annotated output video was not generated.")
            st.stop()

        output_path = avi_path
        conversion_warning = None
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", avi_path, "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", mp4_path],
                check=True, capture_output=True, text=True, timeout=180
            )
            if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                output_path = mp4_path
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            conversion_warning = "H.264 conversion failed; using AVI output. " + str(exc)
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                conversion_warning += "\n" + exc.stderr[-1000:]

        st.header("MOTIONIQ analysis report")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Unique tracked IDs", len(all_ids))
        col2.metric("Potential stop alerts", len(stop_alerted))
        col3.metric("Restricted-zone alerts", len(zone_alerted))
        col4.metric("Total alerts", len(alerts))
        if conversion_warning:
            st.warning(conversion_warning)
        st.subheader("Alert history")
        if alerts:
            st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
        else:
            st.success("No configured anomaly rules were triggered in this footage.")
        st.subheader("Processed video")
        st.video(output_path)
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        is_mp4 = output_path == mp4_path
        st.download_button("⬇ Download MOTIONIQ result", video_bytes,
                           file_name="MOTIONIQ_result.mp4" if is_mp4 else "MOTIONIQ_result.avi",
                           mime="video/mp4" if is_mp4 else "video/x-msvideo",
                           use_container_width=True)
        st.caption("Alerts are rule-based indicators for human review, not proof of malicious intent. Tracker IDs are temporary and can change under occlusion or when a vehicle leaves view. This prototype processes uploaded video; it does not detect bombs, identify suspects, or transmit data to authorities.")
