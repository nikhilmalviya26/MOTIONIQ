🚘 MOTIONIQ
AI-Based Vehicle Monitoring System
MOTIONIQ is a simple project that checks traffic or CCTV video and helps monitor vehicles.

It can find vehicles, give them IDs, and show an alert when a vehicle stays in one place for some time or enters a selected area.
Note: Right now, MOTIONIQ works with uploaded video files. Connecting a live CCTV camera is a future improvement.

📸 Project Screenshots
Main Dashboard

Vehicle Detection and Alerts

🏗️ How MOTIONIQ Works

1. Upload Video
   - Upload a traffic or CCTV video.
2. Find Vehicles
   - The system detects cars, bikes, buses, and trucks.
3. Give Vehicle IDs
   - Each detected vehicle gets a temporary ID.
   - This helps the system follow the vehicle in the video.
4. Check Vehicle Behaviour
   - Checks if a vehicle stays in one place for the set time.
   - Checks if a vehicle enters the selected restricted area.
5. Show Alerts
   - Displays the alerts on the dashboard.
   - Shows the processed video.
⭐ Main Features
- Vehicle Detection
  - Finds cars, bikes, buses, and trucks.
- Vehicle Tracking
  - Shows a tracking ID on each detected vehicle.
- Long Stop Alert
  - Shows an alert if a vehicle stays almost in the same place for the selected time.
- Restricted Area Alert
  - Shows an alert if a vehicle enters the selected area.
- Dashboard
  - Shows the number of tracked vehicles and alerts.
- Processed Video
  - Lets the user watch and download the video with boxes and IDs.
🛠️ Tools Used
- Python – Main programming language.
- YOLO11n – Finds vehicles in the video.
- ByteTrack – Tracks vehicles and gives temporary IDs.
- OpenCV – Reads and processes video frames.
- Streamlit – Creates the web app and dashboard.
- FFmpeg – Helps convert the processed video.
▶️ How to Run
1. Download or clone this repository.
2. Open the project folder.
3. Install the required packages:
   pip install -r requirements.txt
4. Run the app:
   streamlit run app.py
🔮 Future Improvements
- Connect the system to an authorised live CCTV camera.
- Save alerts so they can be checked later.
- Add more vehicle behaviour checks.
- Improve the dashboard.
⚠️ Important Note
- The current version checks uploaded videos only. It is not connected to live CCTV.
- Alerts are only warnings for a person to review. They do not prove that a vehicle is dangerous.
- Vehicle IDs are temporary and may change if a vehicle is hidden or leaves the camera view.
- This project does not detect bombs or confirm security threats.
MOTIONIQ – Detecting Vehicles. Monitoring Their Behaviour.
