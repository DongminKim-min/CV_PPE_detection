from ultralytics import YOLO

model = YOLO("models/ppe_detector_ver2.pt")


# results = model.track("input_videos/ppe_input_video_2.mp4", save = True)
results = model.track("input_videos/construction_site_video_22.mp4", save = True)

print(results)
print("===================")
for box in results[0].boxes:
    print(box)

