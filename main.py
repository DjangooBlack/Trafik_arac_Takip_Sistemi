import cv2
from ultralytics import YOLO

# YOLOv11 modeli
model = YOLO("./assets/yolo11n.pt")

# Video dosyası
video_path = "./assets/video.mp4"
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)

# Takip verileri
track_history = {}
pixel_to_meter = 0.05  # Kalibrasyon yapmadıysan tahmini bir oran kullan

while True:
    cam_control, frame = video.read()
    if not cam_control:
        break

    # Takip ile birlikte tespit (bytetrack.yaml gerektirir)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()

        for box, track_id, conf, cls in zip(boxes, ids, confs, clss):
            label = model.names[int(cls)]
            if label not in ['car', 'truck', 'bus', 'motorcycle', 'bicycle'] or conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Takip geçmişini güncelle
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((video.get(cv2.CAP_PROP_POS_FRAMES), center_x, center_y))

            # Hız hesapla
            history = track_history[track_id]
            if len(history) >= 2:
                f1, x1_hist, y1_hist = history[-2]
                f2, x2_hist, y2_hist = history[-1]
                time_diff = (f2 - f1) / fps
                if time_diff > 0:
                    dx = x2_hist - x1_hist
                    dy = y2_hist - y1_hist
                    pixel_distance = (dx**2 + dy**2) ** 0.5
                    meter_distance = pixel_distance * pixel_to_meter
                    speed_mps = meter_distance / time_diff
                    speed_kmph = speed_mps * 3.6

                    # Hızı çiz
                    cv2.putText(frame, f"{speed_kmph:.1f} km/h", (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Kutu çiz ve etiketle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            satirlar = ["Elektronik Hiz Tespit Sistemi",
                        "Arac Hizi: {:.1f} km/h".format(speed_kmph) if 'speed_kmph' in locals() else "Hız Hesaplanamadı",
                        "Arac ID: {}".format(track_id),
                        "Konum: ({:.1f}, {:.1f})".format(center_x, center_y),
                        "Exit: 'q' tusu ile cikis yapabilirsiniz."]
            
            def yonergeler():
                y_Koordinati = 20
                # Ekranda talimatları göster
                for satir in satirlar:
                    cv2.putText(frame, satir, (10,y_Koordinati),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
                    y_Koordinati += 20
            yonergeler()

    # Göster
    cv2.imshow("YOLOv11 Detection + Speed", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
