from picamera2 import Picamera2
import cv2
import numpy as np
import time
import math

'''TARGET_LAB_COLORS = {
    "R": (141, 181, 154),
    "Y": (200, 97, 193),
    "O": (161, 165, 182),
    "W": (172, 127, 122),
    "B": (111, 161, 62),
    "G": (178, 69, 169)
}'''
TARGET_LAB_COLORS = {
    "R": (95, 187, 47),
    "Y": (209, 82, 131),
    "O": (158, 140, 69),
    "W": (208, 125, 132),
    "B": (152, 162, 183),
    "G": (201, 71, 194),
}



def get_closest_color_lab(observed_lab):
    min_dist = float('inf')
    closest_name = "Unknown"

    A_obs = float(observed_lab[1])
    B_obs = float(observed_lab[2])

    for name, target_val in TARGET_LAB_COLORS.items():
        A_target = float(target_val[1])
        B_target = float(target_val[2])
        dist = math.sqrt((A_obs - A_target) ** 2 + (B_obs - B_target) ** 2)

        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name, min_dist

def process_region(lab_frame, x, y, region_size=30):
    half_size = region_size // 2
    y1 = max(0, y - half_size)
    y2 = min(lab_frame.shape[0], y + half_size)
    x1 = max(0, x - half_size)
    x2 = min(lab_frame.shape[1], x + half_size)

    region = lab_frame[y1:y2, x1:x2]
    if region.size == 0:
        return "Error", 0

    avg_lab = np.mean(region, axis=(0, 1)).astype(int)
    return get_closest_color_lab(avg_lab)

class RubikCamera:
    def __init__(self, resolution=(320, 240)):
        self.resolution = resolution
        self.picam2 = None
        self.points = [(130, 170), (130, 100), (180, 170),(180, 100)]
        self.running = False

    def start(self):
        if self.running:
            return True
        if self.picam2 is not None:
            self.close()

        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"format": "BGR888", "size": (1280, 720)}
            )
            self.picam2.configure(config)
            self.picam2.start()

            self.picam2.set_controls({
                "AfMode": 0,
                "ExposureValue": -0.2,
                "Brightness": 0.04,
                "Saturation": 1.1,
            })

            time.sleep(2)
            self.running = True
            print("[Camera] Ready")
            return True
        except Exception as e:
            print(f"[Camera] Error: {e}")
            self.close()
            return False

    def get_frame(self):
        if not self.running:
            return None
        try:
            frame = self.picam2.capture_array()
            frame = frame.copy()
            frame = cv2.resize(frame, self.resolution)
            return frame
        except Exception:
            return None

    def stop(self):
        if self.running and self.picam2:
            try:
                self.picam2.stop()
                self.running = False
                print("[Camera] Stream Stopped")
            except Exception as e:
                print(f"[Camera] Stop Error: {e}")

    def close(self):
        self.stop()
        if self.picam2:
            try:
                self.picam2.close()
                print("[Camera] Hardware Released")
            except Exception:
                pass
            self.picam2 = None

if __name__ == "__main__":
    camera = RubikCamera()

    if not camera.start():
        exit(1)

    print("Press ESC to exit")

    try:
        while True:
            frame = camera.get_frame()

            if frame is not None:
                frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

                for (x, y) in camera.points:
                    detected_color, dist = process_region(frame_lab, x, y, region_size=30)

                    top_left = (x - 15, y - 15)
                    bottom_right = (x + 15, y + 15)

                    box_color = (0, 255, 0) if dist < 40 else (0, 0, 255)
                    cv2.rectangle(frame, top_left, bottom_right, box_color, 2)

                    cv2.putText(frame, detected_color, (x + 30, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                cv2.imshow("Rubik LAB Detection", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()
