from picamera2 import Picamera2
import cv2
import numpy as np
import time

COLOR_KEYS = {
    ord('r'): 'R',
    ord('g'): 'G',
    ord('b'): 'B',
    ord('o'): 'O',
    ord('y'): 'Y',
    ord('w'): 'W',
}

POINTS_2X2 = [(130, 170), (130, 100), (180, 170),(180, 100)]
REGION_SIZE = 30
VIEW_RES = (320, 240)

def region_mean_lab_from_bgr(bgr_frame, x, y, region_size=REGION_SIZE):
    half = region_size // 2
    h, w = bgr_frame.shape[:2]
    x1 = max(0, x - half); x2 = min(w, x + half)
    y1 = max(0, y - half); y2 = min(h, y + half)
    roi = bgr_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    mean = lab.reshape(-1, 3).mean(axis=0)
    return mean

def summarize(samples_by_color):
    print("\n=== SUMMARY (OpenCV Lab 8-bit, computed from BGR input) ===")
    target = {}
    for name, arr_list in samples_by_color.items():
        if len(arr_list) == 0:
            print(f"{name}: no samples")
            continue
        arr = np.stack(arr_list, axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        print(f"{name}: mean=({mean[0]:.1f},{mean[1]:.1f},{mean[2]:.1f})  std=({std[0]:.1f},{std[1]:.1f},{std[2]:.1f})  n={len(arr_list)}")
        target[name] = tuple(int(round(v)) for v in mean)

    if target:
        print("\nCopy/paste into your camera.py:")
        print("TARGET_LAB_COLORS = {")
        for k in ["R", "Y", "O", "W", "B", "G"]:
            if k in target:
                print(f'    "{k}": {target[k]},')
        print("}")

def main():
    print("Rubik 2x2 LAB calibrator (BGR888)")
    print("Keys: R/G/B/O/Y/W select | C capture | S summarize | X reset | ESC quit")

    selected_color = "W"
    samples = {k: [] for k in ["R", "G", "B", "O", "Y", "W"]}

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (1280, 720)}
    )
    picam2.configure(config)
    picam2.start()

    time.sleep(2.0)
    try:
        picam2.set_controls({"AfMode": 2})
    except Exception:
        pass

    while True:
        bgr = picam2.capture_array()
        bgr = cv2.resize(bgr, VIEW_RES)

        for i, (x, y) in enumerate(POINTS_2X2):
            lab = region_mean_lab_from_bgr(bgr, x, y)
            half = REGION_SIZE // 2
            cv2.rectangle(bgr, (x - half, y - half), (x + half, y + half), (255, 255, 255), 2)
            if lab is not None:
                cv2.putText(bgr, f"{i+1}:{int(lab[0])},{int(lab[1])},{int(lab[2])}",
                            (x - half, y - half - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.putText(bgr, f"Selected color: {selected_color} | samples: {len(samples[selected_color])}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(bgr, "Press R/G/B/O/Y/W then C, S summarize, X reset, ESC quit",
                    (5, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv2.imshow("Rubik 2x2 LAB Calibrate (BGR888)", bgr)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

        if k in COLOR_KEYS:
            selected_color = COLOR_KEYS[k].upper()
            print(f"[SELECT] {selected_color}")

        if k in (ord('x'), ord('X')):
            samples = {k: [] for k in ["R", "G", "B", "O", "Y", "W"]}
            print("[RESET] cleared all samples")

        if k in (ord('c'), ord('C')):
            captured = 0
            for (x, y) in POINTS_2X2:
                lab = region_mean_lab_from_bgr(bgr, x, y)
                if lab is not None:
                    samples[selected_color].append(lab.copy())
                    captured += 1
            print(f"[CAPTURE] color {selected_color}: +{captured} samples (total {len(samples[selected_color])})")

        if k in (ord('s'), ord('S')):
            summarize(samples)

    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
