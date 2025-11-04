import cv2
import numpy as np

# ---------------------- IMAGE QUALITY CHECK ----------------------
def check_image_quality(image, blur_threshold=50, brightness_range=(20, 200)):
    """
    Check if image is blurry or faulty.
    Returns (status, reason)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur detection (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        return False, f"Blurry image detected (variance={laplacian_var:.2f})"

    # Brightness check
    mean_brightness = np.mean(gray)
    if mean_brightness < brightness_range[0]:
        return False, f"Image too dark (mean brightness={mean_brightness:.2f})"
    elif mean_brightness > brightness_range[1]:
        return False, f"Image too bright (mean brightness={mean_brightness:.2f})"

    return True, "Image quality OK"

# ---------------------- MAIN DETECTION ----------------------
def detect_cells(image_path):
    image = cv2.imread(image_path)
    output = image.copy()

    # Quality check
    ok, reason = check_image_quality(image)
    if not ok:
        print("❌ Image rejected:", reason)
        cv2.putText(output, reason, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Faulty Image", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print("✅ Proceeding:", reason)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)

    # ---------------- WBC Detection ----------------
    lower_wbc = np.array([120, 30, 40])
    upper_wbc = np.array([160, 255, 255])
    mask_wbc = cv2.inRange(hsv, lower_wbc, upper_wbc)
    mask_wbc = cv2.morphologyEx(mask_wbc, cv2.MORPH_OPEN, kernel, iterations=2)

    contours_wbc, _ = cv2.findContours(mask_wbc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wbc_count = 0
    for cnt in contours_wbc:
        area = cv2.contourArea(cnt)
        if area > 1500:  # WBCs are large
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(output, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            cv2.putText(output, "WBC", (int(x)-20, int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            wbc_count += 1

    # ---------------- RBC Detection ----------------
    lower_rbc1 = np.array([0, 20, 50])
    upper_rbc1 = np.array([10, 180, 255])
    lower_rbc2 = np.array([170, 20, 50])
    upper_rbc2 = np.array([180, 180, 255])

    mask_rbc1 = cv2.inRange(hsv, lower_rbc1, upper_rbc1)
    mask_rbc2 = cv2.inRange(hsv, lower_rbc2, upper_rbc2)
    mask_rbc = cv2.bitwise_or(mask_rbc1, mask_rbc2)
    mask_rbc = cv2.morphologyEx(mask_rbc, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_rbc = cv2.morphologyEx(mask_rbc, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Watershed for overlapping RBCs
    dist_transform = cv2.distanceTransform(mask_rbc, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(mask_rbc, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)

    rbc_count = 0
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if 200 < area < 2000:  # RBC range
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(output, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                cv2.putText(output, "RBC", (int(x)-15, int(y)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                rbc_count += 1

    # ---------------- Platelet Detection ----------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours_platelet, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    platelet_count = 0
    for cnt in contours_platelet:
        area = cv2.contourArea(cnt)
        if 20 < area < 80:  # Platelets small
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(output, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(output, "Platelet", (int(x)-20, int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            platelet_count += 1

    # ---------------- Results ----------------
    print(f"RBC count: {rbc_count}, WBC count: {wbc_count}, Platelet count: {platelet_count}")
    cv2.putText(output, f"RBC: {rbc_count} | WBC: {wbc_count} | Platelets: {platelet_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Detected Blood Cells", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------------- RUN ----------------------
detect_cells("test1.jpg")
