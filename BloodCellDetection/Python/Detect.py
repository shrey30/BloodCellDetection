import os
import cv2
import numpy as np
from math import pi
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

file_path = "out.png"  # your image file

if os.path.exists(file_path):
    os.remove(file_path)
    print("File deleted successfully")
else:
    print("File not found")
file_path = "step.jpg"  # your image file

if os.path.exists(file_path):
    os.remove(file_path)
    print("File deleted successfully")
else:
    print("File not found")
file_path = "final.jpg"  # your image file

if os.path.exists(file_path):
    os.remove(file_path)
    print("File deleted successfully")
else:
    print("File not found")    
# ---------------------- HELPERS ----------------------
def normalize_uint8(img):
    if img.dtype != np.uint8:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

def ensure_color(img):
    if img.ndim == 2:
        return cv2.cvtColor(normalize_uint8(img), cv2.COLOR_GRAY2BGR)
    return img

def color_mask(mask, color=(0,0,255)):
    """Return a BGR image where mask>0 is painted with color."""
    mask_u = normalize_uint8(mask)
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask_u > 0] = color
    return out

def visualize_markers(markers):
    """Convert watershed markers (ints) to a color image for visualization."""
    h, w = markers.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    unique = np.unique(markers)
    rng = np.random.RandomState(0)
    color_map = {}
    for u in unique:
        if u == -1:            # boundary
            color_map[u] = (255, 255, 255)
        elif u <= 1:           # background or unknown
            color_map[u] = (0, 0, 0)
        else:
            color_map[u] = tuple(int(x) for x in rng.randint(0, 256, size=3))
    for u, col in color_map.items():
        out[markers == u] = col
    return out

def make_montage(image_dict, cols=3, thumb_w=360):
    """Create a tiled montage from a dict {title: image}."""
    thumbs = []
    titles = []
    for title, img in image_dict.items():
        img_c = ensure_color(img.copy())
        h, w = img_c.shape[:2]
        scale = thumb_w / float(w)
        new_h = int(h * scale)
        thumb = cv2.resize(img_c, (thumb_w, new_h), interpolation=cv2.INTER_AREA)
        # put title text
        cv2.rectangle(thumb, (0,0), (thumb_w, 28), (0,0,0), -1)
        cv2.putText(thumb, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    # pad thumbs to full rows and equal heights in each row
    rows = []
    for i in range(0, len(thumbs), cols):
        row_imgs = thumbs[i:i+cols]
        heights = [im.shape[0] for im in row_imgs]
        max_h = max(heights)
        # pad each to max_h
        row_padded = []
        for im in row_imgs:
            h,w = im.shape[:2]
            if h < max_h:
                pad = np.zeros((max_h-h, w, 3), dtype=np.uint8)
                im2 = np.vstack([im, pad])
            else:
                im2 = im
            row_padded.append(im2)
        # if row has fewer than cols, pad with black images
        while len(row_padded) < cols:
            row_padded.append(np.zeros_like(row_padded[0]))
        row = cv2.hconcat(row_padded)
        rows.append(row)
    if len(rows) == 0:
        return None
    montage = cv2.vconcat(rows)
    return montage

# ---------------------- DETECTION PIPELINE ----------------------
def detect_cells(image_path, show_steps=True, save_steps=False, output_dir="outputs"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    orig = img.copy()
    H, W = img.shape[:2]

    # basic denoise + contrast equalization
    denoised = cv2.bilateralFilter(img, 9, 75, 75)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v2 = clahe.apply(v)
    hsv2 = cv2.merge([h, s, v2])
    img_clahe = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

    # morphological kernels
    kernel_small = np.ones((3,3), np.uint8)
    kernel_med   = np.ones((5,5), np.uint8)
    kernel_big   = np.ones((9,9), np.uint8)

    # ---------------- RBC mask (red hues) ----------------
    lower_r1 = np.array([0, 20, 30])
    upper_r1 = np.array([10, 255, 255])
    lower_r2 = np.array([170, 20, 30])
    upper_r2 = np.array([180, 255, 255])
    mask_r1 = cv2.inRange(hsv2, lower_r1, upper_r1)
    mask_r2 = cv2.inRange(hsv2, lower_r2, upper_r2)
    mask_rbc = cv2.bitwise_or(mask_r1, mask_r2)
    mask_rbc = cv2.morphologyEx(mask_rbc, cv2.MORPH_CLOSE, kernel_med, iterations=2)
    mask_rbc = cv2.morphologyEx(mask_rbc, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask_rbc = cv2.morphologyEx(mask_rbc, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    mask_rbc = normalize_uint8(mask_rbc)

    # ---------------- WBC mask (bluish/purple region) ----------------
    # adjust ranges if your stain colors differ
    lower_wbc = np.array([90, 30, 20])
    upper_wbc = np.array([170, 255, 255])
    mask_wbc = cv2.inRange(hsv2, lower_wbc, upper_wbc)
    mask_wbc = cv2.morphologyEx(mask_wbc, cv2.MORPH_CLOSE, kernel_med, iterations=2)
    mask_wbc = cv2.morphologyEx(mask_wbc, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask_wbc = normalize_uint8(mask_wbc)

    # remove overlaps: where WBC is detected, avoid counting RBC there
    mask_rbc = cv2.bitwise_and(mask_rbc, cv2.bitwise_not(mask_wbc))

    # ---------------- Platelet detection (small dark blobs) ----------------
    gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
    plate_thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    # remove bigger structures (RBC/WBC) from platelet mask
    plate_mask = cv2.bitwise_and(plate_thr, cv2.bitwise_not(mask_rbc))
    plate_mask = cv2.bitwise_and(plate_mask, cv2.bitwise_not(mask_wbc))
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    plate_mask = normalize_uint8(plate_mask)

    # ---------------- Watershed on RBC mask for separation ----------------
    dist = cv2.distanceTransform((mask_rbc>0).astype(np.uint8)*255, cv2.DIST_L2, 5)
    dist_vis = normalize_uint8(dist)  # for display
    # pick a relatively low threshold to capture small centers too
    _, sure_fg = cv2.threshold(dist, 0.30 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(mask_rbc, kernel_big, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_clahe, markers.astype(np.int32))

    # ---------------- Analyze markers -> RBC instances ----------------
    rbc_count = 0
    rbc_boxes = []
    rbc_info = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask_label = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 30:           # tiny noise
                continue
            perim = cv2.arcLength(cnt, True)
            circularity = (4 * pi * area / (perim * perim)) if perim > 0 else 0
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull) if hull is not None else 0
            solidity = (area / hull_area) if hull_area > 0 else 0
            x, y, w, h = cv2.boundingRect(cnt)
            # relaxed filters to include slightly-deformed RBCs; adjust thresholds if needed
            if 80 < area < 4000 and (circularity > 0.30 or solidity > 0.55):
                rbc_count += 1
                rbc_boxes.append((x, y, w, h))
                rbc_info.append({'label': rbc_count, 'area': area, 'circ': circularity, 'solidity': solidity, 'bbox':(x,y,w,h)})
            else:
                # treat as cluster / ambiguous RBC — still mark for review
                rbc_boxes.append((x, y, w, h))
                rbc_info.append({'label': None, 'area': area, 'circ': circularity, 'solidity': solidity, 'bbox':(x,y,w,h)})

    # ---------------- WBC detection (contours on mask_wbc) ----------------
    wbc_count = 0
    wbc_info = []
    cnts_wbc, _ = cv2.findContours(mask_wbc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts_wbc:
        area = cv2.contourArea(cnt)
        if area < 400:
            continue
        perim = cv2.arcLength(cnt, True)
        circularity = (4 * pi * area / (perim * perim)) if perim > 0 else 0
        x, y, w, h = cv2.boundingRect(cnt)
        wbc_count += 1
        wbc_info.append({'label': wbc_count, 'area': area, 'circ': circularity, 'bbox':(x,y,w,h)})

    # ---------------- Platelet detection (small contours) ----------------
    platelet_count = 0
    platelet_info = []
    cnts_plate, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts_plate:
        area = cv2.contourArea(cnt)
        if 8 < area < 200:
            perim = cv2.arcLength(cnt, True)
            circularity = (4 * pi * area / (perim * perim)) if perim > 0 else 0
            if circularity < 0.25:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            platelet_count += 1
            platelet_info.append({'label': platelet_count, 'area': area, 'circ': circularity, 'bbox':(x,y,w,h)})

    # ---------------- Prepare final overlay ----------------
    out = orig.copy()
    # RBCs: red
    idx = 1
    for info in rbc_info:
        x,y,w,h = info['bbox']
        if info['label'] is not None:
            cv2.rectangle(out, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(out, f"RBC-{info['label']}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1, cv2.LINE_AA)
        else:
            # ambiguous cluster -> yellow
            cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,255), 1)
            cv2.putText(out, f"RBC(?)", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,255,255), 1, cv2.LINE_AA)
    # WBCs: blue
    for winfo in wbc_info:
        x,y,w,h = winfo['bbox']
        cv2.rectangle(out, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(out, f"WBC-{winfo['label']}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 1, cv2.LINE_AA)
    # Platelets: green
    for pinfo in platelet_info:
        x,y,w,h = pinfo['bbox']
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 1)
        cv2.putText(out, f"P-{pinfo['label']}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1, cv2.LINE_AA)

    # summary text
    cv2.putText(out, f"RBC: {rbc_count}  WBC: {wbc_count}  Platelets: {platelet_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)

    # ---------------- Build step images to show ----------------
    steps = {
        "Original": orig,
        "CLAHE (preproc)": img_clahe,
        "RBC mask (raw)": color_mask(mask_rbc, color=(0,0,255)),
        "RBC closd/open": color_mask(cv2.morphologyEx(mask_rbc, cv2.MORPH_CLOSE, kernel_med, iterations=1), (0,0,255)),
        "Distance (norm)": dist_vis,
        "Sure FG": normalize_uint8(sure_fg),
        "Markers (watershed)": visualize_markers(markers),
        "WBC mask": color_mask(mask_wbc, color=(255,0,0)),
        "Platelets": color_mask(plate_mask, color=(0,255,0)),
        "Final overlay": out
    }

    # optionally save steps
    if save_steps:
        os.makedirs(output_dir, exist_ok=True)
        for k, im in steps.items():
            fn = os.path.join(output_dir, f"{k.replace(' ','_')}.png")
            cv2.imwrite(fn, ensure_color(im))
        print(f"Saved step images to: {output_dir}")

    # show montage then final
    if show_steps:
        montage = make_montage(steps, cols=3, thumb_w=360)
        if montage is not None:
            cv2.imwrite("step.jpg",montage)
            #cv2.imshow("Processing Steps (montage) - press any key", montage)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        # show final full-size for detailed inspection
        #cv2.imshow("Final Detected Cells", out)
        cv2.imwrite("final.jpg",out)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    print(f"RBC count: {rbc_count}, WBC count: {wbc_count}, Platelet count: {platelet_count}")
    return {
        'rbc_count': rbc_count,
        'wbc_count': wbc_count,
        'platelet_count': platelet_count,
        'rbc_info': rbc_info,
        'wbc_info': wbc_info,
        'platelet_info': platelet_info,
        'final_image': out
    }
def classy():
    # Load the saved model
    model = tf.keras.models.load_model('Blood1.keras')
    x=450
    def preprocess_image(img_path):
        img = image.load_img(img_path, target_size=(x, x))  # Adjust size to match model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize to [0, 1]
        return img_array

    def predict_image(img_path):
        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        return predicted_class, predictions
    # Specify the path to the image you want to predict
    image_path = 'test.jpg'
    className=['Eosinophil','Lymphocyte','Basophils','Erythroblast']
    #className=['Basophils','Eosinophil','Erythroblast','Lymphocyte']
    # Make a prediction
    predicted_class, predictions = predict_image(image_path)
    print(predicted_class)
    # Display the image
    img = image.load_img(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted Class: {className[predicted_class[0]]}')
    #plt.show()
    plt.savefig("out.png")

    # Show the prediction probabilities for each class
    print('Prediction probabilities:', predictions)
# ---------------------- RUN ----------------------
if __name__ == "__main__":
    result = detect_cells("test.jpg", show_steps=True, save_steps=False)
    classy()
    
