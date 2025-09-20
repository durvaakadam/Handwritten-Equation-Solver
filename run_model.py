import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import re

# ------------------- Helper functions -------------------

def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def detect_contours_merge(img_gray):
    """Adaptive threshold + invert + find contours + merge overlapping boxes."""
    binarized = cv2.adaptiveThreshold(img_gray, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    inverted = ~binarized
    contours, _ = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    l = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        l.append([x, y, w, h])

    # Merge overlapping boxes
    lcopy = l.copy()
    keep = []
    while lcopy:
        curr_x, curr_y, curr_w, curr_h = lcopy.pop(0)
        if curr_w * curr_h < 20:
            continue
        throw = []
        for i, (x, y, w, h) in enumerate(lcopy):
            curr_interval = [curr_x, curr_x + curr_w]
            next_interval = [x, x + w]
            if getOverlap(curr_interval, next_interval) > 1:
                new_interval_x = [min(curr_x, x), max(curr_x + curr_w, x + w)]
                new_interval_y = [min(curr_y, y), max(curr_y + curr_h, y + h)]
                curr_x, curr_y = new_interval_x[0], new_interval_y[0]
                curr_w, curr_h = new_interval_x[1] - new_interval_x[0], new_interval_y[1] - new_interval_y[0]
                throw.append(i)
        for ind in sorted(throw, reverse=True):
            lcopy.pop(ind)
        keep.append([curr_x, curr_y, curr_w, curr_h])
    return inverted, keep

def resize_pad(img, size=(45,45), pad_value=0):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    sh, sw = size
    interp = cv2.INTER_AREA if (h > sh or w > sw) else cv2.INTER_CUBIC
    aspect = w / h
    if aspect > 1:
        new_w = sw
        new_h = int(round(new_w / aspect))
        pad_top = (sh - new_h) // 2
        pad_bot = sh - new_h - pad_top
        pad_left = pad_right = 0
    elif aspect < 1:
        new_h = sh
        new_w = int(round(new_h * aspect))
        pad_left = (sw - new_w) // 2
        pad_right = sw - new_w - pad_left
        pad_top = pad_bot = 0
    else:
        new_h, new_w = sh, sw
        pad_left = pad_right = pad_top = pad_bot = 0
    scaled = cv2.resize(img, (new_w, new_h), interpolation=interp)
    result = np.full((sh, sw), pad_value, dtype=np.uint8)
    result[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = scaled
    return result

# ------------------- Load model -------------------

model_path = "my_model.h5"  # put your trained model here
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

model = tf.keras.models.load_model(model_path)
print("✅ Loaded model from:", model_path)

class_names = ['+', '-', '0','1','2','3','4','5','6','7','8','9','=','times']

# ------------------- Run detection -------------------

IMAGE = "62729ec9_06_proc1.png"  # place image in folder `equation_images`
img_folder = "equation_images"
img_path = os.path.join(img_folder, IMAGE)
if not os.path.exists(img_path):
    raise FileNotFoundError("Equation image not found at: " + img_path)

img_gray = cv2.imread(img_path, 0)
inverted, boxes = detect_contours_merge(img_gray)

# visualize merged boxes
os.makedirs("debug_inputs", exist_ok=True)
img_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
for (x,y,w,h) in boxes:
    cv2.rectangle(img_vis, (x,y), (x+w, y+h), (0,0,255), 1)
plt.figure(figsize=(6,6))
plt.imshow(img_vis[:,:,::-1])
plt.title("Merged boxes")
plt.axis('off')
plt.savefig("debug_inputs/merged_boxes.png")
plt.close()

print("Total boxes (before filtering):", len(boxes))

# Filter tiny / noise boxes
areas = [w*h for x,y,w,h in boxes] if boxes else [0]
median_area = np.median(areas) if areas else 0
min_area_thresh = max(80, median_area * 0.02)
filtered = [b for b in boxes if (b[2] > 3 and b[3] > 3 and (b[2]*b[3]) >= min_area_thresh)]
print(f"Boxes after filter (min_area_thresh={int(min_area_thresh)}):", len(filtered))

# Sort left→right
filtered = sorted(filtered, key=lambda b: b[0])

# ------------------- Predict -------------------

eqn_list = []
for i, (x,y,w,h) in enumerate(filtered):
    roi = inverted[y:y+h, x:x+w]
    if roi is None or roi.size == 0:
        continue

    proc = resize_pad(roi, (45,45), pad_value=0)
    if proc is None:
        continue

    inp = tf.expand_dims(tf.expand_dims(proc, 0), -1)
    logits = model.predict(inp, verbose=0)
    probs = tf.nn.softmax(logits[0]).numpy()
    top_idx = probs.argmax()

    proc_inv = 255 - proc
    inp2 = tf.expand_dims(tf.expand_dims(proc_inv, 0), -1)
    logits2 = model.predict(inp2, verbose=0)
    probs2 = tf.nn.softmax(logits2[0]).numpy()
    top_idx2 = probs2.argmax()

    if probs2[top_idx2] > probs[top_idx] + 0.08:
        chosen = class_names[top_idx2]
    else:
        chosen = class_names[top_idx]
    chosen = "*" if chosen == "times" else chosen
    eqn_list.append(chosen)

final_eq = "".join(eqn_list)
print("🔎 Final predicted equation:", final_eq)

# ------------------- Solve -------------------

expr = final_eq.replace("times", "*").replace("×", "*")
expr = expr.replace("=", "").replace(" ", "")

if re.fullmatch(r"[0-9+\-*/().]+", expr):
    try:
        result = eval(expr, {"__builtins__": None}, {})
        print("✅ Solved:", expr, "=", result)
    except Exception as e:
        print("⚠️ Could not solve equation:", e)
else:
    print("⚠️ Equation contains unsupported characters, not solving.")
