"""
ANPR core: extract frames from a video, find likely license-plate regions,
OCR them with Tesseract, clean/validate the text, and deduplicate readings
across frames into a final list of unique plate numbers.
"""
import cv2
import re
import numpy as np
import pytesseract
from difflib import SequenceMatcher

TESS_CONFIG = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Indian plate pattern, e.g. MH12AB1234, DL8CAF5678, KA01MJ1234
PLATE_RE = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$')

# Loose shape check: 2 letters, then a run of alnum, ending in digits.
# Used as a fallback when the strict format doesn't match due to OCR noise
# (O/0, I/1, S/8, B/8 confusions are extremely common in plate OCR).
LOOSE_RE = re.compile(r'^[A-Z]{2}[A-Z0-9]{4,8}[0-9]{3,4}$')


def clean_text(raw):
    t = raw.upper()
    t = re.sub(r'[^A-Z0-9]', '', t)
    return t


def is_plausible_plate(t):
    if len(t) < 7 or len(t) > 11:
        return False
    if not re.search(r'[A-Z]', t) or not re.search(r'[0-9]', t):
        return False
    if PLATE_RE.match(t):
        return True
    if LOOSE_RE.match(t):
        return True
    return False


def candidate_plate_regions(frame):
    """Return list of (x, y, w, h) candidate boxes likely to contain a plate,
    combining edge/contour geometry with a Haar cascade pass so it generalizes
    across plate styles, distances and angles from a handheld phone video."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_f = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray_f, 30, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 50 or h < 15:
            continue
        aspect = w / float(h)
        if not (1.8 <= aspect <= 6.0):
            continue
        area_ratio = (w * h) / float(w_img * h_img)
        if area_ratio < 0.001 or area_ratio > 0.25:
            continue
        boxes.append((x, y, w, h))

    cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    detected = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(50, 15))
    for (x, y, w, h) in detected:
        boxes.append((int(x), int(y), int(w), int(h)))

    return boxes


def dedupe_boxes(boxes, tol=15):
    kept = []
    for x, y, w, h in boxes:
        dup = False
        for (sx, sy, sw, sh) in kept:
            if abs(x - sx) < tol and abs(y - sy) < tol:
                dup = True
                break
        if not dup:
            kept.append((x, y, w, h))
    return kept


def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
    return thresh


def ocr_region(frame, box, pad=6):
    x, y, w, h = box
    H, W = frame.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return []

    processed = preprocess_for_ocr(crop)
    try:
        data = pytesseract.image_to_data(processed, config=TESS_CONFIG,
                                          output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        return []

    out = []
    n = len(data.get('text', []))
    line_text = ''.join(data['text'][i] for i in range(n) if data['text'][i])
    confs = [float(data['conf'][i]) for i in range(n) if data['conf'][i] not in ('-1', -1)]
    if line_text.strip():
        avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.5
        out.append((clean_text(line_text), avg_conf))
    return out


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def merge_readings(readings):
    """readings: list of (text, confidence, frame_idx, timestamp_sec).
    Groups near-duplicate plate strings (OCR noise across frames of the
    same car) and keeps the highest-confidence reading per group, along
    with a count of how many frames supported it."""
    groups = []
    for text, conf, frame_idx, ts in readings:
        placed = False
        for g in groups:
            if text == g['text'] or similar(text, g['text']) >= 0.75:
                g['count'] += 1
                g['variants'].add(text)
                if conf > g['conf']:
                    g['conf'] = conf
                    g['text'] = text
                placed = True
                break
        if not placed:
            groups.append({'text': text, 'conf': conf, 'count': 1,
                            'first_ts': ts, 'variants': {text}})
    return groups


def process_video(video_path, sample_fps=3, progress_cb=None):
    """Process video, return list of dicts with plate_number, confidence,
    frames_detected, first_seen_seconds — sorted by frames_detected/confidence desc."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Could not open video file')

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(native_fps / sample_fps)))

    readings = []
    frame_idx = 0
    while True:    
        ret, frame = cap.read()
        if not ret:
            break
       if frame_idx % step == 0:        
            ts = frame_idx / native_fps            
            boxes = dedupe_boxes(candidate_plate_regions(frame))
            print(f"[DEBUG] frame {frame_idx} ts={ts:.1f}s -> {len(boxes)} candidate boxes")
            for box in boxes:
                for text, conf in ocr_region(frame, box):
                    print(f"[DEBUG]   OCR raw='{text}' conf={conf:.2f} plausible={is_plausible_plate(text)}")
                    if is_plausible_plate(text) and conf >= 0.30:
                        readings.append((text, conf, frame_idx, ts))
            if progress_cb and total_frames:
                progress_cb(min(99, int(100 * frame_idx / total_frames)))
        frame_idx += 1

    cap.release()
    groups = merge_readings(readings)
    groups.sort(key=lambda g: (-g['count'], -g['conf']))

    result = []
    for g in groups:
        result.append({
            'plate_number': g['text'],
            'confidence': round(g['conf'] * 100, 1),
            'frames_detected': g['count'],
            'first_seen_seconds': round(g['first_ts'], 1),
        })
    return result
