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

PLATE_RE = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$')
LOOSE_RE = re.compile(r'^[A-Z]{2}[A-Z0-9]{4,8}[0-9]{3,4}$')

MAX_FRAME_WIDTH = 640      # smaller frame = fewer contours + faster OCR crops
MAX_BOXES_PER_FRAME = 6    # only OCR the most plate-shaped candidates, not every contour


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


def resize_if_needed(frame):
    h, w = frame.shape[:2]
    if w > MAX_FRAME_WIDTH:
        scale = MAX_FRAME_WIDTH / float(w)
        frame = cv2.resize(frame, (MAX_FRAME_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def candidate_plate_regions(frame):
    """Return the most plate-shaped candidate boxes, ranked and capped so
    we don't waste OCR calls on every stray contour in a busy scene."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_f = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray_f, 30, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    scored = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 35 or h < 10:
            continue
        aspect = w / float(h)
        if not (1.5 <= aspect <= 7.0):
            continue
        area_ratio = (w * h) / float(w_img * h_img)
        if area_ratio < 0.0005 or area_ratio > 0.30:
            continue
        # score: prefer aspect ratios closest to a typical plate (~3.5) and larger area
        aspect_score = -abs(aspect - 3.5)
        scored.append((aspect_score + area_ratio, (x, y, w, h)))

    scored.sort(key=lambda s: s[0], reverse=True)
    boxes = [b for _, b in scored[:MAX_BOXES_PER_FRAME]]

    cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(35, 10))
    for (x, y, w, h) in list(detected)[:MAX_BOXES_PER_FRAME]:
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
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)
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


def process_video(video_path, sample_fps=1, progress_cb=None):
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
            small_frame = resize_if_needed(frame)
            boxes = dedupe_boxes(candidate_plate_regions(small_frame))
            for box in boxes:
                for text, conf in ocr_region(small_frame, box):
                    if is_plausible_plate(text) and conf >= 0.25:
                        readings.append((text, conf, frame_idx, ts))
            if progress_cb and total_frames:
                progress_cb(min(99, int(100 * frame_idx / total_frames)))
        frame_idx += 1
        del frame

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


def process_images(image_paths, progress_cb=None):
    """Process a list of still image file paths, return the same shape of
    result as process_video: list of dicts with plate_number, confidence,
    frames_detected (here: number of images it appeared in), first_seen_seconds
    (here: index of the image it was first seen in, for reference)."""
    readings = []
    total = len(image_paths)
    for idx, path in enumerate(image_paths):
        frame = cv2.imread(path)
        if frame is None:
            continue
        small_frame = resize_if_needed(frame)
        boxes = dedupe_boxes(candidate_plate_regions(small_frame))
        for box in boxes:
            for text, conf in ocr_region(small_frame, box):
                if is_plausible_plate(text) and conf >= 0.25:
                    readings.append((text, conf, idx, idx))
        del frame
        if progress_cb and total:
            progress_cb(min(99, int(100 * (idx + 1) / total)))

    groups = merge_readings(readings)
    groups.sort(key=lambda g: (-g['count'], -g['conf']))

    result = []
    for g in groups:
        result.append({
            'plate_number': g['text'],
            'confidence': round(g['conf'] * 100, 1),
            'frames_detected': g['count'],
            'first_seen_seconds': g['first_ts'],
        })
    return result
