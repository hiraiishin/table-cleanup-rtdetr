import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# =========================
# Аргументы
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прототип детекции событий для одного столика по видео с несколькими ROI."
    )
    parser.add_argument("--video", required=True, help="Путь к входному видео")
    parser.add_argument("--output", default="output.mp4", help="Путь к выходному видео")
    parser.add_argument("--events_csv", default="events.csv", help="CSV с событиями")
    parser.add_argument("--report_txt", default="report.txt", help="Текстовый отчет")
    parser.add_argument("--problem_frame", default="problem_frame.jpg", help="Кадр с проблемным кейсом")

    parser.add_argument(
        "--seat_rois",
        default=None,
        help="Несколько seat ROI через ; в формате x,y,w,h;x,y,w,h",
    )
    parser.add_argument(
        "--approach_rois",
        default=None,
        help="Несколько approach ROI через ; в формате x,y,w,h;x,y,w,h",
    )
    parser.add_argument(
        "--select_rois",
        action="store_true",
        help="Интерактивно выбрать несколько seat и approach ROI",
    )

    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Вес модели Ultralytics: yolov8s.pt, yolov8n.pt, rtdetr-l.pt и т.д.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Порог confidence")
    parser.add_argument("--imgsz", type=int, default=1280, help="Размер инференса")
    parser.add_argument("--device", default=None, help="Устройство: cpu, 0, 0,1")

    parser.add_argument("--debounce_sec", type=float, default=1.0, help="Подтверждение смены состояния")
    parser.add_argument("--empty_hold_sec", type=float, default=1.5, help="Сколько держать пустое состояние")
    parser.add_argument("--min_ioa", type=float, default=0.12, help="Минимальная IoA человека в ROI")
    parser.add_argument("--min_box_area", type=int, default=3500, help="Минимальная площадь bbox человека")
    parser.add_argument("--show", action="store_true", help="Показывать обработку в окне")
    return parser.parse_args()


# =========================
# Утилиты
# =========================

def ensure_parent_dir(path_str: str) -> None:
    path = Path(path_str)
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)


def parse_roi(roi_text: str) -> Tuple[int, int, int, int]:
    parts = [int(float(x.strip())) for x in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI должен быть в формате x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise ValueError("Ширина и высота ROI должны быть положительными")
    return x, y, w, h


def parse_multiple_rois(text: str) -> List[Tuple[int, int, int, int]]:
    if not text:
        return []
    items = [x.strip() for x in text.split(";") if x.strip()]
    return [parse_roi(item) for item in items]


def format_ts(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def rect_intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0
    return int((x_right - x_left) * (y_bottom - y_top))


def point_in_roi(point: Tuple[float, float], roi: Tuple[int, int, int, int]) -> bool:
    px, py = point
    rx, ry, rw, rh = roi
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def roi_to_box(roi: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = roi
    return x, y, x + w, y + h


# =========================
# Интерактивная разметка нескольких ROI
# =========================

def select_multiple_rois_from_first_frame(video_path: str, window_name: str) -> List[Tuple[int, int, int, int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Не удалось прочитать первый кадр для выбора ROI")

    original = frame.copy()
    rois: List[Tuple[int, int, int, int]] = []

    drawing = False
    start_pt = None
    current_pt = None

    instructions = [
        "LMB drag: draw ROI",
        "Enter/Space: finish",
        "u: undo last ROI",
        "c: clear all",
        "Esc: cancel",
    ]

    def redraw():
        canvas = original.copy()

        for i, (x, y, w, h) in enumerate(rois):
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                canvas,
                f"{i+1}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if drawing and start_pt is not None and current_pt is not None:
            x1, y1 = start_pt
            x2, y2 = current_pt
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w > 0 and h > 0:
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)

        y0 = 25
        for line in instructions:
            cv2.putText(
                canvas,
                line,
                (15, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y0 += 25

        cv2.imshow(window_name, canvas)

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_pt, current_pt, rois

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
            current_pt = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            current_pt = (x, y)

            x1, y1 = start_pt
            x2, y2 = current_pt
            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)

            if rw > 5 and rh > 5:
                rois.append((rx, ry, rw, rh))

            start_pt = None
            current_pt = None

        redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # Enter / Space
            break
        elif key == ord("u"):
            if rois:
                rois.pop()
            redraw()
        elif key == ord("c"):
            rois.clear()
            redraw()
        elif key == 27:  # Esc
            cv2.destroyWindow(window_name)
            raise RuntimeError("Выбор ROI отменен пользователем")

    cv2.destroyWindow(window_name)

    if not rois:
        raise RuntimeError("Не выбрано ни одного ROI")

    return rois


def pick_rois(args: argparse.Namespace) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    if args.select_rois:
        seat_rois = select_multiple_rois_from_first_frame(args.video, "Select SEAT ROIs")
        approach_rois = select_multiple_rois_from_first_frame(args.video, "Select APPROACH ROIs")
        return seat_rois, approach_rois

    seat_rois = parse_multiple_rois(args.seat_rois)
    approach_rois = parse_multiple_rois(args.approach_rois)

    if not seat_rois:
        raise ValueError("Нужно указать --seat_rois или использовать --select_rois")

    return seat_rois, approach_rois


# =========================
# Модель
# =========================

def load_model(model_name: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Не найден пакет ultralytics. Установи зависимости: pip install ultralytics opencv-python pandas numpy"
        ) from exc
    return YOLO(model_name)


def detect_people(model, frame: np.ndarray, conf: float, imgsz: int, device: Optional[str]) -> List[Dict]:
    kwargs = {
        "conf": conf,
        "verbose": False,
        "classes": [0],
        "imgsz": imgsz,
    }
    if device is not None:
        kwargs["device"] = device

    result = model.predict(frame, **kwargs)[0]
    people = []

    if result.boxes is None:
        return people

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0].item())
        people.append({"bbox": (x1, y1, x2, y2), "conf": score})

    return people


# =========================
# Геометрия
# =========================

def classify_person_for_rois(
    person_box: Tuple[int, int, int, int],
    rois: List[Tuple[int, int, int, int]],
    min_ioa: float,
    min_box_area: int,
) -> Dict:
    x1, y1, x2, y2 = person_box
    area = max(1, (x2 - x1) * (y2 - y1))

    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    foot = ((x1 + x2) / 2.0, y1 + 0.88 * (y2 - y1))

    best_ioa = 0.0
    matched = False
    matched_idx = -1

    if area < min_box_area:
        return {
            "match": False,
            "best_ioa": 0.0,
            "matched_idx": -1,
            "center": center,
            "foot": foot,
            "reason": "small_box",
        }

    for idx, roi in enumerate(rois):
        inter = rect_intersection_area(person_box, roi_to_box(roi))
        ioa = inter / area
        best_ioa = max(best_ioa, ioa)

        center_in = point_in_roi(center, roi)
        foot_in = point_in_roi(foot, roi)

        ok = (
            foot_in
            or (center_in and ioa >= min_ioa)
            or (ioa >= max(0.22, min_ioa + 0.08))
        )

        if ok:
            matched = True
            matched_idx = idx
            break

    return {
        "match": matched,
        "best_ioa": best_ioa,
        "matched_idx": matched_idx,
        "center": center,
        "foot": foot,
        "reason": "matched" if matched else "no_match",
    }


def draw_label(frame: np.ndarray, text: str, org: Tuple[int, int], color=(255, 255, 255)) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x, y - h - baseline - 6), (x + w + 8, y + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 4, y - 4), font, scale, color, thickness, cv2.LINE_AA)


# =========================
# Пайплайн
# =========================

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Видео не найдено: {args.video}")

    ensure_parent_dir(args.output)
    ensure_parent_dir(args.events_csv)
    ensure_parent_dir(args.report_txt)
    ensure_parent_dir(args.problem_frame)

    seat_rois, approach_rois = pick_rois(args)

    if args.device is None:
        try:
            import torch
            args.device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    print(f"[INFO] seat_rois = {seat_rois}")
    print(f"[INFO] approach_rois = {approach_rois}")
    print(f"[INFO] model = {args.model}")
    print(f"[INFO] device = {args.device}")

    model = load_model(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать выходное видео: {args.output}")

    events: List[Dict] = []
    delays_sec: List[float] = []
    pending_empty_ts: Optional[float] = None

    stable_state: Optional[str] = None
    candidate_state: Optional[str] = None
    candidate_since: Optional[float] = None
    no_person_since: Optional[float] = None
    problem_frame_saved = False

    def log_event(event_type: str, timestamp_sec: float, frame_number: int, note: str = "", delay_sec: Optional[float] = None) -> None:
        events.append(
            {
                "frame": frame_number,
                "timestamp_sec": round(timestamp_sec, 3),
                "timestamp_hms": format_ts(timestamp_sec),
                "event": event_type,
                "note": note,
                "delay_from_last_empty_sec": None if delay_sec is None else round(delay_sec, 3),
            }
        )

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        timestamp_sec = frame_idx / fps

        people = detect_people(model, frame, conf=args.conf, imgsz=args.imgsz, device=args.device)

        seat_people = []
        approach_people = []
        suspicious_case = False

        for person in people:
            seat_geom = classify_person_for_rois(
                person["bbox"], seat_rois, args.min_ioa, args.min_box_area
            )

            if approach_rois:
                approach_geom = classify_person_for_rois(
                    person["bbox"], approach_rois, args.min_ioa, args.min_box_area
                )
            else:
                x1, y1, x2, y2 = person["bbox"]
                approach_geom = {
                    "match": False,
                    "best_ioa": 0.0,
                    "matched_idx": -1,
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "foot": ((x1 + x2) / 2.0, y1 + 0.88 * (y2 - y1)),
                    "reason": "no_approach_rois",
                }

            person["seat_match"] = seat_geom["match"]
            person["approach_match"] = approach_geom["match"]
            person["ioa_seat"] = seat_geom["best_ioa"]
            person["ioa_approach"] = approach_geom["best_ioa"]
            person["center"] = seat_geom["center"]
            person["foot"] = seat_geom["foot"]
            person["seat_idx"] = seat_geom["matched_idx"]
            person["approach_idx"] = approach_geom["matched_idx"]

            if person["seat_match"]:
                seat_people.append(person)
            elif person["approach_match"]:
                approach_people.append(person)

            if (person["ioa_seat"] > 0.02 and not person["seat_match"]) or (
                person["ioa_approach"] > 0.02 and not person["approach_match"]
            ):
                suspicious_case = True

        if len(seat_people) > 0:
            raw_state = "occupied"
            no_person_since = None
        elif len(approach_people) > 0:
            raw_state = "approach"
            no_person_since = None
        else:
            if no_person_since is None:
                no_person_since = timestamp_sec
            raw_state = "empty" if (timestamp_sec - no_person_since) >= args.empty_hold_sec else (stable_state or "empty")

        if stable_state is None:
            stable_state = raw_state
            log_event(stable_state, timestamp_sec, frame_idx, note="initial_state")

        if raw_state != stable_state:
            if candidate_state != raw_state:
                candidate_state = raw_state
                candidate_since = timestamp_sec
            elif candidate_since is not None and (timestamp_sec - candidate_since) >= args.debounce_sec:
                prev_state = stable_state
                stable_state = raw_state
                candidate_state = None
                candidate_since = None

                if stable_state == "approach":
                    delay_val = None
                    if pending_empty_ts is not None:
                        delay_val = timestamp_sec - pending_empty_ts
                        delays_sec.append(delay_val)
                        pending_empty_ts = None
                    log_event("approach", timestamp_sec, frame_idx, note="person_in_approach_zone", delay_sec=delay_val)

                elif stable_state == "occupied":
                    if prev_state == "empty":
                        delay_val = None
                        if pending_empty_ts is not None:
                            delay_val = timestamp_sec - pending_empty_ts
                            delays_sec.append(delay_val)
                            pending_empty_ts = None
                        log_event("approach", timestamp_sec, frame_idx, note="directly_to_seat", delay_sec=delay_val)
                    log_event("occupied", timestamp_sec, frame_idx, note="person_in_seat_zone")

                elif stable_state == "empty":
                    pending_empty_ts = timestamp_sec
                    log_event("empty", timestamp_sec, frame_idx, note="table_became_empty")
        else:
            candidate_state = None
            candidate_since = None

        # Рисуем ROI
        for i, (x, y, w, h) in enumerate(seat_rois):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"S{i+1}", (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        for i, (x, y, w, h) in enumerate(approach_rois):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
            cv2.putText(frame, f"A{i+1}", (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2, cv2.LINE_AA)

        # Рисуем людей
        for person in people:
            x1, y1, x2, y2 = person["bbox"]

            if person["seat_match"]:
                color = (0, 0, 255)
            elif person["approach_match"]:
                color = (255, 165, 0)
            else:
                color = (180, 180, 180)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cx, cy = map(int, person["center"])
            fx, fy = map(int, person["foot"])
            cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)
            cv2.circle(frame, (fx, fy), 4, (0, 255, 255), -1)

            txt = f"p {person['conf']:.2f} s:{person['ioa_seat']:.2f} a:{person['ioa_approach']:.2f}"
            cv2.putText(frame, txt, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        draw_label(frame, f"state: {stable_state.upper()}", (20, 32))
        draw_label(frame, f"time: {format_ts(timestamp_sec)}", (20, 58))
        draw_label(frame, f"seat people: {len(seat_people)}", (20, 84))
        draw_label(frame, f"approach people: {len(approach_people)}", (20, 110))
        draw_label(frame, f"seat zones: {len(seat_rois)}", (20, 136))
        draw_label(frame, f"approach zones: {len(approach_rois)}", (20, 162))
        draw_label(frame, f"mean delay: {np.mean(delays_sec):.2f}s" if delays_sec else "mean delay: n/a", (20, 188))

        if suspicious_case and not problem_frame_saved:
            cv2.imwrite(args.problem_frame, frame)
            problem_frame_saved = True

        writer.write(frame)

        if args.show:
            cv2.imshow("Table event detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        if frame_idx % 150 == 0:
            print(
                f"[INFO] frame={frame_idx}, t={timestamp_sec:.1f}s, "
                f"state={stable_state}, seat={len(seat_people)}, approach={len(approach_people)}"
            )

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    events_df = pd.DataFrame(events)
    events_df.to_csv(args.events_csv, index=False, encoding="utf-8-sig")

    mean_delay = float(np.mean(delays_sec)) if delays_sec else float("nan")
    median_delay = float(np.median(delays_sec)) if delays_sec else float("nan")

    report_lines = [
        "Готово.",
        "Прототип детекции событий для одного столика",
        f"Видео: {args.video}",
        f"Seat ROIs: {seat_rois}",
        f"Approach ROIs: {approach_rois}",
        f"Модель: {args.model}",
        f"Порог confidence: {args.conf}",
        f"Размер инференса: {args.imgsz}",
        f"Debounce: {args.debounce_sec} сек",
        f"Empty hold: {args.empty_hold_sec} сек",
        f"Всего событий: {len(events_df)}",
        f"Количество интервалов empty -> approach: {len(delays_sec)}",
        f"Среднее время между уходом гостя и подходом следующего человека: {mean_delay:.3f} сек" if delays_sec else "Среднее время: n/a",
        f"Медиана времени между уходом гостя и подходом следующего человека: {median_delay:.3f} сек" if delays_sec else "Медиана времени: n/a",
        f"События сохранены в: {args.events_csv}",
        f"Выходное видео сохранено в: {args.output}",
        f"Проблемный кадр: {args.problem_frame if problem_frame_saved else 'не сохранен'}",
    ]

    with open(args.report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()