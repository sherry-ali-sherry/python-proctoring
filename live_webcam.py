"""
live_webcam.py

The original ExamVision Phase 1 real-time webcam mode, kept as an
optional tool. The main application (main.py) now analyzes uploaded
videos instead; the only reason to run this is to produce a saved
personal calibration (calibration_data.json) that video analysis can
use via "--calibration file" when the video was recorded on this same
computer/webcam setup.

Controls (debug UI):
    c  - run/redo calibration
    l  - toggle landmark overlay        i - toggle iris overlay
    e  - toggle eye-center overlay      g - toggle gaze-ray overlay
    h  - toggle head-axis overlay       n - toggle numeric overlay
    r  - start/stop session recording (CSV + event JSON)
    q  - quit
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2

from config import AppConfig
from camera.camera_manager import CameraManager
from pipeline import GazePipeline
from visualization.overlay import OverlayRenderer
from proctor_logging.session_logger import SessionLogger

PROJECT_DIR = Path(__file__).resolve().parent


class ExamVisionApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.camera = CameraManager(cfg.camera)
        self.pipeline = GazePipeline(cfg)
        self.overlay = OverlayRenderer(cfg.visualization)
        self.logger: SessionLogger | None = None
        self.calibration_path = str(PROJECT_DIR / cfg.calibration_file)

        self.pipeline.calibration.load(self.calibration_path)

    # ------------------------------------------------------------------ #
    def process_frame(self, frame_bgr) -> dict:
        """Run the full per-frame pipeline. Returns a dict with every
        intermediate result needed for overlay/logging, so nothing has
        to be recomputed for drawing."""
        now = time.time()
        m = self.pipeline.measure(frame_bgr, now)
        d = self.pipeline.decide(m, now)
        return {
            "now": now, "num_faces": m.num_faces, "face": m.face, "pose": m.pose,
            "head_pose": m.head_pose, "eye_centers": m.eye_centers,
            "left_ray": m.left_ray, "right_ray": m.right_ray, "combined_ray": m.combined_ray,
            "gaze_result": d.gaze_result, "iris_left_valid": m.iris_left_valid,
            "iris_right_valid": m.iris_right_valid, "smoothed": d.smoothed,
        }

    # ------------------------------------------------------------------ #
    def run_calibration(self, cap_read_fn) -> None:
        """Blocking calibration sequence using OpenCV's own window for
        both video feed and calibration dots (spec section 14)."""
        calibration = self.pipeline.calibration
        calibration.reset()
        window = "ExamVision Calibration"
        cfg = self.cfg.calibration

        for target in calibration.target_points:
            collected = 0
            attempts = 0
            max_attempts = cfg.samples_per_point * 4
            while collected < cfg.samples_per_point and attempts < max_attempts:
                attempts += 1
                frame_result = cap_read_fn()
                if not frame_result.ok:
                    continue
                frame = frame_result.frame_bgr
                pipeline_out = self.process_frame(frame)
                self.overlay.draw_calibration_target(frame, target, collected / cfg.samples_per_point)
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC aborts calibration
                    cv2.destroyWindow(window)
                    return
                ray = pipeline_out.get("combined_ray")
                if ray is not None and pipeline_out.get("gaze_result") and pipeline_out["gaze_result"].confidence > 0.3:
                    if calibration.add_sample(target, ray):
                        collected += 1

        result = calibration.compute()
        if result.success:
            calibration.save(self.calibration_path)
        else:
            self.pipeline.event_engine.on_calibration_failed(result.message, time.time())
        cv2.destroyWindow(window)
        print(f"[Calibration] {result.quality}: {result.message} (score={result.quality_score:.2f})")

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        if not self.camera.open():
            print("ERROR: could not open webcam.")
            return

        window = "ExamVision"
        prev_t = time.time()
        fps = 0.0
        recording = False

        print("Press 'c' to calibrate (recommended before first use), 'q' to quit.")

        try:
            while True:
                frame_result = self.camera.read()
                if not frame_result.ok:
                    continue
                frame = frame_result.frame_bgr

                out = self.process_frame(frame)

                now = time.time()
                dt = now - prev_t
                prev_t = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                calibration = self.pipeline.calibration
                calibration_status = calibration.result.quality if calibration.result else "NOT_CALIBRATED"

                display = self.overlay.draw(
                    frame, out["face"], out["pose"], out["head_pose"], out["eye_centers"],
                    out["left_ray"], out["right_ray"], out["combined_ray"], out["gaze_result"],
                    self.pipeline.state_manager.state, out["num_faces"], calibration_status, fps,
                )
                cv2.imshow(window, display)

                if recording and self.logger is not None:
                    self._log_frame(out)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.run_calibration(self.camera.read)
                elif key == ord('l'):
                    self.cfg.visualization.show_landmarks = not self.cfg.visualization.show_landmarks
                elif key == ord('i'):
                    self.cfg.visualization.show_iris = not self.cfg.visualization.show_iris
                elif key == ord('e'):
                    self.cfg.visualization.show_eye_centers = not self.cfg.visualization.show_eye_centers
                elif key == ord('g'):
                    self.cfg.visualization.show_gaze_rays = not self.cfg.visualization.show_gaze_rays
                elif key == ord('h'):
                    self.cfg.visualization.show_head_axes = not self.cfg.visualization.show_head_axes
                elif key == ord('n'):
                    self.cfg.visualization.show_numeric_overlay = not self.cfg.visualization.show_numeric_overlay
                elif key == ord('r'):
                    if not recording:
                        self.logger = SessionLogger(str(PROJECT_DIR / self.cfg.session_log_dir))
                        recording = True
                        print(f"[Recording] started -> {self.logger.csv_path}")
                    else:
                        self.logger.close(self.pipeline.event_engine.events)
                        print(f"[Recording] stopped -> {self.logger.json_path}")
                        recording = False
        finally:
            if recording and self.logger is not None:
                self.logger.close(self.pipeline.event_engine.events)
            self.camera.release()
            self.pipeline.close()
            cv2.destroyAllWindows()

    def _log_frame(self, out: dict) -> None:
        gr = out.get("gaze_result")
        hp = out.get("head_pose")
        ec = out.get("eye_centers")
        lr, rr, cr = out.get("left_ray"), out.get("right_ray"), out.get("combined_ray")

        def xyz(vec, prefix):
            if vec is None:
                return {f"{prefix}_x": "", f"{prefix}_y": "", f"{prefix}_z": ""}
            return {f"{prefix}_x": vec[0], f"{prefix}_y": vec[1], f"{prefix}_z": vec[2]}

        row = {
            "timestamp": out["now"],
            "state": self.pipeline.state_manager.state.value,
            "confidence": gr.confidence if gr else "",
            "yaw": hp.yaw_deg if hp else "",
            "pitch": hp.pitch_deg if hp else "",
            "roll": hp.roll_deg if hp else "",
            "screen_intersection_x": gr.screen_x if gr else "",
            "screen_intersection_y": gr.screen_y if gr else "",
            "face_detected": out["face"] is not None,
            "iris_detected": out["iris_left_valid"] or out["iris_right_valid"],
            "number_of_faces": out["num_faces"],
        }
        row.update(xyz(ec.left_eye_center_3d if ec else None, "left_eye"))
        row.update(xyz(ec.right_eye_center_3d if ec else None, "right_eye"))
        row.update(xyz(lr.direction if lr else None, "left_gaze"))
        row.update(xyz(rr.direction if rr else None, "right_gaze"))
        row.update(xyz(cr.direction if cr else None, "combined_gaze"))
        self.logger.log_frame(row)


if __name__ == "__main__":
    app = ExamVisionApp(AppConfig())
    app.run()
