"""
video_analysis/analyzer.py

VideoAnalyzer: the uploaded-video counterpart of the live webcam loop.

    1. Measure   Decode the video frame by frame and run the existing
                 per-frame geometry (GazePipeline.measure) on every frame,
                 time-stamped with the video's own media time.
    2. Calibrate Fit the screen position from the video itself
                 (auto_calibration.py), or use the saved webcam
                 calibration_data.json ("file" mode).
    3. Decide    Replay the existing GazeClassifier -> TemporalFilter ->
                 GazeStateManager -> EventEngine over the measurements.
    4. Incidents Extract timestamped "not looking at the screen"
                 incidents, statistics and the verdict (incidents.py).
    5. Export    Create results/<Candidate_Name>/, write one clip and one
                 snapshot per incident, the data logs and the reports.

Nothing is written to disk until step 5, so cancelling or failing
earlier leaves no half-finished results folder behind; a cancel during
step 5 removes the folder that this run created.
"""
from __future__ import annotations

import dataclasses
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from config import AppConfig
from gaze.calibration import CalibrationManager
from gaze.eye_direction import eyes_away_direction
from pipeline import FrameMeasurement, GazePipeline
from video_analysis.auto_calibration import fit_auto_calibration
from video_analysis.exporter import export_media
from video_analysis.incidents import (
    AnalysisStats, FrameRecord, Incident, Verdict, compute_stats, compute_verdict, extract_incidents,
)
from video_analysis.output_layout import ResultPaths, create_result_folders, validate_candidate_name
from video_analysis.report import ReportContext, write_data_files, write_reports
from video_analysis.video_source import VideoError, VideoFrameReader, VideoInfo, probe_video

CALIBRATION_AUTO = "auto"
CALIBRATION_FILE = "file"

# Overall progress share of each stage (measuring dominates the runtime).
_P_MEASURE = (0.00, 0.82)
_P_DECIDE = (0.82, 0.85)
_P_EXPORT = (0.85, 0.98)

ProgressCallback = Callable[[float, str], None]


class AnalysisCancelled(Exception):
    pass


class AnalysisError(Exception):
    """A problem the user can act on (bad name, unreadable video, ...)."""


@dataclass
class AnalysisResult:
    candidate_name: str
    paths: ResultPaths
    report_html: Path
    verdict: Verdict
    stats: AnalysisStats
    incidents: List[Incident]
    warnings: List[str] = field(default_factory=list)


class VideoAnalyzer:
    def __init__(self, cfg: AppConfig, project_dir: Path):
        self.cfg = cfg
        self.project_dir = Path(project_dir)

    @property
    def results_root(self) -> Path:
        return self.project_dir / self.cfg.video.results_dir

    @property
    def calibration_file(self) -> Path:
        return self.project_dir / self.cfg.calibration_file

    def saved_calibration_available(self) -> bool:
        probe = CalibrationManager(self.cfg.calibration, self.cfg.screen)
        return probe.load(str(self.calibration_file)) and probe.result is not None and probe.result.success

    # ------------------------------------------------------------------ #
    def run(
        self,
        candidate_name: str,
        video_path: str,
        calibration_mode: str = CALIBRATION_AUTO,
        progress: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> AnalysisResult:
        started = time.time()
        candidate_name = (candidate_name or "").strip()
        report_progress = progress or (lambda fraction, message: None)

        def check_cancel() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise AnalysisCancelled()

        def stage(bounds):
            lo, hi = bounds
            return lambda f, msg: report_progress(lo + (hi - lo) * min(max(f, 0.0), 1.0), msg)

        error = validate_candidate_name(candidate_name)
        if error:
            raise AnalysisError(error)
        if calibration_mode not in (CALIBRATION_AUTO, CALIBRATION_FILE):
            raise AnalysisError(f"Unknown calibration mode {calibration_mode!r}")
        try:
            info = probe_video(video_path)
        except VideoError as e:
            raise AnalysisError(str(e)) from e

        warnings: List[str] = []
        if not info.fps_reported:
            warnings.append(f"The video did not report a frame rate; {info.fps:.0f} fps was assumed for clips.")

        # 1. Measure --------------------------------------------------------
        report_progress(0.0, "Loading the face-landmark model")
        pipeline = GazePipeline(self.cfg)
        try:
            measurements, times = self._measure_all(pipeline, info, stage(_P_MEASURE), check_cancel)
        finally:
            pipeline.close()
        if not measurements:
            raise AnalysisError("No frames could be decoded from the video.")
        if info.frame_count and len(measurements) < 0.95 * info.frame_count:
            warnings.append(f"Only {len(measurements)} of the {info.frame_count} frames the file reports "
                            f"could be decoded; the end of the video may be damaged.")

        # 2. Calibrate ------------------------------------------------------
        report_progress(_P_DECIDE[0], "Locating the screen")
        gaze_available, calibration_desc, calibration_data = self._calibrate(
            pipeline, calibration_mode, measurements, warnings)

        # 3 + 4. Decide, incidents, verdict ---------------------------------
        report_progress(_P_DECIDE[0], "Classifying gaze frame by frame")
        records = self._decide_all(pipeline, measurements, times, gaze_available)
        del measurements
        incidents = extract_incidents(records, self.cfg, info.fps, gaze_available)
        stats = compute_stats(records, incidents, info.fps)
        eye_rule_available = (self.cfg.video.incidents.enable_eye_direction_rule
                              and any(r.eye_x is not None for r in records))
        verdict = compute_verdict(stats, self.cfg, gaze_available or eye_rule_available)
        check_cancel()

        # 5. Export ---------------------------------------------------------
        paths = create_result_folders(self.results_root, candidate_name)
        try:
            write_data_files(paths, records, pipeline.event_engine.events, calibration_data)
            exported = export_media(info, incidents, records, paths, self.cfg, stage(_P_EXPORT), check_cancel)
            warnings += exported.warnings
            if self.cfg.video.copy_source_video:
                report_progress(_P_EXPORT[1], "Copying the original video")
                paths.source.mkdir(parents=True, exist_ok=True)
                shutil.copy2(info.path, paths.source / Path(info.path).name)
            check_cancel()
            report_progress(_P_EXPORT[1], "Writing the report")
            ctx = ReportContext(
                candidate_name=candidate_name, video=info, analyzed_at=started,
                processing_seconds=time.time() - started, calibration_mode=calibration_mode,
                calibration_description=calibration_desc, gaze_available=gaze_available,
                stats=stats, verdict=verdict, incidents=incidents, warnings=warnings,
                review_video=exported.review_video,
            )
            report_html = write_reports(paths, ctx, self.cfg)
        except BaseException:
            # Cancelled or failed mid-export: don't leave a half-written
            # results folder that looks like a finished analysis.
            shutil.rmtree(paths.root, ignore_errors=True)
            raise

        report_progress(1.0, "Done")
        return AnalysisResult(candidate_name, paths, report_html, verdict, stats, list(incidents), warnings)

    # ------------------------------------------------------------------ #
    def _measure_all(self, pipeline: GazePipeline, info: VideoInfo, progress: ProgressCallback,
                     check_cancel: Callable[[], None]):
        target_w = self.cfg.video.processing_width
        measurements: List[FrameMeasurement] = []
        times: List[float] = []
        expected = info.frame_count
        with VideoFrameReader(info) as reader:
            for idx, t, frame in reader.frames():
                check_cancel()
                h, w = frame.shape[:2]
                if w > target_w:
                    frame = cv2.resize(frame, (target_w, int(round(h * target_w / w))), interpolation=cv2.INTER_AREA)
                m = pipeline.measure(frame, t, timestamp_ms=int(round(t * 1000)))
                measurements.append(m.compact())
                times.append(t)
                if idx % 10 == 0:
                    if expected:
                        progress(idx / expected, f"Analyzing frame {idx + 1} of {expected}")
                    else:
                        progress(0.0, f"Analyzing frame {idx + 1}")
        progress(1.0, f"Analyzed {len(measurements)} frames")
        return measurements, times

    def _calibrate(self, pipeline: GazePipeline, mode: str, measurements: List[FrameMeasurement],
                   warnings: List[str]):
        """Installs a calibration into pipeline.calibration. Returns
        (gaze_available, human description, dict for calibration.json)."""
        if mode == CALIBRATION_FILE:
            cal = pipeline.calibration
            if cal.load(str(self.calibration_file)) and cal.result is not None and cal.result.success:
                r = cal.result
                desc = (f"Saved webcam calibration ({self.calibration_file.name}, quality {r.quality}, "
                        f"score {r.quality_score:.2f}). Only valid if the video was recorded on the same "
                        f"computer, webcam and seating position.")
                return True, desc, {
                    "mode": CALIBRATION_FILE, "file": str(self.calibration_file), "quality": r.quality,
                    "quality_score": r.quality_score, "affine_matrix": r.affine_matrix.tolist(),
                    "affine_offset": r.affine_offset.tolist(), "plane_distance_mm": r.plane_distance_mm,
                }
            pipeline.calibration.reset()
            warnings.append(f"No usable saved calibration was found at {self.calibration_file}; "
                            f"the screen was located automatically from the video instead.")

        summary = fit_auto_calibration(measurements, self.cfg)
        data = {"mode": CALIBRATION_AUTO, **summary.to_dict()}
        if not summary.success:
            warnings.append(summary.message)
            return False, summary.message, data
        pipeline.calibration.result = summary.result
        r = summary.result
        data.update({"quality": r.quality, "affine_matrix": r.affine_matrix.tolist(),
                     "affine_offset": r.affine_offset.tolist(), "plane_distance_mm": r.plane_distance_mm})
        if r.quality != "AUTO":
            warnings.append(summary.message)
        return True, f"Located automatically from the video. {summary.message}", data

    def _decide_all(self, pipeline: GazePipeline, measurements: List[FrameMeasurement], times: List[float],
                    gaze_available: bool) -> List[FrameRecord]:
        ic = self.cfg.video.incidents
        head = [m.head_pose for m in measurements if m.pose_ok and m.head_pose is not None]
        base_yaw = float(np.median([h.yaw_deg for h in head])) if head else 0.0
        base_pitch = float(np.median([h.pitch_deg for h in head])) if head else 0.0

        pipeline.reset_temporal_state()
        records: List[FrameRecord] = []
        for idx, (m, t) in enumerate(zip(measurements, times)):
            # Without a calibration the classifier would compare raw,
            # unnormalized plane coordinates against [-1, 1]; withholding
            # the ray makes it return UNCERTAIN instead of a false verdict.
            d = pipeline.decide(m if gaze_available else dataclasses.replace(m, combined_ray=None), t)
            hp = m.head_pose if m.pose_ok else None
            yaw_off = hp.yaw_deg - base_yaw if hp else None
            pitch_off = hp.pitch_deg - base_pitch if hp else None
            gr, sm = d.gaze_result, d.smoothed
            eye = m.eye if m.num_faces > 0 else None
            records.append(FrameRecord(
                index=idx,
                time_s=t,
                num_faces=m.num_faces,
                state=d.state,
                away_direction=gr.away_direction if gr else None,
                head_turned=bool(hp) and (abs(yaw_off) > ic.head_turn_yaw_deg or abs(pitch_off) > ic.head_turn_pitch_deg),
                head_yaw_offset=yaw_off,
                head_pitch_offset=pitch_off,
                gaze_confidence=gr.confidence if gr else None,
                screen_x=gr.screen_x if gr else None,
                screen_y=gr.screen_y if gr else None,
                smoothed_x=sm.screen_x if sm else None,
                smoothed_y=sm.screen_y if sm else None,
                smoothed_confidence=sm.confidence if sm else None,
                head_yaw=hp.yaw_deg if hp else None,
                head_pitch=hp.pitch_deg if hp else None,
                head_roll=hp.roll_deg if hp else None,
                eye_x=eye.x if eye else None,
                eye_up=eye.up if eye else None,
                eye_down=eye.down if eye else None,
                eyes_away=eyes_away_direction(eye, ic.eye_side_threshold, ic.eye_down_threshold,
                                              ic.eye_up_threshold, ic.eye_blink_ignore),
            ))
        return records
