"""
Centralized configuration for ExamVision.

Every threshold, duration, and geometric constant that affects
behavior lives here. Nothing in the rest of the codebase should
contain a bare magic number for these concerns.

Units:
    - All 3D geometry (face model, eye offsets, eyeball radius) is
      expressed in the same arbitrary-but-consistent millimeter-like
      scale as the canonical face model below. This scale is generic
      (not measured per-subject), which is documented as a known
      limitation in README.md.
    - Durations are in seconds.
    - Angles are in degrees unless the variable name says "_rad".
"""
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Canonical 3D face model (generic, average adult face).
# This is the widely-used 6-point model from classic OpenCV head-pose
# tutorials. It is NOT subject-specific -- solvePnP only needs a
# consistent model to recover rotation/translation, not an anatomically
# exact one. Points are in "model millimeters", origin at nose tip.
# ---------------------------------------------------------------------------
FACE_MODEL_3D = {
    "nose_tip": (0.0, 0.0, 0.0),
    "chin": (0.0, -330.0, -65.0),
    "left_eye_outer_corner": (-225.0, 170.0, -135.0),
    "right_eye_outer_corner": (225.0, 170.0, -135.0),
    "left_mouth_corner": (-150.0, -150.0, -125.0),
    "right_mouth_corner": (150.0, -150.0, -125.0),
}

# MediaPipe FaceMesh landmark indices for the same six stable points.
# These are deliberately NOT eyelid/iris points -- see README section on
# stable-landmark selection.
FACE_MODEL_LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer_corner": 33,
    "right_eye_outer_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}

# Extra landmarks used elsewhere (not for pose solving).
LEFT_EYE_INNER_CORNER = 133
RIGHT_EYE_INNER_CORNER = 362
LEFT_EYE_OUTER_CORNER = 33
RIGHT_EYE_OUTER_CORNER = 263

# MediaPipe refined iris landmarks (requires refine_landmarks=True).
# Each eye has 5: the iris CENTER (468 / 473) followed by 4 CONTOUR points.
# The full ring is for drawing only; iris-center/validity geometry must use
# the contour alone -- the center point sits at radius ~0 from the centroid,
# which would cap the ring-roundness validity score at ~0.5 on every frame.
LEFT_IRIS_RING = [468, 469, 470, 471, 472]
RIGHT_IRIS_RING = [473, 474, 475, 476, 477]
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_IRIS_CONTOUR = [469, 470, 471, 472]
RIGHT_IRIS_CONTOUR = [474, 475, 476, 477]

# Eyelid landmarks used only for eye-openness / blink estimation
# (never for gaze-center geometry).
LEFT_EYE_LID = {"top": 159, "bottom": 145}
RIGHT_EYE_LID = {"top": 386, "bottom": 374}


@dataclass
class EyeGeometryConfig:
    # Eyeball center offset from the corresponding outer-corner model
    # point, expressed in the FACE model's own local axes
    # (medial_shift along +/-X toward the nose, backward along -Z into
    # the socket, slight downward along -Y). Approximate anatomical
    # values; documented as a simplification (no per-subject orbit scan).
    medial_shift_mm: float = 67.5   # ~0.3 * interocular distance (450mm model)
    backward_shift_mm: float = 20.0
    downward_shift_mm: float = 8.0
    # Approximate adult eyeball radius, same model scale.
    eyeball_radius_mm: float = 60.0


@dataclass
class ConfidenceConfig:
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Below this, iris measurements are not trusted for gaze math.
    min_iris_validity: float = 0.35
    # Below this pnp-reprojection quality, head pose/eye geometry is
    # not trusted this frame.
    max_pnp_reprojection_error_px: float = 12.0
    # Overall gaze confidence floor to leave "UNCERTAIN".
    min_gaze_confidence: float = 0.4
    # Skew between the two eyes' gaze rays, as an angle seen from the
    # camera, at which binocular confidence reaches its maximum penalty.
    max_binocular_disagreement_deg: float = 10.0


@dataclass
class HeadPoseConfig:
    # Beyond this yaw/pitch, head pose is considered too extreme for
    # reliable eye-geometry estimation this frame.
    max_reliable_yaw_deg: float = 45.0
    max_reliable_pitch_deg: float = 40.0
    # How much head pose is allowed to shift the fused confidence,
    # relative to eye-gaze evidence (eye gaze must dominate).
    head_pose_influence_weight: float = 0.25


@dataclass
class ScreenConfig:
    # Fraction of screen half-width/half-height treated as margin
    # before "outside" is declared (avoids flip-flopping at the edge).
    margin_fraction: float = 0.12
    # Assumed plane distance (model units) used as the *starting*
    # value before calibration refines it via least squares.
    initial_plane_distance_mm: float = 600.0


@dataclass
class CalibrationConfig:
    # Normalized screen-space targets in [-1, 1] x [-1, 1], (x right, y up).
    points: tuple = (
        (0.0, 0.0),     # center
        (-0.85, 0.0),   # left
        (0.85, 0.0),    # right
        (0.0, 0.85),    # up
        (0.0, -0.85),   # down
        (-0.85, 0.85),
        (0.85, 0.85),
        (-0.85, -0.85),
        (0.85, -0.85),
    )
    samples_per_point: int = 25
    max_sample_std_normalized: float = 0.18  # quality gate, see calibration.py
    min_valid_sample_fraction: float = 0.6


@dataclass
class TemporalConfig:
    # Exponential moving average smoothing factor for the intersection
    # point and confidence. Higher = more responsive, less smooth.
    ema_alpha: float = 0.25
    # Hysteresis durations before a state transition is committed.
    seconds_to_confirm_away: float = 3.5
    seconds_to_confirm_screen: float = 0.5
    seconds_to_confirm_face_lost: float = 1.5
    seconds_to_confirm_multi_face: float = 2.0
    seconds_to_clear_face_lost: float = 0.3


@dataclass
class CameraConfig:
    device_index: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    processing_width: int = 640  # landmark inference runs at this width
    target_fps: int = 30


@dataclass
class VisualizationConfig:
    show_landmarks: bool = True
    show_iris: bool = True
    show_eye_centers: bool = True
    show_gaze_rays: bool = True
    show_head_axes: bool = True
    show_screen_plane: bool = True
    show_numeric_overlay: bool = True
    show_fps: bool = True


@dataclass
class AutoCalibrationConfig:
    """Self-calibration from the uploaded video itself (see
    video_analysis/auto_calibration.py). A recorded video has no
    calibration dots, so the screen is located from the candidate's own
    dominant gaze direction; the screen's angular size is a prior."""
    # Plane z (model units). 0 = the plane containing the camera, which
    # is where a laptop/monitor-mounted webcam's screen physically is.
    plane_distance_mm: float = 0.0
    # Half-angles of the screen as seen from the candidate's eyes. A
    # 15.6" laptop at ~60 cm is ~16 x 9.5 deg; a 24" monitor at ~70 cm
    # is ~21 x 12 deg. The defaults add headroom for iris-landmark noise.
    # 25 (was 22): looking at a screen corner on the user-labelled
    # 'sherry30' video measured up to ~1.27 half-widths with 22; side
    # glances measured 1.3-1.7. The eye rule is the main sideways check.
    screen_half_angle_x_deg: float = 25.0
    screen_half_angle_y_deg: float = 15.0
    # Minimum confident frames needed to locate the screen.
    min_samples: int = 30
    # Below this fraction of confident frames landing inside the fitted
    # screen, the "mostly looking at the screen" assumption is suspect.
    min_screen_consistency: float = 0.5
    # --- Screen-reading eye position (gaze/eye_direction.py fit_eye_baseline)
    # Only roughly-ahead frames inside this loose prior (blendshape units:
    # eye_x sideways, eye_v = up - down) are candidates for "the screen".
    # Among them, the common eye positions (clusters) are found. The largest
    # is the screen, unless another substantial cluster sits clearly below
    # it (more than cluster_radius lower, within cluster_max_side_offset
    # sideways): a webcam sits above the screen, so then the lower one is
    # the screen and the larger one is looking above it.
    screen_prior_max_side: float = 0.30
    screen_prior_max_up: float = 0.45     # sanity bound only; screen reading measured -0.33..+0.08
    screen_prior_max_down: float = 0.70   # eye_v below -this is looking far below the screen
    # Frames within this distance of a cluster centre belong to it.
    cluster_radius: float = 0.12
    # A cluster counts as "substantial" with at least this share of the
    # largest cluster's frames (and at least min_samples frames).
    cluster_min_share: float = 0.25
    cluster_max_side_offset: float = 0.15
    # Below this share of open-eye frames inside the prior, the baseline
    # falls back to the densest cluster of ALL frames and the report warns.
    min_screen_prior_fraction: float = 0.15
    # Smoothing width of the density estimate (blendshape units).
    baseline_bandwidth: float = 0.06
    # The 3D screen position is fitted from frames whose eyes are within
    # this fraction of the on-screen ellipse around the baseline.
    near_baseline_fraction: float = 0.5


@dataclass
class IncidentConfig:
    # Incidents separated by a gap shorter than this are merged into one.
    merge_gap_s: float = 1.5
    # Merged incidents shorter than this are dropped as noise.
    min_incident_duration_s: float = 1.5
    # Supporting rule: sustained head rotation far from the candidate's
    # own median pose counts as not looking at the screen, even when the
    # eyes cannot be measured. Persistence reuses
    # TemporalConfig.seconds_to_confirm_away.
    enable_head_turn_rule: bool = True
    head_turn_yaw_deg: float = 35.0
    head_turn_pitch_deg: float = 30.0
    # Face absent from the frame (persistence reuses
    # TemporalConfig.seconds_to_confirm_face_lost) counts as not looking.
    count_face_lost_as_not_looking: bool = True
    # Eye-direction rule (gaze/eye_direction.py): eyes held turned away from
    # the candidate's OWN screen-reading position (fit_eye_baseline) for >=
    # seconds_to_confirm_away, where glances back shorter than merge_gap_s
    # do not reset the timer (the pattern of reading notes).
    # Limits are deviations from that baseline in blendshape units
    # (eye_x sideways, eye_v = up - down), tested as an ELLIPSE so diagonal
    # looks toward a corner count. Tuned 2026-09-30 on four labelled test
    # videos (sherry30 flagged by the user; sir, sherry, ali); re-tune with
    # tools/tune_eye_limits.py as more labelled videos become available.
    # The baseline is the LOWER part of the screen when a candidate reads at
    # several heights, so "up" must cover the rest of the screen (0.34);
    # looking at a screen corner measured up to ~0.46 sideways, notes and
    # side glances 0.55-0.95 (0.45).
    enable_eye_direction_rule: bool = True
    eye_limit_side: float = 0.45
    eye_limit_up: float = 0.34
    eye_limit_down: float = 0.35
    # Label a look as diagonal (e.g. DOWN-LEFT) when the smaller normalized
    # component is at least this fraction of the larger one (0.6 ~ 31 deg).
    eye_diagonal_min_ratio: float = 0.6
    # "Away" runs shorter than this are treated as landmark flicker and
    # ignored (so they cannot stretch a real incident's start or end).
    eye_min_run_s: float = 0.4
    eye_blink_ignore: float = 0.50        # frames this closed are skipped (blinks) ...
    # ... unless the eyes stay this closed for at least this long: looking
    # far down lowers the lids enough to read as a blink. 0 disables.
    eye_closed_as_away_s: float = 1.0


@dataclass
class VerdictConfig:
    """CHEATING SUSPECTED is returned when ANY rule below is met."""
    min_incidents: int = 3
    min_single_incident_s: float = 8.0
    min_not_looking_fraction: float = 0.10
    # With no rule met, the verdict is INCONCLUSIVE instead of NO
    # CHEATING when gaze could be measured on less than this fraction of
    # the frames where a face was visible.
    min_gaze_coverage: float = 0.4


@dataclass
class ObjectDetectionConfig:
    """Phone, book and extra-person detection (detection/object_detector.py,
    detection/evidence.py). Any of them makes the verdict CHEATING when its
    *_is_cheating switch is on."""
    enabled: bool = True
    # efficientdet_lite2 (more accurate, ~110 ms/frame) or efficientdet_lite0
    # (faster, misses more small phones).
    model: str = "efficientdet_lite2"
    # Frames checked per second of video (the detector is too slow for every frame).
    samples_per_second: float = 2.0
    # Minimum detector scores. Books are higher because shelves and papers
    # in the background score 0.2-0.35.
    phone_min_score: float = 0.40
    book_min_score: float = 0.45
    person_min_score: float = 0.60
    # A second person must fill at least this share of the frame (ignores
    # tiny figures far in the background).
    person_min_area: float = 0.02
    # A phone or book must be seen this long (short gaps up to
    # IncidentConfig.merge_gap_s allowed) before it counts.
    seconds_to_confirm_object: float = 1.0
    # A book in the same place for at least this share of the video is
    # treated as background (a shelf), reported as a note, not flagged.
    static_book_fraction: float = 0.8
    phone_is_cheating: bool = True
    book_is_cheating: bool = True
    multiple_people_is_cheating: bool = True


@dataclass
class VideoAnalysisConfig:
    auto_calibration: AutoCalibrationConfig = field(default_factory=AutoCalibrationConfig)
    incidents: IncidentConfig = field(default_factory=IncidentConfig)
    verdict: VerdictConfig = field(default_factory=VerdictConfig)
    objects: ObjectDetectionConfig = field(default_factory=ObjectDetectionConfig)
    # Frames wider than this are downscaled for landmark inference only
    # (gaze geometry is angle-based, so it is resolution-independent).
    # Clips and snapshots are always cut from the original frames.
    processing_width: int = 960
    # Context kept before/after each incident in its recording clip.
    clip_padding_s: float = 1.0
    # Also write one full-length video with the verdict banner burned in.
    export_annotated_video: bool = False
    # Browser-playable (H.264) copy of the whole video for the web review
    # player, saved as data/review_video.mp4, downscaled to this height.
    export_review_video: bool = True
    review_video_max_height: int = 720
    # Copy the uploaded video into the candidate's results folder.
    copy_source_video: bool = False
    results_dir: str = "results"
    video_extensions: tuple = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg", ".3gp")


@dataclass
class AppConfig:
    eye: EyeGeometryConfig = field(default_factory=EyeGeometryConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    video: VideoAnalysisConfig = field(default_factory=VideoAnalysisConfig)
    calibration_file: str = "calibration_data.json"
    session_log_dir: str = "session_logs"
