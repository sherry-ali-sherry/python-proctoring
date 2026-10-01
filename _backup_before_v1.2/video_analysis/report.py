"""
video_analysis/report.py

Writes the report and data files into a candidate's results folder:

    report/report.json    machine-readable source of truth, including
                          every setting used
    report/report.html    human-readable report with verdict, timestamps,
                          snapshots and links to each clip (printable)
    report/report.txt     the same content as plain text
    report/incidents.csv  one row per incident (timestamps, reasons, files)
    report/review.json    the reviewer's decisions (web app), if any
    data/frame_log.csv    one row per analyzed frame
    data/events.json      proctoring events from EventEngine (video time)
    data/calibration.json the calibration the gaze classification used

report.html / report.txt are rendered from report.json (+ review.json),
so they can be re-rendered at any time, e.g. after a reviewer confirms
incidents or overrides the verdict.
"""
from __future__ import annotations

import csv
import dataclasses
import html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from config import AppConfig
from proctoring.event_engine import ProctorEvent
from video_analysis.incidents import (
    AnalysisStats, FrameRecord, Incident, Verdict, VERDICT_CHEATING, VERDICT_NO_CHEATING, timecode,
)
from video_analysis.output_layout import ResultPaths
from video_analysis.video_source import VideoInfo

DISCLAIMER = (
    "This is an automated assessment based only on where the candidate was looking. "
    "It flags behaviour for review; a person should watch the clips before any decision is made."
)
DIRECTION_NOTE = (
    "Directions are as seen in the video (camera view): the candidate's own left appears on the right."
)

INCIDENT_UNREVIEWED = "unreviewed"
INCIDENT_CONFIRMED = "confirmed"
INCIDENT_FALSE_ALARM = "false_alarm"
INCIDENT_STATUS_TEXT = {
    INCIDENT_UNREVIEWED: "Not reviewed",
    INCIDENT_CONFIRMED: "Confirmed by reviewer",
    INCIDENT_FALSE_ALARM: "False alarm (reviewer)",
}


@dataclass
class ReportContext:
    candidate_name: str
    video: VideoInfo
    analyzed_at: float
    processing_seconds: float
    calibration_mode: str               # "auto" | "file"
    calibration_description: str
    gaze_available: bool
    stats: AnalysisStats
    verdict: Verdict
    incidents: Sequence[Incident]
    warnings: List[str] = field(default_factory=list)
    review_video: Optional[str] = None  # relative to the results folder


def fmt_duration(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)} min {s:.0f} s"
    h, m = divmod(m, 60)
    return f"{int(h)} h {int(m)} min {s:.0f} s"


# --------------------------------------------------------------------------- #
# Data files
# --------------------------------------------------------------------------- #
def _event_to_dict(e: ProctorEvent) -> dict:
    def t(v):
        return None if v is None else round(v, 3)
    d = {
        "time_s": t(e.timestamp),
        "timecode": timecode(e.timestamp),
        "event": e.event_type,
        "direction": e.direction,
        "start_s": t(e.start_time),
        "end_s": t(e.end_time),
        "duration_s": t(e.duration),
        "confidence": None if e.confidence is None else round(e.confidence, 3),
    }
    d.update(e.metadata)
    return d


def write_data_files(paths: ResultPaths, records: Sequence[FrameRecord], events: Sequence[ProctorEvent],
                     calibration: dict) -> None:
    def num(v, digits=4):
        return "" if v is None else round(v, digits)

    with open(paths.data / "frame_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "time_s", "timecode", "num_faces", "state", "gaze_confidence",
                    "screen_x", "screen_y", "smoothed_x", "smoothed_y", "smoothed_confidence",
                    "away_direction", "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
                    "head_turned", "eye_x", "eye_up", "eye_down", "eyes_away",
                    "not_looking", "incident_number"])
        for r in records:
            w.writerow([r.index, round(r.time_s, 4), timecode(r.time_s), r.num_faces, r.state.value,
                        num(r.gaze_confidence), num(r.screen_x), num(r.screen_y), num(r.smoothed_x),
                        num(r.smoothed_y), num(r.smoothed_confidence), r.away_direction or "",
                        num(r.head_yaw, 2), num(r.head_pitch, 2), num(r.head_roll, 2),
                        int(r.head_turned), num(r.eye_x, 3), num(r.eye_up, 3), num(r.eye_down, 3),
                        r.eyes_away or "", int(r.incident_number is not None), r.incident_number or ""])

    # LOOKING_AWAY_CONTINUED fires on every frame of an away period; the
    # STARTED/RESUMED pair already carries its start, end and duration.
    kept = [_event_to_dict(e) for e in events if e.event_type != "LOOKING_AWAY_CONTINUED"]
    with open(paths.data / "events.json", "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2)
    with open(paths.data / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)


# --------------------------------------------------------------------------- #
# report.json
# --------------------------------------------------------------------------- #
def build_report_dict(ctx: ReportContext, cfg: AppConfig) -> dict:
    return {
        "candidate_name": ctx.candidate_name,
        "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ctx.analyzed_at)),
        "processing_seconds": round(ctx.processing_seconds, 1),
        "video": {**dataclasses.asdict(ctx.video), "description": ctx.video.describe()},
        "review_video": ctx.review_video,
        "verdict": ctx.verdict.label,
        "verdict_reasons": ctx.verdict.reasons,
        "statistics": ctx.stats.to_dict(),
        "incidents": [i.to_dict() for i in ctx.incidents],
        "calibration": {"mode": ctx.calibration_mode, "description": ctx.calibration_description,
                        "gaze_detection_available": ctx.gaze_available},
        "settings": {
            "temporal": dataclasses.asdict(cfg.temporal),
            "screen_margin_fraction": cfg.screen.margin_fraction,
            "min_gaze_confidence": cfg.confidence.min_gaze_confidence,
            "auto_calibration": dataclasses.asdict(cfg.video.auto_calibration),
            "incidents": dataclasses.asdict(cfg.video.incidents),
            "verdict": dataclasses.asdict(cfg.video.verdict),
            "clip_padding_s": cfg.video.clip_padding_s,
        },
        "warnings": ctx.warnings,
        "notes": [DISCLAIMER, DIRECTION_NOTE],
    }


def _incident_review(review: Optional[dict], number: int) -> dict:
    if not review:
        return {"status": INCIDENT_UNREVIEWED, "note": ""}
    return review.get("incidents", {}).get(str(number)) or {"status": INCIDENT_UNREVIEWED, "note": ""}


def final_verdict(report: dict, review: Optional[dict]) -> str:
    if review and review.get("final_verdict"):
        return review["final_verdict"]
    return report["verdict"]


# --------------------------------------------------------------------------- #
# Plain text
# --------------------------------------------------------------------------- #
def render_text(report: dict, review: Optional[dict] = None) -> str:
    s = report["statistics"]
    st = report["settings"]
    vc = st["verdict"]
    lines = [
        "EXAMVISION - VIDEO PROCTORING REPORT",
        "=" * 60,
        f"Candidate        : {report['candidate_name']}",
        f"Analyzed at      : {report['analyzed_at']}",
        f"Video file       : {report['video']['path']}",
        f"Video            : {report['video']['description']}",
        f"Duration analyzed: {fmt_duration(s['analyzed_duration_s'])} ({s['frames_analyzed']} frames)",
        "",
        f"AUTOMATIC VERDICT: {report['verdict']}",
        *[f"  - {r}" for r in report["verdict_reasons"]],
    ]
    if review and review.get("reviewer"):
        lines += ["", f"REVIEWER DECISION: {final_verdict(report, review)}"
                  + ("  (overrides the automatic verdict)" if review.get("final_verdict") else "  (automatic verdict accepted)"),
                  f"  Reviewer   : {review.get('reviewer', '')}",
                  f"  Reviewed at: {review.get('reviewed_at', '')}"]
        if review.get("verdict_note"):
            lines.append(f"  Note       : {review['verdict_note']}")
    lines += [
        "",
        "SUMMARY",
        "-" * 60,
        f"Periods of not looking at the screen : {s['incident_count']}",
        f"Total time not looking               : {fmt_duration(s['total_not_looking_s'])} ({s['not_looking_fraction']:.1%} of the video)",
        f"Longest period                       : {fmt_duration(s['longest_incident_s'])}",
        f"Gaze measurable (face visible frames): {s['gaze_coverage']:.0%}",
    ]
    if s["multiple_faces_s"] > 0:
        lines.append(f"Additional observation               : more than one face visible for {fmt_duration(s['multiple_faces_s'])}")
    lines += ["", "TIMESTAMPS WHEN THE CANDIDATE WAS NOT LOOKING AT THE SCREEN", "-" * 60]
    if not report["incidents"]:
        lines.append("None.")
    for i in report["incidents"]:
        lines.append(f"#{i['number']:<3} {i['start_timecode']} -> {i['end_timecode']}   "
                     f"({i['duration_s']:.1f} s)   {i['reason_text']}" + (f"   [{i['direction']}]" if i["direction"] else ""))
        r = _incident_review(review, i["number"])
        if review and review.get("reviewer"):
            lines.append(f"      review  : {INCIDENT_STATUS_TEXT.get(r['status'], r['status'])}"
                         + (f" - {r['note']}" if r.get("note") else ""))
        if i["clip_file"]:
            lines.append(f"      clip    : {i['clip_file']}")
        if i["snapshot_file"]:
            lines.append(f"      snapshot: {i['snapshot_file']}")
    lines += ["", "METHOD", "-" * 60,
              f"Calibration: {report['calibration']['mode']} - {report['calibration']['description']}",
              f"Gaze must stay off the screen for {st['temporal']['seconds_to_confirm_away']:.1f} s before it counts; "
              f"a missing face counts after {st['temporal']['seconds_to_confirm_face_lost']:.1f} s.",
              f"Eyes held turned away (sideways/down/up) count after {st['temporal']['seconds_to_confirm_away']:.1f} s; "
              f"glances back shorter than {st['incidents']['merge_gap_s']:.1f} s don't reset the count.",
              f"CHEATING if any: >= {vc['min_incidents']} periods, a single period >= "
              f"{vc['min_single_incident_s']:.0f} s, or >= {vc['min_not_looking_fraction']:.0%} of the video.",
              "", DIRECTION_NOTE, DISCLAIMER]
    if report["warnings"]:
        lines += ["", "WARNINGS", "-" * 60, *[f"  - {w}" for w in report["warnings"]]]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_CSS = """
:root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#5d6675; --line:#e2e5ea;
        --bad:#b42318; --bad-bg:#fdecea; --ok:#1e7a34; --ok-bg:#e8f5ec; --warn:#8a5a00; --warn-bg:#fff4dc; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif; }
main { max-width:1040px; margin:0 auto; padding:28px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:0 0 12px; }
.sub { color:var(--muted); margin:0 0 20px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin-bottom:16px; break-inside:avoid; }
.verdict { border-width:2px; }
.verdict .kicker { font-size:12px; font-weight:600; letter-spacing:.6px; text-transform:uppercase; color:var(--muted); }
.verdict .label { font-size:28px; font-weight:700; letter-spacing:.5px; }
.v-bad { border-color:var(--bad); background:var(--bad-bg); } .v-bad .label { color:var(--bad); }
.v-ok { border-color:var(--ok); background:var(--ok-bg); } .v-ok .label { color:var(--ok); }
.v-warn { border-color:var(--warn); background:var(--warn-bg); } .v-warn .label { color:var(--warn); }
.verdict ul { margin:8px 0 0; padding-left:20px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; }
.stat .n { font-size:22px; font-weight:650; font-variant-numeric:tabular-nums; }
.stat .k { color:var(--muted); font-size:13px; }
dl { display:grid; grid-template-columns:max-content 1fr; gap:4px 16px; margin:0; }
dt { color:var(--muted); } dd { margin:0; overflow-wrap:anywhere; }
.table-wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; }
th, td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-size:13px; color:var(--muted); font-weight:600; }
td.num { font-variant-numeric:tabular-nums; white-space:nowrap; }
tr.false-alarm td { color:var(--muted); } tr.false-alarm td.what { text-decoration:line-through; }
.pill { display:inline-block; font-size:12px; padding:1px 8px; border-radius:999px; border:1px solid var(--line); white-space:nowrap; }
.pill.confirmed { color:var(--bad); border-color:var(--bad); } .pill.false_alarm { color:var(--ok); border-color:var(--ok); }
img.thumb { width:180px; max-width:100%; border-radius:6px; border:1px solid var(--line); display:block; }
a { color:#1a56b0; }
.note { color:var(--muted); font-size:13px; }
.warn li { color:var(--warn); }
@media print {
  body { background:#fff; font-size:12px; } main { padding:0; max-width:none; }
  .card { border-color:#bbb; } a { color:inherit; text-decoration:none; } img.thumb { width:120px; }
}
"""


def _verdict_class(label: str) -> str:
    return {VERDICT_CHEATING: "v-bad", VERDICT_NO_CHEATING: "v-ok"}.get(label, "v-warn")


def render_html(report: dict, review: Optional[dict] = None) -> str:
    e = html.escape
    s = report["statistics"]
    st = report["settings"]
    vc = st["verdict"]
    reviewed = bool(review and review.get("reviewer"))

    decision = ""
    if reviewed:
        final = final_verdict(report, review)
        how = "Overrides the automatic verdict" if review.get("final_verdict") else "Automatic verdict accepted"
        note = f"<p>{e(review['verdict_note'])}</p>" if review.get("verdict_note") else ""
        decision = (f'<div class="card verdict {_verdict_class(final)}"><div class="kicker">Reviewer decision</div>'
                    f'<div class="label">{e(final)}</div><p class="note">{e(how)} &middot; '
                    f'{e(review.get("reviewer", ""))} &middot; {e(review.get("reviewed_at", ""))}</p>{note}</div>')

    rows = []
    for i in report["incidents"]:
        r = _incident_review(review, i["number"])
        snap = (f'<a href="../{e(i["snapshot_file"])}"><img class="thumb" src="../{e(i["snapshot_file"])}" '
                f'alt="Snapshot of incident {i["number"]}"></a>') if i["snapshot_file"] else "&ndash;"
        clip = f'<a href="../{e(i["clip_file"])}">Open clip</a>' if i["clip_file"] else "not available"
        review_cell = ""
        if reviewed:
            review_cell = (f"<td><span class='pill {e(r['status'])}'>{e(INCIDENT_STATUS_TEXT.get(r['status'], r['status']))}</span>"
                           + (f"<div class='note'>{e(r['note'])}</div>" if r.get("note") else "") + "</td>")
        cls = " class='false-alarm'" if r["status"] == INCIDENT_FALSE_ALARM else ""
        rows.append(
            f"<tr{cls}><td class='num'>{i['number']}</td><td class='num'>{e(i['start_timecode'])}</td>"
            f"<td class='num'>{e(i['end_timecode'])}</td><td class='num'>{i['duration_s']:.1f} s</td>"
            f"<td class='what'>{e(i['reason_text'])}</td><td>{e(i['direction'] or '–')}</td>{review_cell}"
            f"<td>{clip}</td><td>{snap}</td></tr>")
    review_th = "<th>Review</th>" if reviewed else ""
    table = (
        "<div class='table-wrap'><table><thead><tr><th>#</th><th>Start</th><th>End</th><th>Duration</th>"
        f"<th>What happened</th><th>Direction</th>{review_th}<th>Recording</th><th>Snapshot</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        if rows else "<p>No periods of not looking at the screen were found.</p>")

    multi = (f"<div class='stat'><div class='n'>{e(fmt_duration(s['multiple_faces_s']))}</div>"
             f"<div class='k'>More than one face visible (not part of the verdict)</div></div>"
             if s["multiple_faces_s"] > 0 else "")
    warnings = ("<div class='card'><h2>Warnings</h2><ul class='warn'>"
                + "".join(f"<li>{e(w)}</li>" for w in report["warnings"]) + "</ul></div>") if report["warnings"] else ""
    video_name = Path(report["video"]["path"]).name

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Proctoring Report - {e(report['candidate_name'])}</title>
<style>{_CSS}</style></head>
<body><main>
<h1>Proctoring Report: {e(report['candidate_name'])}</h1>
<p class="sub">Analyzed {e(report['analyzed_at'])} &middot; {e(video_name)}</p>

{decision}
<div class="card verdict {_verdict_class(report['verdict'])}">
  <div class="kicker">Automatic verdict</div>
  <div class="label">{e(report['verdict'])}</div>
  <ul>{''.join(f'<li>{e(r)}</li>' for r in report['verdict_reasons'])}</ul>
</div>

<div class="card"><h2>Summary</h2><div class="grid">
  <div class="stat"><div class="n">{s['incident_count']}</div><div class="k">Periods of not looking at the screen</div></div>
  <div class="stat"><div class="n">{e(fmt_duration(s['total_not_looking_s']))}</div><div class="k">Total time not looking ({s['not_looking_fraction']:.1%} of video)</div></div>
  <div class="stat"><div class="n">{e(fmt_duration(s['longest_incident_s']))}</div><div class="k">Longest period</div></div>
  <div class="stat"><div class="n">{s['gaze_coverage']:.0%}</div><div class="k">Frames where gaze was measurable</div></div>
  {multi}
</div></div>

<div class="card"><h2>Timestamps when the candidate was not looking at the screen</h2>
{table}
<p class="note">{e(DIRECTION_NOTE)} Each recording includes {st['clip_padding_s']:.0f} s of context before and after.</p>
</div>

<div class="card"><h2>Video</h2><dl>
  <dt>File</dt><dd>{e(report['video']['path'])}</dd>
  <dt>Format</dt><dd>{e(report['video']['description'])}</dd>
  <dt>Analyzed</dt><dd>{e(fmt_duration(s['analyzed_duration_s']))}, {s['frames_analyzed']} frames, frame by frame</dd>
  <dt>Processing time</dt><dd>{e(fmt_duration(report['processing_seconds']))}</dd>
</dl></div>

<div class="card"><h2>How the verdict is decided</h2><dl>
  <dt>Calibration</dt><dd>{e(report['calibration']['mode'])} &ndash; {e(report['calibration']['description'])}</dd>
  <dt>Looking away</dt><dd>Counts once the gaze stays off the screen for {st['temporal']['seconds_to_confirm_away']:.1f} s</dd>
  <dt>Eyes turned away</dt><dd>Counts once the eyes stay turned sideways, down or up for {st['temporal']['seconds_to_confirm_away']:.1f} s; glances back shorter than {st['incidents']['merge_gap_s']:.1f} s don't reset it (reading notes)</dd>
  <dt>Face missing</dt><dd>Counts once no face is visible for {st['temporal']['seconds_to_confirm_face_lost']:.1f} s</dd>
  <dt>CHEATING if any</dt><dd>{vc['min_incidents']} or more periods &middot; one period of {vc['min_single_incident_s']:.0f} s or more &middot; {vc['min_not_looking_fraction']:.0%} or more of the video</dd>
</dl><p class="note">{e(DISCLAIMER)}</p></div>
{warnings}
</main></body></html>
"""


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def _write_incidents_csv(path: Path, report: dict, review: Optional[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["incident", "start_timecode", "end_timecode", "start_s", "end_s", "duration_s",
                    "reasons", "direction", "start_frame", "end_frame", "clip_file", "snapshot_file",
                    "review_status", "review_note"])
        for i in report["incidents"]:
            r = _incident_review(review, i["number"])
            w.writerow([i["number"], i["start_timecode"], i["end_timecode"], i["start_s"], i["end_s"],
                        i["duration_s"], i["reason_text"], i["direction"] or "", i["start_frame"],
                        i["end_frame"], i["clip_file"] or "", i["snapshot_file"] or "",
                        r["status"], r.get("note", "")])


def render_report_files(report_dir: Path, report: dict, review: Optional[dict] = None) -> Path:
    """(Re-)renders report.html, report.txt and incidents.csv. Returns
    the path of report.html."""
    report_dir = Path(report_dir)
    _write_incidents_csv(report_dir / "incidents.csv", report, review)
    (report_dir / "report.txt").write_text(render_text(report, review), encoding="utf-8")
    html_path = report_dir / "report.html"
    html_path.write_text(render_html(report, review), encoding="utf-8")
    return html_path


def write_reports(paths: ResultPaths, ctx: ReportContext, cfg: AppConfig) -> Path:
    """Writes all report files and returns the path of report.html."""
    report = build_report_dict(ctx, cfg)
    with open(paths.report / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return render_report_files(paths.report, report)


def load_report(results_root: Path) -> dict:
    with open(Path(results_root) / "report" / "report.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_review(results_root: Path, review: dict) -> Path:
    """Stores the reviewer's decisions next to the report and re-renders
    report.html / report.txt / incidents.csv to include them."""
    report_dir = Path(results_root) / "report"
    with open(report_dir / "review.json", "w", encoding="utf-8") as f:
        json.dump(review, f, indent=2, ensure_ascii=False)
    return render_report_files(report_dir, load_report(results_root), review)
