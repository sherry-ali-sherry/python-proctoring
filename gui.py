"""
gui.py

Desktop front end (tkinter, part of the Python standard library):

    Step 1  Enter the candidate's name.
    Step 2  Upload (choose) the recorded exam video.
    Then    Progress while the video is analyzed frame by frame, and a
            results screen with the verdict, every timestamp, and buttons
            to open the report / clips / results folder.

The analysis runs on a worker thread; it reports progress through a
queue that the Tk main loop polls, so the window stays responsive and
tkinter is only ever touched from the main thread.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from config import AppConfig
from video_analysis.analyzer import (
    CALIBRATION_AUTO, CALIBRATION_FILE, AnalysisCancelled, AnalysisError, AnalysisResult, VideoAnalyzer,
)
from video_analysis.incidents import VERDICT_CHEATING, VERDICT_NO_CHEATING, timecode
from video_analysis.output_layout import sanitize_folder_name, validate_candidate_name
from video_analysis.video_source import VideoError, VideoInfo, probe_video

_VERDICT_COLORS = {VERDICT_CHEATING: "#b42318", VERDICT_NO_CHEATING: "#1e7a34"}
_INCONCLUSIVE_COLOR = "#8a5a00"
_MUTED = "#5d6675"
_ERROR = "#b42318"


def open_path(path: Path) -> None:
    """Open a file or folder with the operating system's default app."""
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # noqa: S606 - local file chosen/created by this app
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class ExamVisionGUI:
    def __init__(self, cfg: AppConfig, project_dir: Path):
        self.cfg = cfg
        self.analyzer = VideoAnalyzer(cfg, project_dir)

        self.root = tk.Tk()
        self.root.title("ExamVision - Video Proctoring")
        self.root.geometry("760x560")
        self.root.minsize(620, 480)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Step.TLabel", foreground=_MUTED)
        style.configure("Muted.TLabel", foreground=_MUTED)
        style.configure("Error.TLabel", foreground=_ERROR)
        style.configure("Verdict.TLabel", font=("Segoe UI", 24, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        self.container = ttk.Frame(self.root, padding=24)
        self.container.pack(fill="both", expand=True)

        self.candidate_name = tk.StringVar()
        self.video_info: Optional[VideoInfo] = None
        self.calibration_mode = tk.StringVar(value=CALIBRATION_AUTO)
        self.export_full_video = tk.BooleanVar(value=cfg.video.export_annotated_video)

        self._worker: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._messages: "queue.Queue[tuple]" = queue.Queue()
        self._started_at = 0.0
        self._closing = False
        self._result: Optional[AnalysisResult] = None

        self.show_name_step()

    def run(self) -> None:
        self.root.mainloop()

    def _clear(self) -> ttk.Frame:
        for child in self.container.winfo_children():
            child.destroy()
        frame = ttk.Frame(self.container)
        frame.pack(fill="both", expand=True)
        return frame

    # -- Step 1: candidate name ---------------------------------------------
    def show_name_step(self) -> None:
        f = self._clear()
        ttk.Label(f, text="Step 1 of 2", style="Step.TLabel").pack(anchor="w")
        ttk.Label(f, text="Who is this exam video for?", style="Title.TLabel").pack(anchor="w", pady=(2, 16))
        ttk.Label(f, text="Candidate's full name").pack(anchor="w")

        entry = ttk.Entry(f, textvariable=self.candidate_name, width=48, font=("Segoe UI", 11))
        entry.pack(anchor="w", fill="x", pady=(4, 6))
        folder_hint = ttk.Label(f, style="Muted.TLabel")
        folder_hint.pack(anchor="w")
        error = ttk.Label(f, style="Error.TLabel")
        error.pack(anchor="w", pady=(6, 0))

        def update_hint(*_):
            folder = sanitize_folder_name(self.candidate_name.get())
            folder_hint.configure(text=f"Results will be saved in: {self.cfg.video.results_dir}/{folder}/" if folder else "")
            error.configure(text="")

        def submit(*_):
            message = validate_candidate_name(self.candidate_name.get())
            if message:
                error.configure(text=message)
                entry.focus_set()
                return
            self.candidate_name.set(self.candidate_name.get().strip())
            self.show_video_step()

        trace_id = self.candidate_name.trace_add("write", update_hint)
        f.bind("<Destroy>", lambda _e: self._safe_trace_remove(trace_id))
        update_hint()

        buttons = ttk.Frame(f)
        buttons.pack(side="bottom", fill="x")
        ttk.Button(buttons, text="Next  >", style="Accent.TButton", command=submit).pack(side="right")
        entry.bind("<Return>", submit)
        entry.focus_set()
        entry.icursor("end")

    def _safe_trace_remove(self, trace_id: str) -> None:
        try:
            self.candidate_name.trace_remove("write", trace_id)
        except tk.TclError:
            pass

    # -- Step 2: upload video -------------------------------------------------
    def show_video_step(self) -> None:
        f = self._clear()
        ttk.Label(f, text="Step 2 of 2", style="Step.TLabel").pack(anchor="w")
        ttk.Label(f, text="Upload the exam video", style="Title.TLabel").pack(anchor="w", pady=(2, 4))
        ttk.Label(f, text=f"Candidate: {self.candidate_name.get()}", style="Muted.TLabel").pack(anchor="w", pady=(0, 16))

        pick = ttk.Frame(f)
        pick.pack(anchor="w", fill="x")
        ttk.Button(pick, text="Choose video file...", command=lambda: self._choose_video(file_label, info_label, analyze_btn)).pack(side="left")
        file_label = ttk.Label(pick, text="No video selected", style="Muted.TLabel")
        file_label.pack(side="left", padx=12)
        info_label = ttk.Label(f, style="Muted.TLabel", wraplength=640, justify="left")
        info_label.pack(anchor="w", pady=(8, 16))

        options = ttk.LabelFrame(f, text="Screen calibration", padding=12)
        options.pack(anchor="w", fill="x")
        ttk.Radiobutton(options, text="Locate the screen automatically from the video (recommended)",
                        variable=self.calibration_mode, value=CALIBRATION_AUTO).pack(anchor="w")
        saved = self.analyzer.saved_calibration_available()
        rb = ttk.Radiobutton(options, text="Use the saved webcam calibration (only if the video was recorded "
                                           "on this computer's webcam)",
                             variable=self.calibration_mode, value=CALIBRATION_FILE)
        rb.pack(anchor="w", pady=(4, 0))
        if not saved:
            rb.state(["disabled"])
            self.calibration_mode.set(CALIBRATION_AUTO)
            ttk.Label(options, text="No saved calibration found (run live_webcam.py and press 'c' to create one).",
                      style="Muted.TLabel").pack(anchor="w", padx=(22, 0))
        ttk.Checkbutton(f, text="Also export the full video with the verdict banner burned in (slower)",
                        variable=self.export_full_video).pack(anchor="w", pady=(12, 0))

        buttons = ttk.Frame(f)
        buttons.pack(side="bottom", fill="x")
        analyze_btn = ttk.Button(buttons, text="Analyze video", style="Accent.TButton", command=self.start_analysis)
        analyze_btn.pack(side="right")
        ttk.Button(buttons, text="<  Back", command=self.show_name_step).pack(side="left")

        if self.video_info is not None:
            file_label.configure(text=Path(self.video_info.path).name)
            info_label.configure(text=self.video_info.describe())
        else:
            analyze_btn.state(["disabled"])

    def _choose_video(self, file_label: ttk.Label, info_label: ttk.Label, analyze_btn: ttk.Button) -> None:
        patterns = " ".join(f"*{ext}" for ext in self.cfg.video.video_extensions)
        path = filedialog.askopenfilename(
            parent=self.root, title="Choose the exam video",
            filetypes=[("Video files", patterns), ("All files", "*.*")],
        )
        if not path:
            return
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            info = probe_video(path)
        except VideoError as e:
            self.video_info = None
            file_label.configure(text=Path(path).name)
            info_label.configure(text=str(e), style="Error.TLabel")
            analyze_btn.state(["disabled"])
            return
        finally:
            self.root.configure(cursor="")
        self.video_info = info
        file_label.configure(text=Path(path).name)
        info_label.configure(text=info.describe(), style="Muted.TLabel")
        analyze_btn.state(["!disabled"])

    # -- Analysis -----------------------------------------------------------
    def start_analysis(self) -> None:
        if self.video_info is None:
            return
        self.cfg.video.export_annotated_video = bool(self.export_full_video.get())
        self._cancel.clear()
        self._messages = queue.Queue()
        self._started_at = time.time()

        f = self._clear()
        ttk.Label(f, text="Analyzing the video", style="Title.TLabel").pack(anchor="w")
        ttk.Label(f, text=f"{self.candidate_name.get()}  -  {Path(self.video_info.path).name}",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 24))
        self._bar = ttk.Progressbar(f, mode="determinate", maximum=1000)
        self._bar.pack(fill="x")
        self._status = ttk.Label(f, text="Starting...")
        self._status.pack(anchor="w", pady=(10, 0))
        self._eta = ttk.Label(f, style="Muted.TLabel")
        self._eta.pack(anchor="w", pady=(4, 0))
        buttons = ttk.Frame(f)
        buttons.pack(side="bottom", fill="x")
        self._cancel_btn = ttk.Button(buttons, text="Cancel", command=self._request_cancel)
        self._cancel_btn.pack(side="right")

        name, path, mode = self.candidate_name.get(), self.video_info.path, self.calibration_mode.get()
        self._worker = threading.Thread(target=self._work, args=(name, path, mode), daemon=True)
        self._worker.start()
        self.root.after(100, self._poll)

    def _work(self, name: str, path: str, mode: str) -> None:
        try:
            result = self.analyzer.run(
                name, path, mode,
                progress=lambda fraction, msg: self._messages.put(("progress", fraction, msg)),
                cancel_event=self._cancel,
            )
            self._messages.put(("done", result))
        except AnalysisCancelled:
            self._messages.put(("cancelled",))
        except AnalysisError as e:
            self._messages.put(("error", str(e), None))
        except Exception as e:  # unexpected: show it, and keep details on the console
            traceback.print_exc()
            self._messages.put(("error", f"{type(e).__name__}: {e}", traceback.format_exc()))

    def _poll(self) -> None:
        latest_progress = None
        try:
            while True:
                msg = self._messages.get_nowait()
                if msg[0] == "progress":
                    latest_progress = msg
                    continue
                self._finish(msg)
                return
        except queue.Empty:
            pass
        if latest_progress is not None and not self._closing:
            _, fraction, text = latest_progress
            self._bar["value"] = int(fraction * 1000)
            self._status.configure(text=text)
            elapsed = time.time() - self._started_at
            if 0.03 < fraction < 1.0:
                remaining = elapsed * (1.0 - fraction) / fraction
                self._eta.configure(text=f"Elapsed {int(elapsed // 60)}:{int(elapsed % 60):02d}  -  "
                                         f"about {int(remaining // 60)}:{int(remaining % 60):02d} remaining")
        self.root.after(100, self._poll)

    def _request_cancel(self) -> None:
        if messagebox.askyesno("Cancel analysis", "Stop analyzing this video? Nothing will be saved.", parent=self.root):
            self._cancel.set()
            self._cancel_btn.state(["disabled"])
            self._status.configure(text="Cancelling...")

    def _finish(self, msg: tuple) -> None:
        self._worker = None
        if self._closing:
            self.root.destroy()
            return
        kind = msg[0]
        if kind == "done":
            self._result = msg[1]
            self.show_result_step(msg[1])
        elif kind == "cancelled":
            self.show_video_step()
        else:
            _, text, details = msg
            messagebox.showerror("Analysis failed", text + ("\n\nDetails were printed to the console." if details else ""),
                                 parent=self.root)
            self.show_video_step()

    # -- Results ------------------------------------------------------------
    def show_result_step(self, result: AnalysisResult) -> None:
        f = self._clear()
        v, s = result.verdict, result.stats
        ttk.Label(f, text=f"Result for {result.candidate_name}", style="Muted.TLabel").pack(anchor="w")
        verdict = ttk.Label(f, text=v.label, style="Verdict.TLabel",
                            foreground=_VERDICT_COLORS.get(v.label, _INCONCLUSIVE_COLOR))
        verdict.pack(anchor="w", pady=(0, 4))
        for reason in v.reasons:
            ttk.Label(f, text=f"-  {reason}", wraplength=680, justify="left").pack(anchor="w")
        ttk.Label(f, style="Muted.TLabel", wraplength=680, justify="left", text=(
            f"{s.incident_count} period(s) not looking at the screen, {s.total_not_looking_s:.1f} s in total "
            f"({s.not_looking_fraction:.1%} of {s.analyzed_duration_s:.0f} s). "
            f"Gaze measurable on {s.gaze_coverage:.0%} of frames with a face."
        )).pack(anchor="w", pady=(8, 8))

        table_frame = ttk.Frame(f)
        table_frame.pack(fill="both", expand=True)
        columns = ("n", "start", "end", "duration", "reason")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=7)
        for col, title, width, anchor in (("n", "#", 40, "center"), ("start", "Start", 110, "w"),
                                          ("end", "End", 110, "w"), ("duration", "Duration", 80, "e"),
                                          ("reason", "What happened", 320, "w")):
            tree.heading(col, text=title)
            tree.column(col, width=width, anchor=anchor, stretch=(col == "reason"))
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for inc in result.incidents:
            reason = inc.reason_text() + (f" ({inc.direction})" if inc.direction else "")
            tree.insert("", "end", iid=str(inc.number), values=(
                inc.number, timecode(inc.start_s), timecode(inc.end_s), f"{inc.duration_s:.1f} s", reason))
        if not result.incidents:
            tree.insert("", "end", values=("", "", "", "", "No periods of not looking at the screen"))

        def open_clip(_event=None):
            sel = tree.selection()
            if not sel or not sel[0].isdigit():
                return
            inc = next((i for i in result.incidents if i.number == int(sel[0])), None)
            if inc and inc.clip_file:
                open_path(result.paths.root / inc.clip_file)
        tree.bind("<Double-1>", open_clip)
        if result.incidents:
            ttk.Label(f, text="Double-click a row to play its recording.", style="Muted.TLabel").pack(anchor="w", pady=(4, 0))
        if result.warnings:
            ttk.Label(f, text=f"{len(result.warnings)} warning(s) - see the report.", foreground=_INCONCLUSIVE_COLOR).pack(anchor="w", pady=(4, 0))

        buttons = ttk.Frame(f)
        buttons.pack(side="bottom", fill="x", pady=(12, 0))
        ttk.Button(buttons, text="Open report", style="Accent.TButton",
                   command=lambda: open_path(result.report_html)).pack(side="right")
        ttk.Button(buttons, text="Open results folder",
                   command=lambda: open_path(result.paths.root)).pack(side="right", padx=8)
        ttk.Button(buttons, text="Analyze another video", command=self._restart).pack(side="left")
        ttk.Label(f, text=f"Saved to {result.paths.root}", style="Muted.TLabel", wraplength=680).pack(side="bottom", anchor="w")

    def _restart(self) -> None:
        self.candidate_name.set("")
        self.video_info = None
        self.calibration_mode.set(CALIBRATION_AUTO)
        self._result = None
        self.show_name_step()

    def _on_close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            if not messagebox.askyesno("Quit", "An analysis is still running. Stop it and quit?", parent=self.root):
                return
            self._closing = True
            self._cancel.set()
            return  # _finish() destroys the window once the worker has stopped
        self.root.destroy()


def run_gui(cfg: AppConfig, project_dir: Path) -> None:
    if sys.platform.startswith("win"):
        try:  # crisp text on high-DPI displays
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    ExamVisionGUI(cfg, project_dir).run()
