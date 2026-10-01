/* ExamVision review UI. Plain JavaScript, no build step.
 * Routes (hash based):  #/  analyses list · #/new  upload · #/a/<id>  review · #/settings
 * All text from the server is inserted with textContent (never innerHTML). */
"use strict";

const VERDICT = { CHEATING: "CHEATING", CLEAN: "NO CHEATING", INCONCLUSIVE: "INCONCLUSIVE" };
const INCIDENT = { UNREVIEWED: "unreviewed", CONFIRMED: "confirmed", FALSE_ALARM: "false_alarm" };
const TL_COLORS = { screen: "--tl-screen", glance: "--tl-glance", incident: "--tl-incident", no_face: "--tl-noface", uncertain: "--tl-uncertain" };
const TL_LABELS = { screen: "Looking at screen", glance: "Glancing away (under the time limit)", incident: "Not looking (incident)", no_face: "No face visible", uncertain: "Gaze not measurable" };

const view = document.getElementById("view");
let cleanup = [];          // functions to run when leaving the current view
let status = null;         // /api/status, loaded once

/* ---------------------------------------------------------------- helpers */
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

async function api(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  const res = await fetch(url, opts);
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    const msg = Array.isArray(d) ? d.map(x => (typeof x === "string" ? x : x.msg)).join(" ") : (d || `Request failed (${res.status})`);
    throw new Error(msg);
  }
  return data;
}

function toast(message, isError = false) {
  const t = h("div", { class: "toast" + (isError ? " error" : ""), role: isError ? "alert" : "status" }, message);
  document.getElementById("toasts").append(t);
  setTimeout(() => t.remove(), isError ? 7000 : 3500);
}

function timecode(s) {
  s = Math.max(0, s || 0);
  const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = Math.floor(s % 60);
  const pad = n => String(n).padStart(2, "0");
  return (hh ? pad(hh) + ":" : "") + pad(mm) + ":" + pad(ss);
}
function duration(s) {
  if (s === null || s === undefined) return "–";
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return m < 60 ? `${m} min ${r} s` : `${Math.floor(m / 60)} h ${m % 60} min`;
}
function verdictBadge(label, prefix) {
  if (!label) return h("span", { class: "badge neutral" }, "–");
  const cls = label === VERDICT.CHEATING ? "cheating" : label === VERDICT.CLEAN ? "clean" : "inconclusive";
  return h("span", { class: `badge ${cls}` }, prefix ? `${prefix}: ${label}` : label);
}
/* Chevron badge used inside the main call-to-action buttons. */
function arrowIcon() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 1024 1024");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", "M779.18 473.23 322.35 16.41c-21.41-21.41-56.12-21.41-77.53 0-21.41 21.41-21.41 56.12 0 77.53l418.06 418.06-418.06 418.06c-21.41 21.41-21.41 56.12 0 77.53 10.71 10.71 24.76 16.06 38.77 16.06s28.06-5.35 38.77-16.06l456.83-456.83c21.41-21.41 21.41-56.12 0-77.54z");
  svg.append(path);
  return h("span", { class: "btn-icon" }, svg);
}
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

/* ---------------------------------------------------------------- reviewer identity */
const REVIEWER_KEY = "examvision.reviewer";
function getReviewer() { try { return localStorage.getItem(REVIEWER_KEY) || ""; } catch { return ""; } }
function setReviewer(name) { try { localStorage.setItem(REVIEWER_KEY, name); } catch { /* private mode */ } renderReviewerChip(); }
function renderReviewerChip() {
  const chip = document.getElementById("reviewer-chip");
  const name = getReviewer();
  chip.replaceChildren(name ? h("span", {}, "Reviewer: ", h("strong", {}, name)) : "Set reviewer name");
}
function askReviewer() {
  return new Promise(resolve => {
    const dlg = document.getElementById("reviewer-dialog");
    const input = document.getElementById("reviewer-input");
    input.value = getReviewer();
    dlg.onclose = () => {
      const v = input.value.trim().replace(/\s+/g, " ");
      if (dlg.returnValue === "ok" && v) { setReviewer(v); resolve(v); } else resolve(getReviewer() || null);
    };
    dlg.showModal();
    input.focus();
  });
}
async function requireReviewer() { return getReviewer() || await askReviewer(); }
document.getElementById("reviewer-chip").addEventListener("click", askReviewer);

/* ---------------------------------------------------------------- router */
function route() {
  cleanup.forEach(fn => { try { fn(); } catch { /* ignore */ } });
  cleanup = [];
  const hash = location.hash || "#/";
  let m;
  const nav = hash.startsWith("#/new") ? "new" : hash.startsWith("#/settings") ? "settings" : "list";
  document.querySelectorAll("[data-nav]").forEach(a => a.classList.toggle("active", a.dataset.nav === nav && !hash.startsWith("#/a/")));
  view.replaceChildren();
  if (hash === "#/new") renderNew();
  else if (hash === "#/settings") renderSettings();
  else if ((m = hash.match(/^#\/a\/([0-9a-f]+)$/))) renderReview(m[1]);
  else renderList();
  view.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);

function poll(fn, ms) {
  let stopped = false, timer = null;
  const tick = async () => {
    if (stopped) return;
    let next = ms;
    try { const r = await fn(); if (typeof r === "number") next = r; } catch { /* keep polling */ }
    if (!stopped) timer = setTimeout(tick, next);
  };
  tick();
  cleanup.push(() => { stopped = true; clearTimeout(timer); });
}

/* ---------------------------------------------------------------- list */
function renderList() {
  let filter = sessionStorage.getItem("examvision.filter") || "all";
  const filters = [["all", "All"], ["review", "Needs review"], ["reviewed", "Reviewed"], ["active", "In progress"], ["failed", "Failed"]];
  const filterBar = h("div", { class: "filters", role: "group", "aria-label": "Filter analyses" });
  const body = h("div", { class: "card", style: "padding:0" });
  const mosaic = h("div", { class: "mosaic", "aria-label": "Recent incident snapshots" });
  const activeStrip = h("div", { class: "active-strip", role: "status", "aria-live": "polite" });
  view.append(
    activeStrip,
    h("section", { class: "hero" },
      h("div", {},
        h("h1", {}, "Review exam videos with confidence."),
        h("p", {}, "Upload a recording, let ExamVision find every moment the candidate looked away, then confirm or dismiss each one and record your decision."),
        h("div", { class: "cta" },
          h("a", { class: "btn primary", href: "#/new" }, "Upload a video", arrowIcon()),
          h("a", { class: "text-link", href: "#/settings" }, "Adjust detection settings"))),
      mosaic),
    h("div", { class: "section-head" }, h("h2", {}, "Analyses"), filterBar),
    body);

  let rows = [];
  let mosaicKey = null;
  const drawMosaic = () => {
    const pics = rows.flatMap(r => (r.thumbs || []).map(src => ({ src, id: r.id, name: r.candidate_name }))).slice(0, 12);
    const key = JSON.stringify(pics);
    if (key === mosaicKey) return;
    mosaicKey = key;
    const tiles = pics.map(p => h("div", { class: "tile link", title: `${p.name}: open review`, onclick: () => { location.hash = `#/a/${p.id}`; } },
      h("img", { src: p.src, alt: `Incident snapshot, ${p.name}`, loading: "lazy" })));
    for (let k = tiles.length; k < 12; k++) tiles.push(h("div", { class: "tile blank " + ["", "a", "", "b"][k % 4], "aria-hidden": "true" }));
    mosaic.replaceChildren(...tiles);
  };

  const drawActive = () => {
    const active = rows.filter(r => r.status === "running" || r.status === "queued")
      .sort((a, b) => (a.status === "running" ? -1 : 1) - (b.status === "running" ? -1 : 1) || a.created_at.localeCompare(b.created_at));
    activeStrip.hidden = !active.length;
    activeStrip.replaceChildren(
      h("div", { class: "active-title" }, active.length === 1 ? "1 video is being processed" : `${active.length} videos are being processed`),
      ...active.map(r => h("a", { class: "active-item", href: `#/a/${r.id}` },
        h("span", { class: "who" }, r.candidate_name),
        r.status === "running"
          ? h("span", { class: "progress" }, h("div", { style: `width:${Math.round(r.progress * 100)}%` }))
          : h("span", { class: "badge neutral" }, "Queued"),
        h("span", { class: "pct num" }, r.status === "running" ? `${Math.round(r.progress * 100)}%` : ""))));
  };

  const draw = () => {
    drawActive();
    drawMosaic();
    filterBar.replaceChildren(...filters.map(([key, label]) =>
      h("button", { type: "button", class: filter === key ? "active" : "", "aria-pressed": String(filter === key),
        onclick: () => { filter = key; sessionStorage.setItem("examvision.filter", key); draw(); } }, label)));
    const shown = rows.filter(r =>
      filter === "all" ? true :
      filter === "review" ? r.status === "done" && !r.reviewed :
      filter === "reviewed" ? r.reviewed :
      filter === "active" ? r.status === "queued" || r.status === "running" :
      r.status === "failed" || r.status === "cancelled");
    if (!rows.length) {
      body.replaceChildren(h("div", { class: "empty" }, h("p", {}, "No analyses yet."), h("a", { class: "btn primary", href: "#/new" }, "Upload the first video")));
      return;
    }
    if (!shown.length) { body.replaceChildren(h("div", { class: "empty" }, "Nothing matches this filter.")); return; }
    const queued = rows.filter(r => r.status === "queued").sort((a, b) => a.created_at.localeCompare(b.created_at)).map(r => r.id);
    body.replaceChildren(h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Candidate", "Uploaded", "Status", "Automatic verdict", "Review", "Not looking", ""].map(t => h("th", {}, t)))),
      h("tbody", {}, shown.map(r => listRow(r, queued))))));
  };

  poll(async () => {
    rows = await api("GET", "/api/analyses");
    draw();
    return rows.some(r => r.status === "queued" || r.status === "running") ? 1500 : 10000;
  }, 1500);
}

/* Asks for confirmation, then permanently deletes the analysis.
 * Returns true if it was deleted. */
async function deleteAnalysis(r) {
  const what = r.status === "done"
    ? `This permanently deletes the analysis for ${r.candidate_name}: its results folder (clips, snapshots, `
      + `report, the uploaded video) and all review decisions. This cannot be undone.`
    : `Remove the ${r.status} analysis for ${r.candidate_name} from the list?`;
  if (!confirm(what + "\n\nDelete it?")) return false;
  try {
    await api("DELETE", `/api/analyses/${r.id}`);
    toast(`Deleted the analysis for ${r.candidate_name}.`);
    return true;
  } catch (e) {
    toast(e.message, true);
    return false;
  }
}

function listRow(r, queued) {
  const done = r.status === "done";
  let statusCell;
  if (r.status === "running") {
    statusCell = h("div", {}, h("div", { class: "progress" }, h("div", { style: `width:${Math.round(r.progress * 100)}%` })),
      h("div", { class: "status-line" }, `${Math.round(r.progress * 100)}% · ${r.message || "Analyzing"}`));
  } else if (r.status === "queued") {
    statusCell = h("span", { class: "badge neutral" }, `Queued (#${queued.indexOf(r.id) + 1})`);
  } else if (r.status === "failed") {
    statusCell = h("div", {}, h("span", { class: "badge cheating" }, "Failed"), h("div", { class: "status-line" }, r.error || ""));
  } else if (r.status === "cancelled") {
    statusCell = h("span", { class: "badge neutral" }, "Cancelled");
  } else {
    statusCell = h("span", { class: "badge neutral" }, "Analyzed");
  }
  const reviewCell = !done ? "–" : r.reviewed
    ? h("div", {}, verdictBadge(r.effective_verdict), h("div", { class: "status-line" }, `by ${r.reviewer}`))
    : h("span", { class: "badge pending" }, "Needs review");
  const actions = h("td", { style: "text-align:right" });
  if (done) actions.append(h("a", { class: "btn small", href: `#/a/${r.id}` }, r.reviewed ? "Open" : "Review"));
  if (r.status === "queued" || r.status === "running") {
    actions.append(h("button", { class: "btn small danger", type: "button", onclick: async ev => {
      ev.stopPropagation();
      if (!confirm(`Cancel the analysis for ${r.candidate_name}? The uploaded video will be discarded.`)) return;
      try { await api("POST", `/api/analyses/${r.id}/cancel`); toast("Cancelling…"); } catch (e) { toast(e.message, true); }
    } }, "Cancel"));
  }
  if (r.status === "done" || r.status === "failed" || r.status === "cancelled") {
    actions.append(" ", h("button", { class: "btn small danger", type: "button", title: "Delete this analysis",
      onclick: async ev => { ev.stopPropagation(); if (await deleteAnalysis(r)) route(); } }, "Delete"));
  }
  return h("tr", { class: done ? "clickable" : "", onclick: done ? () => { location.hash = `#/a/${r.id}`; } : null },
    h("td", {}, h("div", { class: "name" }, r.candidate_name), h("div", { class: "file" }, r.video_filename)),
    h("td", { class: "num" }, r.created_at || ""),
    h("td", {}, statusCell),
    h("td", {}, done ? verdictBadge(r.auto_verdict) : "–"),
    h("td", {}, reviewCell),
    h("td", { class: "num" }, done ? `${r.incident_count} × · ${duration(r.not_looking_s)}` : "–"),
    actions);
}

/* ---------------------------------------------------------------- new analysis */
async function renderNew() {
  status = status || await api("GET", "/api/status").catch(() => null);
  const exts = status ? status.video_extensions : [".mp4", ".avi", ".mov", ".mkv", ".webm"];
  let file = null;

  const name = h("input", { id: "cand", maxlength: "80", autocomplete: "off", placeholder: "e.g. Ali Khan" });
  const nameErr = h("div", { class: "error-text", role: "alert" });
  const picker = h("input", { type: "file", accept: exts.join(","), hidden: true });
  const chosen = h("div", { class: "chosen" });
  const zone = h("div", { class: "dropzone", tabindex: "0", role: "button", "aria-label": "Choose the exam video" },
    h("div", { class: "big" }, "Drop the exam video here"),
    h("div", { class: "muted" }, "or click to choose a file · ", exts.join(" ")), chosen);
  const fileErr = h("div", { class: "error-text", role: "alert" });
  const calib = h("select", { id: "calib" },
    h("option", { value: "auto" }, "Locate the screen automatically from the video (recommended)"),
    h("option", { value: "file", disabled: !(status && status.saved_calibration_available) },
      "Use the saved webcam calibration" + (status && status.saved_calibration_available ? "" : " (none saved)")));
  const bar = h("div", { class: "progress big", hidden: true }, h("div", { style: "width:0%" }));
  const barText = h("div", { class: "status-line" });
  const submit = h("button", { class: "btn primary", type: "submit" }, "Upload and analyze", arrowIcon());

  const setFile = f => {
    fileErr.textContent = "";
    if (!f) return;
    const ext = "." + (f.name.split(".").pop() || "").toLowerCase();
    if (!exts.includes(ext)) { file = null; chosen.textContent = ""; fileErr.textContent = `“${f.name}” is not a supported video type.`; return; }
    file = f;
    chosen.textContent = `${f.name} · ${(f.size / 1024 / 1024).toFixed(1)} MB`;
  };
  picker.addEventListener("change", () => setFile(picker.files[0]));
  zone.addEventListener("click", () => picker.click());
  zone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); picker.click(); } });
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", e => { e.preventDefault(); zone.classList.remove("over"); setFile(e.dataTransfer.files[0]); });

  let xhr = null;
  cleanup.push(() => { if (xhr) xhr.abort(); });

  const form = h("form", { class: "card", novalidate: true, onsubmit: async e => {
    e.preventDefault();
    if (xhr || submit.disabled) return;            // already uploading: ignore double clicks
    nameErr.textContent = ""; fileErr.textContent = "";
    const cand = name.value.trim().replace(/\s+/g, " ");
    let bad = false;
    if (!cand) { nameErr.textContent = "Enter the candidate's name."; bad = true; }
    if (!file) { fileErr.textContent = "Choose the exam video."; bad = true; }
    if (bad) return;

    // Same candidate already waiting or being analyzed? Probably a repeat upload.
    submit.disabled = true;
    const busy = await api("GET", "/api/analyses")
      .then(rows => rows.filter(r => (r.status === "queued" || r.status === "running")
                                  && r.candidate_name.toLowerCase() === cand.toLowerCase()))
      .catch(() => []);
    submit.disabled = false;
    if (busy.length && !confirm(`A video for ${cand} is already ${busy[0].status === "running" ? "being analyzed" : "waiting in the queue"}`
                                + ` (${busy[0].video_filename}).\n\nUpload another one anyway?`)) {
      location.hash = `#/a/${busy[0].id}`;
      return;
    }

    const fd = new FormData();
    fd.append("candidate_name", cand);
    fd.append("calibration_mode", calib.value);
    fd.append("file", file);
    xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/analyses");
    bar.hidden = false; submit.disabled = true;
    xhr.upload.onprogress = ev => {
      if (!ev.lengthComputable) return;
      const p = ev.loaded / ev.total;
      bar.firstChild.style.width = `${Math.round(p * 100)}%`;
      barText.textContent = p < 1 ? `Uploading… ${Math.round(p * 100)}%` : "Checking the video…";
    };
    xhr.onload = () => {
      const ok = xhr.status === 201;
      let data = {};
      try { data = JSON.parse(xhr.responseText || "{}"); } catch { /* non-JSON error page */ }
      xhr = null;
      if (ok) {
        toast(`${cand}: video uploaded. Analysis started.`);
        location.hash = `#/a/${data.id}`;          // live progress, then the review, on one page
        return;
      }
      submit.disabled = false; bar.hidden = true; barText.textContent = "";
      const d = data.detail;
      const msg = Array.isArray(d) ? d.map(x => (typeof x === "string" ? x : x.msg)).join(" ") : d;
      (/name/i.test(msg || "") ? nameErr : fileErr).textContent = msg || "Upload failed.";
    };
    xhr.onerror = () => { xhr = null; submit.disabled = false; bar.hidden = true; barText.textContent = ""; fileErr.textContent = "Upload failed: the server could not be reached."; };
    xhr.send(fd);
  } },
    h("div", { class: "field" }, h("label", { for: "cand" }, "Candidate's full name"), name,
      h("div", { class: "hint" }, "Results are saved in a folder named after the candidate."), nameErr),
    h("div", { class: "field" }, h("label", {}, "Exam video"), zone, picker, fileErr),
    h("div", { class: "field" }, h("label", { for: "calib" }, "Screen calibration"), calib,
      h("div", { class: "hint" }, "Use the saved calibration only if the video was recorded on this computer's own webcam.")),
    h("div", { class: "field" }, bar, barText),
    h("div", { class: "form-actions" }, h("a", { class: "btn ghost", href: "#/" }, "Cancel"), submit));

  view.append(h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "New analysis"),
    h("p", { class: "sub" }, "The video is analyzed frame by frame in the background. You can upload several; they are processed one at a time."))),
    h("div", { class: "new-grid" }, form));
  name.focus();
}

/* ---------------------------------------------------------------- review */
async function renderReview(id) {
  const holder = h("div", {});
  view.append(holder);
  let data;
  poll(async () => {
    data = await api("GET", `/api/analyses/${id}`);
    if (data.status === "done" && data.report) { buildReview(holder, data); return 1e9; }
    holder.replaceChildren(pendingCard(data));
    return 1500;
  }, 1500);
}

function pendingCard(d) {
  const kids = [h("a", { href: "#/", class: "back" }, "← All analyses"), h("h1", { style: "margin:8px 0 16px" }, d.candidate_name)];
  const card = h("div", { class: "card" });
  if (d.status === "running" || d.status === "queued") {
    card.append(h("h2", {}, d.status === "queued" ? "Upload complete. Waiting in the queue" : "Upload complete. Analyzing the video"),
      h("p", { class: "muted", style: "margin:-6px 0 14px" }, d.video_filename),
      h("div", { class: "progress big" }, h("div", { style: `width:${Math.round(d.progress * 100)}%` })),
      h("div", { class: "status-line" }, d.status === "queued" ? "It starts when the analyses before it finish." : `${Math.round(d.progress * 100)}% · ${d.message || ""}`),
      h("p", { class: "hint", style: "margin-top:14px" },
        "This page turns into the review screen by itself when the analysis finishes. You can also leave; it keeps running in the background."),
      h("div", { class: "form-actions", style: "justify-content:flex-start" },
        h("button", { class: "btn danger small", type: "button", onclick: async () => {
          if (!confirm(`Cancel the analysis for ${d.candidate_name}? The uploaded video will be discarded.`)) return;
          try { await api("POST", `/api/analyses/${d.id}/cancel`); toast("Cancelling…"); } catch (err) { toast(err.message, true); }
        } }, "Cancel analysis")));
  } else {
    card.append(h("h2", {}, d.status === "failed" ? "The analysis failed" : d.status === "cancelled" ? "The analysis was cancelled" : "Results unavailable"),
      h("p", {}, d.error || ""),
      h("button", { class: "btn danger", type: "button", onclick: async () => { if (await deleteAnalysis(d)) location.hash = "#/"; } },
        "Delete this analysis"));
  }
  return h("div", {}, ...kids, card);
}

function buildReview(holder, d) {
  const report = d.report, review = d.review, files = d.files_url;
  const stats = report.statistics;
  const incidents = report.incidents;
  const reviews = review.incidents;
  const statusOf = n => (reviews[String(n)] || {}).status || INCIDENT.UNREVIEWED;

  /* --- player + timeline --- */
  let video = null;
  const playerCard = h("div", { class: "card player-card" });
  if (report.review_video) {
    video = h("video", { controls: true, preload: "metadata", src: files + report.review_video });
    playerCard.append(video);
  } else {
    playerCard.append(h("div", { class: "no-video" },
      "This analysis has no browser-playable review video (it was made before the web app, or H.264 was unavailable). ",
      "Use the clip links, or open the report."));
  }
  const canvas = h("canvas", {});
  const playhead = h("div", { class: "playhead", hidden: !video });
  const tip = h("div", { class: "tip" });
  const tl = h("div", { class: "timeline", role: "slider", "aria-label": "Video timeline", tabindex: "0",
    "aria-valuemin": "0", "aria-valuemax": String(Math.round(stats.analyzed_duration_s)) }, canvas, playhead, tip);
  const markers = h("div", { class: "markers" });
  const clock = h("span", { class: "clock" }, "00:00 / " + timecode(stats.analyzed_duration_s));
  const legend = h("div", { class: "legend" }, Object.entries(TL_LABELS).map(([k, label]) => h("span", { style: `--c: var(${TL_COLORS[k]})` }, label)));
  let current = -1;
  const prevBtn = h("button", { class: "btn small", type: "button", onclick: () => jumpIncident(-1), disabled: !incidents.length }, "◀ Previous incident");
  const nextBtn = h("button", { class: "btn small", type: "button", onclick: () => jumpIncident(1), disabled: !incidents.length }, "Next incident ▶");
  playerCard.append(h("div", { class: "timeline-wrap" }, tl, markers, legend,
    h("div", { class: "player-tools" }, h("div", {}, prevBtn, " ", nextBtn), clock)));

  const total = Math.max(stats.analyzed_duration_s, 0.001);
  let segments = [];
  const drawTimeline = () => {
    const r = tl.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = cssVar("--tl-uncertain");
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (const [a, b, kind] of segments) {
      ctx.fillStyle = cssVar(TL_COLORS[kind] || "--tl-uncertain");
      const x = Math.floor((a / total) * canvas.width);
      const w = Math.max(1, Math.ceil(((b - a) / total) * canvas.width));
      ctx.fillRect(x, 0, w, canvas.height);
    }
  };
  api("GET", `/api/analyses/${d.id}/timeline`).then(t => { segments = t.segments; drawTimeline(); }).catch(e => toast(e.message, true));
  const ro = new ResizeObserver(drawTimeline);
  ro.observe(tl);
  const mq = matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", drawTimeline);
  cleanup.push(() => { ro.disconnect(); mq.removeEventListener("change", drawTimeline); });

  const timeAt = clientX => { const r = tl.getBoundingClientRect(); return Math.min(Math.max((clientX - r.left) / r.width, 0), 1) * total; };
  const kindAt = t => { const s = segments.find(([a, b]) => t >= a && t < b); return s ? TL_LABELS[s[2]] : ""; };
  let stopAt = null;
  const seek = (t, play = false, until = null) => {
    if (!video) return;
    video.currentTime = Math.min(Math.max(t, 0), total);
    stopAt = until;
    if (play) video.play().catch(() => {});
  };
  tl.addEventListener("click", e => seek(timeAt(e.clientX)));
  tl.addEventListener("mousemove", e => {
    const t = timeAt(e.clientX);
    tip.style.display = "block";
    tip.style.left = `${(t / total) * 100}%`;
    tip.textContent = `${timecode(t)} · ${kindAt(t)}`;
  });
  tl.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  tl.addEventListener("keydown", e => {
    if (!video) return;
    if (e.key === "ArrowRight") { seek(video.currentTime + 5); e.preventDefault(); }
    if (e.key === "ArrowLeft") { seek(video.currentTime - 5); e.preventDefault(); }
  });
  if (video) {
    const onTime = () => {
      playhead.style.left = `${(video.currentTime / total) * 100}%`;
      clock.textContent = `${timecode(video.currentTime)} / ${timecode(total)}`;
      tl.setAttribute("aria-valuenow", String(Math.round(video.currentTime)));
      if (stopAt !== null && video.currentTime >= stopAt) { video.pause(); stopAt = null; }
      const idx = incidents.findIndex(i => video.currentTime >= i.start_s && video.currentTime < i.end_s);
      incidentEls.forEach((el, k) => el.classList.toggle("active", k === idx));
    };
    video.addEventListener("timeupdate", onTime);
    video.addEventListener("seeked", onTime);
    video.addEventListener("pause", () => { stopAt = null; });
  }
  const pad = report.settings.clip_padding_s || 1;
  const playIncident = i => { current = incidents.indexOf(i); seek(i.start_s - pad, true, i.end_s + pad); };
  function jumpIncident(dir) {
    if (!incidents.length) return;
    const t = video ? video.currentTime : 0;
    let k;
    if (dir > 0) { k = incidents.findIndex(i => i.start_s - pad > t + 0.3); if (k < 0) k = 0; }
    else { k = -1; incidents.forEach((i, j) => { if (i.start_s - pad < t - 0.3) k = j; }); if (k < 0) k = incidents.length - 1; }
    playIncident(incidents[k]);
    incidentEls[k].scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  const drawMarkers = () => markers.replaceChildren(...incidents.map(i =>
    h("button", { type: "button", class: statusOf(i.number), style: `left:${((i.start_s + i.end_s) / 2 / total) * 100}%`,
      title: `Incident ${i.number}: ${timecode(i.start_s)}–${timecode(i.end_s)}`, onclick: () => playIncident(i) }, String(i.number))));

  /* --- incidents --- */
  const incidentEls = [];
  const reviewCount = h("span", { class: "review-progress" });
  const updateCount = () => {
    const done = incidents.filter(i => statusOf(i.number) !== INCIDENT.UNREVIEWED).length;
    reviewCount.textContent = incidents.length ? `${done} of ${incidents.length} reviewed` : "";
    drawMarkers();
  };
  const saveIncident = async (i, status, note, savedEl, card) => {
    const reviewer = await requireReviewer();
    if (!reviewer) return false;
    try {
      const r = await api("PUT", `/api/analyses/${d.id}/incidents/${i.number}/review`, { status, note, reviewer });
      Object.assign(reviews, r.incidents);
      card.className = "incident " + statusOf(i.number);
      savedEl.textContent = `Saved · ${reviewer}`;
      updateCount();
      return true;
    } catch (e) { toast(e.message, true); return false; }
  };
  const incidentList = h("div", {}, incidents.length ? incidents.map(i => {
    const r = reviews[String(i.number)] || { status: INCIDENT.UNREVIEWED, note: "" };
    const saved = h("span", { class: "saved" }, r.reviewer ? `Saved · ${r.reviewer}` : "");
    const note = h("textarea", { placeholder: "Note (optional), e.g. what the candidate was doing", maxlength: "4000", "aria-label": `Note for incident ${i.number}` });
    note.value = r.note || "";
    const card = h("div", { class: "incident " + r.status });
    const confirmBtn = h("button", { type: "button", class: "confirm", "aria-pressed": String(r.status === INCIDENT.CONFIRMED) }, "Confirm");
    const dismissBtn = h("button", { type: "button", class: "dismiss", "aria-pressed": String(r.status === INCIDENT.FALSE_ALARM) }, "False alarm");
    const setStatus = async target => {
      const now = statusOf(i.number);
      const next = now === target ? INCIDENT.UNREVIEWED : target;
      if (await saveIncident(i, next, note.value, saved, card)) {
        confirmBtn.setAttribute("aria-pressed", String(next === INCIDENT.CONFIRMED));
        dismissBtn.setAttribute("aria-pressed", String(next === INCIDENT.FALSE_ALARM));
      }
    };
    confirmBtn.onclick = () => setStatus(INCIDENT.CONFIRMED);
    dismissBtn.onclick = () => setStatus(INCIDENT.FALSE_ALARM);
    note.addEventListener("change", () => saveIncident(i, statusOf(i.number), note.value, saved, card));
    const thumb = i.snapshot_file ? h("img", { src: files + i.snapshot_file, alt: `Snapshot of incident ${i.number}`, onclick: () => playIncident(i) })
      : h("div", {});
    card.append(thumb,
      h("div", {},
        h("div", { class: "head" }, h("span", { class: "title" }, `#${i.number}`),
          h("button", { class: "btn small ghost when", type: "button", onclick: () => playIncident(i), title: "Play this incident" },
            `▶ ${timecode(i.start_s)} – ${timecode(i.end_s)}`)),
        h("div", { class: "what" }, `${duration(i.duration_s)} · ${i.reason_text}${i.direction ? ` · ${i.direction}` : ""}`)),
      h("div", { class: "controls" }, h("div", { class: "seg", role: "group", "aria-label": `Decision for incident ${i.number}` }, confirmBtn, dismissBtn),
        i.clip_file ? h("a", { class: "btn small ghost", href: files + i.clip_file, target: "_blank", rel: "noopener" }, "Clip ↗") : null, saved),
      note);
    incidentEls.push(card);
    return card;
  }) : h("p", { class: "muted" }, "No periods of not looking at the screen were found."));

  /* --- decision --- */
  const decisionName = "decision";
  const options = [["accept", `Accept the automatic verdict (${report.verdict})`], [VERDICT.CHEATING, "CHEATING"], [VERDICT.CLEAN, "NO CHEATING"], [VERDICT.INCONCLUSIVE, "INCONCLUSIVE"]];
  const initial = review.decision_made ? (review.final_verdict || "accept") : null;
  const decisionNote = h("textarea", { placeholder: "Reason for your decision (recommended when overriding)", maxlength: "4000", "aria-label": "Decision note" });
  decisionNote.value = review.verdict_note || "";
  const decisionState = h("div", { class: "hint" });
  const headerDecision = h("span", {});
  const showDecision = s => {
    headerDecision.replaceChildren(s.reviewed ? verdictBadge(s.effective_verdict, "Decision") : h("span", { class: "badge pending" }, "Needs review"));
    decisionState.textContent = s.reviewed ? `Saved by ${s.reviewer} at ${s.reviewed_at}.` : "No decision recorded yet.";
  };
  showDecision(d);
  const decisionForm = h("form", { class: "decision", onsubmit: async e => {
    e.preventDefault();
    const choice = new FormData(e.target).get(decisionName);
    if (!choice) { toast("Choose a decision first.", true); return; }
    const pending = incidents.filter(i => statusOf(i.number) === INCIDENT.UNREVIEWED).length;
    if (pending && !confirm(`${pending} incident(s) are not reviewed yet. Save the decision anyway?`)) return;
    const reviewer = await requireReviewer();
    if (!reviewer) return;
    try {
      const s = await api("PUT", `/api/analyses/${d.id}/review`, { decision: choice, note: decisionNote.value, reviewer });
      showDecision(s);
      toast("Decision saved. The report was updated.");
    } catch (err) { toast(err.message, true); }
  } },
    options.map(([value, label]) => h("label", { class: "opt" },
      h("input", { type: "radio", name: decisionName, value, checked: initial === value }), h("span", {}, label))),
    h("div", { class: "field", style: "margin-top:12px" }, decisionNote),
    h("div", { class: "form-actions", style: "justify-content:space-between;margin-top:12px" }, decisionState,
      h("button", { class: "btn primary", type: "submit" }, "Save decision")));

  const warnings = report.warnings.length ? h("div", { class: "warn-box" }, h("strong", {}, "Warnings"),
    h("ul", {}, report.warnings.map(w => h("li", {}, w)))) : null;

  holder.replaceChildren(
    h("div", { class: "page-head" },
      h("div", {}, h("a", { href: "#/", class: "back" }, "← All analyses"),
        h("h1", { style: "margin-top:6px" }, report.candidate_name),
        h("p", { class: "sub" }, `${d.video_filename} · analyzed ${report.analyzed_at} · ${duration(stats.analyzed_duration_s)}`)),
      h("div", { style: "display:flex;gap:8px;align-items:center;flex-wrap:wrap" },
        verdictBadge(report.verdict, "Automatic"), headerDecision,
        h("a", { class: "btn", href: files + "report/report.html", target: "_blank", rel: "noopener" }, "Open report ↗"),
        h("button", { class: "btn danger", type: "button", onclick: async () => {
          const v = playerCard.querySelector("video");
          if (v) { v.pause(); v.removeAttribute("src"); v.load(); }   // release the file before deleting
          if (await deleteAnalysis(d)) location.hash = "#/";
          else if (v && report.review_video) { v.src = files + report.review_video; }
        } }, "Delete"))),
    h("div", { class: "review-grid" },
      h("div", {}, playerCard,
        h("div", { class: "card" }, h("div", { class: "head", style: "display:flex;justify-content:space-between;align-items:baseline" },
          h("h2", {}, "Incidents"), reviewCount), incidentList)),
      h("div", {},
        h("div", { class: "card" }, h("h2", {}, "Automatic verdict"), verdictBadge(report.verdict),
          h("ul", { class: "reasons" }, report.verdict_reasons.map(r => h("li", {}, r))),
          h("div", { class: "stats", style: "margin-top:14px" },
            stat(String(stats.incident_count), "Periods not looking"),
            stat(duration(stats.total_not_looking_s), `Total (${(stats.not_looking_fraction * 100).toFixed(1)}% of video)`),
            stat(duration(stats.longest_incident_s), "Longest period"),
            stat(`${Math.round(stats.gaze_coverage * 100)}%`, "Gaze measurable"),
            stats.multiple_faces_s > 0 ? stat(duration(stats.multiple_faces_s), "More than one face (not in verdict)") : null)),
        h("div", { class: "card" }, h("h2", {}, "Your decision"), decisionForm),
        warnings ? h("div", { class: "card" }, warnings) : null,
        h("div", { class: "card" }, h("h2", {}, "Results folder"), h("p", { class: "muted", style: "overflow-wrap:anywhere;margin:0" }, d.results_dir)))));
  updateCount();
  requestAnimationFrame(drawTimeline);
}

function stat(n, k) { return h("div", { class: "stat" }, h("div", { class: "n" }, n), h("div", { class: "k" }, k)); }

/* ---------------------------------------------------------------- settings */
async function renderSettings() {
  view.append(h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Settings"),
    h("p", { class: "sub" }, "Detection limits used for new analyses. Existing results keep the settings they were analyzed with."))));
  const holder = h("div", {});
  view.append(holder);
  let fields;
  try { fields = await api("GET", "/api/settings"); } catch (e) { holder.append(h("p", { class: "error-text" }, e.message)); return; }
  const draw = () => {
    const inputs = new Map();
    const groups = [...new Set(fields.map(f => f.group))];
    const errors = h("div", { class: "error-text", role: "alert" });
    const form = h("form", { novalidate: true, onsubmit: async e => {
      e.preventDefault();
      errors.textContent = "";
      const values = {};
      for (const f of fields) {
        const el = inputs.get(f.key);
        if (f.type === "bool") values[f.key] = el.checked;
        else {
          const raw = el.value.trim();
          if (raw === "" || Number.isNaN(Number(raw))) { errors.textContent = `${f.label}: enter a number.`; el.focus(); return; }
          values[f.key] = f.percent ? Number(raw) / 100 : Number(raw);
        }
      }
      try { fields = await api("PUT", "/api/settings", { values }); draw(); toast("Settings saved. They apply to the next analysis."); }
      catch (err) { errors.textContent = err.message; }
    } },
      groups.map(g => h("div", { class: "card settings-group" }, h("h2", {}, g), fields.filter(f => f.group === g).map(f => {
        const id = "s-" + f.key.replace(/\./g, "-");
        let input;
        if (f.type === "bool") input = h("input", { type: "checkbox", id, class: "switch", checked: f.value });
        else {
          const show = v => f.percent ? +(v * 100).toFixed(2) : v;
          input = h("input", { type: "number", id, step: f.type === "int" ? "1" : "any",
            min: f.min === null ? null : show(f.min), max: f.max === null ? null : show(f.max) });
          input.value = show(f.value);
        }
        inputs.set(f.key, input);
        const def = f.type === "bool" ? (f.default ? "on" : "off") : (f.percent ? `${+(f.default * 100).toFixed(2)}%` : f.default);
        return h("div", { class: "setting" + (JSON.stringify(f.value) !== JSON.stringify(f.default) ? " changed" : "") },
          h("label", { for: id }, f.label),
          h("div", { class: "input-wrap" }, input, f.percent ? "%" : ""),
          h("div", { class: "hint" }, [f.help, `Default: ${def}.`].filter(Boolean).join(" ")));
      }))),
      errors,
      h("div", { class: "form-actions" },
        h("button", { class: "btn ghost", type: "button", onclick: async () => {
          if (!confirm("Reset every setting to its default?")) return;
          try { fields = await api("POST", "/api/settings/reset"); draw(); toast("Settings reset to defaults."); } catch (e) { toast(e.message, true); }
        } }, "Reset to defaults"),
        h("button", { class: "btn primary", type: "submit" }, "Save settings")));
    holder.replaceChildren(form);
  };
  draw();
}

/* ---------------------------------------------------------------- start */
renderReviewerChip();
route();
