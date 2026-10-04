"""
AI-Assisted Counter-Drone Decision Trainer
Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

import engine as E
import viz as V

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "Sample_Image")
MODEL_PATHS = [os.path.join(HERE, p) for p in ("best.pt", "models/best.pt", "weights/best.pt")]

st.set_page_config(page_title="C-UAS Decision Trainer", page_icon="🛡️", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem;
       font-weight:700; letter-spacing:.4px; color:#111;}
.pill-open {background:#ffbe00;} .pill-drone {background:#ff4646; color:#fff;}
.pill-done {background:#6c757d; color:#fff;}
.hero {padding:1.1rem 1.3rem; border-radius:14px; border:1px solid #2b3a4a;
       background:linear-gradient(135deg,#0e1720,#16232f);}
.small {font-size:0.85rem; opacity:.8;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- cached helpers
@st.cache_data(show_spinner=False)
def load_ds(folder: str):
    return E.discover_dataset(folder)


@st.cache_data(show_spinner=False)
def display_image(path: str):
    return V.load_display(path, 1600)


@st.cache_data(show_spinner=False)
def zoom_image(path: str, xywhn: tuple, fov: float):
    return V.crop_zoom(path, xywhn, fov)


@st.cache_data(show_spinner=False)
def gt_thumb(path: str, boxes: tuple, W: int, H: int):
    return V.render_gt(V.load_display(path, 900), boxes, W, H)


@st.cache_resource(show_spinner=False)
def load_model():
    """Optional: real YOLO weights (best.pt) turn the AI assistant from simulated to real."""
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                from ultralytics import YOLO
                return YOLO(p), os.path.relpath(p, HERE)
            except Exception as ex:  # noqa: BLE001
                return None, f"found {os.path.basename(p)} but could not load it ({type(ex).__name__})"
    return None, None


@st.cache_data(show_spinner=False)
def detect(_model, path: str):
    r = _model.predict(path, verbose=False, conf=0.15, imgsz=1920)[0]
    out = []
    for b in r.boxes:
        x, y, w, h = (float(v) for v in b.xywhn[0])
        out.append((int(b.cls[0]), float(b.conf[0]), (x, y, w, h)))
    return out


# ---------------------------------------------------------------- state callbacks
def _M():
    return st.session_state.M


def _frame(M):
    return M["frames"][M["fi"]]


def cb_select(i):
    M = _M()
    M["sel"] = i
    E.touch(_frame(M)["contacts"][i], time.time())


def cb_ai(i):
    M = _M()
    E.do_ask_ai(M, _frame(M)["contacts"][i])


def cb_classify(i, choice):
    M = _M()
    f = _frame(M)
    E.do_classify(M, f, f["contacts"][i], choice, time.time())


def cb_assess(i, kind):
    M = _M()
    f = _frame(M)
    E.do_assess(M, f, f["contacts"][i], kind, time.time())


def cb_respond(i, action):
    M = _M()
    f = _frame(M)
    E.do_respond(M, f, f["contacts"][i], action, time.time())


def cb_next_open(i):
    M = _M()
    j = E.next_open(_frame(M), i)
    if j is not None:
        cb_select(j)


def cb_next_frame():
    E.advance_frame(_M(), time.time())


def cb_new():
    st.session_state.M = None


# ---------------------------------------------------------------- sidebar
model, model_info = load_model()

with st.sidebar:
    st.title("🛡️ C-UAS Trainer")
    M0 = st.session_state.get("M")
    active = bool(M0) and M0["phase"] == "active"

    folder = st.text_input("Dataset folder", DEFAULT_DATA, key="folder", disabled=active,
                           help="Folder with images and YOLO .txt labels (class 0 = drone, 1 = bird).")
    items = load_ds(folder)
    st.caption(f"{len(items)} labelled frames found" if items else "No labelled frames found")

    callsign = st.text_input("Operator callsign", "ALPHA-1", key="callsign", disabled=active)
    difficulty = st.selectbox("Difficulty", list(E.DIFFICULTY), index=1, key="difficulty", disabled=active)
    roe = st.selectbox("Rules of engagement", E.ROE_LEVELS, index=1, key="roe", disabled=active)
    n_frames = st.slider("Scenarios per mission", 3, 12, 6, key="nframes", disabled=active)
    timers_on = st.checkbox("Time pressure", value=True, key="timers", disabled=active,
                            help="Off = training mode with no clocks.")
    seed_in = st.number_input("Scenario seed (0 = random)", min_value=0, value=0, step=1,
                              key="seed", disabled=active)

    if model is not None:
        st.success(f"AI engine: YOLO ({model_info})")
    else:
        st.info("AI engine: simulated assistant calibrated on your dataset.\n\n"
                + (model_info or "Drop a trained `best.pt` next to app.py to use real detections."))

    start = st.button("▶ START MISSION", type="primary", disabled=active or not items)
    abort = st.button("■ Abort mission", disabled=not active)

if start:
    seed = int(seed_in) if seed_in else int(time.time()) % 100000

    def _model_fn(path):
        if model is None:
            return None
        try:
            return detect(model, path)
        except Exception:  # noqa: BLE001
            return None

    st.session_state.M = E.build_mission(items, n_frames, difficulty, roe, callsign, seed,
                                         timers_on, _model_fn if model is not None else None)
    st.rerun()

if abort and active:
    E.finish(M0, "ABORTED")
    st.rerun()

if not items:
    st.error("No labelled images found. Put your images and YOLO .txt files in a folder called "
             "`Sample_Image` next to app.py, or change the path in the sidebar.")
    st.stop()


# ---------------------------------------------------------------- UI pieces
LEVEL_ICON = {"good": "✅", "bad": "❌", "warn": "⚠️", "info": "ℹ️"}


def render_events(M, n=7):
    with st.container(border=True):
        st.markdown("**📻 Comms log**")
        for ev in reversed(M["events"][-n:]):
            t = int(ev["ts"] - M["t0"])
            st.markdown(f"`{t // 60:02d}:{t % 60:02d}` {LEVEL_ICON[ev['level']]} {ev['text']}")


def clock_panel(M):
    """Live ETAs (only for contacts you have tracked) + automatic expiry. Runs as a fragment."""
    frame = _frame(M)
    now = time.time()
    changed = E.expire_overdue(M, frame, now)
    E.check_end(M)
    shown = False
    for c in frame["contacts"]:
        if (c["status"] != "RESOLVED" and c["tracked"] and c["heading"] == "INBOUND"
                and c["deadline"] and c["limit"]):
            rem = max(0.0, c["deadline"] - now)
            st.progress(min(1.0, rem / c["limit"]), text=f"⏱️ {c['cid']} inbound - ETA to perimeter {int(rem)} s")
            shown = True
    if not shown:
        st.caption("⏱️ No ETA yet. Identify a drone, then TRACK it to see whether it is on an intercept course.")
    if changed or M["phase"] != "active":
        st.rerun()


def contact_buttons(M, frame):
    cols = st.columns(len(frame["contacts"]))
    for k, c in enumerate(frame["contacts"]):
        mark = {"OPEN": "❔", "CLASSIFIED": "🛸", "RESOLVED": "✔"}[c["status"]]
        cols[k].button(f"{mark} {c['cid']}", key=f"sel_{frame['id']}_{k}",
                       type="primary" if k == M["sel"] else "secondary",
                       on_click=cb_select, args=(k,))


def contact_panel(M, frame, c):
    i, fid = c["idx"], frame["id"]
    pill = {"OPEN": ("UNKNOWN", "pill-open"), "CLASSIFIED": ("DRONE", "pill-drone"),
            "RESOLVED": ("RESOLVED", "pill-done")}[c["status"]]
    st.markdown(f"### Contact {c['cid']} &nbsp;<span class='pill {pill[1]}'>{pill[0]}</span>",
                unsafe_allow_html=True)

    fov = st.slider("Field of view (× target size)", 2, 16, 4, key="fov")
    st.image(zoom_image(frame["path"], tuple(c["xywhn"]), float(fov)),
             caption=f"Full-resolution crop from the {frame['W']}×{frame['H']} frame")
    pct = c["px"] / frame["W"] * 100
    st.caption(f"Apparent size **{c['px']} px** ({pct:.1f}% of frame width) · range band **{c['band']}**")

    # ---- AI assistant
    if not c["ai_shown"]:
        if c["status"] != "RESOLVED":
            st.button("🤖 Ask AI classifier", key=f"ai_{fid}_{i}", on_click=cb_ai, args=(i,))
    else:
        st.markdown(f"🤖 AI suggests **{c['ai_label']}** · confidence **{c['ai_conf']:.0%}**")
        st.progress(float(min(max(c["ai_conf"], 0.0), 1.0)))
        if c["status"] != "OPEN":
            if c["ai_label"] == c["truth"]:
                st.caption("AI assessment was correct.")
            else:
                st.caption("⚠️ AI assessment was WRONG - this is why you stay in the loop.")

    # ---- step 1: identify
    if c["status"] == "OPEN":
        st.markdown("**Step 1 - Identify**")
        b1, b2 = st.columns(2)
        b1.button("🛸 DRONE", key=f"cd_{fid}_{i}", on_click=cb_classify, args=(i, "Drone"))
        b2.button("🐦 BIRD", key=f"cb_{fid}_{i}", on_click=cb_classify, args=(i, "Bird"))
        return

    # ---- resolved
    if c["status"] == "RESOLVED":
        msg = f"**{c['outcome']}** · {c['verdict']} · {c['points']:+d} pts\n\n{c['note']}"
        {"OPTIMAL": st.success, "ACCEPTABLE": st.info, "SUBOPTIMAL": st.warning,
         "VIOLATION": st.error, "FAILED": st.error}.get(c["verdict"], st.info)(msg)
        if not E.frame_complete(frame):
            st.button("Next open contact ▶", key=f"no_{fid}_{i}", on_click=cb_next_open, args=(i,))
        return

    # ---- step 2: assess
    st.markdown("**Step 2 - Assess** *(each check costs time on the clock)*")
    a1, a2 = st.columns(2)
    a1.button("📍 Track target", key=f"tr_{fid}_{i}", disabled=c["tracked"],
              on_click=cb_assess, args=(i, "TRACK"))
    a2.button("🪪 Registry / Remote-ID", key=f"rg_{fid}_{i}", disabled=c["registry"] is not None,
              on_click=cb_assess, args=(i, "REGISTRY"))
    if c["tracked"]:
        st.markdown(f"📍 Heading **{c['heading']}** · speed ≈ **{c['speed']} m/s** (sim. estimate)")
    if c["registry"]:
        icon = "🟦" if c["registry"] == "REGISTERED" else "🟥"
        st.markdown(f"🪪 Registry: {icon} **{c['registry']}**")

    # ---- step 3: respond
    st.markdown("**Step 3 - Decide your response**")
    for r in range(2):
        cols = st.columns(3)
        for j in range(3):
            a = E.ACTION_ORDER[r * 3 + j]
            ok, why = E.action_gate(M, frame, c, a)
            meta = E.ACTIONS[a]
            cols[j].button(f"{meta['icon']} {meta['label']}", key=f"ac_{a}_{fid}_{i}", disabled=not ok,
                           help=meta["desc"] if ok else f"Blocked: {why}",
                           on_click=cb_respond, args=(i, a))
    blocked = [(a, E.action_gate(M, frame, c, a)[1]) for a in ("NET", "KIN")
               if not E.action_gate(M, frame, c, a)[0]]
    for a, why in blocked:
        st.caption(f"🔒 {E.ACTIONS[a]['label']}: {why}")
    if c["attempts"]:
        st.warning(f"{c['attempts']} failed attempt(s) - the drone is still active.")


def render_active(M):
    frame = _frame(M)
    E.expire_overdue(M, frame, time.time())
    E.check_end(M)
    if M["phase"] != "active":
        return render_debrief(M)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Score", M["score"])
    m2.metric("Asset integrity", f"{M['asset']}%")
    m3.metric("Scenario", f"{M['fi'] + 1}/{len(M['frames'])}")
    m4.metric("Net rounds", M["inv"]["NET"])
    m5.metric("Kinetic rounds", M["inv"]["KIN"])
    st.progress(M["asset"] / 100)

    left, right = st.columns([3, 2])
    with left:
        st.image(V.render_feed(display_image(frame["path"]), frame, M, M["sel"]))
        contact_buttons(M, frame)
        needs_tick = any(c["deadline"] and c["status"] != "RESOLVED" for c in frame["contacts"])
        st.fragment(run_every=1 if needs_tick else None)(clock_panel)(M)
        if not M["timers_on"]:
            st.caption("Training mode: time pressure is off.")

        if E.frame_complete(frame):
            pts = sum(c["points"] for c in frame["contacts"])
            st.success(f"Scenario cleared - {pts:+d} pts this scenario.")
            last = M["fi"] + 1 >= len(M["frames"])
            st.button("🏁 Finish mission - view after-action review" if last else "Next scenario ▶",
                      key=f"nf_{frame['id']}", type="primary", on_click=cb_next_frame)
        render_events(M)
    with right:
        contact_panel(M, frame, frame["contacts"][M["sel"]])


def render_debrief(M):
    S = E.summarize(M)
    m, rows = S["m"], S["rows"]
    dur = int((M["t1"] or time.time()) - M["t0"])
    st.header(f"After-action review · {M['callsign']}")
    if M["end_reason"] == "ASSET DESTROYED":
        st.error("MISSION FAILED - the protected asset was destroyed.")
    elif M["end_reason"] == "ABORTED":
        st.warning("Mission aborted.")
    else:
        st.success("Mission complete.")
    st.markdown(f"### Readiness: **{m['grade']}** &nbsp; ({m['index']:.0%})")
    st.caption(f"{M['difficulty']} · {M['roe']} · seed {M['seed']} · {dur // 60}m {dur % 60}s")

    c = st.columns(5)
    c[0].metric("Score", m["score"])
    c[1].metric("Asset integrity", f"{m['asset']}%")
    c[2].metric("ID accuracy", f"{m['id_acc']:.0%}")
    c[3].metric("Hostiles stopped", f"{m['neutralised']}/{m['hostile']}")
    c[4].metric("Avg. ID time", f"{m['avg_react']:.1f}s")
    c = st.columns(5)
    c[0].metric("Friendly fire", m["fratricide"])
    c[1].metric("Impacts", m["impacts"])
    c[2].metric("False alarms", m["false_alarms"])
    c[3].metric("AI wrong & followed", m["followed_wrong"])
    c[4].metric("AI right & overridden", m["overrode_right"])

    st.subheader("Coaching points")
    for ls in S["lessons"]:
        st.markdown(f"- {ls}")

    df = pd.DataFrame(rows)
    show = df.drop(columns=[x for x in df.columns if x.startswith("_")])
    st.subheader("Decision log")
    st.dataframe(show)
    if show["ID time (s)"].notna().any():
        st.caption("Time to identify each contact (seconds)")
        st.bar_chart(show.set_index("Contact")["ID time (s)"])

    report = dict(callsign=M["callsign"], difficulty=M["difficulty"], roe=M["roe"], seed=M["seed"],
                  end_reason=M["end_reason"], duration_s=dur, metrics=m, decisions=rows)
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Decision log (CSV)", show.to_csv(index=False).encode("utf-8"),
                       file_name="after_action_log.csv", mime="text/csv")
    d2.download_button("⬇ Full report (JSON)", json.dumps(report, indent=2, default=str).encode("utf-8"),
                       file_name="after_action_report.json", mime="application/json")
    st.button("↺ New mission", type="primary", on_click=cb_new)


def render_briefing():
    st.markdown(
        """
<div class="hero">
<h2 style="margin:0">🛡️ AI-Assisted Counter-Drone Decision Trainer</h2>
<p style="margin:.4rem 0 0 0">Real sky footage. Real labels. The AI spots contacts - <b>you</b> decide what they are and what to do about them.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1 · Detect**\n\nThe AI cues every aerial contact in the live frame.")
    c2.markdown("**2 · Identify**\n\nZoom into the full-resolution crop. Drone or bird? Optionally ask the AI - it can be wrong.")
    c3.markdown("**3 · Assess**\n\nTrack the heading and run a registry check. Is it friendly, recon or attack?")
    c4.markdown("**4 · Respond**\n\nPick a proportionate response within your ROE. Wrong calls cost the asset, or a friendly drone.")
    st.info("Set difficulty and rules of engagement in the sidebar, then press **START MISSION**.")


# ---------------------------------------------------------------- dataset + doctrine tabs
def dataset_tab():
    st.subheader("Dataset behind the scenarios")
    rows = []
    for it in items:
        for cls, x, y, w, h in it["boxes"]:
            px = E.box_px((cls, x, y, w, h), it["W"], it["H"])
            rows.append(dict(Image=it["name"], Class="Drone" if cls == E.DRONE_CLASS else "Bird",
                             Pixels=px, PctWidth=round(px / it["W"] * 100, 2)))
    df = pd.DataFrame(rows)
    thr = E.drone_size_thresholds(items)
    c = st.columns(4)
    c[0].metric("Frames", len(items))
    c[1].metric("Resolution", f"{items[0]['W']}×{items[0]['H']}")
    c[2].metric("Drones", int((df.Class == "Drone").sum()))
    c[3].metric("Birds", int((df.Class == "Bird").sum()))
    st.caption(f"Range bands come from the data: drone apparent size < {thr[0]:.0f} px = FAR, "
               f"{thr[0]:.0f}-{thr[1]:.0f} px = MID, above = NEAR. "
               f"Median drone is {df[df.Class == 'Drone'].Pixels.median():.0f} px in a "
               f"{items[0]['W']} px wide frame - tiny targets, which is why the AI helps and why zoom matters.")
    a, b = st.columns(2)
    a.caption("Objects per class")
    a.bar_chart(df.groupby("Class").size())
    hist, edges = np.histogram(df.Pixels, bins=8)
    b.caption("Apparent target size (px)")
    b.bar_chart(pd.DataFrame({"count": hist}, index=[f"{int(edges[k])}-{int(edges[k + 1])}" for k in range(len(hist))]))

    with st.expander("How the AI assistant works in this prototype", expanded=False):
        st.markdown(
            "- **Real:** frames, boxes and labels from your dataset. The boxes play the role of the detector's cues.\n"
            "- **Simulated by default:** the AI's drone/bird opinion. Its confidence rises with apparent target size, "
            "and it is sometimes confidently wrong, so trainees learn not to follow it blindly.\n"
            "- **Real when you add weights:** put a trained `best.pt` (Ultralytics YOLO, class 0 = drone, 1 = bird) "
            "next to `app.py`. The AI panel then shows the model's own prediction for each contact.\n"
            "- **Synthetic overlay:** intent, heading, registry result, civilians and ROE are generated so there are "
            "real decisions to make. They are not in your dataset."
        )
    st.subheader("Annotated frames (red = drone, green = bird)")
    cols = st.columns(3)
    for k, it in enumerate(items):
        cols[k % 3].image(gt_thumb(it["path"], tuple(it["boxes"]), it["W"], it["H"]),
                          caption=f"{it['name']} · {len(it['boxes'])} objects")


def doctrine_tab():
    st.subheader("Decision rules used in this trainer")
    st.markdown(
        """
**Threat types (hidden until you assess):** *Attack* (inbound, reaches the asset when its clock runs out),
*Recon* (orbits and collects), *Friendly* (registered, sometimes returning to base).

| Response | Use it when | Notes |
|---|---|---|
| Report & Monitor | Friendly or low-risk contact | Does not stop a hostile drone |
| Alert & Shelter | You cannot stop it in time | Halves impact damage |
| Stand Down | Registry says **REGISTERED** | Unverified stand-downs score less |
| Electronic Countermeasure | Most hostile drones | Works at all ranges; may disrupt civilian comms |
| Net Interceptor | Hostile drone at **close** range | Limited stock; poor at long range |
| Kinetic | Positively hostile, clear fire zone, ROE allows | Limited rounds; disproportionate vs recon |

| ROE | Kinetic allowed when |
|---|---|
| Weapons Hold | Never |
| Weapons Tight | Registry check returned NOT REGISTERED, and no civilians in the fire zone |
| Weapons Free | No civilians in the fire zone |

A **registry result is not infallible**: a friendly drone can fail to respond. Heading, speed and context all matter.

**Scoring highlights:** correct drone ID +25 (+ speed bonus) · correct bird +20 · drone called a bird −50 ·
neutralise attack drone +80 to +90 (+ time bonus) · effector on a friendly −80 / −150 ·
attack drone reaches asset −80 and 35% asset damage.
        """
    )


# ---------------------------------------------------------------- main
tab_console, tab_data, tab_doc = st.tabs(["🎯 Mission console", "📊 Dataset & AI", "📘 Doctrine & ROE"])
with tab_console:
    M = st.session_state.get("M")
    if M is None:
        render_briefing()
    elif M["phase"] == "debrief":
        render_debrief(M)
    else:
        render_active(M)
with tab_data:
    dataset_tab()
with tab_doc:
    doctrine_tab()
