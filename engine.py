"""
engine.py - dataset loading, scenario generation, rules of engagement, scoring.

No Streamlit in here, so every rule can be unit-tested and reused (e.g. in a
Flask/FastAPI backend or a notebook).

What is REAL and what is SIMULATED
----------------------------------
REAL      : images, YOLO boxes and class labels from your dataset
            (class 0 = drone, class 1 = bird); apparent target size in pixels.
SIMULATED : the AI classifier's opinion (unless a YOLO weights file is supplied,
            then it is real model output), plus the tactical overlay
            (intent, heading, registry, civilians, ROE). These exist so the
            trainee has real decisions to make.
"""
from __future__ import annotations

import glob
import os
import random
import time
from typing import Callable, Optional

from PIL import Image

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
DRONE_CLASS = 0                      # dataset: 0 = drone, 1 = bird
IMG_EXT = (".jpg", ".jpeg", ".png")

ROE_LEVELS = ["Weapons Hold", "Weapons Tight", "Weapons Free"]

# weights are (ATTACK, RECON, FRIENDLY)
DIFFICULTY = {
    "Easy":   dict(time=1.5,  ai_bonus=0.06,  civ=0.25, net=6, kin=6, w=(0.40, 0.30, 0.30)),
    "Normal": dict(time=1.0,  ai_bonus=0.00,  civ=0.35, net=4, kin=5, w=(0.45, 0.30, 0.25)),
    "Hard":   dict(time=0.75, ai_bonus=-0.08, civ=0.45, net=3, kin=4, w=(0.50, 0.25, 0.25)),
}

ASSETS = ["Forward Operating Base", "Command Post", "Supply Depot",
          "Airstrip", "Checkpoint Alpha", "Field Hospital"]

# Seconds until an attacking drone reaches the asset, by apparent range band
BAND_IMPACT_S = {"NEAR": 28, "MID": 45, "FAR": 70}
RECON_DWELL_S = 100
ASSET_DAMAGE = 35

ACTIONS = {
    "REPORT": dict(label="Report & Monitor", icon="📡",
                   desc="Log the track and pass it to command. Has no effect on the drone."),
    "ALERT":  dict(label="Alert & Shelter", icon="🚨",
                   desc="Sound the alarm and send personnel to cover. Reduces harm, does not stop the drone."),
    "STAND":  dict(label="Stand Down (Friendly)", icon="✅",
                   desc="Clear the contact as a friendly drone."),
    "ECM":    dict(label="Electronic Countermeasure", icon="📶",
                   desc="Non-kinetic: disrupt the drone's control link. Works at all ranges; may disrupt civilian comms nearby."),
    "NET":    dict(label="Net Interceptor", icon="🕸️",
                   desc="Non-lethal capture. Effective at close range only. Limited stock."),
    "KIN":    dict(label="Kinetic Engagement", icon="🎯",
                   desc="Hard-kill. Needs ROE clearance and a clear fire zone. Limited rounds."),
}
ACTION_ORDER = ["REPORT", "ALERT", "STAND", "ECM", "NET", "KIN"]

# chance an effector works, by action and range band
SUCCESS = {
    "ECM": {"NEAR": 0.85, "MID": 0.80, "FAR": 0.60},
    "NET": {"NEAR": 0.85, "MID": 0.55, "FAR": 0.15},
    "KIN": {"NEAR": 0.85, "MID": 0.70, "FAR": 0.35},
}
# base points for a successful neutralisation, by hidden intent
BASE_POINTS = {
    "ATTACK": {"ECM": 80, "NET": 85, "KIN": 90},
    "RECON":  {"ECM": 70, "NET": 65, "KIN": 35},   # kinetic vs recon = disproportionate
}


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def parse_yolo(txt_path: str):
    """Return [(cls, x, y, w, h)] with normalised YOLO coordinates."""
    boxes = []
    try:
        with open(txt_path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return boxes
    for ln in lines:
        p = ln.split()
        if len(p) < 5:
            continue
        try:
            cls = int(float(p[0]))
            x, y, w, h = (float(v) for v in p[1:5])
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        boxes.append((cls, x, y, w, h))
    return boxes


def _nat_key(name: str):
    stem = os.path.splitext(name)[0]
    return (0, int(stem), "") if stem.isdigit() else (1, 0, stem)


def discover_dataset(folder: str):
    """Find image + YOLO .txt pairs in a folder."""
    items = []
    if not folder or not os.path.isdir(folder):
        return items
    for p in glob.glob(os.path.join(folder, "*")):
        if not p.lower().endswith(IMG_EXT):
            continue
        txt = os.path.splitext(p)[0] + ".txt"
        if not os.path.exists(txt):
            continue
        boxes = parse_yolo(txt)
        if not boxes:
            continue
        try:
            with Image.open(p) as im:
                W, H = im.size
        except OSError:
            continue
        items.append(dict(name=os.path.basename(p), path=p, W=W, H=H, boxes=boxes))
    items.sort(key=lambda d: _nat_key(d["name"]))
    return items


def box_px(box, W, H) -> int:
    _, _, _, w, h = box
    return int(max(w * W, h * H))


def drone_size_thresholds(items):
    """Tertiles of drone apparent size -> (FAR/MID cut, MID/NEAR cut)."""
    sizes = sorted(box_px(b, it["W"], it["H"])
                   for it in items for b in it["boxes"] if b[0] == DRONE_CLASS)
    if len(sizes) < 3:
        return (90.0, 110.0)
    return (float(sizes[len(sizes) // 3]), float(sizes[(2 * len(sizes)) // 3]))


def band_for(px: float, thr) -> str:
    if px < thr[0]:
        return "FAR"
    if px < thr[1]:
        return "MID"
    return "NEAR"


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------
# Scenario generation
# --------------------------------------------------------------------------
def _simulate_ai(truth: str, px: int, cfg, rng: random.Random):
    """AI opinion whose confidence grows with apparent target size (real data)."""
    s = min(max((px - 60) / 140.0, 0.0), 1.0)
    conf = min(max(0.55 + 0.38 * s + rng.uniform(-0.06, 0.06), 0.50), 0.98)
    p_correct = min(0.97, 0.55 + 0.45 * conf + cfg["ai_bonus"])
    if rng.random() < p_correct:
        return truth, conf
    other = "Bird" if truth == "Drone" else "Drone"
    return other, rng.uniform(0.55, 0.90)       # wrong AND confident


def _make_contact(i, box, W, H, thr, cfg, rng, timers_on, dets):
    cls, x, y, w, h = box
    truth = "Drone" if cls == DRONE_CLASS else "Bird"
    px = box_px(box, W, H)
    band = band_for(px, thr)

    # --- AI opinion: real YOLO output if available, otherwise simulated
    if dets is not None:
        best, best_iou = None, 0.0
        for d in dets:
            v = _iou((x, y, w, h), d[2])
            if v > best_iou:
                best, best_iou = d, v
        if best is not None and best_iou > 0.1:
            ai_label = "Drone" if best[0] == DRONE_CLASS else "Bird"
            ai_conf = float(best[1])
        else:
            ai_label, ai_conf = "No detection", 0.0
    else:
        ai_label, ai_conf = _simulate_ai(truth, px, cfg, rng)

    c = dict(
        cid=f"C{i + 1}", idx=i, cls=cls, truth=truth, xywhn=(x, y, w, h), px=px, band=band,
        ai_label=ai_label, ai_conf=ai_conf, ai_shown=False,
        intent=None, heading=None, speed=None, limit=None, deadline=None,
        status="OPEN",                       # OPEN -> CLASSIFIED -> RESOLVED
        soldier_class=None, misid=False, first_view=None, class_react=None, followed_ai=False,
        tracked=False, registry=None, attempts=0,
        action=None, outcome=None, verdict=None, points=0, note="",
    )

    if truth == "Drone":
        c["intent"] = rng.choices(["ATTACK", "RECON", "FRIENDLY"], weights=cfg["w"])[0]
        r = rng.random()
        if c["intent"] == "ATTACK":
            c["heading"] = "INBOUND" if r < 0.85 else "CROSSING"
            c["speed"] = rng.randint(15, 25)
        elif c["intent"] == "RECON":
            c["heading"] = "ORBITING" if r < 0.80 else "CROSSING"
            c["speed"] = rng.randint(2, 6)
        else:
            c["heading"] = "CROSSING" if r < 0.60 else ("ORBITING" if r < 0.85 else "INBOUND")
            c["speed"] = rng.randint(6, 12)

        if timers_on:
            f = cfg["time"]
            if c["intent"] == "ATTACK":
                c["limit"] = BAND_IMPACT_S[band] * f
            elif c["intent"] == "RECON":
                c["limit"] = RECON_DWELL_S * f
            elif c["heading"] == "INBOUND":          # friendly returning to base
                c["limit"] = BAND_IMPACT_S[band] * 1.2 * f
    return c


def build_mission(items, n_frames, difficulty, roe, callsign, seed, timers_on,
                  model_fn: Optional[Callable] = None):
    cfg = DIFFICULTY[difficulty]
    rng = random.Random(seed)
    pool = list(items)
    rng.shuffle(pool)
    chosen = pool[:n_frames]
    while len(chosen) < n_frames:
        chosen.append(rng.choice(items))
    chosen.sort(key=lambda it: len(it["boxes"]))       # stable: difficulty ramps up
    thr = drone_size_thresholds(items)

    frames = []
    for fi, it in enumerate(chosen):
        frng = random.Random(f"{seed}|{fi}|{it['name']}")
        dets = model_fn(it["path"]) if model_fn else None
        contacts = [
            _make_contact(i, b, it["W"], it["H"], thr, cfg,
                          random.Random(f"{seed}|{fi}|{i}"), timers_on, dets)
            for i, b in enumerate(it["boxes"])
        ]
        frames.append(dict(id=fi, name=it["name"], path=it["path"], W=it["W"], H=it["H"],
                           civilians=frng.random() < cfg["civ"],
                           asset=frng.choice(ASSETS), contacts=contacts))

    M = dict(callsign=callsign, difficulty=difficulty, roe=roe, seed=seed, timers_on=timers_on,
             frames=frames, fi=0, sel=0, score=0, asset=100,
             inv=dict(NET=cfg["net"], KIN=cfg["kin"]),
             events=[], phase="active", end_reason=None,
             t0=time.time(), t1=None, thr=thr)
    event(M, f"Mission start - ROE: {roe}. Sector scan in progress.", "info")
    start_frame(M, time.time())
    return M


# --------------------------------------------------------------------------
# State helpers
# --------------------------------------------------------------------------
def event(M, text, level="info"):
    M["events"].append(dict(ts=time.time(), text=text, level=level))
    del M["events"][:-40]


def touch(c, now):
    if c["first_view"] is None:
        c["first_view"] = now


def start_frame(M, now):
    frame = M["frames"][M["fi"]]
    M["sel"] = 0
    for c in frame["contacts"]:
        if c["limit"]:
            c["deadline"] = now + c["limit"]
    touch(frame["contacts"][0], now)
    n = len(frame["contacts"])
    extra = " Civilians reported near the perimeter." if frame["civilians"] else ""
    event(M, f"Frame {M['fi'] + 1}/{len(M['frames'])}: {n} aerial contact(s) near the {frame['asset']}.{extra}", "info")


def frame_complete(frame) -> bool:
    return all(c["status"] == "RESOLVED" for c in frame["contacts"])


def next_open(frame, after: int):
    n = len(frame["contacts"])
    for k in range(1, n + 1):
        j = (after + k) % n
        if frame["contacts"][j]["status"] != "RESOLVED":
            return j
    return None


def finish(M, reason):
    if M["phase"] != "debrief":
        M["phase"] = "debrief"
        M["end_reason"] = reason
        M["t1"] = time.time()


def check_end(M):
    if M["asset"] <= 0:
        finish(M, "ASSET DESTROYED")


def advance_frame(M, now):
    M["fi"] += 1
    if M["fi"] >= len(M["frames"]):
        M["fi"] = len(M["frames"]) - 1
        finish(M, "MISSION COMPLETE")
    else:
        start_frame(M, now)


def _award(M, c, pts):
    c["points"] += pts
    M["score"] += pts


def _resolve(M, c, outcome, verdict, pts, msg, level, asset_damage=0):
    _award(M, c, pts)
    c["status"] = "RESOLVED"
    c["outcome"], c["verdict"], c["note"] = outcome, verdict, msg
    if asset_damage:
        M["asset"] = max(0, M["asset"] - asset_damage)
    sign = f"{pts:+d}" if pts else "0"
    event(M, f"{c['cid']}: {msg} ({sign})", level)
    check_end(M)


# --------------------------------------------------------------------------
# Trainee actions
# --------------------------------------------------------------------------
def do_ask_ai(M, c):
    c["ai_shown"] = True


def do_classify(M, frame, c, choice, now):
    if c["status"] != "OPEN":
        return
    touch(c, now)
    c["soldier_class"] = choice
    react = now - c["first_view"]
    c["class_react"] = react
    c["followed_ai"] = bool(c["ai_shown"]) and choice == c["ai_label"]
    truth = c["truth"]

    if choice == truth and truth == "Drone":
        c["status"] = "CLASSIFIED"
        pts = 25 + max(0, int(15 - react))
        _award(M, c, pts)
        event(M, f"{c['cid']}: identified as DRONE ({react:.1f}s) (+{pts}). Assess, then decide.", "good")
    elif choice == truth:
        pts = 20 + max(0, int(10 - react))
        _resolve(M, c, "CLEARED", "OPTIMAL", pts, f"Bird confirmed ({react:.1f}s) - no threat", "good")
    elif truth == "Drone":
        c["misid"] = True
        c["status"] = "CLASSIFIED"
        _award(M, c, -50)
        event(M, f"{c['cid']}: you called it a BIRD - it is a DRONE (-50). Cross-cue confirms; engage now!", "bad")
    else:
        _resolve(M, c, "FALSE ALARM", "SUBOPTIMAL", -20,
                 "False alarm - it was a bird. Time and attention wasted", "warn")


def do_assess(M, frame, c, kind, now):
    if c["status"] != "CLASSIFIED":
        return
    touch(c, now)
    if kind == "TRACK" and not c["tracked"]:
        c["tracked"] = True
    elif kind == "REGISTRY" and c["registry"] is None:
        rng = random.Random(f"{M['seed']}|{frame['id']}|{c['cid']}|registry")
        if c["intent"] == "FRIENDLY" and rng.random() < 0.9:
            c["registry"] = "REGISTERED"
        else:
            c["registry"] = "NOT REGISTERED"      # hostile, or a friendly with a transponder fault
    else:
        return
    if c["deadline"]:
        c["deadline"] -= 3                         # assessing costs time
    if kind == "TRACK":
        event(M, f"{c['cid']}: track - {c['heading']}, ~{c['speed']} m/s.", "info")
    else:
        event(M, f"{c['cid']}: registry check - {c['registry']}.", "info")


def action_gate(M, frame, c, action):
    """(allowed, reason). This is the fire-control interlock."""
    if c["status"] != "CLASSIFIED":
        return False, "Classify the contact as a drone first"
    if action == "NET" and M["inv"]["NET"] <= 0:
        return False, "No net rounds left"
    if action == "KIN":
        if M["inv"]["KIN"] <= 0:
            return False, "No rounds left"
        if M["roe"] == "Weapons Hold":
            return False, "Weapons Hold: kinetic engagement prohibited"
        if frame["civilians"]:
            return False, "Civilians in the fire zone"
        if M["roe"] == "Weapons Tight" and c["registry"] != "NOT REGISTERED":
            return False, "Weapons Tight: needs a registry check showing NOT REGISTERED"
    return True, ""


def do_respond(M, frame, c, action, now):
    ok, why = action_gate(M, frame, c, action)
    if not ok:
        event(M, f"{c['cid']}: {ACTIONS[action]['label']} blocked - {why}", "warn")
        return
    touch(c, now)
    intent, band, cid = c["intent"], c["band"], c["cid"]
    c["action"] = ACTIONS[action]["label"]
    remaining = (c["deadline"] - now) if c["deadline"] else None

    if action in ("NET", "KIN"):
        M["inv"][action] -= 1
    if action == "ECM" and frame["civilians"]:
        _award(M, c, -10)
        event(M, f"{cid}: ECM is disrupting civilian comms nearby (-10)", "warn")

    # ---------------- non-engagement responses ----------------
    if action == "STAND":
        if intent == "FRIENDLY":
            verified = c["registry"] == "REGISTERED"
            _resolve(M, c, "STOOD DOWN", "OPTIMAL" if verified else "ACCEPTABLE",
                     60 if verified else 25,
                     "Friendly drone cleared" + ("" if verified else " (unverified - run a registry check next time)"),
                     "good")
        elif intent == "ATTACK":
            _resolve(M, c, "IMPACT", "FAILED", -120,
                     "You stood down a hostile attack drone - it hit the asset", "bad", ASSET_DAMAGE)
        else:
            _resolve(M, c, "INTEL COMPROMISED", "FAILED", -70,
                     "You stood down a hostile recon drone - it kept collecting", "bad")
        return

    if action == "REPORT":
        if intent == "FRIENDLY":
            _resolve(M, c, "MONITORED", "ACCEPTABLE", 15, "Friendly drone logged and monitored", "info")
        elif intent == "RECON":
            _resolve(M, c, "MONITORED", "SUBOPTIMAL", 20,
                     "Recon drone reported but still overhead - it keeps collecting", "warn")
        else:
            _resolve(M, c, "IMPACT", "FAILED", -30,
                     "Reporting alone did not stop the attack drone - impact", "bad", ASSET_DAMAGE)
        return

    if action == "ALERT":
        if intent == "FRIENDLY":
            _resolve(M, c, "FALSE ALERT", "SUBOPTIMAL", -10, "Base alerted over a friendly drone", "warn")
        elif intent == "RECON":
            _resolve(M, c, "MONITORED", "SUBOPTIMAL", 10,
                     "Base alerted; recon drone still collecting", "warn")
        else:
            _resolve(M, c, "IMPACT MITIGATED", "SUBOPTIMAL", 20,
                     "Personnel sheltered - impact damage halved", "warn", ASSET_DAMAGE // 2)
        return

    # ---------------- effectors (ECM / NET / KIN) ----------------
    label = ACTIONS[action]["label"]
    if intent == "FRIENDLY":
        pts = -150 if action == "KIN" else -80
        _resolve(M, c, "BLUE-ON-BLUE", "VIOLATION", pts,
                 f"{label} used on a registered friendly drone - blue-on-blue", "bad")
        return

    rng = random.Random(f"{M['seed']}|{frame['id']}|{cid}|{action}|{c['attempts']}")
    if rng.random() < SUCCESS[action][band]:
        base = BASE_POINTS[intent][action]
        mult = 1.0 if c["attempts"] == 0 else 0.6
        bonus = 0
        if intent == "ATTACK" and remaining is not None and c["limit"]:
            bonus = int(20 * max(0.0, remaining) / c["limit"])
        pts = int(base * mult) + bonus
        verdict = "OPTIMAL" if base >= 60 else "ACCEPTABLE"
        extra = " (disproportionate force for a recon drone)" if (intent == "RECON" and action == "KIN") else ""
        again = " on re-engagement" if c["attempts"] else ""
        _resolve(M, c, "NEUTRALISED", verdict, pts, f"{label}: drone neutralised{again}{extra}", "good")
    else:
        c["attempts"] += 1
        _award(M, c, -5)
        if c["deadline"]:
            c["deadline"] -= 4
        event(M, f"{cid}: {label} FAILED at {band} range (-5). Re-engage or change method.", "warn")


def expire_overdue(M, frame, now) -> bool:
    """Resolve contacts whose clock ran out. Returns True if anything changed."""
    changed = False
    for c in frame["contacts"]:
        if c["status"] == "RESOLVED" or not c["deadline"] or now < c["deadline"]:
            continue
        changed = True
        c["action"] = "NO DECISION"
        if c["intent"] == "ATTACK":
            _resolve(M, c, "IMPACT", "FAILED", -80,
                     "No decision in time - attack drone hit the asset", "bad", ASSET_DAMAGE)
        elif c["intent"] == "RECON":
            _resolve(M, c, "INTEL COMPROMISED", "FAILED", -40,
                     "No decision in time - recon drone collected intelligence", "bad")
        else:
            _resolve(M, c, "ARRIVED SAFELY", "-", 0, "Friendly drone recovered safely", "info")
    return changed


# --------------------------------------------------------------------------
# After-action review
# --------------------------------------------------------------------------
def summarize(M):
    rows = []
    for f in M["frames"][: M["fi"] + 1]:
        for c in f["contacts"]:
            ai_wrong = c["ai_shown"] and c["ai_label"] != c["truth"]
            ai_right = c["ai_shown"] and c["ai_label"] == c["truth"]
            rows.append({
                "Contact": f"F{f['id'] + 1}-{c['cid']}",
                "Truth": c["truth"],
                "Your ID": c["soldier_class"] or "-",
                "AI said": (f"{c['ai_label']} {c['ai_conf']:.0%}" if c["ai_shown"] else "not asked"),
                "AI correct": ("-" if not c["ai_shown"] else ("yes" if ai_right else "NO")),
                "Followed AI": ("-" if not c["ai_shown"] else ("yes" if c["followed_ai"] else "no")),
                "Hidden intent": c["intent"] or "-",
                "Heading": (c["heading"] if c["tracked"] else "not tracked"),
                "Registry": c["registry"] or "not checked",
                "Action": c["action"] or "-",
                "Attempts": c["attempts"] + (1 if c["outcome"] in ("NEUTRALISED",) else 0),
                "Outcome": c["outcome"] or "UNRESOLVED",
                "Verdict": c["verdict"] or "-",
                "Points": c["points"],
                "ID time (s)": round(c["class_react"], 1) if c["class_react"] is not None else None,
                "_ai_wrong": bool(ai_wrong), "_ai_right": bool(ai_right),
                "_followed": c["followed_ai"], "_misid": c["misid"],
                "_class_ok": (c["soldier_class"] == c["truth"]) if c["soldier_class"] else None,
                "_hostile": c["intent"] in ("ATTACK", "RECON"),
            })

    classified = [r for r in rows if r["_class_ok"] is not None]
    hostile = [r for r in rows if r["_hostile"]]
    outc = [r["Outcome"] for r in rows]
    times = [r["ID time (s)"] for r in rows if r["ID time (s)"] is not None]

    m = dict(
        score=M["score"], asset=M["asset"],
        contacts=len(rows), classified=len(classified),
        id_acc=(sum(1 for r in classified if r["_class_ok"]) / len(classified)) if classified else 0.0,
        hostile=len(hostile),
        neutralised=sum(1 for r in hostile if r["Outcome"] == "NEUTRALISED"),
        impacts=sum(1 for o in outc if o.startswith("IMPACT")),
        intel=outc.count("INTEL COMPROMISED"),
        fratricide=outc.count("BLUE-ON-BLUE"),
        false_alarms=outc.count("FALSE ALARM"),
        missed_ids=sum(1 for r in rows if r["_misid"]),
        avg_react=(sum(times) / len(times)) if times else 0.0,
        ai_asked=sum(1 for r in rows if r["AI correct"] != "-"),
        ai_wrong=sum(1 for r in rows if r["_ai_wrong"]),
        followed_wrong=sum(1 for r in rows if r["_ai_wrong"] and r["_followed"]),
        overrode_right=sum(1 for r in rows if r["_ai_right"] and not r["_followed"]),
        unresolved=outc.count("UNRESOLVED"),
        net_left=M["inv"]["NET"], kin_left=M["inv"]["KIN"],
    )
    neut_rate = (m["neutralised"] / m["hostile"]) if m["hostile"] else 1.0
    g = 0.3 * m["id_acc"] + 0.3 * neut_rate + 0.3 * (m["asset"] / 100) + 0.1 * (1 if not m["fratricide"] else 0)
    g -= 0.15 * m["fratricide"]
    m["index"] = max(0.0, min(1.0, g))
    m["grade"] = ("A - Mission ready" if g >= 0.85 else
                  "B - Proficient" if g >= 0.70 else
                  "C - Needs more practice" if g >= 0.50 else
                  "D - Not ready for live operations")

    lessons = []
    if m["fratricide"]:
        lessons.append("🔴 You engaged a registered friendly drone. Always run the registry check before using any effector - and remember a friendly can fail to respond.")
    if m["missed_ids"]:
        lessons.append(f"🔴 You called a drone a bird {m['missed_ids']} time(s). Zoom in and look for rotor arms and a symmetric body before clearing a contact.")
    if m["impacts"]:
        lessons.append(f"🟠 {m['impacts']} drone(s) reached the asset. Hostile attack drones need an effector, not just a report - and every assessment step costs seconds.")
    if m["intel"]:
        lessons.append(f"🟠 {m['intel']} recon drone(s) were left collecting. Orbiting drones with no registration are still hostile.")
    if m["followed_wrong"]:
        lessons.append(f"🟠 You followed the AI when it was wrong {m['followed_wrong']} time(s). Treat its confidence as a hint, not proof.")
    if m["overrode_right"]:
        lessons.append(f"🟡 You overrode the AI when it was right {m['overrode_right']} time(s). Check what made you doubt it.")
    if m["false_alarms"]:
        lessons.append(f"🟡 {m['false_alarms']} false alarm(s) on birds. Wing flapping and an irregular outline point to a bird.")
    if not lessons:
        lessons.append("🟢 Clean run: correct identifications, proportionate responses, no friendly losses.")
    return dict(rows=rows, m=m, lessons=lessons)
