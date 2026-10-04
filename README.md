# 🛡️ AI-Assisted Counter-Drone Decision Trainer

Real sky footage, real labels. The AI spots aerial contacts - **the soldier decides** what each one is
and what to do about it, under rules of engagement, time pressure and limited effectors.

## Run it in VS Code

```bash
# 1. open this folder in VS Code, then open a terminal (Ctrl+`)
python -m venv .venv
.venv\Scripts\activate          # Windows      (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

The browser opens at http://localhost:8501. Needs Python 3.9+ and Streamlit 1.37+ (live countdowns use `st.fragment`).

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit UI: mission console, dataset tab, doctrine tab, after-action review |
| `engine.py` | Dataset loader, scenario generator, ROE interlock, scoring, debrief (no UI code, unit-testable) |
| `viz.py` | Annotated sensor feed and full-resolution optic zoom |
| `Sample_Image/` | Your dataset: images + YOLO `.txt` labels (class 0 = drone, 1 = bird) |

## The soldier's loop

1. **Detect** - the AI cues every contact on the 4K frame.
2. **Identify** - zoom into a full-resolution crop; call it Drone or Bird. Optionally ask the AI (it can be confidently wrong).
3. **Assess** - *Track* (heading, speed, ETA) and *Registry / Remote-ID check*. Each costs seconds on the clock.
4. **Respond** - report, alert & shelter, stand down, electronic countermeasure, net interceptor, or kinetic.
   Kinetic is gated by ROE (Hold / Tight / Free), civilians in the fire zone, and limited rounds.
5. **Debrief** - readiness grade, coaching points, AI-trust analysis, CSV + JSON export.

Mistakes teach: engaging a friendly drone, calling a drone a bird, following a wrong AI, or leaving a recon drone
overhead all cost points, and attack drones that reach the asset damage it. Lose the asset and the mission fails.

## What is real and what is simulated

| Real (from your data) | Simulated (so there are decisions to make) |
|---|---|
| Images, boxes, drone/bird labels | Intent (attack / recon / friendly), heading, speed |
| Apparent target size → range band (data-driven tertiles) | Registry result, civilians, ETA clocks |
| AI confidence scales with real target size | AI opinion - **unless you add trained weights (below)** |

## Make the AI real (recommended before demoing)

Your 10 images are a sample. With the full dataset, train a detector and drop the weights in:

```bash
pip install ultralytics
# data.yaml:  train/val paths, names: {0: drone, 1: bird}
yolo detect train data=data.yaml model=yolov8n.pt imgsz=1920 epochs=50
# copy runs/detect/train/weights/best.pt  ->  best.pt  (next to app.py)
```

Restart the app: the sidebar switches to **AI engine: YOLO** and each contact's AI panel shows the model's own
prediction. Targets are tiny (about 100 px in a 3840 px frame), so train at high `imgsz` or use tiled inference.
This path is wired up but untested here, because there are no trained weights yet.

To use a different dataset, change the path in the sidebar (any folder of images with same-named YOLO `.txt` files).

## 3-minute demo script

1. Dataset tab: show real frames and the target-size histogram - "tiny targets, hard for humans alone".
2. Start a Normal mission on *Weapons Tight*. Zoom into a contact, **don't** ask the AI, identify it.
3. Next scenario: ask the AI, show it being confidently wrong, then correct it.
4. Track + registry check a drone; show the ETA clock, then neutralise with ECM.
5. Try Kinetic on an unchecked drone - show the interlock blocking it. Then run the registry check and engage.
6. Finish: debrief with coaching points and the "AI wrong & followed" metric.

## Configuration worth knowing

- Difficulty changes timers, AI reliability, civilians, effector stock and threat mix (`DIFFICULTY` in `engine.py`).
- Untick **Time pressure** for a no-clock training mode.
- Use a fixed **Scenario seed** to replay the identical mission for every trainee (fair scoring).
- All effector odds, points and ROE rules live at the top of `engine.py`.

## Scope note

This is a training and decision-support prototype with abstract effectors. It contains no real countermeasure
parameters and is not a fire-control system.
