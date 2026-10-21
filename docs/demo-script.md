# Demo Script

A repeatable sequence for a live demonstration. Everything runs locally,
with no network dependency and no artificial load.

There are two ways to show PRISM working: **live** on the machine's real
usage, or **replayed** from a recorded session that is known to contain a
pressure event. Live is more convincing; replay is guaranteed to show a
warning on cue.

## Option A: live

```bash
aiperfopt doctor
```
Shows the resolved paths, the active model version and whether the system
can predict right now, with no server running.

```bash
aiperfopt run
```
Starts the server (collection plus the 5-second live-inference loop) and
opens the dashboard in the browser. Prediction needs about two minutes of
history after a cold start.

1. **Live view.** CPU and memory with the dashed forecast running ahead of
   the actual line, the contention risk against its threshold, and the
   top processes.
2. **Create real memory pressure** by using the machine normally: open
   many browser tabs or several heavy apps. Do not run a synthetic load
   generator.
3. **Warning.** When the risk crosses the threshold, "What's happening"
   shows the diagnosed cause in plain language and three recommended
   actions. If the workload is unusual, a banner says prediction
   confidence has been reduced.
4. **Act on the advice** (for example, close the tabs) and watch memory
   fall and the risk return to 0 % with "All clear".
5. **Accuracy view.** The same period's forecasts plotted against what
   actually happened.
6. **Compare view.** Click **Generate comparison graph** to show the
   LightGBM vs Random Forest vs Ridge results the model was chosen from.
7. **Model view.** The active version and the deployment gate.

## Option B: replay a recorded event

Record once, ahead of time, from a period that contains a pressure event:

```bash
aiperfopt record --minutes 15 --out demo-session.json
```

Stop any running collector first: replay refuses to run while a live
collector writes to the primary database. Then, in one terminal:

```bash
AIPERFOPT_STORAGE__DATABASE_PATH="$HOME/Library/Application Support/aiperfopt/data/aiperfopt-replay.db" \
AIPERFOPT_STORAGE__IS_REPLAY=true \
aiperfopt serve --no-monitor
```

and in another:

```bash
aiperfopt replay --session demo-session.json \
  --replay-db "$HOME/Library/Application Support/aiperfopt/data/aiperfopt-replay.db" \
  --speed 10
```

Open `http://127.0.0.1:8000/`. The status bar shows a purple **REPLAY**
badge, so the room knows the data is replayed. Then follow steps 1 and 3–7
of Option A.

## Afterwards

```bash
aiperfopt compare        # re-run the three-model comparison (several minutes)
python -m pytest -q      # Python tests
node --test src/aiperfopt/web/static/js/   # dashboard JS tests
```

## Rehearsal log

| Date | Run by | Mode | Notes |
|---|---|---|---|
| 2026-09-13 | rrsd2769 | Replay, Qt UI | REPLAY badge confirmed; forecast rose before the actual CPU line. No classifier was deployed yet. |
| 2026-10-20 | rrsd2769 | Live, web dashboard | Risk reached 70 % (threshold 50 %) with memory at 82 %; diagnosis "memory climbing steadily", advice to close tabs. After closing them memory fell to 63 % and risk to 0 %. |
