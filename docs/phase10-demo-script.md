# Demo Script (Phase 10.5)

A rehearsed, repeatable sequence for the live demonstration (Review 2
Requirement 6). Everything here is local — no network dependency, no
artificial load, deterministic every run (`docs/architecture.md`,
`docs/plans/2026-09-12-presentable-predictive-monitor.md`).

**Before the room:** stop the standing background collector if one is
running (`ps aux | grep "aiperfopt collect"`, then `kill <pid>`) — a real
collector writing to the primary database will make `aiperfopt replay`
refuse to run (`ReplayError`, by design: replay and a live collector must
never write to the same database concurrently). Record a session once,
ahead of time, from a period that actually contains a pressure event
(e.g. a build, a large `git clone`, opening several heavy apps):

```bash
aiperfopt record --minutes 15 --out demo-session.json
```

Keep `demo-session.json` under version control or at least backed up —
it *is* the deterministic fixture the whole demo depends on.

## Sequence

**1. `doctor` — environment and readiness, cold.**
```bash
aiperfopt doctor
```
Shows the resolved paths, the active model version, and the current
`/readiness` state with no server running at all — the same function
(`aiperfopt.readiness.assess_readiness`) the dashboard banner calls.

**2. Show readiness in the UI.**
```bash
aiperfopt serve --no-monitor &
python3 -m aiperfopt.frontend.main
```
(`--no-monitor`: the server must not also poll this real machine into the
replay database it's about to serve — see `docs/architecture.md`'s notes
on why replay and the bundled collector can never share a database.)
With nothing replayed yet, the dashboard banner should read a real state
derived from whatever the configured database currently holds — **not**
a hardcoded green "Live" (this is Phase 9's automated regression test,
`tests/frontend/test_main_window.py::test_start_polls_readiness_and_updates_the_dashboard_banner`).

**3. Start the replay.**
```bash
AIPERFOPT_STORAGE__DATABASE_PATH="$HOME/Library/Application Support/aiperfopt/data/aiperfopt-replay.db" \
AIPERFOPT_STORAGE__IS_REPLAY=true \
aiperfopt replay --session demo-session.json \
  --replay-db "$HOME/Library/Application Support/aiperfopt/data/aiperfopt-replay.db" \
  --speed 10
```
The dashboard banner switches to the distinct purple **REPLAY** state
immediately — an unmistakable indicator that what follows is replayed,
not live, data (`readiness.py`'s `is_replay` override, unconditional even
over `NO_DATA`).

**4. Watch the forecast lead the actual.** As the recorded pressure event
plays back, the live chart's dashed forecast segment should visibly rise
*before* the solid actual CPU/memory line catches up to it — the
project's single flagship visual claim.

**5. Watch the contention probability cross its threshold ahead of the
spike.** The contention strip's rolling probability line should cross its
marked decision threshold before the actual pressure event lands.

**6. Diagnosis resolves to an actionable category.** When contention
predicts positive, the dashboard's diagnosis panel should resolve to one
of the seven actionable categories (`DiagnosisCategory`, `taxonomy.py`) —
not the informational-only `UNUSUAL_PATTERN_ANOMALY` catch-all, which
would indicate the anomaly detector is over-firing (Bug F1/F2, fixed in
an earlier phase).

**7. A recommendation appears** — the category's curated actions from
`config/recommendations.yaml`, rendered in the recommendation panel.

**8. Open Predicted vs. Actual.** Switch tabs; the same event should show
the model's forecast plotted against the actual series with the
prediction visibly leading the event, confirming the live chart's story
against the persisted `forecast_predictions` history rather than only a
rolling in-memory buffer.

**9. `aiperfopt evaluate --report`.** In a terminal, regenerate the full
evidence pack live:
```bash
aiperfopt evaluate --ablation --report --out-dir results
```
Walks the room through the printed ablation summary (system-only vs.
system+process features), then points at `results/` for the PR curve,
confusion matrix, SHAP feature-importance plot, and overhead measurement
it just wrote.

**10. Clean test run.**
```bash
pytest --ignore=tests/frontend -q && QT_QPA_PLATFORM=offscreen pytest tests/frontend -q
```

## Rehearsal log

Record each full run-through here (date, who ran it, anything that
didn't go as scripted) — the plan's manual success criterion is "the demo
has been rehearsed end-to-end at least twice from a cold start."

| Date | Run by | Cold start? | Notes |
|---|---|---|---|
| 2026-09-13 | rrsd2769 | No — hit and fixed two bugs mid-run (`copy_model_metadata` missing, `serve` needed `--no-monitor`) | REPLAY badge confirmed purple; dashed CPU forecast segment visibly rose before the solid actual line's bump. Contention strip had nothing to show — `phase7-real-deploy` has no trained classifier (see `docs/phase10-limitations.md`), not a bug in this run. |
| | | | |
