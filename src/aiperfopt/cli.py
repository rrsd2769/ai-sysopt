"""The command-line entrypoint: ``aiperfopt <command>``.

Before this existed the project had no entrypoint at all -- no
``console_scripts``, no ``python -m aiperfopt``. Getting from a fresh
install to a working system required pasting a multi-line Python snippet
out of the README to train a model, because the live inference loop
returns early forever until one is deployed. That is the gap this module
closes.

Every subcommand is a thin wrapper over an existing subsystem entrypoint;
no orchestration logic lives here that is not about argument parsing and
user-facing output.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from aiperfopt.paths import HOME_ENV_VAR, app_home


def _bootstrap(args: argparse.Namespace):
    """Bootstrap with the CLI's log level applied.

    Report-style commands (``doctor``) default to WARNING so their output
    is readable; long-running commands keep INFO. Set via the environment
    rather than by mutating settings, so it flows through the same layered
    configuration as everything else instead of becoming a second,
    special-cased path.
    """
    import os

    level = getattr(args, "log_level", None)
    if level is not None and "AIPERFOPT_LOGGING__LEVEL" not in os.environ:
        os.environ["AIPERFOPT_LOGGING__LEVEL"] = level

    from aiperfopt.bootstrap import bootstrap

    return bootstrap(args.config)


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report environment, resolved paths and readiness. Exit non-zero if unusable."""
    import platform as _platform

    print("aiperfopt doctor")
    print("=" * 60)
    print(f"python           : {sys.version.split()[0]} ({sys.executable})")
    print(f"platform         : {_platform.system()} {_platform.release()} ({sys.platform})")
    print(f"{HOME_ENV_VAR:<17}: {app_home()}")

    # Everything below is diagnostic, not authoritative -- `doctor`'s whole
    # purpose is to explain what's wrong with an install, so it must never
    # itself end in an unhandled traceback. A previous version only guarded
    # the bootstrap call; a readiness-check failure on a real CI runner
    # (exit code 1, i.e. an uncaught exception -- `doctor` never *returns*
    # 1 itself) escaped uncaught because of that gap. Everything after
    # bootstrap is now inside the same guard.
    try:
        ctx = _bootstrap(args)

        s = ctx.settings
        print(f"database         : {s.storage.database_path}")
        print(f"models dir       : {s.forecasting.models_dir}")
        print(f"active version   : {s.retraining.active_version_file}")
        print(f"log dir          : {s.logging.log_dir}")

        from aiperfopt.features.config import FeatureEngineeringConfig
        from aiperfopt.readiness import assess_readiness

        result = assess_readiness(ctx.storage_engine, FeatureEngineeringConfig(), s.retraining)
        print("-" * 60)
        print(f"state            : {result.state.value}")
        print(f"model version    : {result.active_model_version or '(none deployed)'}")
        print(f"history          : {result.history_seconds:.0f}s of {result.history_seconds_required:.0f}s needed")
        print(f"\n{result.detail}")
    except Exception as exc:  # noqa: BLE001 - doctor reports failures, never raises them
        import traceback

        print(f"\nFAILED: {exc}")
        traceback.print_exc()
        return 2

    # "Not ready yet" is a normal, actionable state, not a broken install --
    # only the failure handled above is worth a non-zero exit, so that
    # `aiperfopt doctor` in CI checks installability rather than whether
    # someone has collected data yet.
    return 0


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------


def _cmd_collect(args: argparse.Namespace) -> int:
    """Run monitoring + ingestion, with no API server.

    ``--forever`` is the mode the presentable-predictive-monitor plan's
    Phase 2 actually wants: passive collection running unattended across
    normal daily use for days, not a fixed lab-style window. A SIGTERM
    handler is installed so stopping a backgrounded (``nohup ... &``)
    process still flushes the pipeline's in-flight batch and stops the
    monitoring threads cleanly, the same as Ctrl-C does for a foreground
    run -- the default SIGTERM behavior would otherwise kill the process
    without giving IngestionConsumer a chance to flush.
    """
    import signal

    from aiperfopt.pipeline import create_monitoring_pipeline
    from aiperfopt.storage.reader import get_recent_system_metrics

    ctx = _bootstrap(args)
    pipeline = create_monitoring_pipeline(ctx)

    stop_requested = threading.Event()

    def _handle_sigterm(signum, frame) -> None:  # type: ignore[no-untyped-def]
        stop_requested.set()

    previous_handler = signal.signal(signal.SIGTERM, _handle_sigterm)

    if args.forever:
        print(f"Collecting indefinitely into {ctx.settings.storage.database_path} (Ctrl-C or SIGTERM to stop) ...")
    else:
        seconds = args.minutes * 60.0
        print(f"Collecting for {args.minutes:g} minute(s) into {ctx.settings.storage.database_path} ...")

    pipeline.start()
    started = time.time()
    try:
        if args.forever:
            while not stop_requested.is_set():
                time.sleep(0.5)
        else:
            while time.time() - started < seconds and not stop_requested.is_set():
                time.sleep(min(5.0, seconds - (time.time() - started)))
                elapsed = time.time() - started
                print(f"  {elapsed:6.0f}s / {seconds:.0f}s", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted - stopping cleanly.")
    finally:
        pipeline.stop()
        signal.signal(signal.SIGTERM, previous_handler)

    elapsed_total = time.time() - started
    if not args.forever:
        collected = get_recent_system_metrics(ctx.storage_engine, window_seconds=elapsed_total + 60)
        print(f"\nDone. {len(collected)} system samples collected over {elapsed_total:.0f}s.")
    else:
        print(f"\nStopped after {elapsed_total:.0f}s.")
    return 0


# --------------------------------------------------------------------------
# export-training-window
# --------------------------------------------------------------------------


def _cmd_export_training_window(args: argparse.Namespace) -> int:
    """Export system metrics + the durable process summary series for a window.

    Process-level detail (``raw_process_metrics``/``process_events``) is
    pruned after a few hours (Phase 2, NFR-4.3) so multi-day passive
    collection stays storage-bounded, but the *derived* process summary
    series survives that pruning in ``process_feature_summaries`` (see
    ``aiperfopt.features.storage_bridge.summarize_and_persist_process_features``,
    run automatically on every retention check). This command is the
    explicit, on-demand escape hatch for pulling a training window's
    worth of that already-durable data out to a portable file -- e.g.
    ahead of a training run on another machine, or as a backup.
    """
    import json
    from dataclasses import asdict

    from aiperfopt.storage.reader import get_process_feature_summaries_range, get_system_metrics_range

    ctx = _bootstrap(args)
    end = time.time()

    system_metrics = get_system_metrics_range(ctx.storage_engine, start=args.since, end=end)
    summaries = get_process_feature_summaries_range(ctx.storage_engine, start=args.since, end=end)

    payload = {
        "exported_at": end,
        "since": args.since,
        "system_metrics": [asdict(s) for s in system_metrics],
        "process_feature_summaries": [
            {"timestamp": s.timestamp, "computed_at": s.computed_at, **s.summary_values}
            for s in summaries
        ],
    }

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    print(
        f"Exported {len(system_metrics)} system metric row(s) and {len(summaries)} "
        f"process summary row(s) since {args.since:.0f} to {out_path}"
    )
    return 0


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> int:
    """Train a candidate model and, with --deploy, activate it."""
    from aiperfopt.common.exceptions import RetrainingError
    from aiperfopt.contention.config import ContentionLabelingConfig
    from aiperfopt.features.config import FeatureEngineeringConfig
    from aiperfopt.retraining.deployment import confirm_deployment
    from aiperfopt.retraining.pipeline import (
        run_retraining_on_real_usage_data,
        run_retraining_on_synthetic_data,
    )

    ctx = _bootstrap(args)
    feature_config = FeatureEngineeringConfig()
    labeling_config = ContentionLabelingConfig()
    targets = args.metrics

    print(f"Training model version {args.version!r} from {args.source} data ...")
    try:
        if args.source == "real":
            candidate = run_retraining_on_real_usage_data(
                ctx.storage_engine, feature_config, ctx.settings.forecasting, ctx.settings.contention,
                labeling_config, ctx.settings.retraining, new_model_version=args.version,
                target_metrics=targets,
            )
        else:
            candidate = run_retraining_on_synthetic_data(
                ctx.storage_engine, feature_config, ctx.settings.forecasting, ctx.settings.contention,
                labeling_config, ctx.settings.retraining, new_model_version=args.version,
                target_metrics=targets,
            )
    except RetrainingError as exc:
        print(f"Training failed: {exc}")
        return 1

    comparison = candidate.comparison
    print("-" * 60)
    print(f"recommendation   : {comparison.recommendation}")
    for reason in comparison.reasons:
        print(f"  - {reason}")
    for m in comparison.metric_comparisons:
        print(
            f"  {m.metric_name}/{m.horizon_label}: mae={m.candidate_mae} "
            f"directional={m.candidate_directional_accuracy} regressed={m.regressed}"
        )
    if comparison.contention_comparison is not None:
        print(f"  contention f1  : {comparison.contention_comparison.candidate_f1}")

    if not args.deploy:
        print("\nNot deployed (pass --deploy to activate this version).")
        return 0

    record = confirm_deployment(
        ctx.storage_engine, ctx.settings.retraining, candidate.model_version, comparison,
        confirmed=True, confirmed_by=args.confirmed_by,
    )
    print(f"\nDeployed {record.candidate_version!r} (previous: {record.previous_version!r}).")
    return 0


# --------------------------------------------------------------------------
# serve / ui / run
# --------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    from aiperfopt.api.server import main as serve_main

    serve_main()
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    from aiperfopt.frontend.main import main as ui_main

    ui_main()
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Start the backend, wait for it to answer, then open the UI.

    The backend runs in a child process rather than a thread: the Qt event
    loop wants the main thread, and a crashing UI must not take the
    monitoring pipeline down with it.
    """
    import subprocess

    import httpx

    ctx = _bootstrap(args)
    base_url = f"http://{ctx.settings.api.host}:{ctx.settings.api.port}"

    print(f"Starting backend on {base_url} ...")
    backend = subprocess.Popen([sys.executable, "-m", "aiperfopt.api.server"])
    try:
        deadline = time.time() + 30.0
        while time.time() < deadline:
            if backend.poll() is not None:
                print("Backend exited before becoming ready.")
                return 1
            try:
                httpx.get(f"{base_url}/health", timeout=0.5)
                break
            except httpx.TransportError:
                time.sleep(0.2)
        else:
            print("Backend did not become ready within 30s.")
            return 1

        print("Backend ready. Opening the UI ...")
        from aiperfopt.frontend.main import main as ui_main

        ui_main()
    except KeyboardInterrupt:
        pass
    finally:
        backend.terminate()
        try:
            backend.wait(timeout=10)
        except subprocess.TimeoutExpired:
            backend.kill()
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiperfopt",
        description="Predictive, explainable system health monitor (Windows and macOS).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional user override YAML file (highest-priority file layer).",
    )
    parser.add_argument(
        "--log-level", default=None,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Override the log level for this command.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="Report environment, paths and readiness.")
    p_doctor.set_defaults(func=_cmd_doctor, default_log_level="WARNING")

    p_collect = sub.add_parser("collect", help="Collect metrics for a fixed duration.")
    p_collect.add_argument("--minutes", type=float, default=15.0)
    p_collect.add_argument(
        "--forever", action="store_true",
        help="Run until interrupted (Ctrl-C or SIGTERM) instead of --minutes -- "
        "for the multi-day passive collection window.",
    )
    p_collect.set_defaults(func=_cmd_collect)

    p_export = sub.add_parser(
        "export-training-window",
        help="Export system metrics + the durable process summary series for a time window.",
    )
    p_export.add_argument(
        "--since", type=float, required=True,
        help="Unix epoch seconds; export everything from here to now.",
    )
    p_export.add_argument("--out", type=Path, required=True, help="Output JSON file path.")
    p_export.set_defaults(func=_cmd_export_training_window, default_log_level="WARNING")

    p_train = sub.add_parser("train", help="Train a candidate model, optionally deploying it.")
    p_train.add_argument("--source", choices=("synthetic", "real"), default="synthetic")
    p_train.add_argument("--version", default="v1", help="Model version label to create.")
    p_train.add_argument(
        "--metrics", nargs="*", default=["cpu_percent", "mem_percent"],
        help="Target metrics to train forecasters for.",
    )
    p_train.add_argument("--deploy", action="store_true", help="Activate the trained version.")
    p_train.add_argument("--confirmed-by", default="cli", dest="confirmed_by")
    p_train.set_defaults(func=_cmd_train, default_log_level="WARNING")

    p_serve = sub.add_parser("serve", help="Run the API server with monitoring and live inference.")
    p_serve.set_defaults(func=_cmd_serve)

    p_ui = sub.add_parser("ui", help="Open the desktop UI against a running backend.")
    p_ui.set_defaults(func=_cmd_ui)

    p_run = sub.add_parser("run", help="Start the backend and open the UI (the one command).")
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.log_level is None:
        args.log_level = getattr(args, "default_log_level", None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
