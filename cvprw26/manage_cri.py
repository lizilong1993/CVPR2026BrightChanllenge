import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


CLASS_TARGETS = {
    "intact": 0.4000,
    "damaged": 0.2000,
    "destroyed": 0.2500,
}
DEFAULT_FINAL_MAP_THRESHOLD = 0.3500
DEFAULT_FINAL_CRI_THRESHOLD = 100.0


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def extract_metric(text, labels):
    for label in labels:
        match = re.search(rf"{re.escape(label)}\s*=\s*([0-9.]+)", text)
        if match:
            return float(match.group(1))
    return None


def parse_summary_metrics(summary_path):
    if not summary_path.exists():
        return None

    text = summary_path.read_text(encoding="utf-8")
    metrics = {
        "mAP": extract_metric(text, ["Best segm_AP", "segm_AP"]),
        "AP50": extract_metric(text, ["segm_AP50", "AP50"]),
        "AP75": extract_metric(text, ["segm_AP75", "AP75"]),
        "intact": extract_metric(text, ["intact"]),
        "damaged": extract_metric(text, ["damaged"]),
        "destroyed": extract_metric(text, ["destroyed"]),
    }
    if metrics["mAP"] is None:
        return None
    return metrics


def parse_checkpoint_metrics(run_dir):
    checkpoint_path = run_dir / "best_model.pth"
    if not checkpoint_path.exists():
        return None

    try:
        import torch
    except Exception:
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None

    metric_blob = checkpoint.get("metrics") or {}
    return {
        "mAP": float(metric_blob.get("segm_AP", 0.0)),
        "AP50": float(metric_blob.get("segm_AP50", 0.0)),
        "AP75": float(metric_blob.get("segm_AP75", 0.0)),
        "intact": float(metric_blob.get("segm_AP_intact", 0.0)),
        "damaged": float(metric_blob.get("segm_AP_damaged", 0.0)),
        "destroyed": float(metric_blob.get("segm_AP_destroyed", 0.0)),
    }


def collect_run_metrics(project_root, exp_id, run_index):
    output_dir = project_root / "outputs" / f"{exp_id}_run{run_index}"
    sync_dir = project_root / "experiment_sync" / "runs" / f"{exp_id}_run{run_index}"
    summary_path = sync_dir / "latest_summary.md"

    metrics = parse_checkpoint_metrics(output_dir)
    source = "output_checkpoint" if metrics else None

    if not metrics:
        metrics = parse_summary_metrics(summary_path)
        if metrics:
            source = "experiment_sync_summary"

    run_exists = metrics is not None
    latest_checkpoint = output_dir / "latest.pth"
    best_checkpoint = output_dir / "best_model.pth"

    return {
        "run_index": run_index,
        "run_exists": run_exists,
        "output_dir": str(output_dir),
        "sync_dir": str(sync_dir),
        "metrics_source": source,
        "resume_checkpoint_exists": latest_checkpoint.exists(),
        "best_checkpoint_exists": best_checkpoint.exists(),
        "metrics": metrics,
    }


def average_metrics(run_metrics):
    metric_keys = ("mAP", "AP50", "AP75", "intact", "damaged", "destroyed")
    available = [item["metrics"] for item in run_metrics if item["metrics"]]
    if not available:
        return None

    return {
        key: sum(metrics[key] for metrics in available) / len(available)
        for key in metric_keys
    }


def calculate_cri(map_confirm, averaged_metrics, reproducibility=1.0):
    term1 = 0.70 * (map_confirm / DEFAULT_FINAL_MAP_THRESHOLD)
    ratios = [
        averaged_metrics["intact"] / CLASS_TARGETS["intact"],
        averaged_metrics["damaged"] / CLASS_TARGETS["damaged"],
        averaged_metrics["destroyed"] / CLASS_TARGETS["destroyed"],
    ]
    term2 = 0.20 * min(ratios)
    term3 = 0.10 * reproducibility
    return 100.0 * (term1 + term2 + term3)


def build_evaluation(project_root, exp_id, final_map_threshold, final_cri_threshold):
    run1 = collect_run_metrics(project_root, exp_id, 1)
    run2 = collect_run_metrics(project_root, exp_id, 2)
    averaged_metrics = average_metrics([run1, run2])

    map_confirm = None
    class_ap_average = None
    cri = None
    meets_final_target = False
    decision = "continue_training"
    reason = ""

    if averaged_metrics:
        class_ap_average = (
            averaged_metrics["intact"] + averaged_metrics["damaged"] + averaged_metrics["destroyed"]
        ) / 3.0

    if run1["run_exists"] and run2["run_exists"]:
        map_confirm = (run1["metrics"]["mAP"] + run2["metrics"]["mAP"]) / 2.0
        cri = calculate_cri(map_confirm, averaged_metrics, reproducibility=1.0)
        meets_final_target = map_confirm >= final_map_threshold and cri >= final_cri_threshold
        if meets_final_target:
            decision = "pass_final_target"
            reason = (
                f"mAP_confirm={map_confirm:.4f} and CRI={cri:.2f} meet final targets "
                f"(mAP_confirm>={final_map_threshold:.4f}, CRI>={final_cri_threshold:.1f})."
            )
        else:
            reason = (
                f"Two-run confirmation is available, but mAP_confirm={map_confirm:.4f} and "
                f"CRI={cri:.2f} do not meet final targets "
                f"(mAP_confirm>={final_map_threshold:.4f}, CRI>={final_cri_threshold:.1f})."
            )
    elif run1["run_exists"] and not run2["run_exists"]:
        reason = "run1 exists but run2 is missing; continue the approved confirmation training."
    elif not run1["run_exists"] and run2["run_exists"]:
        reason = "run2 exists but run1 is missing; the confirmation pair is incomplete."
    else:
        reason = "Neither confirmed run is complete yet."

    return {
        "experiment_id": exp_id,
        "generated_at": utc_now(),
        "project_root": str(project_root),
        "run1_exists": run1["run_exists"],
        "run2_exists": run2["run_exists"],
        "mAP_confirm": map_confirm,
        "class_ap_average": class_ap_average,
        "CRI": cri,
        "meets_final_target": meets_final_target,
        "final_target_thresholds": {
            "mAP_confirm": final_map_threshold,
            "CRI": final_cri_threshold,
        },
        "decision": decision,
        "reason": reason,
        "runs": {
            "run1": run1,
            "run2": run2,
        },
        "averaged_metrics": averaged_metrics,
    }


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate confirmation metrics and CRI.")
    parser.add_argument("exp_id", nargs="?", default="exp001")
    parser.add_argument("--json-out", dest="json_out", default=None)
    parser.add_argument("--project-root", dest="project_root", default=None)
    parser.add_argument(
        "--final-map-threshold",
        dest="final_map_threshold",
        type=float,
        default=DEFAULT_FINAL_MAP_THRESHOLD,
    )
    parser.add_argument(
        "--final-cri-threshold",
        dest="final_cri_threshold",
        type=float,
        default=DEFAULT_FINAL_CRI_THRESHOLD,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parent
    evaluation = build_evaluation(
        project_root=project_root,
        exp_id=args.exp_id,
        final_map_threshold=args.final_map_threshold,
        final_cri_threshold=args.final_cri_threshold,
    )

    if args.json_out:
        save_json(Path(args.json_out), evaluation)

    print(json.dumps(evaluation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
