import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_event(path, event):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def normalize_command(command):
    parts = shlex.split(command)
    if parts and parts[0] in {"python", "python3"}:
        parts[0] = sys.executable
    return parts


def read_queue(queue_path):
    queue = load_json(queue_path)
    if not queue:
        raise RuntimeError(f"Missing queue file: {queue_path}")
    if not isinstance(queue.get("steps"), list):
        raise RuntimeError(f"Invalid queue format: {queue_path}")
    return queue


def resolve_step_paths(project_root, step):
    exp_id = step["experiment_id"]
    artifacts = step.get("artifacts") or {}
    run1_dir = project_root / artifacts.get("run1_dir", f"outputs/{exp_id}_run1")
    run2_dir = project_root / artifacts.get("run2_dir", f"outputs/{exp_id}_run2")
    evaluation_path = project_root / artifacts.get(
        "evaluation_path",
        f"experiment_sync/evaluations/{exp_id}.json",
    )
    log_path = project_root / "experiment_sync" / "watchdog_runs" / f"{step['step_id']}.log"
    return {
        "run1_dir": run1_dir,
        "run2_dir": run2_dir,
        "evaluation_path": evaluation_path,
        "log_path": log_path,
    }


def find_step(queue, step_id):
    if not step_id:
        return None
    for step in queue["steps"]:
        if step.get("step_id") == step_id and step.get("approved", False):
            return step
    return None


def choose_current_step(queue, project_root):
    approved_steps = [step for step in queue["steps"] if step.get("approved", False)]
    if not approved_steps:
        return None

    step = approved_steps[0]
    visited = set()
    while step and step["step_id"] not in visited:
        visited.add(step["step_id"])
        paths = resolve_step_paths(project_root, step)
        evaluation = read_evaluation(paths)
        if not evaluation or evaluation.get("decision") == "pass_final_target":
            return step
        next_step = find_step(queue, step.get("next_step_on_fail"))
        if not next_step:
            return step
        step = next_step
    return step


def find_active_training_processes(project_root):
    user = os.environ.get("USER")
    command = ["ps", "-eo", "pid=,args="]
    if user:
        command = ["ps", "-u", user, "-o", "pid=,args="]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    active = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_text, args = line.split(" ", 1)
        if int(pid_text) == os.getpid():
            continue
        if "remote_training_watchdog.py" in args:
            continue
        if "run_confirm.py" in args or "src.train" in args or "manage_cri.py" in args:
            active.append({"pid": int(pid_text), "args": args})
    return active


def latest_checkpoint_exists(run_dir):
    return (run_dir / "latest.pth").exists()


def best_checkpoint_exists(run_dir):
    return (run_dir / "best_model.pth").exists()


def step_has_resume_candidate(paths):
    for run_dir in (paths["run1_dir"], paths["run2_dir"]):
        if latest_checkpoint_exists(run_dir) and not best_checkpoint_exists(run_dir):
            return True
    return False


def step_runs_complete(paths):
    return best_checkpoint_exists(paths["run1_dir"]) and best_checkpoint_exists(paths["run2_dir"])


def read_evaluation(paths):
    return load_json(paths["evaluation_path"])


def write_state(state_path, current_state, update):
    next_state = dict(current_state or {})
    next_state.update(update)
    next_state["last_checked_at"] = utc_now()
    if next_state != current_state:
        save_json(state_path, next_state)
    return next_state


def build_event(state, step, message, extra=None):
    payload = {
        "timestamp": utc_now(),
        "state": state,
        "message": message,
        "step_id": step.get("step_id") if step else None,
        "experiment_id": step.get("experiment_id") if step else None,
    }
    if extra:
        payload.update(extra)
    return payload


def launch_background_command(project_root, command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        normalize_command(command),
        cwd=project_root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    return process.pid


def run_evaluation(project_root, step, paths):
    subprocess.run(
        normalize_command(step["evaluation_command"]),
        cwd=project_root,
        check=True,
    )
    evaluation = read_evaluation(paths)
    if not evaluation:
        raise RuntimeError(f"Evaluation file was not created: {paths['evaluation_path']}")
    return evaluation


def main():
    project_root = Path(__file__).resolve().parents[1]
    sync_root = project_root / "experiment_sync"
    queue_path = sync_root / "training_queue.json"
    state_path = sync_root / "watchdog_state.json"
    events_path = sync_root / "watchdog_events.jsonl"
    (sync_root / "evaluations").mkdir(parents=True, exist_ok=True)
    (sync_root / "watchdog_runs").mkdir(parents=True, exist_ok=True)

    current_state = load_json(state_path, {})

    try:
        queue = read_queue(queue_path)
    except Exception as exc:
        write_state(
            state_path,
            current_state,
            {
                "state": "critical_blocker",
                "current_step_id": None,
                "active_experiment_id": None,
                "last_action": "queue_read_failed",
                "last_decision": "critical_blocker",
                "last_evaluation_path": None,
                "blocker": str(exc),
            },
        )
        append_event(events_path, build_event("critical_blocker", None, str(exc)))
        raise

    step = choose_current_step(queue, project_root)
    if not step:
        message = "No approved training step is available in training_queue.json."
        write_state(
            state_path,
            current_state,
            {
                "state": "awaiting_local_codex",
                "current_step_id": None,
                "active_experiment_id": None,
                "last_action": "queue_empty",
                "last_decision": "awaiting_local_codex",
                "last_evaluation_path": None,
                "blocker": message,
            },
        )
        append_event(events_path, build_event("awaiting_local_codex", None, message))
        return

    paths = resolve_step_paths(project_root, step)
    active_processes = find_active_training_processes(project_root)
    if active_processes:
        write_state(
            state_path,
            current_state,
            {
                "state": "running",
                "current_step_id": step["step_id"],
                "active_experiment_id": step["experiment_id"],
                "last_action": "detected_active_training",
                "last_decision": "running",
                "last_evaluation_path": str(paths["evaluation_path"]),
                "blocker": None,
            },
        )
        return

    if step_runs_complete(paths) and not paths["evaluation_path"].exists():
        evaluation = run_evaluation(project_root, step, paths)
        current_state = write_state(
            state_path,
            current_state,
            {
                "state": "evaluated",
                "current_step_id": step["step_id"],
                "active_experiment_id": step["experiment_id"],
                "last_action": "evaluated_current_step",
                "last_decision": evaluation["decision"],
                "last_evaluation_path": str(paths["evaluation_path"]),
                "blocker": None,
            },
        )
        append_event(
            events_path,
            build_event(
                "evaluated",
                step,
                evaluation["reason"],
                {
                    "evaluation_path": str(paths["evaluation_path"]),
                    "decision": evaluation["decision"],
                },
            ),
        )

    evaluation = read_evaluation(paths)
    if evaluation and evaluation.get("decision") == "pass_final_target":
        write_state(
            state_path,
            current_state,
            {
                "state": "success",
                "current_step_id": step["step_id"],
                "active_experiment_id": step["experiment_id"],
                "last_action": "final_target_met",
                "last_decision": evaluation["decision"],
                "last_evaluation_path": str(paths["evaluation_path"]),
                "blocker": None,
            },
        )
        append_event(
            events_path,
            build_event(
                "success",
                step,
                evaluation["reason"],
                {"evaluation_path": str(paths["evaluation_path"])},
            ),
        )
        return

    if evaluation and evaluation.get("decision") != "pass_final_target" and step_runs_complete(paths):
        next_step = find_step(queue, step.get("next_step_on_fail"))
        if next_step:
            next_paths = resolve_step_paths(project_root, next_step)
            pid = launch_background_command(project_root, next_step["command"], next_paths["log_path"])
            write_state(
                state_path,
                current_state,
                {
                    "state": "running",
                    "current_step_id": next_step["step_id"],
                    "active_experiment_id": next_step["experiment_id"],
                    "last_action": "started_next_approved_step",
                    "last_decision": evaluation["decision"],
                    "last_evaluation_path": str(paths["evaluation_path"]),
                    "blocker": None,
                },
            )
            append_event(
                events_path,
                build_event(
                    "running",
                    next_step,
                    "Current step missed final targets. Starting next approved step.",
                    {
                        "spawned_pid": pid,
                        "previous_step_id": step["step_id"],
                        "evaluation_path": str(paths["evaluation_path"]),
                    },
                ),
            )
            return

        message = (
            "Current step finished and evaluation did not reach final targets. "
            "No next approved step is available; waiting for local Codex."
        )
        write_state(
            state_path,
            current_state,
            {
                "state": "awaiting_local_codex",
                "current_step_id": step["step_id"],
                "active_experiment_id": step["experiment_id"],
                "last_action": "no_next_approved_step",
                "last_decision": "awaiting_local_codex",
                "last_evaluation_path": str(paths["evaluation_path"]),
                "blocker": message,
            },
        )
        append_event(
            events_path,
            build_event(
                "awaiting_local_codex",
                step,
                message,
                {"evaluation_path": str(paths["evaluation_path"])},
            ),
        )
        return

    resume = step_has_resume_candidate(paths)
    pid = launch_background_command(project_root, step["command"], paths["log_path"])
    action = "resumed_current_step" if resume else "started_current_step"
    write_state(
        state_path,
        current_state,
        {
            "state": "running",
            "current_step_id": step["step_id"],
            "active_experiment_id": step["experiment_id"],
            "last_action": action,
            "last_decision": "running",
            "last_evaluation_path": str(paths["evaluation_path"]),
            "blocker": None,
        },
    )
    append_event(
        events_path,
        build_event(
            "running",
            step,
            "Resuming current step from latest checkpoint." if resume else "Starting current approved step.",
            {
                "spawned_pid": pid,
                "log_path": str(paths["log_path"]),
            },
        ),
    )


if __name__ == "__main__":
    main()
