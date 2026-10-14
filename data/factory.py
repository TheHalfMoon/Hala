"""Deterministic original tasks; never import benchmark trajectories for training."""
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from evaluation.coordinates import normalize_box, denormalize_box

SEED = 20261014
REVISION = "hala-synthetic-v1"
FROZEN = Path(__file__).parent / "frozen" / "tasks.json"
FROZEN_SOURCE_SHA256 = "bdde4ea5326b391172435e29cee77701135c8bfa73805856ffa5dab538bbd35e"
WORKFLOWS = ("save", "open", "archive", "export", "toggle", "form", "filter", "drag_drop", "recovery", "scroll_confirm", "rename", "delete_confirm")
RIGHTS = {"source": "original_synthetic", "generator_revision": REVISION, "license": "CC0-1.0", "commercial_training_allowed": True, "contains_personal_data": False, "benchmark_evaluation": False}
LABELS = {
    "save": ("Save", "حفظ"), "open": ("Open", "فتح"), "archive": ("Archive", "أرشفة"),
    "export": ("Export", "تصدير"), "toggle": ("Enable", "تفعيل"), "form": ("Name", "الاسم"),
    "filter": ("Filter", "تصفية"), "drag_drop": ("Move", "نقل"), "recovery": ("Retry", "إعادة المحاولة"),
    "scroll_confirm": ("Scroll", "تمرير"), "rename": ("Rename", "إعادة التسمية"), "delete_confirm": ("Delete", "حذف"),
}


def family_split(workflow):
    return {0: "train", 1: "train", 2: "train", 3: "dev", 4: "test"}[WORKFLOWS.index(workflow) % 5]


def manifest_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def semantic_goal(goal):
    """Collapse incidental goal formatting without rewriting value labels."""
    return " ".join(unicodedata.normalize("NFKC", goal).split())


def _task(suite, index, workflow, language, rng):
    object_id = f"{suite}-{workflow}-{index:02d}-{rng.randrange(100000, 999999)}"
    value = f"Hala-{rng.randrange(1000, 9999)}"
    label = LABELS[workflow][language == "ar"] + " " + object_id
    controls = []
    steps = []
    initial = {"document": object_id}
    desired = {"document": object_id}

    def button(ref, name, updates, requires=None, visible_if=None, transient=0):
        controls.append({"ref": ref, "name": name, "role": "Button", "updates": updates, "requires": requires or {}, "visible_if": visible_if or {}, "transient_failures": transient})

    def edit(ref, name, key):
        controls.append({"ref": ref, "name": name, "role": "Edit", "input_key": key, "updates": {}, "requires": {}, "visible_if": {}, "transient_failures": 0})

    def step(target, action="click", text=None):
        item = {"action": action, "target": target}
        if text is not None:
            item["text"] = text
        steps.append(item)

    if suite == "intent":
        # Intent fixtures isolate a single language/target binding. GUI fixtures below
        # exercise the workflow; both suites share the same family and split.
        if workflow == "form":
            initial["name"] = ""
            desired["name"] = value
            edit("target", label, "name")
            step("target", "set_value", value)
            goal = f"Set {label} to {value}" if language == "en" else f"عيّن {label} إلى {value}"
        else:
            initial["activated"] = False
            desired["activated"] = True
            button("target", label, {"activated": True})
            step("target", "semantic_invoke")
            goal = f"Click {label}" if language == "en" else f"انقر {label}"
    else:
        goal = f"Complete {workflow} for {object_id}" if language == "en" else f"أكمل {LABELS[workflow][1]} للمستند {object_id}"
        if workflow in ("save", "open", "archive", "export", "toggle"):
            key = {"save": "saved", "open": "opened", "archive": "archived", "export": "exported", "toggle": "enabled"}[workflow]
            initial[key], desired[key] = False, True
            button("target", label, {key: True})
            step("target")
        elif workflow in ("form", "rename"):
            initial.update(name="", saved=False)
            desired.update(name=value, saved=True)
            edit("field", label, "name")
            button("submit", "Save" if language == "en" else "حفظ", {"saved": True}, {"name": value})
            step("field", "set_value", value)
            step("submit")
        elif workflow == "filter":
            initial.update(filter="", applied=False)
            desired.update(filter=value, applied=True)
            edit("field", label, "filter")
            button("apply", "Apply" if language == "en" else "تطبيق", {"applied": True}, {"filter": value})
            step("field", "set_value", value)
            step("apply")
        elif workflow == "drag_drop":
            initial.update(selection="", location="Inbox")
            desired.update(selection=object_id, location="Archive")
            button("source", label, {"selection": object_id})
            button("destination", "Drop in Archive" if language == "en" else "إفلات في الأرشيف", {"location": "Archive"}, {"selection": object_id})
            step("source", "semantic_invoke")
            step("destination", "semantic_invoke")
        elif workflow == "recovery":
            initial["saved"], desired["saved"] = False, True
            button("target", label, {"saved": True}, transient=1)
            step("target")
            steps.append({"action": "observe"})
            step("target")
        elif workflow == "scroll_confirm":
            initial.update(scrolled=False, confirmed=False)
            desired.update(scrolled=True, confirmed=True)
            controls.append({"ref": "pane", "name": label, "role": "Pane", "updates": {"scrolled": True}, "requires": {}, "visible_if": {}, "transient_failures": 0})
            button("confirm", "Confirm" if language == "en" else "تأكيد", {"confirmed": True}, {"scrolled": True}, {"scrolled": True})
            step("pane", "scroll")
            step("confirm")
        elif workflow == "delete_confirm":
            initial.update(dialog=False, deleted=False)
            desired.update(dialog=False, deleted=True)
            button("target", label, {"dialog": True})
            button("confirm", "Confirm deletion" if language == "en" else "تأكيد الحذف", {"dialog": False, "deleted": True}, {"dialog": True}, {"dialog": True})
            step("target")
            step("confirm")
    # Wrong but observed control is executable and changes unrelated state. The
    # checker must detect that the intended final state was never reached.
    initial["distractor_activated"] = desired["distractor_activated"] = False
    button("distractor", label + (" copy" if language == "en" else " نسخة"), {"distractor_activated": True})
    scale = (1, 1.5, 2)[rng.randrange(3)]
    width, height = int(800 * scale), int(600 * scale)
    cells = [(col, row) for row in range(4) for col in range(3)]
    rng.shuffle(cells)
    for control, (col, row) in zip(controls, cells):
        x = (30 + col * 250 + rng.randrange(0, 15)) * scale
        y = (70 + row * 110 + rng.randrange(0, 15)) * scale
        w, h = 210 * scale, 60 * scale
        if language == "ar":
            x = width - x - w
        control["bounds"] = [x, y, w, h]
        control["normalized_bounds"] = normalize_box(control["bounds"], width, height)
    split = family_split(workflow)
    return {"task_id": object_id, "family_id": f"workflow-{workflow}", "suite": suite, "workflow": workflow, "language": language, "rtl": language == "ar", "goal": goal, "split": split, "provenance": dict(RIGHTS), "frame": {"width": width, "height": height, "scale": scale, "coordinate_space": "window"}, "controls": controls, "initial_state": initial, "desired_state": desired, "oracle_steps": steps, "max_steps": len(steps) + 2, "synthetic_drag_abstraction": workflow == "drag_drop" and suite == "gui"}


def generate_manifest(seed=SEED):
    rng = random.Random(seed)
    tasks = []
    for suite, families, per_family in [("intent", WORKFLOWS[:10], 6), ("gui", WORKFLOWS, 10)]:
        for workflow in families:
            for index in range(per_family):
                tasks.append(_task(suite, index, workflow, "en" if index % 2 == 0 else "ar", rng))
    manifest = {"schema": "hala.synthetic.manifest.v1", "revision": REVISION, "seed": seed, "rights_notice": "Original synthetic tasks and labels generated for Hala; dedicated to CC0-1.0. No benchmark trajectories or personal data.", "tasks": tasks}
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    if manifest.get("schema") != "hala.synthetic.manifest.v1" or not manifest.get("tasks"):
        raise ValueError("Unknown or empty manifest")
    ids, fingerprints, family_splits = set(), set(), {}
    for case in manifest["tasks"]:
        if case["task_id"] in ids:
            raise ValueError("Duplicate task ID")
        ids.add(case["task_id"])
        if case["split"] not in ("train", "dev", "test") or case["language"] not in ("en", "ar") or case["suite"] not in ("intent", "gui"):
            raise ValueError("Invalid split, language or suite")
        if family_splits.setdefault(case["family_id"], case["split"]) != case["split"]:
            raise ValueError("Family leaks across splits")
        if case["family_id"] != "workflow-" + case["workflow"] or case["workflow"] not in WORKFLOWS:
            raise ValueError("Workflow family mismatch")
        if case["split"] != family_split(case["workflow"]):
            raise ValueError("Frozen family allocation cannot be relabeled into training")
        rights = case["provenance"]
        if rights != RIGHTS:
            raise ValueError("Unknown, unlicensed, personal or benchmark source rejected")
        semantic_fields = {k: case[k] for k in ("initial_state", "desired_state", "oracle_steps", "workflow", "language")}
        semantic_fields["goal"] = semantic_goal(case["goal"])
        fingerprint = json.dumps(semantic_fields, ensure_ascii=False, sort_keys=True)
        if fingerprint in fingerprints:
            raise ValueError("Duplicate semantic task")
        fingerprints.add(fingerprint)
        if type(case["max_steps"]) is not int or not 1 <= case["max_steps"] <= 60:
            raise ValueError("Invalid task budget")
        if len(case["oracle_steps"]) > case["max_steps"]:
            raise ValueError("Annotated path exceeds admitted task budget")
        identity = re.fullmatch(rf"{re.escape(case['suite'])}-{re.escape(case['workflow'])}-([0-9]{{2}})-([1-9][0-9]{{5}})", case["task_id"])
        if identity is None or int(identity.group(1)) >= (6 if case["suite"] == "intent" else 10):
            raise ValueError("Task identity disagrees with suite/workflow/index")
        if case["initial_state"].get("document") != case["task_id"] or case["desired_state"].get("document") != case["task_id"]:
            raise ValueError("Original document identity disagrees with task identity")
        frame = case["frame"]
        if frame["coordinate_space"] != "window" or type(frame["scale"]) not in (int, float) or frame["scale"] not in (1, 1.5, 2) or type(case["rtl"]) is not bool or case["rtl"] != (case["language"] == "ar"):
            raise ValueError("Frame/RTL label mismatch")
        controls = {c["ref"]: c for c in case["controls"]}
        if len(controls) != len(case["controls"]) or len(set(c["name"] for c in case["controls"])) != len(controls):
            raise ValueError("Duplicate controls or ambiguous labels")
        for control in controls.values():
            calculated = normalize_box(control["bounds"], frame["width"], frame["height"])
            restored = denormalize_box(control["normalized_bounds"], frame["width"], frame["height"])
            if any(abs(a - b) > 1e-9 for a, b in zip(calculated, control["normalized_bounds"])) or any(abs(a - b) > 1e-7 for a, b in zip(restored, control["bounds"])):
                raise ValueError("Normalized coordinates disagree with captured pixels")
            if control["role"] not in ("Button", "Edit", "Pane") or not control["name"]:
                raise ValueError("Invalid control label")
            if type(control["transient_failures"]) is not int or not 0 <= control["transient_failures"] <= 3:
                raise ValueError("Invalid bounded transient failure")
            keys = set(control["updates"]) | set(control["requires"]) | set(control["visible_if"])
            if not keys <= set(case["initial_state"]):
                raise ValueError("Control references undeclared state")
            if control["role"] == "Edit" and control.get("input_key") not in case["initial_state"]:
                raise ValueError("Input references undeclared state")
        # Reference label replay is separate from the evaluation executor. It
        # validates annotation reachability, never a provider's claimed success.
        replay = copy.deepcopy(case["initial_state"])
        attempts = {}
        for step in case["oracle_steps"]:
            if step["action"] not in ("semantic_invoke", "click", "set_value", "scroll", "observe"):
                raise ValueError("Invalid labeled action")
            if step["action"] != "observe":
                target = controls.get(step.get("target"))
                if target is None:
                    raise ValueError("Label references nonexistent target")
                if (step["action"] == "set_value" and (target["role"] != "Edit" or not isinstance(step.get("text"), str))) or (step["action"] == "scroll" and target["role"] != "Pane"):
                    raise ValueError("Label action/role mismatch")
                if step["action"] in ("click", "semantic_invoke") and target["role"] != "Button":
                    raise ValueError("Invoke label is not a button")
                if not all(replay.get(k) == v for k, v in target["visible_if"].items()):
                    raise ValueError("Label references a hidden target at this step")
                count = attempts.get(target["ref"], 0)
                attempts[target["ref"]] = count + 1
                if count >= target["transient_failures"] and all(replay.get(k) == v for k, v in target["requires"].items()):
                    if step["action"] == "set_value":
                        replay[target["input_key"]] = step["text"]
                    else:
                        replay.update(target["updates"])
        if not case["desired_state"] or set(case["desired_state"]) != set(case["initial_state"]):
            raise ValueError("Invalid final-state labels")
        if any(replay.get(k) != v for k, v in case["desired_state"].items()):
            raise ValueError("Labeled actions do not reach the labeled final state")
    return manifest


def training_records(manifest):
    manifest = copy.deepcopy(manifest)
    validate_manifest(manifest)
    if manifest.get("revision") != REVISION or type(manifest.get("seed")) is not int:
        raise ValueError("Training importer requires original generator revision and integer seed")
    # This is an original-synthetic-source importer, not a metadata-trusting
    # generic dataset importer. Regeneration has no dependency on this function.
    source = generate_manifest(manifest["seed"])
    if manifest["seed"] == SEED and hashlib.sha256(manifest_bytes(source)).hexdigest() != FROZEN_SOURCE_SHA256:
        raise ValueError("Pinned original source digest changed")
    original_by_id = {c["task_id"]: hashlib.sha256(manifest_bytes(c)).digest() for c in source["tasks"]}
    for case in manifest["tasks"]:
        if original_by_id.get(case["task_id"]) != hashlib.sha256(manifest_bytes(case)).digest():
            raise ValueError("Training importer rejects unknown identities or changed canonical source content")
    # Exact source content includes the original family and allocation. Relabeling
    # IDs, goals or entire heldout families cannot create admitted train records.
    return copy.deepcopy([c for c in manifest["tasks"] if c["split"] == "train"])


def freeze(directory=FROZEN.parent):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = manifest_bytes(generate_manifest())
    target = directory / "tasks.json"
    target.write_bytes(payload)
    (directory / "tasks.sha256").write_text(hashlib.sha256(payload).hexdigest() + "  tasks.json\n", encoding="ascii")
    return target


def load_frozen(path=FROZEN):
    path = Path(path)
    payload = path.read_bytes()
    digest = path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Frozen manifest digest mismatch")
    return validate_manifest(json.loads(payload))


if __name__ == "__main__":
    print(json.dumps({"path": str(freeze()), "seed": SEED, "revision": REVISION}))
