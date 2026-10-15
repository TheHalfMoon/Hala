"""Small offline state machine. Nothing here operates a real GUI or computer."""
import copy
import hashlib

from hala import HalaType
from hala.contracts import ContractError, Proposal, Request
from hala.contracts.mock import binding, metadata


def check_final_state(case, state):
    """Read actual environment state; ignore proposals and model postconditions."""
    return all(key in state and state[key] == value for key, value in case["desired_state"].items())


class Environment:
    def __init__(self, case):
        self.case = copy.deepcopy(case)
        self.state = copy.deepcopy(case["initial_state"])
        self.generation = 0
        self.steps = 0
        self.history = []
        self.attempts = {}

    def advance_generation(self):
        self.generation += 1

    def observe(self):
        frame = self.case["frame"]
        visible = [c for c in self.case["controls"] if all(self.state.get(k) == v for k, v in c["visible_if"].items())]
        raw = {
            "schema": "hala.request.v1", "request_id": f"{self.case['task_id']}:request:{self.steps}:{self.generation}",
            "tenant_id": "synthetic-tenant", "task": {"task_id": self.case["task_id"], "goal": self.case["goal"], "step_index": self.steps, "max_steps": self.case["max_steps"]},
            "device": {"device_id": "synthetic-device", "session_id": self.case["task_id"] + ":session", "os": "synthetic", "locale": "ar-SA" if self.case["language"] == "ar" else "en-US"},
            "observation": {"generation": str(self.generation), "window": {"window_ref": "synthetic-window", "title": "Synthetic fixture", "process": "synthetic-only"},
                            "accessibility": [{"element_ref": c["ref"], "name": c["name"], "role": c["role"], "bounds": c["bounds"]} for c in visible],
                            "image": {"format": "png", "width": frame["width"], "height": frame["height"], "coordinate_space": frame["coordinate_space"], "scale": frame["scale"], "redactions": []}, "truncation": {"elements_omitted": 0}},
            "allowed_actions": ["semantic_invoke", "set_value", "click", "scroll", "observe", "wait", "abstain", "done_proposed"],
            "method_ceiling": "WINDOW_TARGETED_INPUT", "history": copy.deepcopy(self.history),
        }
        return Request.from_dict(raw)

    def _consume(self, action, target, result):
        self.history.append({"step": self.steps, "action": action, "target": target, "result": result, "postcondition": "unknown"})
        self.steps += 1
        self.advance_generation()

    def consume_invalid(self):
        # Every failed attempt consumes a step, so a provider cannot loop forever.
        self._consume("observe", "", "failed")

    def execute(self, proposal, request):
        if request.task.task_id != self.case["task_id"] or request.observation.generation != str(self.generation) or request.task.step_index != self.steps:
            raise ValueError("Stale observation or executor binding mismatch")
        current = self.observe()
        parsed = Proposal.from_dict(proposal.to_dict() if type(proposal) is Proposal else proposal, current)
        if len(parsed.actions) != 1:
            raise ValueError("Synthetic executor accepts one action per fresh observation")
        action = parsed.actions[0]
        if action.type == "done_proposed":
            self._consume(action.type, "", "completed")
            return {"done": True, "wrong_target": False, "effect_applied": False}
        if action.type in ("abstain",):
            self._consume(action.type, "", "cancelled")
            return {"stop": True, "wrong_target": False, "effect_applied": False}
        if action.type in ("observe", "wait"):
            self._consume(action.type, "", "completed")
            return {"wrong_target": False, "effect_applied": False}
        ref = action.element_ref_hint if action.type == "click" else action.element_ref
        control = next(c for c in self.case["controls"] if c["ref"] == ref)
        kind_matches = (action.type in ("click", "semantic_invoke") and control["role"] == "Button") or (action.type == "set_value" and control["role"] == "Edit") or (action.type == "scroll" and control["role"] == "Pane")
        if not kind_matches:
            raise ValueError("Synthetic control role/action mismatch")
        allowed = all(self.state.get(k) == v for k, v in control["requires"].items())
        attempts = self.attempts.get(ref, 0)
        self.attempts[ref] = attempts + 1
        applied = allowed and attempts >= control["transient_failures"]
        if applied:
            if action.type == "set_value":
                self.state[control["input_key"]] = action.text
            else:
                self.state.update(copy.deepcopy(control["updates"]))
        self._consume(action.type, ref, "completed" if applied else "failed")
        return {"wrong_target": ref == "distractor", "effect_applied": applied}


def proposal_dict(request, action, status="continue"):
    info = metadata().to_dict()
    info.update(model_id="synthetic-infrastructure-provider", model_revision=hashlib.sha256(b"synthetic-infrastructure-v1").hexdigest())
    return {"schema": "hala.proposal.v1", **binding(request), "actions": [action], "status": status, "confidence": 0.0,
            "rationale_short": "Scripted synthetic infrastructure baseline; not trained inference.", "metadata": info,
            "expected_postcondition": {"kind": "final_state_check"}}


class OracleProvider:
    """Uses privileged fixture labels, deliberately not a model-quality baseline."""
    name = "oracle"

    def __init__(self, case):
        self.case = copy.deepcopy(case)

    def propose(self, request):
        index = request.task.step_index
        if index >= len(self.case["oracle_steps"]):
            return proposal_dict(request, {"type": "done_proposed"}, "verification_recommended")
        step = self.case["oracle_steps"][index]
        action = {"type": step["action"]}
        if step["action"] == "observe":
            return proposal_dict(request, action)
        ref = step["target"]
        if step["action"] == "click":
            target = next(e for e in request.observation.accessibility if e.element_ref == ref)
            x, y, width, height = target.bounds
            action.update(point=[x + width / 2, y + height / 2], coordinate_space=request.observation.image.coordinate_space, image_scale=request.observation.image.scale, element_ref_hint=ref)
        else:
            action["element_ref"] = ref
            if step["action"] == "set_value":
                action.update(text=step["text"], secret=False)
            elif step["action"] == "scroll":
                action.update(delta_x=0, delta_y=300)
        return proposal_dict(request, action)


class ControlProvider:
    def __init__(self, mode, case):
        if mode not in ("wrong_target", "stale", "invalid", "false_done"):
            raise ValueError("Unknown negative control")
        self.name = mode
        self.case = case

    def propose(self, request):
        if self.name == "false_done" or request.task.step_index > 0:
            return proposal_dict(request, {"type": "done_proposed"}, "verification_recommended")
        if self.name == "wrong_target":
            return proposal_dict(request, {"type": "semantic_invoke", "element_ref": "distractor"})
        proposal = OracleProvider(self.case).propose(request)
        if self.name == "stale":
            proposal["observation_generation"] = "old-generation"
        elif self.name == "invalid":
            proposal["success"] = True
        return proposal


class MockProvider:
    name = "hala_contract_mock"

    def propose(self, request):
        return HalaType().action(request).to_dict()


def run_case(case, provider):
    # Providers may retain the caller's object. Neither success labels nor the
    # admitted budget/identity can come from it after provider execution begins.
    case = copy.deepcopy(case)
    env = Environment(case)
    result = {"task_id": case["task_id"], "family_id": case["family_id"], "language": case["language"], "suite": case["suite"], "split": case["split"], "provider": provider.name,
              "wrong_target": 0, "schema_failures": 0, "stale_failures": 0, "invalid_outputs": 0, "false_done": False, "trace": []}
    done = False
    while env.steps < case["max_steps"]:
        request = env.observe()
        try:
            raw = provider.propose(Request.from_dict(request.to_dict()))
            # Serializer and validator are distinct from provider and executor.
            validated = Proposal.from_dict(raw, request)
            event = env.execute(validated, request)
            result["wrong_target"] += int(event["wrong_target"])
            result["trace"].append({"step": request.task.step_index, "action": validated.actions[0].to_dict(), **event})
            if event.get("done"):
                done = True
                break
            if event.get("stop"):
                break
        except (ContractError, ValueError, TypeError, KeyError, StopIteration) as error:
            result["invalid_outputs"] += 1
            category = "stale_failures" if "generation" in str(error).lower() or "binding" in str(error).lower() else "schema_failures"
            result[category] += 1
            result["trace"].append({"step": request.task.step_index, "rejected": category, "reason": str(error)[:200]})
            env.consume_invalid()
    result.update(steps=env.steps, budget_exhausted=env.steps >= case["max_steps"], verified_success=check_final_state(case, env.state), final_state=copy.deepcopy(env.state))
    result["false_done"] = done and not result["verified_success"]
    return result
