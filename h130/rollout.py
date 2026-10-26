"""H130 screenshot-only feedback loop for future computer-use models.

The callable receives ONLY screenshot, task instruction and previous outcome,
never browser DOM, hidden target coordinates, selectors or JS access.
No implicit model integration, new trained weights or skill qualification.
"""
from dataclasses import dataclass
from h130.browser_env import BrowserDragEnv, MAX_ACTIONS, TASKS

@dataclass(frozen=True)
class AttemptReceipt:
    action_index: int
    before_image_sha256: str
    after_image_sha256: str
    executed: bool
    success: bool

@dataclass(frozen=True)
class EpisodeReceipt:
    task: str
    success: bool
    attempts: tuple
    policy_calls: int
    actual_browser_pointer_execution: bool
    trained_model_inference_verified: bool = False

def run_episode(task, propose_action, *, factory=BrowserDragEnv,
                max_actions=MAX_ACTIONS):
    """propose_action(observation, previous_feedback) -> (start_xy, end_xy).

    The policy is injected and untrusted; it never receives a Playwright Page.
    No raw images or target geometry are stored in the small receipt.
    """
    if task not in TASKS or not callable(propose_action):
        raise ValueError("H130 named GUI task + explicit external policy required")
    if type(max_actions) is not int or not 1<=max_actions<=MAX_ACTIONS:
        raise ValueError("H130 bounded closed-loop attempts required")
    history=[]
    with factory(task) as env:
        observation=env.observe()
        previous=None
        for n in range(max_actions):
            answer=propose_action(observation, previous)
            if type(answer) not in (list,tuple) or len(answer)!=2:
                raise ValueError("H130 model must return (start_xy, end_xy)")
            feedback=env.execute_drag(answer[0],answer[1])
            history.append(AttemptReceipt(
                action_index=n+1,
                before_image_sha256=observation.image_sha256,
                after_image_sha256=feedback.observation.image_sha256,
                executed=feedback.action_applied,
                success=feedback.success))
            observation=feedback.observation
            previous=feedback
            if feedback.success or feedback.terminal:
                break
    return EpisodeReceipt(
        task=task,success=bool(history and history[-1].success),
        attempts=tuple(history),policy_calls=len(history),
        actual_browser_pointer_execution=True,
        trained_model_inference_verified=False)
