# H161 — cross-application native action training source

After H160 failed real TodoMVC transfer (**0/16 for H156 and H157** despite H157 MiniWoB++ 60/72), the next improvement is **new, genuinely cross-app training data**. No H160/EVAL source images or model predictions can be reused for training.

## Frozen source admission

Original licensed unmodified MIT applications:
- `tastejs/todomvc` ES6 original app, git `ff43b02e59dfa604386bb382034b2cd07c2bcd8a`, three real native **click** task families.
- `react-dnd-kit-board`, git `553d4e92be3e86e0deb7045f4968152868399eae`, native persistent **drag** from Todo to In Progress.

`TRAIN`: 24 independent screenshot/task examples per app = **48**. `DEV`: 12 independent examples per app = **24**. Total **72** scenes and native source-reference actions, split across Arabic/English. DEV screenshots are **excluded from optimizer**; no H158/H159/H160 consumed screenshot SHA overlap. Every media item and reference action is stored only in `C:\Users\Shehr\work\Hala\private-h161\`; weights and app source stay private. All apps retain upstream license/copyright and unchanged source. Real browser action execution and persisted/in-session state verification are mandatory, as are independent exact screenshot replay and SHA provenance.

The task contract is `click(point=[x,y])` or `drag(start=[x,y],end=[x,y])`, not an owner-DOM oracle at inference. Training or score reporting **must not proceed** until source, split and image/label audits pass. A third application separate from BOTH TRAIN apps is to be reserved as unseen-app OOD; no existing H138 evaluations are a new independent BLIND claim.

Windows owner commands (no GPU):
```powershell
cd C:\Users\Shehr\work\Hala\h154-isolated
python -m unittest tests.test_h161_two_original_apps_source
python -m h161.original_apps_source smoke --todo-root C:\Users\Shehr\work\Hala\private-h158\upstream\todomvc --board-root C:\Users\Shehr\work\Hala\private-h138\vendor-apps\react-dnd-kit-board --owner C:\Users\Shehr\work\Hala
python -m h161.original_apps_source capture --todo-root C:\Users\Shehr\work\Hala\private-h158\upstream\todomvc --board-root C:\Users\Shehr\work\Hala\private-h138\vendor-apps\react-dnd-kit-board --owner C:\Users\Shehr\work\Hala
python -m h161.audit_source --owner C:\Users\Shehr\work\Hala
python -m h161.original_apps_source verify --todo-root C:\Users\Shehr\work\Hala\private-h158\upstream\todomvc --board-root C:\Users\Shehr\work\Hala\private-h138\vendor-apps\react-dnd-kit-board --owner C:\Users\Shehr\work\Hala
```

No paid GPU, model release, main merge, Hala 35B changes or H140 sealed blind calls. Next phase: train typed mixed click/drag on free Kaggle T4 only after TRAIN-only package provenance, and evaluate on fresh DEV once. If blocked, publish nothing and document the failure.
