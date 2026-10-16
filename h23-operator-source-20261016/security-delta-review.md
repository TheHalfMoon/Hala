# H23 operator cleanup security delta

Reviewed operator SHA256 `fff4235e6e81ac325b0456805d1d7d18c59a1d41a450482db7c82f1b37b3f1a9` against the prior operator monitoring-exception finding. Independent byte check matches.

Verdict: previous P2 resolved; no remaining must-fix security finding in this narrow delta.

- The full post-spawn monitor is now inside `try/finally`. Read, selector, log and other monitor failures enter bounded owned-group cleanup, close stdout and record errors before retention.
- `finish_owned()` targets only the session/group created by this invocation's `Popen(start_new_session=True)`. Abort sends TERM, waits up to 5 seconds, escalates to KILL and waits up to 2 seconds; group-presence probes and short final group cleanup are bounded. No caller/external PID or process-name kill is added.
- Leader reap and group quiescence are separate explicit evidence fields. Cleanup failure cannot become `finished`: status remains `incomplete_cleanup`, and normal ZIP export is blocked while `producer_quiescent=false`. A `HALA_H23_RETENTION_BLOCKED` diagnostic retains raw-evidence location and cleanup evidence without claiming a stable archive.
- Existing fixed timeout, six-profile prerequisite barrier, source/pins/cache/hardware identity and unqualified flags remain intact. No scientific-source or native-execution scope expansion.

Validation: `rtk proxy python outputs/test_h23_operator.py -v` passed 12 tests in 1.452s, including monitor read/selector/log failure cleanup and double-timeout archive blocking. All process/group/network behavior in these tests is mocked; no actual operator invocation, model/GPU work, cache move, publication or live Linux cleanup experiment was performed. Actual TEST/H23 evidence and qualification remain separate root-owned gates.

panel: operator security delta ✓ independent session-model judge.
