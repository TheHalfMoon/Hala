# 0008. Isolate the private serving experiment worker

Status: accepted (2026-10-16)

## Context
H16 threads preserve admission until model work exits but cannot recover a failed CUDA context. Lifecycle evidence needs bounded recovery without extending the public API or enabling native execution.

## Decision
An optional operator engine owns one fixed-module child process over bounded JSON pipes. Cancel/deadline/failure ends that child; explicit restart reloads the pinned model. Existing authenticated v2 validation remains authoritative. Evaluation admission is separate from training and default RemoteQwenEngine behavior.

## Consequences
Cancellation cleanup has a hard process boundary and no generic command surface. Reload is expensive and actual CUDA work/cancellation proof still requires remote evidence. Two synthetic outcomes cannot qualify native or production use.
