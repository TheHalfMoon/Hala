"""Transparent scenarios, never a substitute for measured serving economics."""

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class CostInputs:
    users: int = 10
    tasks_per_user: int = 100
    model_calls_per_task: float = 2
    images_per_task: float = 1
    tokens_per_task: float = 1000
    gpu_seconds_per_call: float = 3
    retry_rate: float = 0.1
    success_rate: float = 0.8
    gpu_hourly_usd: float = 1
    warm_workers: int = 0
    month_hours: float = 720
    cold_starts: int = 0
    cold_gpu_seconds: float = 0
    storage_gb: float = 0
    storage_usd_per_gb: float = 0
    network_gb_per_task: float = 0
    network_usd_per_gb: float = 0
    failure_cost_usd: float = 0
    concurrency: int = 1
    wall_seconds_per_call: float = 3
    measured_run_id: str | None = None


def estimate(inputs: CostInputs) -> dict:
    values = asdict(inputs)
    for name, value in values.items():
        if name == 'measured_run_id':
            if value is not None:
                raise ValueError('Attach measured evidence through a reviewed result manifest, not a label')
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a number')
        if value < 0 or value > 1e12 or not math.isfinite(value):
            raise ValueError(f'{name} must be finite and in [0, 1e12]')
    for name in ('users', 'tasks_per_user', 'warm_workers', 'cold_starts', 'concurrency'):
        if not isinstance(values[name], int):
            raise ValueError(f'{name} must be an integer')
    if inputs.month_hours <= 0 or inputs.concurrency <= 0:
        raise ValueError('Month duration and concurrency must be positive')
    if inputs.retry_rate > 1 or inputs.success_rate > 1:
        raise ValueError('Rates must be in [0, 1]')
    tasks = inputs.users * inputs.tasks_per_user
    calls = tasks * inputs.model_calls_per_task * (1 + inputs.retry_rate)
    active = calls * inputs.gpu_seconds_per_call / 3600
    cold = inputs.cold_starts * inputs.cold_gpu_seconds / 3600
    warm = inputs.warm_workers * inputs.month_hours
    # GPU-seconds are already summed worker time; concurrency does not divide their price.
    billed = max(active + cold, warm)
    storage = inputs.storage_gb * inputs.storage_usd_per_gb
    network = tasks * inputs.network_gb_per_task * inputs.network_usd_per_gb
    failed = tasks * (1 - inputs.success_rate)
    failure = failed * inputs.failure_cost_usd
    total = billed * inputs.gpu_hourly_usd + storage + network + failure
    verified = tasks * inputs.success_rate
    offered_load = (calls * inputs.wall_seconds_per_call /
                    (inputs.month_hours * 3600 * inputs.concurrency))
    per_verified = total / verified if verified else None
    derived = (tasks, calls, active, cold, warm, billed, storage, network,
               failed, failure, total, verified, offered_load)
    if (any(not math.isfinite(value) for value in derived) or
            (per_verified is not None and not math.isfinite(per_verified))):
        raise ValueError('Scenario exceeds finite numerical range')
    # M/M/c or burst predictions need an arrival distribution. Report load only.
    return {
        'evidence': 'scenario_assumptions', 'inputs': values, 'attempted_tasks': tasks,
        'verified_tasks_assumed': verified, 'model_calls': calls,
        'images': tasks * inputs.images_per_task, 'tokens': tasks * inputs.tokens_per_task,
        'active_gpu_hours': active, 'cold_gpu_hours': cold,
        'idle_gpu_hours': max(0, warm - active - cold), 'billed_gpu_hours': billed,
        'storage_usd': storage, 'network_usd': network, 'failure_usd': failure,
        'total_usd': total, 'usd_per_verified_task': per_verified,
        'offered_concurrency_load': offered_load,
        'capacity_exceeded': offered_load >= 1,
        'queue_latency_seconds': None,
        'limitations': 'Inputs are assumptions. Image/token costs must be included in measured GPU time. '
                       'No burst distribution, tail latency, taxes, support or provider quote is inferred.',
    }


def scenario_table(inputs: CostInputs) -> list[dict]:
    rows = []
    for users in (10, 50, 100, 500, 1000):
        result = estimate(replace(inputs, users=users))
        for price in (5, 8, 10, 15, 20):
            revenue = users * price
            rows.append({'users': users, 'price_usd': price, 'revenue_usd': revenue,
                         'modeled_cost_usd': result['total_usd'],
                         'modeled_margin_usd': revenue - result['total_usd'],
                         'usd_per_verified_task': result['usd_per_verified_task'],
                         'evidence': 'scenario_assumptions', 'pricing_adopted': False})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', help='JSON file of explicit scenario assumptions')
    parser.add_argument('--table', action='store_true')
    args = parser.parse_args()
    values = {}
    if args.inputs:
        with open(args.inputs, encoding='utf-8') as handle:
            values = json.load(handle)
    inputs = CostInputs(**values)
    output = scenario_table(inputs) if args.table else estimate(inputs)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
