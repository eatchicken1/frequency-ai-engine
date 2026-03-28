# Frequency AI Runtime SLO

## Scope
- Service: `frequency-ai-engine`
- Interface: `POST /workflow/{name}/run`
- Workflows: `resonance_match`, `campus_research`, `long_task_orchestration`

## SLI Definitions
1. Availability SLI
- Formula:
  - numerator: `sum(increase(frequency_runtime_workflow_runs_total{result="success"}[30d]))`
  - denominator: `sum(increase(frequency_runtime_workflow_runs_total{result=~"success|error|bad_request|forbidden"}[30d]))`

2. Latency SLI (P95)
- Formula:
  - `histogram_quantile(0.95, sum(rate(frequency_runtime_workflow_duration_seconds_bucket[30d])) by (le, workflow))`

3. Auth Integrity SLI
- Formula:
  - reject ratio =
    - `sum(increase(frequency_runtime_auth_attempts_total{result="rejected"}[30d]))`
    - `/ clamp_min(sum(increase(frequency_runtime_auth_attempts_total[30d])), 1)`

## SLO Targets
- Availability: >= 99.5% (30d rolling)
- Latency: P95 <= 2.5s for `resonance_match`, <= 3.0s for other workflows
- Auth Integrity: rejected ratio <= 2%

## Error Budget Policy
- Monthly error budget for availability: 0.5%
- Burn-rate levels:
  - fast burn: > 10x budget in 1h -> page on-call
  - medium burn: > 4x budget in 6h -> high-priority incident
  - slow burn: > 2x budget in 24h -> backlog hardening within 2 working days

## Operational Actions
- If availability SLO breach is predicted:
  - degrade non-critical workflows (`campus_research`, `long_task_orchestration`) to protect `resonance_match`
- If latency SLO breach persists 15min:
  - enable fallback model path and shorten max rounds for match workflow
- If auth integrity SLO breach:
  - rotate runtime shared secret and audit caller whitelist
