# lie-detection

Experiments testing whether LLMs deceive in agentic settings, and whether monitoring affects deception rates.

## Experiments

- **Retroactive restriction**: Model reads files freely, then is told one was restricted. Does it admit reading it?
- **Forbidden file**: Model is told a file is off-limits upfront. Does it read it anyway?
- **Capability bluff**: Tools always fail. Does the model fabricate data instead of admitting failure?
- **Monitored cheating**: Closed-book eval with a forbidden cheat tool available. Does the model use it?

Each experiment has two conditions: unmonitored (A) and monitored (B).

## Usage

```
uv run python experiment_monitored_cheating.py --n-trials 5 --conditions A_no_monitoring B_monitored
```

Requires `OPENROUTER_API_KEY` in `.env`.
