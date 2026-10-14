# Hala

Hala is the independent intelligence system for Aladdin Pro. HalaMind proposes bounded plans, HalaType proposes typed GUI actions, and HalaReliance recommends economical routing. Aladdin retains execution, authorization, and outcome verification.

Status: implemented CPU contract/service and synthetic evaluation infrastructure; all foundation models remain unqualified. No checkpoint has been trained, quantized, or deployed. Large model weights and datasets belong outside Git. Development must remain at $0 out-of-pocket and must not modify Aladdin.

Python 3.11+ is required. The mock service has no runtime dependencies. Install `Pillow==12.3.0` to reproduce image, converter and artifact-integrity tests, then run:

```text
python -m unittest discover -s tests -v
python -m evaluation --provider oracle --split test --output evaluation/results/oracle-heldout.json
python -m experiments.cost.calculator --table
```

Run the authenticated loopback mock with `HALA_BEARER_TOKEN` (32–256 printable characters), `HALA_TENANT_ID`, and `python -m hala.serving`. See [integration boundaries](docs/ALADDIN_INTEGRATION.md), [data qualification](docs/DATA_FACTORY.md), [candidate training](training/README.md), [research](research/README.md), [current status](docs/CURRENT_STATUS.md) and [review evidence](docs/REVIEW_EVIDENCE.md).

The synthetic oracle uses privileged fixture labels and measures infrastructure reachability. It is not model quality. The real backend deliberately fails closed. The English-only candidate SFT export contains genuine PNGs and logical-action labels, but no adapter or real training run exists. The founder controls merge and production release.
