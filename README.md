# Hala

Hala is an independent intelligence system under development for **Orcel**, the current local-first computer-use and MCP authority platform. HalaMind proposes bounded plans, HalaType proposes typed GUI actions, and HalaReliance recommends economical routing. **Orcel retains local execution authority, approval policy, and independent outcome verification.**

Status: implemented CPU contract/service and synthetic evaluation infrastructure. Real Qwen3-VL-2B experiments and a retained small LoRA pilot have been measured on free GPU, but neither model nor any precision is adopted or qualified for Orcel production. No native Orcel integration or production deployment exists. Large model weights and datasets belong outside Git. Development must target $0 founder out-of-pocket and must not modify Orcel.

Python 3.11+ is required. The mock service has no runtime dependencies. Install `Pillow==12.3.0` to reproduce image, converter and artifact-integrity tests, then run:

```text
python -m unittest discover -s tests -v
python -m evaluation --provider oracle --split test --output evaluation/results/oracle-heldout.json
python -m experiments.cost.calculator --table
```

Run the authenticated loopback mock with `HALA_BEARER_TOKEN` (32â€“256 printable characters), `HALA_TENANT_ID`, and `python -m hala.serving`. See [current Orcel integration boundaries](docs/ORCEL_INTEGRATION.md) and the [historical Aladdin compatibility record](docs/ALADDIN_INTEGRATION.md), [data qualification](docs/DATA_FACTORY.md), [candidate training](training/README.md), [research](research/README.md), [current status](docs/CURRENT_STATUS.md) and [review evidence](docs/REVIEW_EVIDENCE.md).

The synthetic oracle uses privileged fixture labels and measures infrastructure reachability. It is not model quality. The real backend deliberately fails closed. The original English-only candidate SFT export contains genuine PNGs and logical-action labels; a later separate, retained small LoRA engineering/training pilot exists with narrow bilingual held-out measurements, which do not establish Arabic success or release quality. The founder controls merge and production release.
