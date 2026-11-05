# Hala

Hala is the independent intelligence system for Aladdin Pro. HalaMind proposes bounded plans, HalaType proposes typed GUI actions, and HalaReliance recommends economical routing. Aladdin retains execution, authorization, and outcome verification.

Status: implemented CPU contract/service and synthetic evaluation infrastructure; **no foundation-model checkpoint is qualified for product use or distribution**. A real owner-private Qwen3-VL-2B H425 research LoRA was trained and source-development-tested (original 240-image H417 baseline 94 correct; H425 114 correct), and H426 is an **untrained** candidate loss improvement. Those author-accessible data do not establish independent blind bilingual OOD generalization, clinical safety or completed rendered-screenshot ML rights. Source-asset/legal TRAIN admission and independent eval remain open; no weights were published, quantized or deployed. Large model weights and private datasets belong outside Git. Development must remain at $0 out-of-pocket and must not modify Aladdin.

Python 3.11+ is required. The mock service has no runtime dependencies. Install `Pillow==12.3.0` to reproduce image, converter and artifact-integrity tests, then run:

```text
python -m unittest discover -s tests -v
python -m evaluation --provider oracle --split test --output evaluation/results/oracle-heldout.json
python -m experiments.cost.calculator --table
```

Run the authenticated loopback mock with `HALA_BEARER_TOKEN` (32–256 printable characters), `HALA_TENANT_ID`, and `python -m hala.serving`. See [integration boundaries](docs/ALADDIN_INTEGRATION.md), [data qualification](docs/DATA_FACTORY.md), [candidate training](training/README.md), [research](research/README.md), [current status](docs/CURRENT_STATUS.md) and [review evidence](docs/REVIEW_EVIDENCE.md).

The synthetic oracle uses privileged fixture labels and measures infrastructure reachability. It is not model quality. The real backend deliberately fails closed. The English-only candidate SFT export contains genuine PNGs and logical-action labels; this is separate from the completed **private research H425 LoRA** and does not authorize training or release of models derived from any source without independently verified rights. H425's source-only model evidence and H426's unfinished research-only tokenizer/processor work are tracked in [H426 private research](h426/README.md) and the [independent screenshot rights issue](https://github.com/TheHalfMoon/Hala/issues/384) / [independent sealed EN/AR evaluation issue](https://github.com/TheHalfMoon/Hala/issues/385). **Production remains NO-GO**, and the founder controls merge and release.
