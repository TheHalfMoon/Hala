---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen3.6-35B-A3B
- Qwen/Qwen3.5-9B
- Qwen/Qwen3.5-4B
- Qwen/Qwen3.5-0.8B

pipeline_tag: image-text-to-text
library_name: transformers
tags:
- multimodal
- action
- agent
- pytorch
- computer use
- gui agents
- moe
---

# **Holo3.1: Fast & Local Computer Use Agents**


## **Model Description**

**Holo3.1** is our latest family of Vision-Language Models (VLMs) for computer use agents. Building on Holo3, it expands support beyond browser and desktop automation to mobile environments, introduces native function-calling support for seamless integration with agent frameworks, and enables local deployment through optimized quantized checkpoints.

The Holo3.1 family spans model sizes from 0.8B to 35B-A3B parameters.
Across computer use, UI grounding, mobile automation, and business workflows, Holo3.1 delivers strong performance while improving deployment flexibility and cost efficiency.
* **Developed by:** [**H Company**](https://www.hcompany.ai/)
* **Model type:** Vision-Language Models for Navigation and Computer Use Agents
* **Available models:** Holo3.1-0.8B, Holo3.1-4B, Holo3.1-9B, Holo3.1-35B-A3B
* **Base models:** Qwen 3.5 family
* **Supported environments:** Web, Desktop, Mobile
* **Available quantizations for Holo3.1-35B-A3B:** BF16, FP8, NVFP4, Q4 GGUF
* **Blog Post:** [hcompany.ai/holo3.1](https://www.hcompany.ai/holo3.1)
* **Quickstart:** [hub.hcompany.ai/quickstart](https://hub.hcompany.ai/quickstart)
* **License:** Apache 2.0 License

### Performance vs Cost

The figure below compares the overall performance and inference cost of the Holo3.1 and Qwen 3.5 families. Overall performance averages computer use, mobile automation, enterprise workflows, and UI grounding benchmarks.

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/69ce2739f4b9146a31e99a2f/0uU-6v68tSl3Kvd9jHWfS.png" width="700"/>
</p>

Holo3.1 establishes a strong Pareto frontier across model sizes, from lightweight local agents to state-of-the-art enterprise deployments.

---

## Benchmark Results

Holo3.1 delivers strong performance across computer use, mobile automation, enterprise workflows, and UI grounding benchmarks.

<div align="center">

**Table 1: Evaluation results across computer use, mobile automation, enterprise workflows, and grounding benchmarks.**

<img width="800" src="https://cdn-uploads.huggingface.co/production/uploads/69ce2739f4b9146a31e99a2f/eKbMPsY32AF4PCNzumeZQ.png"/>

</div>

## Get Started

Explore our [Quickstart guide](https://hub.hcompany.ai/quickstart) to learn how to integrate Holo3.1 into your applications, deploy local agents, or run optimized inference on NVIDIA hardware.

---

## Citation

```bibtex
@misc{hai2026holo31,
      title={Holo3.1: Fast & Local Computer Use Agents},
      author={H Company},
      year={2026},
      url={https://huggingface.co/Hcompany/Holo-3.1-4B},
}