---
library_name: transformers
pipeline_tag: image-text-to-text
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
tags:
- multimodal
- computer-use
- agent
---

# *Holo4*-35B-A3B

> [!IMPORTANT]
> *Holo4* family:
>
> - Holo4-27B: Qwen3.8 dense architecture. [BF16](https://huggingface.co/Hcompany/Holo4-27B), [FP8](https://huggingface.co/Hcompany/Holo4-27B-FP8), [NVFP4](https://huggingface.co/Hcompany/Holo4-27B-NVFP4), and [Q4 GGUF](https://huggingface.co/Hcompany/Holo4-27B-GGUF).
> - Holo4-35B-A3B: Qwen3.6 MoE architecture. [BF16](https://huggingface.co/Hcompany/Holo4-35B-A3B) [current], [FP8](https://huggingface.co/Hcompany/Holo4-35B-A3B-FP8), [NVFP4](https://huggingface.co/Hcompany/Holo4-35B-A3B-NVFP4), and [Q4 GGUF](https://huggingface.co/Hcompany/Holo4-35B-A3B-GGUF).
> - Holotron4-30B-A3B: NemotronH Nano Omni architecture. [BF16](https://huggingface.co/Hcompany/Holotron4-30B-A3B) and [FP8](https://huggingface.co/Hcompany/Holotron4-30B-A3B-FP8).

[![H Models API](https://img.shields.io/badge/H_Models_API-181717?style=for-the-badge)](https://hub.hcompany.ai/models-api/introduction)
[![Agent trajectories](https://img.shields.io/badge/Agent_trajectories-181717?style=for-the-badge)](https://trajectories.hcompany.ai/)

## Model summary

*Holo4*-35B-A3B is a vision-language model (VLM) for Computer Use, built on [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and developed by [H Company](https://hcompany.ai/). Used with the [hai-agents harness](https://github.com/hcompai/hai-agents-python), it can send screenshots and tool results to the model, then execute its requested clicks, typing, code, and tool calls.

<table width="100%">
<thead>
<tr style="border-bottom: 2px solid rgb(230, 126, 34);"><th style="background-color: rgb(255, 248, 239); color: #1f2937;">Specification</th><th style="background-color: rgb(255, 248, 239); color: #1f2937;">Value</th></tr>
</thead>
<tbody>
<tr><td style="background-color: white; color: #1f2937;">Model ID</td><td style="background-color: white; color: #1f2937;"><code>Hcompany/Holo4-35B-A3B</code></td></tr>
<tr><td style="background-color: white; color: #1f2937;">Architecture</td><td style="background-color: white; color: #1f2937;">Qwen3.6 MoE</td></tr>
<tr><td style="background-color: white; color: #1f2937;">Checkpoint format</td><td style="background-color: white; color: #1f2937;">BF16 safetensors</td></tr>
<tr><td style="background-color: white; color: #1f2937;">Maximum context length in config</td><td style="background-color: white; color: #1f2937;">262,144 tokens</td></tr>
</tbody>
</table>

<video src="https://assets.hcompanyprod.fr/holo4/eiffel-holo4.mp4" poster="https://assets.hcompanyprod.fr/holo4/eiffel-holo4-poster.jpg" controls muted loop playsinline width="100%"></video>

This demo shows *Holo4*-27B using FreeCAD to build a replica of the Eiffel Tower. More examples in the [blog post](https://hcompany.ai/newsroom/holo4).

<details>
<summary>Prompt</summary>

<blockquote>
<p>Build a 3D Eiffel Tower in FreeCAD at a scale of 1 mm per metre. Center it on the origin and align it with the X and Y axes. Its plan must stay square at every height. The distance from the center to each corner is 62.5 mm at ground level, 32.5 mm at height 57, 17.5 mm at height 115, and 9.35 mm at height 276. Connect these widths with a smooth curve that narrows quickly near the base and more slowly near the top.</p>

<p>Make four separate, identical square legs, one in each quadrant. Their outer corners follow that curve, and each leg narrows from 14 mm across at ground level to 4 mm at height 276. Leave the space between the legs open. Add centered square platforms measuring 72 mm by 72 mm by 4 mm at height 57, 40 mm by 40 mm by 3 mm at height 115, and 22 mm by 22 mm by 3 mm at height 276. Add a square mast from height 276 to 324, tapering from 8 mm across to 2 mm across. Make every component a closed solid with nonzero volume, without filling the space between the legs.</p>
</blockquote>

</details>

## Usage

The harness sends screenshots and tool results to *Holo4*, executes the model's requested actions, and sends the results back. It can give the model access to application tools and code execution.

Refer to the documentation for more details about:

- [Function calling](https://hub.hcompany.ai/models-api/build-an-agent/function-calling)
- [Element localization](https://hub.hcompany.ai/models-api/element-localization)
- [Document OCR](https://hub.hcompany.ai/models-api/document-ocr)

## Performance

*Holo4* models improve significantly over their Qwen base models. On OSWorld, *Holo4*-27B scores 85.2% at $0.08 per task. We also evaluate them on Agentic Task Factory, a set of held-out business workflows across web, desktop, and MCP tools.

### Benchmark results

![Holo4 benchmark comparison table](https://assets.hcompanyprod.fr/holo4/hf/holo4-benchmark-table.png)

Read the [blog post](https://hcompany.ai/newsroom/holo4) for more tables about the benchmark results.

### Pareto plots

These plots compare benchmark scores with cost per task. On OSWorld 2.0, *Holo4*-27B scores 61.7% at $1.22 per task, and *Holo4*-35B-A3B scores 30.9% at $0.61 per task.

![Cost versus score on OSWorld 2.0](https://assets.hcompanyprod.fr/holo4/hf/holo4-cost-osworld2.png)

On AutomationBench, *Holo4*-27B scores 45.4% at $0.05 per task, and *Holo4*-35B-A3B scores 34.5% at $0.02 per task.

![Cost versus score on AutomationBench](https://assets.hcompanyprod.fr/holo4/hf/holo4-cost-automationbench.png)

### Open-source evaluation traces

For transparency, we share all agent trajectories in the open-source dataset at [Hcompany/trajectories](https://huggingface.co/datasets/Hcompany/trajectories).

## Training

![Holo4 training pipeline](https://assets.hcompanyprod.fr/holo4/hf/holo4-training.png)

## License

The model weights are available under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). This model is built on [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), which Alibaba Cloud also releases under the Apache License 2.0.
