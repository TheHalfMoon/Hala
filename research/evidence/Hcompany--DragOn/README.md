---

license: apache-2.0
pretty_name: DragOn
language:

- en
- fr
task_categories:
- image-to-text
tags:
- gui-agents
- computer-use
- drag-and-drop
- grounding
- vision-language
size_categories:
- 1M<n<10M

---

# DragOn: A Drag-Grounding Benchmark and Training Dataset for GUI Agents

Drag-grounding dataset for GUI agents. Each example = **one screenshot** + **one instruction** +
**start/end bounding boxes** for the drag. Four domains: `text_highlight`, `sheet` (cell
selection), `slide_resize` (element resize/rotate/crop), `slider`.

## Dataset statistics


| Domain            | Train images| Train tasks   | Eval (public) | Dominant resolution        |
| ----------------- | ----------- | ------------- | ------------- | -------------------------- |
| Text Highlighting | 100,000     | 1,000,000     | 250           | 1275×1650 (portrait)       |
| Cell Selection    | 38,332      | 996,632       | 250           | 1920×1080                  |
| Element Resizing  | 82,680      | 992,160       | 250           | 1280×720, 960×720, 960×540 |
| Slider            | 65,000      | 571,350       | 250           | 1280×800                   |
| **Total**         | **286,012** | **3,560,142** | **1,000**     | —                          |


A separate **1,000-example test set is kept private** as the reference benchmark for
cross-model comparison and is **not** included in this repository.

## Repository structure

```
DragOn/
├── train/                      # training data, one tar per domain
│   ├── text_highlight.tar      # <domain>/images/*.jpg + <domain>/tasks/*.json
│   ├── sheet.tar
│   ├── slide_resize.tar
│   └── slider.tar
└── eval/                       # public validation set (250 examples × 4 domains)
    ├── text_highlight/{images,tasks}/
    ├── sheet/{images,tasks}/
    ├── slide_resize/{images,tasks}/
    └── slider/{images,tasks}/
```

Each `images/<id>.jpg` has a matching `tasks/<id>.json`. Extract a training tar with:

```bash
tar -xf train/slider.tar -C train/
```

## Example schema

Each task JSON is a list of one or more examples:

```json
{
  "intent": "Highlight the first paragraph of the document",
  "start_bbox": [666, 232, 688, 254],
  "end_bbox":   [527, 238, 537, 248],
  "domain": "text_highlight",
  "subtype": null,
  "ordered": false,
  "image_id": "00f6cad5-a737-56a5-9533-4f13fdd3f3d5",
  "metadata": { "...": "domain-specific fields" }
}
```

- `start_bbox` / `end_bbox` — `[x0, y0, x1, y1]` in **image-pixel** coordinates.
- `ordered` — whether endpoint order matters. `false` for cell selection and text
highlighting (either drag direction yields the same selection); `true` for resizing,
rotation, and slider manipulation (the action is defined relative to the start handle).
- `metadata` — domain-specific context (e.g. cell contents, slider URL/variant/affordances).

## Evaluation notes

- **acc@5%** is the strict metric (target box = 5% of element size, ±2.5% tolerance);
**acc@10%** and **acc@15%** are reported as relaxed tolerances.
- Resize and rotation targets have an inherent degree of freedom (a line/arc of valid
points); train and eval use the same **canonical** convention to avoid mismatch.
- The public `eval/` split is for development and self-reported results; the private
test set is the reference for cross-model comparison.

## Citation

If you find this dataset or project useful, please consider citing our paper:

```bibtex
@misc{bout2026dragon,
  title        = {DragOn: A Drag-Grounding Benchmark and Training Dataset for GUI Agents},
  author       = {Bout, Nathan and Langevin, Maxime and Riochet, Ronan},
  year         = {2026},
  howpublished = {SCALE Workshop, 43rd International Conference on Machine Learning (ICML), Seoul, South Korea},
  url          = {https://huggingface.co/datasets/Hcompany/DragOn}
}
```
