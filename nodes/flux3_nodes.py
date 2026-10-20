"""FLUX.3 prompt builder.

Same visual editor as the Ideogram 4 builder (web/js/ideogram4_prompt_builder.js, spec in
web/js/flux3_prompt_builder.js). FLUX.3 reads layout boxes as part of the prompt: the prompt text, one
space, then a JSON list of rows with [y_min, x_min, y_max, x_max] on a 0-1000 grid. With reference images
each row says where its object is in a reference (src_bbox) and where it goes in the output (tgt_bbox).

The prompt upsampler keeps boxes only in its box mode (t2i_bbox / i2i_bbox system prompts) with the user
turn from _upsampler_input(); Flux3BoxReconcileKJ then re-imposes the user's boxes on its answer.
"""

import json
import math
import re

import torch

from comfy.utils import common_upscale
from comfy_api.latest import io

from .ideogram4_nodes import _render_preview, _parse_json_list, _norm_bbox, _seed_boxes, _bbox_outputs, _bg_image


MAX_REFS = 10
ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
LOCKED_HEADER = ("LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at "
                 "the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, "
                 "provenance value, null, and coordinate exactly:")
I2I_TURN = ("Reference images follow in order (the first image, the second image, ...).\n"
            "User request: {instruction}\n"
            "Output the caption.")
I2I_BBOX_TURN = ("Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
                 "User editing instruction: {instruction}\n"
                 "Decide the edit, imagine the target image, and output the annotation record.")


def _upsampler_input(prompt, rows, has_refs):
    # The upsampler's user turn: box mode when there are rows (pair with the t2i_bbox / i2i_bbox system prompt)
    if rows:
        text = f"ORIGINAL PROMPT:\n{prompt.strip()}\n\n{LOCKED_HEADER}\n{json.dumps(rows, ensure_ascii=False)}"
        return I2I_BBOX_TURN.format(instruction=text) if has_refs else text
    return I2I_TURN.format(instruction=prompt) if has_refs else prompt


def _ref_slots(references):
    # {"ref_3": IMAGE, ...} -> connected slot numbers, sorted = the order the model receives the references
    slots = []
    for name, ref in (references or {}).items():
        k = name.rsplit("_", 1)[-1]
        if ref is not None and k.isdigit():
            slots.append(int(k))
    return sorted(slots)


def _rows(boxes, slots):
    # Editor boxes (one per id and canvas; view 0 = output, view k = reference input ref_k) -> FLUX.3 rows.
    # Mirrored by rows() in flux3_prompt_builder.js.
    rank = {k: i for i, k in enumerate(slots)}
    order, src, tgt = [], {}, {}
    for b in boxes:
        if not isinstance(b, dict) or b.get("nobbox"):
            continue
        view = int(b.get("view") or 0)
        if view and view not in rank:                        # its reference is no longer connected
            continue
        rid = str(b.get("id") or "")
        if rid not in src and rid not in tgt:
            order.append(rid)
        (src if view else tgt).setdefault(rid, b)
    rows = []
    for rid in order:
        s, t = src.get(rid), tgt.get(rid)
        desc = (t or {}).get("desc") or (s or {}).get("desc") or ""
        if slots:
            rows.append({"id": rid, "from": f"ref_image_{rank[int(s['view'])]}" if s else "target",
                         "src_bbox": _norm_bbox(s) if s else None, "tgt_bbox": _norm_bbox(t) if t else None, "desc": desc})
        elif t:
            rows.append({"id": rid, "bbox": _norm_bbox(t), "desc": desc})
    return rows


def _split_caption(text):
    # "prompt [rows]" -> (prompt, rows): the last "[" from which the rest parses as a JSON list
    for i in range(len(text) - 1, -1, -1):
        if text[i] != "[":
            continue
        try:
            rows = json.loads(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(rows, list):
            return text[:i].rstrip(), rows
    return text.strip(), []


def _grid_box(bb):
    # [y_min, x_min, y_max, x_max] on the 0-1000 grid -> editor {x, y, w, h} fractions, or None
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return None
    try:
        y1, x1, y2, x2 = (float(v) / 1000 for v in bb)
    except (TypeError, ValueError):
        return None
    return {"x": min(x1, x2), "y": min(y1, y2), "w": abs(x2 - x1), "h": abs(y2 - y1)}


def _rows_to_boxes(rows):
    # FLUX.3 rows -> editor boxes; ref_image_N goes to the canvas of input ref_{N+1}
    boxes = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        base = {"id": str(r.get("id") or f"obj_{i + 1}"), "desc": str(r.get("desc") or "")}
        origin = str(r.get("from") or "")
        src = _grid_box(r.get("src_bbox"))
        if src and origin.startswith("ref_image_") and origin[10:].isdigit():
            boxes.append({**src, **base, "view": int(origin[10:]) + 1})
        tgt = _grid_box(r["tgt_bbox"] if "tgt_bbox" in r else r.get("bbox"))
        if tgt:
            boxes.append({**tgt, **base, "view": 0})
    return boxes


def _valid_box(box):
    # A well-formed box on the 0-1000 grid, else None (all values <= 1 = a 0-1 box, not the grid)
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        return None
    y1, x1, y2, x2 = box
    if y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1 or y2 > 1000 or x2 > 1000 or max(box) <= 1:
        return None
    return list(box)


def _iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _row_box(row):
    return row.get("bbox") or row.get("tgt_bbox") or row.get("src_bbox")


def _valid_added(row, caption):
    # A row the upsampler added: kept only when well formed and cited in the caption as <id>
    if not isinstance(row, dict):
        return None
    rid, desc = row.get("id"), row.get("desc")
    if not isinstance(rid, str) or not ID_RE.fullmatch(rid) or f"<{rid}>" not in caption or not isinstance(desc, str) or not desc.strip():
        return None
    box = _valid_box(row.get("bbox"))
    if box is not None:
        return {"id": rid, "bbox": box, "desc": desc.strip()}
    origin, src, tgt = row.get("from"), row.get("src_bbox"), row.get("tgt_bbox")
    src_box = _valid_box(src) if src is not None else None
    tgt_box = _valid_box(tgt) if tgt is not None else None
    if (not isinstance(origin, str) or not re.fullmatch(r"ref_image_\d+|target", origin)
            or (src is not None and src_box is None) or (tgt is not None and tgt_box is None)
            or (src_box is None and tgt_box is None) or (origin == "target") == (src_box is not None)):
        return None
    return {"id": rid, "from": origin, "src_bbox": src_box, "tgt_bbox": tgt_box, "desc": desc.strip()}


def _reconcile(text, locked):
    # The upsampler's answer with the user's rows re-imposed: a row re-planning a locked box under a new id
    # (IoU >= 0.9, e.g. balloon_1 for obj_1) takes the locked id in the rows and the caption's <id> citations;
    # locked rows are kept verbatim with the upsampler's desc, then its own valid rows follow.
    caption, rows = _split_caption(text)
    locked_ids = {r["id"] for r in locked}
    row_ids = {r.get("id") for r in rows if isinstance(r, dict)}
    renamed = {}
    for lock in locked:
        lock_box = _row_box(lock)
        if lock["id"] in row_ids or lock_box is None:
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("id") in renamed or row.get("id") in locked_ids:
                continue
            box = _valid_box(_row_box(row))
            if box is not None and _iou(box, lock_box) >= 0.9:
                renamed[row["id"]] = lock["id"]
                break
    for row in rows:
        if isinstance(row, dict) and row.get("id") in renamed:
            row["id"] = renamed[row["id"]]
    for old, new in renamed.items():
        caption = caption.replace(f"<{old}>", f"<{new}>")
    by_id = {r.get("id"): r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)}
    out = []
    for lock in locked:
        desc = by_id.get(lock["id"], {}).get("desc")
        out.append({**lock, "desc": desc.strip() if isinstance(desc, str) and desc.strip() else lock.get("desc", "")})
    for row in rows:
        added = _valid_added(row, caption)
        if added is not None and added["id"] not in {r["id"] for r in out}:
            out.append(added)
    return f"{caption} {json.dumps(out, ensure_ascii=False)}"


class Flux3PromptBuilderKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Flux3PromptBuilderKJ",
            display_name="FLUX.3 Prompt Builder KJ",
            category="KJNodes/text",
            search_aliases=["flux3", "flux 3", "bbox", "layout", "prompt builder"],
            is_experimental=True,
            description="""
Visual layout builder for FLUX.3: boxes are appended to the prompt as a JSON table on a 0-1000 grid.

Draw boxes on the output canvas; give each an id and a description. Cite an id as <id> in the
prompt to tie words to its box.

With reference images (same order as the references given to the model, ref_1 first): each
reference gets its own canvas (toolbar view selector). A box drawn on a reference marks where an
object is in it; select it, switch to output and draw again to place that object there. A box only
on a reference = removed, only on the output = new.

Canvas controls are the same as in the Ideogram 4 builder; right-click lists the boxes of the
current canvas.""",
            inputs=[
                io.Int.Input("width", default=1024, min=64, max=16384, step=16,
                             tooltip="Output canvas width (the aspect the boxes are drawn on). Multiples of 16."),
                io.Int.Input("height", default=1024, min=64, max=16384, step=16,
                             tooltip="Output canvas height (the aspect the boxes are drawn on). Multiples of 16."),
                io.String.Input("prompt", multiline=True, default="",
                                tooltip="The prompt text; the box table is appended after it. Cite boxes as <id>."),
                io.Image.Input("image", optional=True,
                               tooltip="Optional background for the output canvas (and behind the preview)."),
                io.Autogrow.Input("references", template=io.Autogrow.TemplateNames(
                    io.Image.Input("image"), names=[f"ref_{i}" for i in range(1, MAX_REFS + 1)], min=0),
                    tooltip="The reference images, in the order the model gets them (ref_1 = ref_image_0). "
                            "Each gets its own canvas in the editor."),
                io.String.Input("import_caption", default="", optional=True, force_input=True,
                                tooltip="Optional: a FLUX.3 caption (prompt + box table), e.g. the prompt upsampler's "
                                        "output. When connected, it loads into the editor and drives the output per 'import_mode'."),
                io.String.Input("elements_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized boxes from the editor (managed by the node UI)."),
                io.Int.Input("bg_brightness", default=25, min=0, max=100, socketless=True, advanced=True,
                             tooltip="Background image brightness % (managed by the node UI slider)."),
                io.Combo.Input("import_mode", options=["when empty", "always"], default="when empty",
                               tooltip="How a wired import_caption is used: 'when empty' only seeds the editor while "
                                       "it has no boxes (then the editor wins, so you can edit); 'always' makes "
                                       "the wired caption authoritative."),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True,
                                     tooltip="Optional pixel-space boxes ({x, y, width, height}) used to seed the output "
                                             "canvas when the editor has no boxes."),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Image.Output(display_name="preview"),
                io.BoundingBox.Output(display_name="bboxes"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.String.Output(display_name="upsampler_input",
                                 tooltip="The prompt upsampler's user turn. With boxes: box mode, use the t2i_bbox (or with "
                                         "references i2i_bbox) system prompt, then FLUX.3 Box Reconcile KJ on its answer."),
            ],
        )

    @classmethod
    def execute(cls, width, height, prompt="", elements_data="", import_caption="", import_mode="when empty",
                bboxes=None, image=None, bg_brightness=25, references=None) -> io.NodeOutput:
        if import_mode not in ("when empty", "always"):
            import_mode = "when empty"
        boxes = _parse_json_list(elements_data)
        boxes_seeded = False
        if not boxes and bboxes:
            boxes = [{**b, "id": f"obj_{i + 1}", "view": 0} for i, b in enumerate(_seed_boxes(bboxes, width, height, {"desc": ""}))]
            boxes_seeded = bool(boxes)

        used_import = bool(import_caption and import_caption.strip()) and (import_mode == "always" or not boxes)
        if used_import:
            prompt, imported_rows = _split_caption(import_caption)
            boxes = _rows_to_boxes(imported_rows)

        slots = _ref_slots(references)
        rows = _rows(boxes, slots)
        caption = f"{prompt.rstrip()} {json.dumps(rows, ensure_ascii=False)}" if rows else prompt

        out_boxes = [b for b in boxes if isinstance(b, dict) and not b.get("view")]
        preview = _render_preview(out_boxes, width, height, _bg_image(image), bg_brightness)
        ui = {"dims": [width, height]}
        if boxes_seeded:
            ui["boxes"] = [json.dumps(boxes)]
        if used_import:                                       # mirror the import in the editor (only when used)
            ui["caption"] = [import_caption]
        return io.NodeOutput(caption, preview, _bbox_outputs(out_boxes, width, height), width, height,
                             _upsampler_input(prompt, rows, bool(slots)), ui=ui)


def _pack_collage(sizes, canvas_width, canvas_height):
    gap = 32
    available_width, available_height = canvas_width - gap, canvas_height - gap

    def layout(scale):
        patches = {i: (max(1, round(w * scale)), max(1, round(h * scale))) for i, w, h in sizes}
        ordered = sorted(((i, w + gap, h + gap) for i, (w, h) in patches.items()), key=lambda item: item[2], reverse=True)
        area = sum(w * h for _, w, h in ordered)
        widest = max(w for _, w, _ in ordered)
        root = math.sqrt(area)
        widths = {widest, sum(w for _, w, _ in ordered), available_width}
        widths.update(max(widest, round(root * factor)) for factor in (1, 1.25, 1.5, 2, 3))
        best = None
        for limit in widths:
            if limit < widest or limit > available_width:
                continue
            shelves = []
            positions = {}
            for index, w, h in ordered:
                shelf = next((s for s in shelves if s[0] + w <= limit), None)
                if shelf is None:
                    shelf = [0, h, sum(s[1] for s in shelves)]
                    shelves.append(shelf)
                positions[index] = (shelf[0], shelf[2])
                shelf[0] += w
            width = max(s[0] for s in shelves)
            height = sum(s[1] for s in shelves)
            if height <= available_height and (best is None or (width * height, max(width, height)) <
                                               (best[0] * best[1], max(best[0], best[1]))):
                best = width, height, positions, patches
        return best

    best = layout(1.0)
    if best is None:
        best = layout(0.0)
        if best is None:
            raise ValueError("Canvas too small to fit collage elements with 32 pixel gaps")
        low, high = 0.0, 1.0
        for _ in range(18):
            mid = (low + high) / 2
            candidate = layout(mid)
            if candidate is None:
                high = mid
            else:
                low, best = mid, candidate
    return best


class Flux3BBoxCollageKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Flux3BBoxCollageKJ",
            display_name="FLUX.3 BBox Collage KJ",
            category="KJNodes/image",
            description="Pack cropped references at their target-box sizes with 32 pixel gaps and margins. "
                        "Per-reference scale factors adjust those sizes before fitting; the rewritten prompt uses ref_image_0.",
            inputs=[
                io.String.Input("prompt", force_input=True, tooltip="FLUX.3 Prompt Builder prompt, including its box table."),
                io.Int.Input("width", default=1024, min=1, max=16384, force_input=True),
                io.Int.Input("height", default=1024, min=1, max=16384, force_input=True),
                io.Color.Input("background_color", default="#808080"),
                io.Autogrow.Input("images", template=io.Autogrow.TemplatePrefix(
                    io.Image.Input("image"), prefix="ref_image_", min=0, max=MAX_REFS,
                ), tooltip="References in the model's order: ref_image_0 is the first connected builder reference. "
                           "The first frame of each image is used."),
                *[io.Float.Input(f"ref_image_{i}_scale", default=1.0, min=0.0, step=0.05, socketless=True,
                                 tooltip=f"Size multiplier for ref_image_{i} (default 1.0).") for i in range(MAX_REFS)],
            ],
            outputs=[io.String.Output(display_name="prompt"), io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, prompt, width, height, background_color="#808080", images=None, **scales) -> io.NodeOutput:
        color = background_color.removeprefix("#")
        if len(color) not in (6, 8) or any(c not in "0123456789abcdefABCDEF" for c in color):
            raise ValueError("Background color must be #RRGGBB or #RRGGBBAA")
        rgb = [int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        caption, rows = _split_caption(prompt)
        images = images or {}
        patches = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not str(row.get("from", "")).startswith("ref_image_"):
                continue
            origin = row["from"]
            image = images.get(origin)
            if image is None:
                raise ValueError(f"Missing collage input {origin} for {row.get('id', 'box')}")
            sy1, sx1, sy2, sx2 = row["src_bbox"]
            ih, iw = image.shape[1:3]
            x1, x2 = max(0, round(sx1 * iw / 1000)), min(iw, round(sx2 * iw / 1000))
            y1, y2 = max(0, round(sy1 * ih / 1000)), min(ih, round(sy2 * ih / 1000))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Empty source box for {row.get('id', 'box')}")
            tgt = row.get("tgt_bbox")
            factor = scales.get(f"{origin}_scale", 1.0)
            w = (tgt[3] - tgt[1]) * width / 1000 if tgt else x2 - x1
            h = (tgt[2] - tgt[0]) * height / 1000 if tgt else y2 - y1
            patches.append((index, image, (x1, y1, x2, y2), max(1, round(w * factor)), max(1, round(h * factor))))

        first_image = patches[0][1] if patches else None
        bg = torch.tensor(rgb, device=first_image.device if first_image is not None else "cpu",
                          dtype=first_image.dtype if first_image is not None else torch.float32)
        canvas = bg.reshape(1, 1, 3).expand(height, width, 3).clone()
        if not patches:
            return io.NodeOutput(prompt, canvas.unsqueeze(0))
        cw, ch, positions, patch_sizes = _pack_collage([(i, w, h) for i, _, _, w, h in patches], width, height)
        offset_x = (width - cw + 32) // 2
        offset_y = (height - ch + 32) // 2

        for index, image, (x1, y1, x2, y2), w, h in patches:
            x, y = positions[index]
            pw, ph = patch_sizes[index]
            left, top = offset_x + x, offset_y + y
            right, bottom = left + pw, top + ph
            crop = image[:1, y1:y2, x1:x2, :3].movedim(-1, 1).to(canvas.device)
            fit = min(pw / (x2 - x1), ph / (y2 - y1))
            rw, rh = max(1, min(pw, round((x2 - x1) * fit))), max(1, min(ph, round((y2 - y1) * fit)))
            resized = common_upscale(crop, rw, rh, "bilinear", "disabled").movedim(1, -1)[0]
            canvas[top + (ph - rh) // 2:top + (ph - rh) // 2 + rh,
                   left + (pw - rw) // 2:left + (pw - rw) // 2 + rw] = resized
            rows[index] = {**rows[index], "from": "ref_image_0",
                           "src_bbox": [round(top * 1000 / height), round(left * 1000 / width),
                                        round(bottom * 1000 / height), round(right * 1000 / width)]}

        rewritten = f"{caption} {json.dumps(rows, ensure_ascii=False)}" if rows else prompt
        return io.NodeOutput(rewritten, canvas.unsqueeze(0))


class Flux3BoxReconcileKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Flux3BoxReconcileKJ",
            display_name="FLUX.3 Box Reconcile KJ",
            category="KJNodes/text",
            search_aliases=["flux3", "flux 3", "bbox", "upsampler", "reconcile"],
            is_experimental=True,
            description="Puts the user's boxes back into the prompt upsampler's box-mode answer: boxes it re-planned "
                        "under a new id get the original id back (also in the caption's <id> citations), the user's "
                        "rows keep their exact coordinates with the upsampler's descriptions, and the rows it added "
                        "follow when they are well formed and cited in the caption.",
            inputs=[
                io.String.Input("caption", force_input=True,
                                tooltip="The prompt upsampler's answer (TextGenerate text output)."),
                io.String.Input("prompt", force_input=True,
                                tooltip="The FLUX.3 Prompt Builder's prompt output: its box table is the locked rows."),
            ],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, caption, prompt) -> io.NodeOutput:
        locked = [r for r in _split_caption(prompt)[1] if isinstance(r, dict) and isinstance(r.get("id"), str)]
        return io.NodeOutput(_reconcile(caption.strip(), locked) if locked else caption.strip())
