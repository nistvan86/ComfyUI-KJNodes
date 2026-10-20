import math

import node_helpers
from comfy_api.latest import io
from comfy_extras.nodes_qwen import TextEncodeQwenImage21


def _parse_resolution_schedule(schedule):
    stages = []
    for entry in schedule.split(","):
        start_text, separator, resolution_text = entry.strip().partition(":")
        if not separator:
            raise ValueError("Resolution schedule entries must be start:resolution, e.g. 0:192,0.2:512")
        try:
            start = float(start_text)
            resolution = int(resolution_text.strip())
        except ValueError as e:
            raise ValueError(f"Invalid resolution schedule entry: {entry.strip()}") from e
        if not math.isfinite(start) or not 0 <= start < 1 or resolution < 0:
            raise ValueError(f"Invalid resolution schedule entry: {entry.strip()}")
        if stages and start <= stages[-1][0]:
            raise ValueError("Resolution schedule starts must be strictly increasing")
        stages.append((start, resolution))
    if not stages or stages[0][0] != 0:
        raise ValueError("Resolution schedule must start at 0")
    return stages


class TextEncodeQwenImage21ScheduledKJ(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = TextEncodeQwenImage21.define_schema()
        schema.node_id = "TextEncodeQwenImage21ScheduledKJ"
        schema.display_name = "Text Encode Qwen Image 2.1 Scheduled KJ"
        schema.category = "KJNodes/conditioning"
        schema.description = ("Encode the Qwen Image 2.1 prompt at different reference resolutions over sampling. "
                              "Each start:resolution applies until the next start; the latent output uses the first resolution.")
        schema.inputs = [io.String.Input("resolution_schedule", default="0:1024", tooltip="Sampling fractions and reference resolutions: 0:192,0.2:512,0.6:256. 0 resolution keeps the reference size.")
                         if inp.id == "resolution" else inp for inp in schema.inputs]
        return schema

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, resolution_schedule, vae=None, images=None) -> io.NodeOutput:
        stages = _parse_resolution_schedule(resolution_schedule)
        positive, negative = [], []
        latent = None
        for i, (start, resolution) in enumerate(stages):
            end = stages[i + 1][0] if i + 1 < len(stages) else 1.0
            pos, neg, stage_latent = TextEncodeQwenImage21.execute(
                clip, prompt, negative_prompt, vae=vae, resolution=resolution, images=images).result
            values = {"start_percent": start, "end_percent": end}
            positive.extend(node_helpers.conditioning_set_values(pos, values))
            negative.extend(node_helpers.conditioning_set_values(neg, values))
            if latent is None:
                latent = stage_latent
        return io.NodeOutput(positive, negative, latent)
