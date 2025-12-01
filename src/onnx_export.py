import sys
import argparse
import torch
import torch.nn as nn

from utils.color import get_color_bins, lab_to_rgb

from train import MODELS

class Pipeline(nn.Module):
    def __init__(self, model, bins):
        super().__init__()

        self.luma_mapping = torch.tensor([0.6, 0.83, 0.91, 0.97])
        self.model = model
        self.bins = bins

    def forward(self, input):
        luma = self.luma_mapping[input]

        pred = self.model(input.to(torch.float32) / 3.0)

        color = torch.softmax(pred, dim=1)
        color = torch.argmax(pred, dim=1)
        color = self.bins[color].movedim(3, 1)

        lab = torch.cat([luma, color], dim=1)

        rgb = (
            lab_to_rgb(lab.movedim(0, 1).reshape(3, -1))
            .view(3, input.shape[0], 112, 128)
            .movedim(1, 0)
        )

        return rgb


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--onnx_dest", type=str, required=True)
    args = parser.parse_args()

    model = MODELS[args.model].load_from_checkpoint(args.ckpt_path)

    pipeline = Pipeline(model, get_color_bins())

    example_input = (torch.randint(0, 3, (1, 1, 112, 128), dtype=torch.int64),)
    torch.onnx.export(
        pipeline,
        example_input,
        args.onnx_dest,
        input_names=["gb_image"],
        output_names=["color_image"],
    )
