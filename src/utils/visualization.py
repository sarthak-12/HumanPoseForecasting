from os.path import dirname, join
from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, uint8
from torchvision.io.image import write_png
from torchvision.transforms.functional import resize
from torchvision.utils import draw_keypoints, make_grid

# connection between joints for visualizing skeletons
H36M_SKELETON = [
    [0, 1],
    [1, 2],
    [2, 3],
    [0, 4],
    [4, 5],
    [5, 6],
    [0, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    [8, 11],
    [11, 12],
    [12, 13],
    [8, 14],
    [14, 15],
    [15, 16],
]

path = join(dirname(dirname(__file__)), 'images')
Path(path).mkdir(exist_ok=True)


# outputs a png image of multiple unnormalized skeletons to the `images` folder

def _write_skeleton_png(x: Tensor, output_path: str, input_size: int, output_size: int = 64):
    x = x.reshape(-1, 17, 2)
    skeletons = []
    for frame in x:
        frame = frame.T
        # rescaling the unnormalized joint coordinates to fit any image size
        frame[0] = frame[0] / input_size * output_size
        frame[1] = frame[1] / input_size * output_size
        skeletons.append(
            draw_keypoints(
                image=torch.zeros((3, output_size, output_size), dtype=uint8),
                keypoints=frame.T.unsqueeze(0),
                connectivity=H36M_SKELETON,
                colors='lime',
                radius=output_size // 256 + 1,
                width=output_size // 256 + 1,
            )
        )
    grid = make_grid(skeletons, nrow=10, padding=0)
    write_png(grid, output_path)


# outputs a png image of multiple heatmap frames to the `images` folder

def _write_heatmap_png(x: Tensor, output_path: str, output_size: int = 64):
    x = resize(x, output_size)
    x = x.reshape(-1, 17, output_size, output_size)
    # each frame is a gray-scale image of shape 1*h*w
    x = x.sum(1, keepdim=True)
    grid = make_grid(x, nrow=10, normalize=True, padding=0)
    write_png((grid * 255).to(uint8), output_path)


def visualize_predictions(
    model: LightningModule,
    sequence: Tensor,
    skeleton: bool,
    file_name: str,
    samples: int = 10,
    input_size: int = None,
    output_size: int = 64,
    unnormalize=None,
) -> str:
    output_path = join(path, file_name + '.png')
    seed = sequence[:10]
    ground_truth = sequence[10:]
    predictions = model.predict(seed, samples)
    output = torch.cat((seed, ground_truth, predictions))

    if skeleton:
        output = unnormalize(output)
        _write_skeleton_png(
            output, output_path, input_size, output_size)
    else:
        _write_heatmap_png(output, output_path, output_size)

    return output_path
