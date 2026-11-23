"""Native inference interfaces; callers supply inputs, never labels or references."""
import json
from pathlib import Path
import sys


PROMPT = ('Infer possible emotions experienced by the participant using unrestricted emotion '
          'vocabulary. The stimulus video is context, not a recording of the participant. '
          'EEG measurements do not uniquely establish a specific emotion. ')


def internvl_tiles(image, maximum=6, size=448):
    from PIL import Image
    image = image.convert('RGB')
    width, height = image.size
    candidates = sorted({(columns, rows) for columns in range(1, maximum + 1)
                         for rows in range(1, maximum + 1) if columns * rows <= maximum},
                        key=lambda grid: (grid[0] * grid[1], grid[0], grid[1]))
    best, difference = (1, 1), float('inf')
    for grid in candidates:
        distance = abs(width / height - grid[0] / grid[1])
        if distance < difference or (distance == difference and width * height >
                0.5 * size * size * grid[0] * grid[1]):
            best, difference = grid, distance
    columns, rows = best
    resized = image.resize((size * columns, size * rows), Image.Resampling.BICUBIC)
    tiles = [resized.crop((column * size, row * size, (column + 1) * size, (row + 1) * size))
             for row in range(rows) for column in range(columns)]
    if len(tiles) > 1:
        tiles.append(image.resize((size, size), Image.Resampling.BICUBIC))
    return tiles


class InternVL35:
    def __init__(self, directory):
        self.prompt = PROMPT
        self.video_label = 'Stimulus frame'
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        model_path = Path(directory) / 'model'
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            use_flash_attn=False).eval().requires_grad_(False).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
            local_files_only=True, use_fast=False)

    def prepare_input(self, frames=(), atlases=()):
        from PIL import Image
        from torchvision import transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        images = [(f'{self.video_label} {number + 1}', Image.fromarray(frame), 1)
                  for number, frame in enumerate(frames)]
        images += [(f'EEG atlas {number + 1} (successive five-second intervals)', image, 6)
                   for number, image in enumerate(atlases)]
        if not images:
            raise ValueError('At least one image is required')
        counts, tensors, labels = [], [], []
        for label, image, maximum in images:
            tiles = internvl_tiles(image, maximum=maximum)
            counts.append(len(tiles))
            tensors.extend(transform(tile) for tile in tiles)
            labels.append(label + ': <image>')
        pixels = self.torch.stack(tensors).to(device='cuda', dtype=self.torch.bfloat16)
        question = '\n'.join(labels) + '\n' + self.prompt + 'Return only JSON {"emotions":["emotion expression", ...]}.'
        return pixels, counts, question

    def generate(self, frames=(), atlases=()):
        pixels, counts, question = self.prepare_input(frames, atlases)
        with self.torch.inference_mode():
            raw = self.model.chat(self.tokenizer, pixels, question,
                dict(do_sample=False, max_new_tokens=320), num_patches_list=counts)
        return dict(raw=raw, image_patch_counts=counts, prompt=question)


class R1Omni:
    def __init__(self, directory):
        self.prompt = PROMPT
        directory = Path(directory)
        assets = json.loads((directory / 'assets.json').read_text())
        if not (directory / 'compatibility.json').exists():
            raise RuntimeError('Run path-only compatibility preparation first')
        sys.path.insert(0, str(directory / 'runtime_source'))
        from humanomni import model_init, mm_infer
        from transformers import BertTokenizer
        self.model, self.processor, self.tokenizer = model_init(str(directory / 'runtime_model'),
            use_flash_attn=False, device_map={'': 0})
        self.model.eval().requires_grad_(False)
        self.bert = BertTokenizer.from_pretrained(assets['bert']['path'], local_files_only=True)
        self.infer = mm_infer

    def generate(self, frames, measurements=None):
        if len(frames) != 8:
            raise ValueError('Reuse exactly eight existing sampled frames')
        question = self.prompt
        if measurements is not None:
            if len(measurements) != 2 or any(len(row) != 5 for row in measurements):
                raise ValueError('Expected two five-band measurement rows')
            question += ('EEG log-band-power channel means for successive five-second windows '
                         '(delta, theta, alpha, beta, gamma): ' + json.dumps(measurements) + '. ')
        question += ('Output brief reasoning in <think> </think> and final emotions in '
                     '<answer> </answer>. Use comma-separated emotion expressions; [] is allowed.')
        tensor = self.processor['video'](frames)
        raw = self.infer(tensor, question, model=self.model, tokenizer=self.tokenizer,
            audio=None, modal='video', question=question, bert_tokeni=self.bert,
            do_sample=False, max_new_tokens=512)
        return dict(raw=raw, prompt=question, audio_input=False)
