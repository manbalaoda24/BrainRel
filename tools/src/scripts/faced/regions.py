"""One region contract shared by rendering, prompts and measurement targets."""
import hashlib
import numpy as np
from PIL import Image, ImageDraw

REGIONS = {'UPPER_REGION': (0, 8, 0, 16), 'LOWER_REGION': (8, 16, 0, 16)}
THRESHOLD = .03
CANVAS = (832, 224)


def measured_target(panel):
    means = {}
    for name, (row_start, row_stop, col_start, col_stop) in REGIONS.items():
        means[name] = float(panel[row_start:row_stop, col_start:col_stop].mean())
    gap = means['UPPER_REGION'] - means['LOWER_REGION']
    target = 'SIMILAR' if abs(gap) <= THRESHOLD else ('FRONT' if gap > 0 else 'BACK')
    return target, gap, means


def handles(group, swapped=False):
    reverse = bool(int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 2) ^ swapped
    return dict(zip(REGIONS, ['R2', 'R1'] if reverse else ['R1', 'R2']))


def render(panels, band, group, names, marked=True, swapped=False):
    target, gap, means = measured_target(panels[band])
    mapping = handles(group, swapped)
    pixels = np.uint8(np.rint(np.stack([panels, 1 - np.abs(2 * panels - 1), 1 - panels], -1) * 255))
    image = Image.new('RGB', CANVAS, 'white')
    draw = ImageDraw.Draw(image)
    rectangles = {}
    for number, name in enumerate(names):
        left, top = number * 160 + 16, 48
        tile = Image.fromarray(pixels[number]).resize((128, 128), Image.Resampling.NEAREST)
        image.paste(tile, (left, top))
        draw.text((left, 12), name, fill='black')
        if number == band:
            for region, (row_start, row_stop, col_start, col_stop) in REGIONS.items():
                bounds = [left + col_start * 8, top + row_start * 8, left + col_stop * 8, top + row_stop * 8]
                rectangles[region] = bounds
                if marked:
                    x_right, y_first, y_last = bounds[2] + 2, bounds[1], bounds[3] - 1
                    draw.line([(x_right, y_first), (x_right + 2, y_first), (x_right + 2, y_last), (x_right, y_last)], fill='black')
                    draw.text((x_right + 4, (y_first + y_last) // 2 - 4), mapping[region], fill='black')
        assert np.array_equal(np.asarray(image)[top:top + 128, left:left + 128], np.asarray(tile))
    draw.text((8, 198), 'fixed scale; upper/lower image regions; not anatomical lobes', fill='black')
    expected_id = 'SIMILAR' if target == 'SIMILAR' else mapping['UPPER_REGION' if target == 'FRONT' else 'LOWER_REGION'] + '_HIGHER'
    return image, {'canonical_target': target, 'gap': gap, 'means': means, 'mapping': mapping, 'rectangles_xyxy_exclusive': rectangles, 'id_target': expected_id}
