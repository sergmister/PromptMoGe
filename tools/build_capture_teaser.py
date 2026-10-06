"""Create the website's depth maps and RGB point clouds from one demo capture.

Usage: python tools/build_capture_teaser.py --capture CAPTURE_DIR --promptda DEPTH_NPY --output OUTPUT_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from matplotlib import colormaps


def build(capture, promptda, output):
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((capture / 'meta.json').read_text())
    rgb = cv2.cvtColor(cv2.imread(str(capture / 'rgb.jpg')), cv2.COLOR_BGR2RGB)
    lidar = np.fromfile(capture / 'lidar_depth.bin', np.float32).reshape(meta['lidar'])
    model = meta['models']['A']
    shape = (model['height'], model['width'])
    depth = np.fromfile(capture / 'A_depth_k3.bin', np.float32).reshape(shape)
    mask = np.fromfile(capture / 'A_mask.bin', np.float32).reshape(shape) > 0
    sources = [('arkit', lidar, None), ('promptmoge-a-k3', depth, mask), ('promptda', np.load(promptda), None)]
    palette = (colormaps['turbo'](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    k = np.asarray(meta['intrinsics'])
    ih, iw = meta['image']
    report = {'capture': capture.name, 'date': meta['date'], 'depth_range_m': [0, 6],
              'camera_intrinsics': meta['intrinsics'], 'image_hw': meta['image'],
              'point_encoding': 'uint32 little-endian count; records: int16 little-endian XYZ millimetres, uint8 RGB (9 bytes per point)',
              'point_coordinates': '+X right, +Y up, -Z forward; camera-intrinsic back-projection',
              'promptda': {'model': 'PromptDA-L', 'input_hw': [756, 1008], 'prompt': 'Saved LiDAR depth with Euclidean-nearest fill of missing returns'},
              'sources': {}}
    for name, d, validity in sources:
        h, w = d.shape
        valid = np.isfinite(d) & (d > 0)
        if validity is not None:
            valid &= validity
        indices = np.clip(np.round(np.nan_to_num(d) / 6 * 255), 0, 255).astype(np.uint8)
        visual = palette[indices]
        visual[~valid] = [8, 11, 16]
        # Preserve sensor pixels; display image uses nearest-neighbour sampling.
        cv2.imwrite(str(output / f'{name}-depth.png'), visual[:, :, ::-1])
        step = max(1, int(np.ceil(np.sqrt(h * w / 100000))))
        y, x = np.mgrid[0:h:step, 0:w:step]
        keep = valid[y, x]
        x, y = x[keep], y[keep]
        z = d[y, x]
        xyz = np.stack(((x - k[0, 2] * w / iw) / (k[0, 0] * w / iw) * z,
                        -(y - k[1, 2] * h / ih) / (k[1, 1] * h / ih) * z, -z), axis=-1)
        assert np.isfinite(xyz).all() and np.max(np.abs(xyz)) < 32.767
        rx = np.minimum(iw - 1, ((x + .5) * iw / w).astype(int))
        ry = np.minimum(ih - 1, ((y + .5) * ih / h).astype(int))
        records = np.empty(len(z), dtype=[('xyz', '<i2', (3,)), ('rgb', 'u1', (3,))])
        records['xyz'] = np.round(xyz * 1000).astype('<i2')
        records['rgb'] = rgb[ry, rx]
        payload = np.array([len(records)], dtype='<u4').tobytes() + records.tobytes()
        (output / f'{name}-cloud.bin').write_bytes(payload)
        report['sources'][name] = {'depth_hw': [h, w], 'sample_step': step, 'points': len(records),
                                   'sha256': hashlib.sha256(payload).hexdigest()}
        print(name, f'{len(records):,} points', len(payload), 'bytes')
    report['orbit_distance_m'] = float(np.median(lidar[np.isfinite(lidar) & (lidar > 0)]))
    report['focal_y_normalized'] = float(2 * k[1, 1] / ih)
    (output / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--promptda', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.capture, args.promptda, args.output)
