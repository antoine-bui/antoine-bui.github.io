"""Rebuild models/tracknet.onnx from the official TrackNetV3 checkpoint (MIT, github.com/qaz812345/TrackNetV3).

Usage:
    git clone https://github.com/qaz812345/TrackNetV3   # for model.py
    # download TrackNetV3_ckpts.zip (link in its README) and unzip it
    pip install torch onnx onnxconverter-common onnxruntime
    python export_tracknet.py <TrackNetV3 repo dir> <TrackNet_best.pt> <out.onnx>

Two changes vs. the original network, same maths:
  - the first conv takes 32 input channels instead of 27 (5 extra zero weights): onnxruntime-web's WebGPU conv
    returned wrong heatmaps with 27 channels; the page feeds 5 zero channels.
  - weights in float16 (22.7 MB instead of 45 MB), inputs/outputs stay float32.
"""
import sys, os
import numpy as np
import torch, onnx
import onnxruntime as ort
from onnxconverter_common import float16

repo, ckpt_file, out = sys.argv[1:4]
sys.path.insert(0, repo)
from model import TrackNet  # noqa: E402

CH = 32
ck = torch.load(ckpt_file, map_location='cpu', weights_only=True)
assert ck['param_dict']['seq_len'] == 8 and ck['param_dict']['bg_mode'] == 'concat'
sd = dict(ck['model'])
w = sd['down_block_1.conv_1.conv.weight']  # (64, 27, 3, 3)
sd['down_block_1.conv_1.conv.weight'] = torch.cat([w, torch.zeros(64, CH - 27, 3, 3)], 1)
net = TrackNet(in_dim=CH, out_dim=8); net.load_state_dict(sd); net.eval()
ref = TrackNet(in_dim=27, out_dim=8); ref.load_state_dict(ck['model']); ref.eval()

x27 = torch.rand(1, 27, 288, 512)
x = torch.cat([x27, torch.zeros(1, CH - 27, 288, 512)], 1)
torch.onnx.export(net, x, out, input_names=['frames'], output_names=['heatmaps'], opset_version=17, dynamo=False)
onnx.save(float16.convert_float_to_float16(onnx.load(out), keep_io_types=True), out)

# self-check: the exported fp16 model matches the original 27-channel PyTorch model
with torch.no_grad():
    y_ref = ref(x27).numpy()
y = ort.InferenceSession(out, providers=['CPUExecutionProvider']).run(None, {'frames': x.numpy()})[0]
diff = float(np.abs(y - y_ref).max())
assert diff < 1e-2, diff
print(f'{out}: {os.path.getsize(out) / 1e6:.1f} MB, max abs diff vs PyTorch {diff:.5f}')
