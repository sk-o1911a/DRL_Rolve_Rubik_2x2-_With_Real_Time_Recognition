import torch
import warnings
import logging
logging.getLogger("torch.onnx").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from Resnet import ResNetPolicyValueNet

device = "cpu"
model = ResNetPolicyValueNet(hidden_dim=192, num_res_blocks=8, num_actions=9).to(device)
state = torch.load("pbt_checkpoints/best_latest.pt", map_location=device)

clean_state = {}
for k, v in state.items():
    clean_state[k.replace("_orig_mod.", "")] = v
model.load_state_dict(clean_state)
model.eval()

dummy_input = torch.randn(1, 144).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "rubik2x2.onnx",
    export_params=True,
    opset_version=18,          # dùng 18 — phiên bản PyTorch mới nhất hỗ trợ
    do_constant_folding=True,
    input_names=['input'],
    output_names=['policy', 'value'],
    dynamic_axes={
        'input':  {0: 'batch_size'},
        'policy': {0: 'batch_size'},
        'value':  {0: 'batch_size'},
    }
)
print("Done rubik2x2.onnx")
