import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from datetime import datetime, timedelta

# Set backend etc.
torch.jit.enable_onednn_fusion(True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

# Set seeds
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda:3")
else:
    device = torch.device("cpu")

# Set variables
surface_vars = [
    "EFLUX",
    "GWETROOT",
    "HFLUX",
    "LAI",
    "LWGAB",
    "LWGEM",
    "LWTUP",
    "PS",
    "QV2M",
    "SLP",
    "SWGNT",
    "SWTNT",
    "T2M",
    "TQI",
    "TQL",
    "TQV",
    "TS",
    "U10M",
    "V10M",
    "Z0M",
]
static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
levels = [
    34.0,
    39.0,
    41.0,
    43.0,
    44.0,
    45.0,
    48.0,
    51.0,
    53.0,
    56.0,
    63.0,
    68.0,
    71.0,
    72.0,
]
padding = {"level": [0, 0], "lat": [0, -1], "lon": [0, 0]}

variable_names = surface_vars + [
    f'{var}_level_{level}' for var in vertical_vars for level in levels
]

lead_time = 12  # This variable can be change to change the task
input_time = -6  # This variable can be change to change the task
import argparse

# --------------------------
# 读入 --date 参数
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True, help="格式 YYYY-MM-DD，例如 2025-01-01")
parser.add_argument("--outdir", required=True, help="输出目录")
args = parser.parse_args()

DATE = args.date
OUTDIR = args.outdir

# 自动生成时间范围：00:00 到 18:00
time_range = ("2024-01-01T00:00:00", "2024-01-01T18:00:00")
print(f"[INFO] 使用 time_range = {time_range}")

surf_dir = Path("../data/merra-2")
vert_dir = Path("../data/merra-2")
surf_clim_dir = Path("../data/climatology")
vert_clim_dir = Path("../data/climatology")
positional_encoding = "fourier"

from PrithviWxC.dataloaders.merra2_rollout import Merra2RolloutDataset

dataset = Merra2RolloutDataset(
    time_range=time_range,
    lead_time=lead_time,
    input_time=input_time,
    data_path_surface=surf_dir,
    data_path_vertical=vert_dir,
    climatology_path_surface=surf_clim_dir,
    climatology_path_vertical=vert_clim_dir,
    surface_vars=surface_vars,
    static_surface_vars=static_surface_vars,
    vertical_vars=vertical_vars,
    levels=levels,
    positional_encoding=positional_encoding,
)
assert len(dataset) > 0, "There doesn't seem to be any valid data."

from PrithviWxC.dataloaders.merra2 import (
    input_scalers,
    output_scalers,
    static_input_scalers,
)

surf_in_scal_path = Path("../data/climatology/musigma_surface.nc")
vert_in_scal_path = Path("../data/climatology/musigma_vertical.nc")
surf_out_scal_path = Path("../data/climatology/anomaly_variance_surface.nc")
vert_out_scal_path = Path("../data/climatology/anomaly_variance_vertical.nc")

in_mu, in_sig = input_scalers(
    surface_vars,
    vertical_vars,
    levels,
    surf_in_scal_path,
    vert_in_scal_path,
)

output_sig = output_scalers(
    surface_vars,
    vertical_vars,
    levels,
    surf_out_scal_path,
    vert_out_scal_path,
)

static_mu, static_sig = static_input_scalers(
    surf_in_scal_path,
    static_surface_vars,
)

residual = "climate"
masking_mode = "both"
encoder_shifting = True
decoder_shifting = True
masking_ratio = 0.0

weights_path = Path("../data/weights/prithvi.wxc.rollout.2300m.v1.pt")

import yaml

from PrithviWxC.model import PrithviWxC

with open("../data/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = PrithviWxC(
    in_channels=config["params"]["in_channels"],
    input_size_time=config["params"]["input_size_time"],
    in_channels_static=config["params"]["in_channels_static"],
    input_scalers_mu=in_mu,
    input_scalers_sigma=in_sig,
    input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
    static_input_scalers_mu=static_mu,
    static_input_scalers_sigma=static_sig,
    static_input_scalers_epsilon=config["params"][
        "static_input_scalers_epsilon"
    ],
    output_scalers=output_sig**0.5,
    n_lats_px=config["params"]["n_lats_px"],
    n_lons_px=config["params"]["n_lons_px"],
    patch_size_px=config["params"]["patch_size_px"],
    mask_unit_size_px=config["params"]["mask_unit_size_px"],
    mask_ratio_inputs=masking_ratio,
    mask_ratio_targets=0.0,
    embed_dim=config["params"]["embed_dim"],
    n_blocks_encoder=config["params"]["n_blocks_encoder"],
    n_blocks_decoder=config["params"]["n_blocks_decoder"],
    mlp_multiplier=config["params"]["mlp_multiplier"],
    n_heads=config["params"]["n_heads"],
    dropout=config["params"]["dropout"],
    drop_path=config["params"]["drop_path"],
    parameter_dropout=config["params"]["parameter_dropout"],
    residual=residual,
    masking_mode=masking_mode,
    encoder_shifting=encoder_shifting,
    decoder_shifting=decoder_shifting,
    positional_encoding=positional_encoding,
    checkpoint_encoder=[],
    checkpoint_decoder=[],
)


state_dict = torch.load(weights_path, weights_only=False)
if "model_state" in state_dict:
    state_dict = state_dict["model_state"]
model.load_state_dict(state_dict, strict=True)

if (hasattr(model, "device") and model.device != device) or not hasattr(
    model, "device"
):
    model = model.to(device)

from PrithviWxC.dataloaders.merra2_rollout import preproc
from PrithviWxC.rollout import rollout_iter

data = next(iter(dataset))
batch = preproc([data], padding)

for k, v in data.items():
    if hasattr(v, "shape"):
        print(k, type(v), "shape=", v.shape)
    else:
        print(k, type(v), "value=", v)
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.to(device)

rng_state_1 = torch.get_rng_state()
with torch.no_grad():
    model.eval()
    out = rollout_iter(dataset.nsteps, model, batch)

import os
import xarray as xr
import numpy as np

# ---------------------------
# 1. 重要变量名
# ---------------------------
important_surface = [
    "T2M", "QV2M", "TQV", "U10M", "V10M",
    "GWETROOT", "TS", "LAI", "EFLUX",
    "HFLUX", "SWGNT", "SWTNT", "LWGAB", "LWGEM",
]

# ---------------------------
# 2. 输出路径
# ---------------------------
# -------------------------
# 自动命名输出文件
# -------------------------
yyyymmdd = DATE.replace("-", "")   # 2025-01-01 → 20250101
#out_nc_path = f"{OUTDIR}/pred_{yyyymmdd}_00.nc"
out_nc_path = f"{OUTDIR}/pred_20240101_18.nc"
print(f"[INFO] file save as {out_nc_path}")


# ---------------------------
# 3. 取出预测结果 (out: [T, C, H, W])
# ---------------------------
out_np = out.detach().cpu().numpy().astype("float32")  # 转成 float32 也更安全
T, C, H, W = out_np.shape
print(f"out shape: time={T}, channel={C}, lat={H}, lon={W}")

# 这里先用简单坐标，如果你有真实 lat/lon，再替换
time_array = np.arange(T)
lat = np.linspace(-90, 90, H)
lon = np.linspace(-180, 180, W)

# ---------------------------
# 4. 构建 Dataset
# ---------------------------
data_vars = {}

for name in important_surface:
    if name not in variable_names:
        print(f"[WARN] {name} 不在 variable_names 中，跳过。")
        continue

    ch_idx = variable_names.index(name)
    arr = out_np[:, ch_idx, :, :]  # [time, lat, lon]

    data_vars[name] = xr.DataArray(
        arr,
        coords={"time": time_array, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name=name,
    )

ds_pred = xr.Dataset(data_vars)

# ---------------------------
# 5. 写入 NetCDF 文件（只用 scipy one-shot）
# ---------------------------
if os.path.exists(out_nc_path):
    print(f"[INFO] delete old file: {out_nc_path}")
    os.remove(out_nc_path)

ds_pred.load()  # 确保不是 dask lazy
ds_pred.to_netcdf(out_nc_path, mode="w", engine="scipy")  # 固定用 scipy

print(f"[INFO] ✓ Prediction has been save to: {out_nc_path} (engine=scipy, NetCDF3)")
