# pyhazards/datasets/inspection.py  (或你想放的任何单文件位置，但不要在 __init__ 强制 import 它)

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

def inspect_wxc_rollout(
    date: str,
    outdir: str,
    *,
    time_range: Optional[tuple[str, str]] = None,
    device: Optional[str] = None,
) -> Path:
    """
    Run a minimal Prithvi-WxC rollout inspection and write a NetCDF prediction file.

    This function is SAFE TO IMPORT: it does not execute anything until called.
    """
    # Heavy/optional imports go INSIDE the function
    import random
    import numpy as np
    import torch
    import yaml
    import os
    import xarray as xr

    # ---- deterministic-ish setup (safe defaults) ----
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    # ---- device selection ----
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    # ---- user params ----
    yyyymmdd = date.replace("-", "")
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # NOTE: give a default time_range if not provided
    if time_range is None:
        # TODO: choose based on `date` if you want
        time_range = ("2024-01-01T00:00:00", "2024-01-01T18:00:00")

    # ---- your variable lists (copied from your file) ----
    surface_vars = [
        "EFLUX","GWETROOT","HFLUX","LAI","LWGAB","LWGEM","LWTUP","PS","QV2M","SLP",
        "SWGNT","SWTNT","T2M","TQI","TQL","TQV","TS","U10M","V10M","Z0M",
    ]
    static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
    vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
    levels = [34.0,39.0,41.0,43.0,44.0,45.0,48.0,51.0,53.0,56.0,63.0,68.0,71.0,72.0]
    padding = {"level": [0, 0], "lat": [0, -1], "lon": [0, 0]}

    variable_names = surface_vars + [f"{v}_level_{lv}" for v in vertical_vars for lv in levels]

    lead_time = 12
    input_time = -6
    positional_encoding = "fourier"

    # ---- paths (keep your relative layout, but do not assume cwd) ----
    # If this file sits inside pyhazards/, resolve relative to repo root if needed.
    # Here: assume user runs from repo root.
    surf_dir = Path("data/merra-2")
    vert_dir = Path("data/merra-2")
    clim_dir = Path("data/climatology")
    weights_path = Path("data/weights/prithvi.wxc.rollout.2300m.v1.pt")
    config_path = Path("data/config.yaml")

    # ---- Prithvi-WxC imports (also inside function) ----
    from PrithviWxC.dataloaders.merra2_rollout import Merra2RolloutDataset, preproc
    from PrithviWxC.dataloaders.merra2 import input_scalers, output_scalers, static_input_scalers
    from PrithviWxC.model import PrithviWxC
    from PrithviWxC.rollout import rollout_iter

    dataset = Merra2RolloutDataset(
        time_range=time_range,
        lead_time=lead_time,
        input_time=input_time,
        data_path_surface=surf_dir,
        data_path_vertical=vert_dir,
        climatology_path_surface=clim_dir,
        climatology_path_vertical=clim_dir,
        surface_vars=surface_vars,
        static_surface_vars=static_surface_vars,
        vertical_vars=vertical_vars,
        levels=levels,
        positional_encoding=positional_encoding,
    )
    if len(dataset) == 0:
        raise RuntimeError("No valid data found for the given time_range / paths.")

    surf_in_scal_path = clim_dir / "musigma_surface.nc"
    vert_in_scal_path = clim_dir / "musigma_vertical.nc"
    surf_out_scal_path = clim_dir / "anomaly_variance_surface.nc"
    vert_out_scal_path = clim_dir / "anomaly_variance_vertical.nc"

    in_mu, in_sig = input_scalers(surface_vars, vertical_vars, levels, surf_in_scal_path, vert_in_scal_path)
    out_sig = output_scalers(surface_vars, vertical_vars, levels, surf_out_scal_path, vert_out_scal_path)
    static_mu, static_sig = static_input_scalers(surf_in_scal_path, static_surface_vars)

    with open(config_path, "r") as f:
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
        static_input_scalers_epsilon=config["params"]["static_input_scalers_epsilon"],
        output_scalers=out_sig**0.5,
        n_lats_px=config["params"]["n_lats_px"],
        n_lons_px=config["params"]["n_lons_px"],
        patch_size_px=config["params"]["patch_size_px"],
        mask_unit_size_px=config["params"]["mask_unit_size_px"],
        mask_ratio_inputs=0.0,
        mask_ratio_targets=0.0,
        embed_dim=config["params"]["embed_dim"],
        n_blocks_encoder=config["params"]["n_blocks_encoder"],
        n_blocks_decoder=config["params"]["n_blocks_decoder"],
        mlp_multiplier=config["params"]["mlp_multiplier"],
        n_heads=config["params"]["n_heads"],
        dropout=config["params"]["dropout"],
        drop_path=config["params"]["drop_path"],
        parameter_dropout=config["params"]["parameter_dropout"],
        residual="climate",
        masking_mode="both",
        encoder_shifting=True,
        decoder_shifting=True,
        positional_encoding=positional_encoding,
        checkpoint_encoder=[],
        checkpoint_decoder=[],
    )

    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    model = model.to(dev).eval()

    data = next(iter(dataset))
    batch = preproc([data], padding)
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(dev)

    with torch.no_grad():
        out = rollout_iter(dataset.nsteps, model, batch)

    # ---- write a small set of outputs ----
    important_surface = ["T2M","QV2M","TQV","U10M","V10M","GWETROOT","TS","LAI","EFLUX","HFLUX","SWGNT","SWTNT","LWGAB","LWGEM"]

    out_np = out.detach().cpu().numpy().astype("float32")
    T, C, H, W = out_np.shape

    time_array = np.arange(T)
    lat = np.linspace(-90, 90, H)
    lon = np.linspace(-180, 180, W)

    data_vars = {}
    for name in important_surface:
        if name not in variable_names:
            continue
        ch = variable_names.index(name)
        data_vars[name] = xr.DataArray(out_np[:, ch, :, :], coords={"time": time_array, "lat": lat, "lon": lon}, dims=("time","lat","lon"))

    ds = xr.Dataset(data_vars)

    out_path = outdir_path / f"pred_{yyyymmdd}.nc"
    if out_path.exists():
        os.remove(out_path)

    ds.load()
    ds.to_netcdf(out_path, mode="w", engine="scipy")
    return out_path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default=None, help='e.g., "cuda", "cuda:0", "cpu"')
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    out_path = inspect_wxc_rollout(args.date, args.outdir, device=args.device)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
