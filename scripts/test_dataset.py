print("TEST SCRIPT STARTED")

import torch
from pyhazards.data.load_hydrograph_data import load_hydrograph_data

print("Imports OK")

bundle = load_hydrograph_data(
    era5_path="pyhazards/data/era5_subset",
    mesh_coords=mesh_coords,
)

print("Bundle built")
print(bundle)
print(bundle.feature_spec)
print("DONE")
