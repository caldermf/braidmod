# Data

This repo only tracks the small data files needed for the public story:

- `geordie_kernel.txt`
  A saved kernel word used in the public prefix-confusion examples.
- `geordie_kernel_gnf.json`
  The corresponding Garside factorization metadata.
- `generated/`
  Default local destination for regenerated training datasets.

The large Burau/Garside training corpora are intentionally not versioned in
GitHub. Recreate the main public dataset with:

```bash
bash jobs/build_reference_dataset.sh
```

or directly with:

```bash
.venv/bin/python generate_dataset.py \
  --output-path data/generated/burau_gnf_L30to60_p5_D140_N200000_uniform_corrected.json \
  --num-samples 200000 \
  --length-min 30 \
  --length-max 60 \
  --p 5 \
  --D 140
```
