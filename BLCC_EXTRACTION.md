# Extracting BLCCs (LFCCs and MFCCs) for FVC experiments

Based on the selected full-band CCs stored as `.npz` files in the three batch directories (for example, within `male_int_selected_ccs_16k_n12000`), BLCCs are computed for the specified sub-band. The resulting BLCCs are saved as `.npz` files in the corresponding three batch directories (for example, within `male_int_selected_blccs_16k_n12000`).

1. The BLCC extraction script, for example, takes `0001(1)_int_LFCC_N12000.npz` from the root directory of `male_int_selected_ccs_16k_n12000`, computes BLCCs for a specified sub-band (e.g., 0–600 Hz), and saves the output as `0001(1)_int_LFCC_N12000_omega1_0_omega2_600.npz` in the root directory of `male_int_selected_blccs_16k_n12000`.

## Project structure
The following scripts were used for the above processes:

```
.
|---fvc_blcc_ivector/
|    |---australian_english_database/
|          |---male_int_selected_ccs_16k_n12000/        # Root directory for CCs
|                |---round1/
|                     |---batch1/
|                          |---0001(1)_int_LFCC_N12000.npz
|                          |---0001(2)_int_LFCC_N12000.npz
|                                 :
|                     |---batch2/
|                     |---batch3/
|
|    |---australian_english_database/
|          |---male_int_selected_blccs_16k_n12000/      # Root director for BLCCs  
|                |---round1/
|                     |---batch1/
|                          |---0001(1)_int_LFCC_N12000_omega1_0_omega2_600.npz
|                          |---0001(2)_int_LFCC_N12000_omega1_0_omega2_600.npz
|                                 :
|                     |---batch2/
|                     |---batch3/
|
|    |---extract_blcc_utterances_vectorised.py            # Main code
|    |---run_extract_blcc_utterances_vectorised.sh        # For a single extraction (bash)
|    |---run_batch_extract_blcc_utterances_vectorised.py  # For a batch extraction
     
```

## Workflow

```
# Single extraction (bash)
./run_extract_blcc_utterances_vectorised.sh

# Batch extraction
python3 ./run_batch_extract_blcc_utterances_vectorised.py 
```