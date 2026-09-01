import argparse
import numpy as np

from neurocircuit.data.hemodynamics import load_hemodynamics_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hemodynamics", required=True)
    ap.add_argument("--n-regions", required=True, type=int)
    ap.add_argument("--n-timepoints", required=True, type=int)
    args = ap.parse_args()

    h = load_hemodynamics_npz(args.hemodynamics)
    h.validate(args.n_regions, args.n_timepoints)
    print("PASS: hemodynamic contract is valid")
    print(f"HRF shape: {h.hrf_kernel.shape}")
    print(f"Systemic waveform: {h.systemic_waveform.shape}")
    print(f"Vascular delay: {h.vascular_delay_seconds.shape}")
    print(f"Vascular amplitude: {h.vascular_amplitude.shape}")
    print(f"RAPIDTIDE R2 present: {h.rapidtide_r2 is not None}")


if __name__ == "__main__":
    main()
