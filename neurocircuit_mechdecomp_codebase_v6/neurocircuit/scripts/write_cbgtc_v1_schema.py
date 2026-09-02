from __future__ import annotations

import argparse
from neurocircuit.atlas.cbgtc_v1 import write_schema_bundle


def main() -> None:
    ap = argparse.ArgumentParser(description="Write canonical cbgtc_v1 ROI schema and anatomical graph masks.")
    ap.add_argument("--out-dir", default="atlases/cbgtc_v1")
    args = ap.parse_args()
    out = write_schema_bundle(args.out_dir)
    print(f"wrote cbgtc_v1 schema: {out}")


if __name__ == "__main__":
    main()
