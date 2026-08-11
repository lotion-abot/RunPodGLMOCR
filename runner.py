"""
Config MERGE tool. Runs INSIDE the venv (has yaml + glmocr). Called by pod_setup.sh:

    $VENV/bin/python /workspace/runner.py <overrides_yaml> <merged_yaml_out>

Takes the glmocr PACKAGE default config.yaml as the base (keeps label_task_mapping,
page-loader max_tokens, output format 'both', server section, etc. — native_label
parity depends on those defaults) and deep-merges our overrides on top, then writes
the merged file for `python -m glmocr.server --config <merged>`.

Without this, a hand-written minimal config loses label_task_mapping and EVERY
element comes back native_label='text' — which silently blinds the C# side's
title detection / dedup / header recovery.
"""
import os
import sys
import traceback


def _deep_merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def main():
    overrides_path, out_path = sys.argv[1], sys.argv[2]
    import glob as _glob

    import yaml

    import glmocr as _g

    with open(overrides_path) as f:
        overrides = yaml.safe_load(f) or {}

    pkg_dir = os.path.dirname(_g.__file__)
    base, base_src = None, None
    for cand in _glob.glob(os.path.join(pkg_dir, "**", "*.yaml"), recursive=True):
        try:
            with open(cand) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "pipeline" in data:
                base, base_src = data, cand
                break
        except Exception:
            continue

    if base is None:
        # Fail loudly: silently falling through to overrides-only is exactly the
        # "all labels become text" failure this file exists to prevent.
        raise RuntimeError(
            f"glmocr package default config.yaml NOT FOUND under {pkg_dir} — "
            "refusing to write an overrides-only config (would lose label_task_mapping)"
        )

    print(f"config base: {base_src}", file=sys.stderr)
    merged = _deep_merge(base, overrides)

    with open(out_path, "w") as f:
        yaml.safe_dump(merged, f)
    print(f"merged config written: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
