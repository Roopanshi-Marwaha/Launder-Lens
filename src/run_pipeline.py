"""python -m src.run_pipeline [--quick] [--only amlsim,robustness,loto,temporal,lab]"""
import argparse, time
from . import experiments as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, coarser grids (a few minutes)")
    ap.add_argument("--only", default="amlsim,robustness,loto,temporal,lab")
    a = ap.parse_args()
    steps = a.only.split(","); t0 = time.time()
    if "amlsim" in steps:
        E.exp_amlsim()
    if "robustness" in steps:
        E.exp_robustness(seeds=(100,) if a.quick else (100, 200), c_grid=(0, .5, 1) if a.quick else (0, .25, .5, .75, 1))
    if "loto" in steps:
        E.exp_loto(seeds=(300,) if a.quick else (300, 400))
    if "temporal" in steps:
        E.exp_temporal(list(range(19, 200, 20)) if a.quick else None)
    if "lab" in steps:
        E.build_lab_bundle()
    print(E.write_headline())
    print(f"Pipeline finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
