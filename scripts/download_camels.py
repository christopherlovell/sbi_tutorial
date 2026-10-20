"""Compute summary statistics (GSMF + SHMR) for the CAMELS LH suites at z=0.

For each simulation only three Subfind arrays are read, via HTTP range
requests (or from a local mirror), so no full catalogue is ever downloaded:

    Subhalo/SubhaloMassType[:, 4]   stellar mass       -> GSMF
    Group/Group_M_Crit200           halo mass          -> SHMR
    Group/GroupFirstSub             central subhalo    -> SHMR

Each simulation is reduced immediately to its data vector, cached as a tiny
.npz (so the script can be interrupted and resumed), and the final output is
a single small HDF5 file:

    params      (N_sim, N_param)  LH parameters (attrs: names)
    sim_names   (N_sim,)
    gsmf        (N_sim, N_mbin)   phi [Mpc^-3 dex^-1]          (attrs: bin_edges)
    shmr        (N_sim, N_hbin)   median log10 M*_cen [Msun]   (attrs: bin_edges)

Examples:
    python download_camels.py --suite SIMBA
    python download_camels.py --suite Swift-EAGLE --nsims 100 --workers 32
    python download_camels.py --suite SIMBA --root /path/to/camels/Sims
"""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import fsspec
import h5py
import numpy as np
from tqdm import tqdm

BASE_URL = "https://users.flatironinstitute.org/~camels/Sims"

GSMF_EDGES = np.arange(8.5, 11.76, 0.25)  # log10 M* [Msun]
SHMR_EDGES = np.arange(10.5, 13.01, 0.25)  # log10 M200c [Msun]
SHMR_MIN_HALOS = 3  # bins with fewer halos are set to NaN


def read_params(path):
    """Parse a CosmoAstroSeed_*.txt table into names, sim names and values."""
    with fsspec.open(path, "r") as f:
        lines = f.read().strip().splitlines()
    header = lines[0].lstrip("#").split()
    rows = [line.split() for line in lines[1:] if line.strip()]
    sims = np.array([r[0] for r in rows])
    names = [n for n in header[1:] if n.lower() != "seed"]
    values = np.array([[float(v) for v in r[1 : 1 + len(names)]] for r in rows])
    return names, sims, values


def summarise(path):
    """Read one group catalogue and return its (gsmf, shmr) data vectors."""
    # small blocks: we only need ~1 MB of a ~50 MB file
    with fsspec.open(path, "rb", block_size=2**16) as f, h5py.File(f, "r") as h:
        hub = float(h["Header"].attrs["HubbleParam"])
        volume = (float(h["Header"].attrs["BoxSize"]) / hub / 1e3) ** 3  # Mpc^3
        mstar = h["Subhalo/SubhaloMassType"][:, 4] * 1e10 / hub
        m200 = h["Group/Group_M_Crit200"][:] * 1e10 / hub
        first_sub = h["Group/GroupFirstSub"][:]

    # GSMF
    counts, _ = np.histogram(np.log10(mstar[mstar > 0]), bins=GSMF_EDGES)
    gsmf = counts / volume / np.diff(GSMF_EDGES)

    # SHMR: median central stellar mass in bins of halo mass
    ok = (first_sub >= 0) & (m200 > 0)
    log_mh = np.log10(m200[ok])
    mstar_cen = mstar[first_sub[ok]]
    ibin = np.digitize(log_mh, SHMR_EDGES) - 1
    shmr = np.full(len(SHMR_EDGES) - 1, np.nan)
    for i in range(len(shmr)):
        sel = mstar_cen[ibin == i]
        if sel.size >= SHMR_MIN_HALOS and np.median(sel) > 0:
            shmr[i] = np.log10(np.median(sel))

    return gsmf, shmr


def work(sim, set_dir, snap, cache):
    fname = os.path.join(cache, f"{sim}.npz")
    if not os.path.exists(fname):
        gsmf, shmr = summarise(f"{set_dir}/{sim}/groups_{snap}.hdf5")
        np.savez(fname, gsmf=gsmf, shmr=shmr)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--suite", default="SIMBA", help="SIMBA, Swift-EAGLE, ...")
    p.add_argument("--set", default="LH", help="LH, 1P, CV, ...")
    p.add_argument("--box", default="L25n256")
    p.add_argument("--snap", default="090", help="090 is z=0")
    p.add_argument("--root", default=BASE_URL, help="URL or local Sims dir")
    p.add_argument("--params-file", default=None, help="Override params table")
    p.add_argument("--nsims", type=int, default=None, help="Only first N sims")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--cache-dir", default="camels_cache")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    set_dir = f"{args.root}/{args.suite}/{args.box}/{args.set}"
    params_file = args.params_file or (
        f"{set_dir}/CosmoAstroSeed_{args.suite}_{args.box}_{args.set}.txt"
    )
    names, sims, values = read_params(params_file)
    if args.nsims is not None:
        sims, values = sims[: args.nsims], values[: args.nsims]

    cache = os.path.join(args.cache_dir, f"{args.suite}_{args.set}_{args.snap}")
    os.makedirs(cache, exist_ok=True)

    failed = []
    # processes rather than threads, as h5py serialises calls behind a global
    # lock; "spawn" as fsspec's event loop does not survive a fork
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(work, s, set_dir, args.snap, cache): s for s in sims}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                fut.result()
            except Exception as e:  # keep going; report at the end
                failed.append((futures[fut], repr(e)))

    if failed:
        print(f"{len(failed)} sims failed (re-run to retry), e.g.:")
        for s, e in failed[:5]:
            print(f"  {s}: {e}")

    keep = [i for i, s in enumerate(sims) if os.path.exists(f"{cache}/{s}.npz")]
    vecs = [np.load(f"{cache}/{sims[i]}.npz") for i in keep]

    out = args.out or f"camels_{args.suite}_{args.set}_z{args.snap}.hdf5"
    with h5py.File(out, "w") as h:
        h.attrs.update(suite=args.suite, set=args.set, snap=args.snap)
        h.create_dataset("params", data=values[keep]).attrs["names"] = names
        h.create_dataset("sim_names", data=sims[keep].astype("S"))
        h.create_dataset("gsmf", data=[v["gsmf"] for v in vecs]).attrs[
            "bin_edges"
        ] = GSMF_EDGES
        h.create_dataset("shmr", data=[v["shmr"] for v in vecs]).attrs[
            "bin_edges"
        ] = SHMR_EDGES
    print(f"Wrote {out}: {len(keep)} sims")


if __name__ == "__main__":
    main()
