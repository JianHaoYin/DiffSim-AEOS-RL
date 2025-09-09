# DiffSim-AEOS-RL
Reinforcement learning framework for Agile Earth Observation Satellites (AEOS) scheduling, built on top of Sat-Sim-pytorch with full support for differentiable simulation. This repository provides a research-ready environment to train, evaluate, and benchmark RL schedulers under realistic orbital dynamics, attitude control, energy, and field-of-view constraints.




## Tips: Updating the vendored `Sat-Sim-pytorch` (fork + subtree)

This repository vendors our customized **Sat-Sim-pytorch** under `external/sat-sim-pytorch` using **git subtree**.  
Topology:

- **Fork** (write access): [JianHaoYin/Sat-Sim-pytorch](https://github.com/JianHaoYin/Sat-Sim-pytorch.git)  
- **Vendored path in this repo**: `external/sat-sim-pytorch/`

> We do **not** discuss pushing changes to the original upstream here—only syncing **between this repo and our fork**.

### One-time setup

```bash
# In the AEOS repo root
git remote add satsim-fork https://github.com/JianHaoYin/Sat-Sim-pytorch.git
git fetch satsim-fork
```
## Pull updates from the fork into AEOS (subtree pull)
Use this when the fork has new commits you want to vendor here.
```bash
# Pull latest fork:main into external/sat-sim-pytorch (squashed history recommended)
git subtree pull --prefix=external/sat-sim-pytorch satsim-fork main --squash

# Commit if needed (sometimes subtree creates a commit automatically)
git commit -m "chore: subtree pull Sat-Sim-pytorch from fork:main" || true
```

--squash keeps this repo light by compressing history; remove it if you need the full commit history.

Resolve any merge conflicts only under external/sat-sim-pytorch/, then commit.

## Push local AEOS changes back to the fork (subtree push)
Use this when you edited code inside external/sat-sim-pytorch/ and want those changes reflected in your fork.

```bash
# Push the subtree prefix to a branch on the fork
git subtree push --prefix=external/sat-sim-pytorch satsim-fork aeos-sync-branch
```
Then open a Pull Request on your fork: aeos-sync-branch → main, review, and merge.

You can choose any branch name (e.g., aeos-sync-YYYYMMDD).

After merging on the fork, you can later pull back again via the “Pull updates from the fork” step.

## Editable install for development
```bash
# Make the vendored package importable in your Python env
pip install -e external/sat-sim-pytorch
```