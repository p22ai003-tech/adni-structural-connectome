# SL-H04A-GPU — controlled Eddy GPU transition recommendation

**Status:** recommended transition; wait for the current bounded Phase-A attempt to terminate cleanly  
**Prepared:** 2026-07-18 16:36 UTC; updated 18:00 UTC  
**Current instance:** `c6a.16xlarge`, 64 vCPU, 123 GiB RAM, no NVIDIA device exposed  

## Decision

Do not stop or resize the EC2 instance while the current 15-unit response-calibration attempt is running. After that attempt writes its required terminal record, stop at the existing Phase-A boundary and resize to a single-GPU instance before any pre-tractography continuation, corrected-connectome canary, or larger selected repair lane.

The preferred first choice is `g6.4xlarge` if available in the project Availability Zone; `g5.4xlarge` is the conservative fallback, and `g4dn.4xlarge` is the older-generation fallback. Each provides enough host memory/CPU for the surrounding MRtrix/FSL work while exposing one NVIDIA accelerator. Exact Availability Zone capacity must be checked at transition time.

## Why downstream stages do not start concurrently

The executable workflow has three distinct stages, even though informal discussion has sometimes called both downstream stages “Phase B”:

1. **Response-calibration Phase A (active):** produces one diagnosis-blind WM/GM/CSF response outcome per approved calibration unit and terminally closes the 15-unit denominator.
2. **Pre-tractography canary:** pools at least 12 valid Phase-A responses into one frozen, hash-bound calibration, then computes FOD/tensor, registration, atlas and automated/visual QC products. Its FOD rules consume that exact pooled artifact; a partial pool would change the model and force a rerun.
3. **Launcher `phase-b`:** performs tractography and matrices only for units that pass pre-tractography automated and blinded human QC. It requires a separate H04B continuation attestation.

The current signed recovery decision explicitly sets FOD reconstruction, tractography, matrices and full-cohort execution to false. The launcher also fails closed if the terminal Phase-A completion, frozen-response manifest, or later H04B evidence is absent. Preparation can occur in parallel, but scientific imaging cannot cross these data dependencies without changing the recipe.

## Calculation and local evidence

- The completed CPU reference unit `168_S_6735_I1175371` ran `dwifslpreproc` from the log birth time `13:29:48.405 UTC` to the atomic preprocessed-image completion time `15:29:56.102 UTC`: approximately **2 h 00 min 08 s** for a `116 x 116 x 80 x 55` series.
- At 16:34 UTC, the retry2 controller was active at **81/168 jobs (48%)**, with **eight concurrent `eddy_cpu` processes** and no completed `dwi_preproc.mif` yet. Stopping EC2 would terminate and later restart this in-flight wave.
- The current workflow is not hardware-agnostic: it exports `DWIFSLPREPROC_FORCE_CPU=1`, rejects any `eddy_cuda` appearance, and exposes a CPU-only FSL path. Resizing alone therefore cannot enable GPU processing.
- The host already contains `/home/ec2-user/fsl/bin/eddy_cuda11.0`, its CUDA libraries resolve, and NVIDIA driver module `570.133.20` matches the running kernel. GPU-device and runtime validation still must be repeated after boot on the resized instance.

FSL documents functionally equivalent CPU and CUDA Eddy executables and notes that the CUDA implementation is normally substantially faster unless many CPU cores are effectively used. The current `eddy_cpu` invocations consume approximately one core each despite four-thread Snakemake reservations. See the [FSL Eddy user guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/eddy/users_guide/index.html). AWS accelerator specifications are documented on the [EC2 accelerated-computing instance page](https://aws.amazon.com/ec2/instance-types/accelerated-computing/) and [G4 instance page](https://aws.amazon.com/ec2/instance-types/g4/).

## Required transition gate

After Phase A terminates, the following must all pass before any wider processing:

1. Preserve and validate the current attempt end record and response-calibration outputs.
2. User performs the EC2 stop/resize/start operation; no in-host command initiates an instance stop.
3. Verify the mounted data volume, instance identity/type, `nvidia-smi`, driver/CUDA compatibility, and `eddy_cuda11.0` startup.
4. Create a separately versioned GPU environment/source/recipe binding; never mutate the completed CPU binding.
5. Re-run one already completed unit with GPU Eddy in an isolated root.
6. Compare CPU versus GPU dimensions, gradients, output completeness, outlier/motion/QC summaries, representative voxel statistics, and downstream response-estimation acceptability.
7. Record elapsed time and peak GPU memory, then select safe concurrency from measured evidence.
8. Continue only if the equivalence/QC contract passes; otherwise retain the CPU result and investigate without widening scope.

The one-unit GPU equivalence package is now frozen under execution-contract SHA-256 `776684e5e0a1725e3b33ce4a5332598bfe23f6f9a5bdf10596506a44246cd338` and passes 11/11 static checks. The aggregate gate record at `outputs/gpu_transition_gate_v1/gpu_transition_gate.json` is fail-closed: while Phase A is active it reports `NOT_SAFE_TO_RESIZE`; it can report `SAFE_TO_RESIZE` only after the service is terminal, imaging processes are absent, all 15 terminal outcomes and hashes validate, no forbidden downstream artifact exists, and the frozen GPU package remains intact.

## Scope boundary

This recommendation accelerates a scientifically selected repair/canary path. It does **not** authorize an indiscriminate 530-subject rerun, tractography, matrices, dashboard replacement, or a novelty claim. Those remain governed by the existing canary/QC/human gates and the selective-repair triage.
