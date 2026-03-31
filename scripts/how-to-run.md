⏺ SafeVLA Docker Evaluation — Feasibility Review

  Yes, running evaluation via Docker is feasible — it's actually the recommended path per the README. The pre-built image safevla/safevla:v1 has all
  dependencies pre-installed.

  ---
  What Already Exists

  - scripts/run_docker.sh — launches the container with bind mounts for code and data
  - scripts/eval.sh — the eval launcher (runs inside the container)
  - benchmark/*.jsonl.gz — local eval task specs already in the repo (no download needed)

  ---
  Steps to Run Evaluation via Docker

  1. Fix scripts/run_docker.sh — there's a bug: CODE_PATH points to r2b_safety_lab/ instead of SafeVLA/:
  export CODE_PATH=/Users/rafaelfelix/Projects/r2b/SafeVLA   # fix this
  export DATA_PATH=/path/to/your/data                         # set this

  2. Pull and start the container:
  docker pull safevla/safevla:v1
  bash scripts/run_docker.sh   # drops you into an interactive shell

  3. Inside the container, set env vars:
  export PYTHONPATH=/root/SafeVLA
  export OBJAVERSE_HOUSES_DIR=/root/data/objaverse_houses
  export OBJAVERSE_DATA_DIR=/root/data/objaverse_assets
  export ALLENACT_DEBUG=True
  export ALLENACT_DEBUG_VST_TIMEOUT=2000

  4. Run eval:
  cd /root/SafeVLA
  bash scripts/eval.sh --task_type objectnav --ckpt_path /root/SafeVLA/checkpoints/safe_objnav.pt

  ---
  Key Constraints

  ┌───────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │    Constraint     │                                                   Detail                                                   │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ GPU required      │ Needs NVIDIA GPU + nvidia-container-toolkit; container uses --runtime=nvidia --gpus all                    │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Display/rendering │ AI2-THOR renders 3D scenes — on a headless server you'll need Xvfb or similar; /dev/dri is already mounted │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Objaverse data    │ Large download needed (~GB to TB depending on subset); must be available at DATA_PATH before running       │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Checkpoints       │ Must be inside CODE_PATH or DATA_PATH to be visible inside the container                                   │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Shared memory     │ Already handled — run_docker.sh sets --shm-size 50G for multi-worker eval                                  │
  └───────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  The biggest practical blocker is likely the display/rendering requirement for AI2-THOR. On a desktop Mac this isn't an issue, but on a headless server
  you'd need to configure a virtual display. Do you want me to look into that specifically, or help with any other aspect of the setup?