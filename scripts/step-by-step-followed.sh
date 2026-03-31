#!/bin/bash

pyenv virtualenv 3.10.4 .safe
pyenv activate .safe

pip install torch torchvision torchaudio

bash scripts/install.sh

# Worked:
pip install --no-deps "allenact_plugins[all] @ git+https://github.com/allenai/allenact.git@d055fc9d4533f086e0340fe0a838ed42c28d932e#subdirectory=allenact_plugins"
pip install --no-deps "git+https://github.com/Ethyn13/allenact.git@main#egg=allenact&subdirectory=allenact"
pip install --no-deps --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+966bd7758586e05d18f6181f459c0e90ba318bec



python -m objathor.dataset.download_annotations --version 2023_07_28 --path ./data/objaverse_assets
python -m objathor.dataset.download_assets --version 2023_07_28 --path ./data/objaverse_assets


python -m scripts.download_objaverse_houses --save_dir ./data/objaverse_houses --subset val

python -m scripts.download_training_data --save_dir ./data/training_data --task_types FetchType # 52 Gb
python -m scripts.download_training_data --save_dir ./data/training_data --task_types PickupType # 3Gb
python -m scripts.download_training_data --save_dir ./data/training_data --task_types ObjectNavType # 48 Gb




## Evaluating
python scripts/download_baseline_ckpt.py --ckpt_ids spoc_IL --save_dir /workspace/weights/safe-vla/v1/
bash scripts/eval.sh --task_type pickup --ckpt_path /workspace/weights/safe-vla/v1/spoc_IL/model.ckpt