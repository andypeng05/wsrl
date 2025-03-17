export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python finetune.py \
--agent mca \
--config experiments/configs/train_config.py:antmaze_sac \
--project antmaze-finetune \
--env antmaze-large-diverse-v2 \
--reward_scale 10.0 \
--reward_bias -5.0 \
--utd 1 \
--batch_size 256 \
--num_offline_steps 1_000_000 \
$@
