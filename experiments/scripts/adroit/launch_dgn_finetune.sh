export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl


python3 finetune_dgn.py \
--agent sac \
--config experiments/configs/train_config.py:adroit_rlpd \
--dgn_config experiments/configs/train_config.py:adroit_dgn \
--project dgn-adroit-finetune \
--num_offline_steps 0 \
--reward_scale 10.0 \
--reward_bias 5.0 \
--offline_data_ratio 0.5 \
--env pen-binary-v0 \
--utd 20 \
--batch_size $((20 * 128)) \
--warmup_steps 0 \
$@ 