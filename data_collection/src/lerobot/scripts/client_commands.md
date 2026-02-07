## BiARX5 Robot lerobot-teleoperate command

```python
lerobot-teleoperate \
    --robot.type=bi_arx5 \
    --robot.enable_tactile_sensors=true \
    --teleop.type=mock_teleop \
    --fps=30 \
    --debug_timing=false \
    --display_data=true
```

## TacUMI Robot teleoperate by Mock Teleop command

### 1e892b82baa5 -another mac addr

```python
lerobot-teleoperate \
    --robot.type=xense_flare \
    --robot.mac_addr="6ebbc5f53240" \
    --teleop.type=mock_teleop \
    --fps=30 \
    --display_data=true \
    --debug_timing=true \
    --dryrun=false
```

## TacUMI Robot lerobot-record command

```python
lerobot-record \
    --robot.type=xense_flare \
    --robot.mac_addr=6ebbc5f53240 \
    --dataset.repo_id=<your_repo_id> \
    --dataset.num_episodes=20 \
    --dataset.single_task="your prompt" \
    --dataset.fps=30 \
    --display_data=false \
    --resume=false \
    --dataset.push_to_hub=true
```

## BiARX5 Robot lerobot-record command

```python
lerobot-record \
    --robot.type=bi_arx5 \
    --teleop.type=mock_teleop \
    --dataset.repo_id=<your_repo_id> \
    --dataset.num_episodes=100 \
    --dataset.single_task="your prompt" \
    --dataset.fps=30 \
    --dataset.episode_time_s=300 \
    --display_data=false \
    --resume=true \
    --dataset.push_to_hub=true
```

## BiARX5 Robot lerobot-replay command

```python
lerobot-replay \
    --robot.type=bi_arx5 \
    --dataset.repo_id=<your_repo_id> \
    --dataset.episode=0
```

**Note on preview_time:**

Adjust `--robot.preview_time` to reduce jittering:

- 0.03-0.05s: Smoother motion, more delay (recommended for stable movements)
- 0.01-0.02s: More responsive, but may cause jittering
- 0.0: No preview (only for teleoperation/recording)
