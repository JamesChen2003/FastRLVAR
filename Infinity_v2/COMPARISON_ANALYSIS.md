# Comparison: 2048 RL vs FastVAR RL Training

## 1. Environment Differences

### **My2048Env (RL/HW3)**
- **Observation Space**: `Box(0, 1, (16, 4, 4), dtype=int)` - 16 binary planes (one-hot encoding of tile values)
- **Action Space**: `Discrete(4)` - 4 discrete actions (up, right, down, left)
- **Episode Length**: Variable (game ends when no moves available or max tile reached)
- **Reward Structure**: 
  - **Immediate rewards**: Tile merge score + shape bonuses (row/col monotonicity, empty cells)
  - **Episode rewards**: Sum of all step rewards
  - **Illegal move penalty**: -2.0, terminates after 5 consecutive illegal moves
- **State Representation**: Clean, deterministic board state (4x4 grid)
- **Reset**: Starts fresh game with 2 random tiles
- **Episode Info**: Rich statistics (combine_reward, row_reward, col_reward, empty_cell_reward, illegal_moves)

### **VAREnv (FastVAR)**
- **Observation Space**: `Box(-inf, inf, (32, 1, 64, 64), dtype=float32)` - 32-channel feature maps
- **Action Space**: `Box(0.0, 1.0, (1,), dtype=float32)` - Continuous pruning ratio [0, 1]
- **Episode Length**: Fixed (number of scales in `scale_schedule`, typically ~13 scales)
- **Reward Structure**: 
  - **Current**: `reward = prune_ratio` (just returns the action value!)
  - **No quality metrics**: No image quality, FID, CLIP score, or generation success
- **State Representation**: 
  - **Problem**: `summed_codes` from Infinity model (latent feature maps)
  - **Issue**: Observation is intermediate model state, not interpretable game state
  - **Resize**: Attempts to resize to (64, 64), but can fail with empty tokens
- **Reset**: Resets `scale_index=0`, returns zero observation `(32, 1, 64, 64)`
- **Episode Info**: Empty dict `{}` - no useful statistics

## 2. Training Script Differences

### **RL/HW3/train.py (Well-Configured)**

#### **Environment Setup**
```python
# Simple, clean environment creation
def make_env():
    env = gym.make('2048-v0')
    return env

# Parallel environments
train_env = SubprocVecEnv([make_env for _ in range(16)])  # 16 parallel
eval_env = DummyVecEnv([make_env])  # 1 for eval
```

#### **Model Configuration**
- **Algorithm**: DQN/A2C/PPO with proper hyperparameters
- **Feature Extractor**: Custom CNN for image-like observations (16 planes)
- **Policy**: `CnnPolicy` (appropriate for spatial observations)
- **Hyperparameters**: Well-tuned (learning_rate, buffer_size, batch_size, etc.)
- **Parallelization**: 16 parallel environments (`SubprocVecEnv`)

#### **Training Configuration**
```python
my_config = {
    "num_train_envs": 16,           # Parallel training
    "epoch_num": 1000,
    "timesteps_per_epoch": 2048 * 16,  # Proper scaling
    "eval_episode_num": 10,
}
```

#### **Evaluation**
- **Proper API**: Uses Gymnasium 5-tuple `(obs, reward, done, truncate, info)`
- **Metrics**: Extracts `info['highest']` and `info['score']` correctly
- **Logging**: Wandb integration with custom callbacks

### **FastVAR/train.py (Problematic)**

#### **Environment Setup**
```python
# Complex, requires heavy model loading
def make_env(infinity, vae, scale_schedule, text_tokenizer, text_encoder, prompt):
    env = VAREnv(infinity, vae, scale_schedule, text_tokenizer, text_encoder, prompt)
    return env

# Sequential environments (DummyVecEnv, not SubprocVecEnv)
train_env = DummyVecEnv([...])  # Only 1 environment!
eval_env = DummyVecEnv([...])
```

#### **Model Configuration**
- **Algorithm**: PPO with minimal hyperparameters
- **Feature Extractor**: None (uses `MlpPolicy` for high-dim observations!)
- **Policy**: `MlpPolicy` (inappropriate for spatial feature maps)
- **Hyperparameters**: Default PPO settings (no tuning)
- **Parallelization**: Only 1 environment (`num_train_envs=1`)

#### **Training Configuration**
```python
my_config = {
    "num_train_envs": 1,             # NO parallelization!
    "epoch_num": 50000,               # Extremely high
    "timesteps_per_epoch": 1e6,      # 1 million steps per epoch!
    "eval_episode_num": 10,
}
```

#### **Evaluation**
- **API Mismatch**: Uses old Gym API in `eval()` function
  ```python
  obs, reward, done, info = env.step(action)  # Missing truncate!
  ```
  But `VAREnv.step()` returns `(obs, reward, done, truncate, info)` (5-tuple)
- **Broken Metrics**: Tries to access `info[0]['highest']` and `info[0]['score']`
  But `VAREnv` returns `info={}` (empty dict)!
- **No Logging**: No wandb, no tensorboard callbacks

## 3. Critical Problems with FastVAR Training

### **Problem 1: Reward Function is Meaningless**
```python
# VAREnv.step() line 155
reward = score  # where score = float(prune_ratio)
```
- **Issue**: Reward = action value (prune_ratio)
- **Consequence**: Agent learns to always output `action=1.0` (maximum pruning)
- **Fix Needed**: Reward should be based on:
  - Image quality (FID, CLIP score)
  - Generation success
  - Computational efficiency
  - Trade-off between quality and speed

### **Problem 2: Observation Space Mismatch**
- **Observation**: `(32, 1, 64, 64)` feature maps from Infinity model
- **Policy**: `MlpPolicy` (flattens to 32*64*64 = 131,072 dimensions!)
- **Issue**: 
  - MLP doesn't exploit spatial structure
  - Should use `CnnPolicy` with custom feature extractor
  - Or reduce observation dimensionality

### **Problem 3: No Parallelization**
- **Current**: `num_train_envs=1` (single environment)
- **Issue**: 
  - Extremely slow training
  - No diversity in experience
  - Can't use `SubprocVecEnv` because Infinity/VAE models are too large to fork
- **Fix**: Use `DummyVecEnv` with multiple environments, but share model weights

### **Problem 4: Evaluation Function is Broken**
```python
def eval(env, model, eval_episode_num):
    # ...
    obs, reward, done, info = env.step(action)  # WRONG: Missing truncate
    # ...
    avg_highest += info[0]['highest']  # WRONG: info is {}, not list
    avg_score   += info[0]['score']    # WRONG: info is {}, not list
```
- **Issues**:
  1. API mismatch (4-tuple vs 5-tuple)
  2. `info` is empty dict, not list with `['highest']` or `['score']`
  3. Will crash or return NaN

### **Problem 5: Episode Length is Too Short**
- **2048**: Variable length (can be 100+ steps for good games)
- **FastVAR**: Fixed length (~13 scales)
- **Issue**: Very short episodes = less learning signal per episode
- **Fix**: Consider multi-episode rollouts or longer scale schedules

### **Problem 6: No Proper State Reset**
```python
def reset(self, seed=None, options=None):
    # ...
    obs = np.zeros((32,1,64,64), dtype=np.float32)  # Returns zeros!
    return obs, {}
```
- **Issue**: Returns zero observation, not actual initial state
- **Consequence**: Agent doesn't see meaningful initial state
- **Fix**: Should return initial feature map from first scale

### **Problem 7: Extremely High Training Budget**
- **Epochs**: 50,000 epochs
- **Steps per epoch**: 1,000,000
- **Total**: 50 billion steps!
- **Issue**: 
  - Unrealistic training time
  - No early stopping
  - No learning rate scheduling
- **Fix**: Reduce to reasonable numbers (e.g., 1000 epochs, 10K steps/epoch)

### **Problem 8: Observation Resize Can Fail**
```python
try:
    summed_codes = torch.nn.functional.interpolate(
        summed_codes[:,0], 
        size=(64, 64), 
        mode='area'
    )
except Exception as e:
    print(f"Resize failed with Empty Token")
    summed_codes = torch.zeros((1, 64, 64), device=codes.device)
```
- **Issue**: Falls back to zeros on failure
- **Consequence**: Agent sees invalid state, learns wrong behavior
- **Fix**: Proper error handling or fix dimension mismatch

### **Problem 9: No Reward Shaping**
- **2048**: Multiple reward components (merge, shape, empty cells)
- **FastVAR**: Single meaningless reward (prune_ratio)
- **Fix**: Add rewards for:
  - Image quality (FID/CLIP improvement)
  - Generation success (did image generate?)
  - Efficiency (time/compute saved)
  - Penalty for over-pruning (quality degradation)

### **Problem 10: Model Loading Overhead**
- **Issue**: Infinity, VAE, and text encoder are loaded once and shared
- **But**: Each environment creation still has overhead
- **Fix**: Ensure models are truly shared (not duplicated)

## 4. Recommendations

### **Immediate Fixes**

1. **Fix Reward Function**:
   ```python
   # Compute actual reward based on image quality
   reward = compute_image_quality_reward(generated_image, prompt)
   # Or: reward = -compute_fid_score(...)  # Negative FID (lower is better)
   ```

2. **Fix Evaluation Function**:
   ```python
   obs, reward, done, truncate, info = env.step(action)  # 5-tuple
   # Extract metrics from actual info dict
   ```

3. **Use CNN Policy**:
   ```python
   policy_kwargs = dict(
       features_extractor_class=CustomCNNFeatureExtractor,
       features_extractor_kwargs=dict(features_dim=256),
   )
   model = PPO("CnnPolicy", ...)  # Not MlpPolicy
   ```

4. **Reduce Training Budget**:
   ```python
   "epoch_num": 1000,  # Not 50000
   "timesteps_per_epoch": 10000,  # Not 1e6
   ```

5. **Add Proper Info Dict**:
   ```python
   info = {
       'scale_index': self.scale_index,
       'prune_ratio': prune_ratio,
       'image_quality': quality_score,  # If computed
   }
   ```

6. **Fix Reset**:
   ```python
   def reset(self, seed=None, options=None):
       # ... reset state ...
       # Return actual initial observation, not zeros
       initial_obs = self.get_initial_observation()
       return initial_obs, {}
   ```

### **Planned FastVAR RL Improvements**

1. **Custom CNN policy with bounded action**  
   - Use a CNN encoder similar to the 2048 project (3 conv layers + 2 linear layers) as a `features_extractor_class` for `CnnPolicy`.  
   - Add a final sigmoid on the policy head so the continuous action (pruning ratio) is strictly in \([0, 1]\).

2. **Augmented observation with scale index**  
   - Expose the current `scale_index` to the agent in addition to `summed_codes` (e.g. by concatenating a normalized scale index channel or extra scalar feature).  
   - This lets the policy condition its pruning decision on which scale is being processed.

3. **Reward shaping with quality vs. speed trade-off**  
   - Define \( \text{reward} = \alpha \cdot \text{quality\_score} + (1-\alpha)\cdot \text{speed\_score} \).  
   - **quality\_score**: PSNR between the pruned image and a pre-computed unpruned “golden” image at the same scale (generated once per scale with `save_intermediate_results=True` in `autoregressive_infer_cfg`).  
   - **speed\_score**: current pruning ratio (higher pruning → higher speed reward).

4. **Richer `info` dict from `VAREnv.step`**  
   - Include keys like `scale_index`, `prune_ratio`, `quality_score`, `speed_score`, and the combined `reward` to support better logging and analysis.

### **Long-term Improvements**

1. **Add proper reward computation** (FID, CLIP, generation success)
2. **Implement proper feature extractor** for spatial observations
3. **Add wandb/tensorboard logging**
4. **Implement early stopping** based on validation metrics
5. **Add learning rate scheduling**
6. **Consider multi-objective rewards** (quality vs efficiency)

