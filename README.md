# DiffusionArcade

A project that combines diffusion models with reinforcement learning for Pong game generation.

---

## 📦 Setup Instructions

To set up the environment for DiffusionArcade, follow these steps:

1. Make the setup script executable:
   ```bash
   chmod +x setup_env.sh
   ```

2. Run the setup script to create and activate the conda environment, and install the required packages:
   ```bash
   bash setup_env.sh
   ```

This will create a conda environment named `diffusion_arcade` with Python 3.10, using the dependencies defined in `pyproject.toml`.

> **Note:** Ensure that `conda` is installed and available in your `PATH`.

---

## 💻 Running on SCITAS (EPFL)

To request GPUs on SCITAS for interactive work:

```bash
srun -t 120 -A cs-503 --qos=cs-503 --gres=gpu:2 --mem=16G --pty bash
```

Then activate the environment:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"  # Adjust path if different
conda activate diffusion_arcade
```

---

## 🚀 Usage

### Diffusion Models

To test the diffusion model:

```bash
jupyter lab notebooks/diffusion_model.ipynb
```

> Running this notebook on a **GPU** is strongly recommended.

#### Running Jupyter on SCITAS

1. On the assigned compute node:
   ```bash
   module load tmux
   tmux new -s jupyter
   jupyter lab --no-browser --port=8888 --ip=$(hostname -i)
   ```

2. On your local machine, open a tunnel:
   ```bash
   ssh -L 8888:<node_ip>:8888 -l <scitas_username> izar.epfl.ch -f -N
   ```

3. Before opening the tunnel, make sure port 8888 is free locally:
   ```bash
   lsof -i :8888
   ```

   If any process is using it:
   ```bash
   kill <pid>
   ```

4. Then open `localhost:8888` in your browser to access Jupyter.

---

### 🔧 Configuration

Before running training:

1. Edit `config.yaml`:
   - Replace `<YOUR WANDB USER>` under the `wandb` section with your actual Weights & Biases username.

2. Create a `.env` file in the root directory and add your Hugging Face token:
   ```env
   HUGGINGFACE_TOKEN=your_token_here
   ```

---

## 🕹️ Reinforcement Learning with PLE (Pong)

To train a Q-learning agent on Pong and record game states:

1. Make the PLE setup script executable:
   ```bash
   chmod +x setup_ple_env.sh
   ```

2. Run the script to install dependencies:
   ```bash
   bash setup_ple_env.sh
   ```

3. Activate the virtual environment:
   ```bash
   source /home/runai-home/ple-venv/bin/activate  # Replace path if needed
   ```

4. Run the training script:
   ```bash
   python3 qlearn_pong.py
   ```

> If you're running this on a different system or cluster, adapt the virtual environment path accordingly.

---

# ✅ To-Do List: DiffusionArcade

## 🕹️ Reinforcement Learning (RL)

- [ ] Finalize and debug all three RL algorithms to ensure they work as intended.
- [ ] Collect a sufficient dataset to support diffusion training.
- [ ] Adjust RL strategy (e.g., exploration vs. exploitation balance, curriculum learning) to align with the data needs of the diffusion model.

## 🎨 Diffusion Model

- [ ] Fix CUDA out-of-memory issue when using latent representations (e.g., optimize batch size or memory usage).
- [ ] Train the model on a larger dataset to improve Pong frames quality and consistency.
- [ ] Evaluate and possibly fine-tune the VAE encoder with the new dataset.
- [ ] Implement action conditioning by including player inputs at each time step.
- [ ] Integrate context by combining latent encodings of previous frames with current action embeddings.
