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

## 🔧 Configuration

Before running training:

1. Edit `config.yaml`:
   - Replace `<YOUR WANDB USER>` under the `wandb` section with your actual Weights & Biases username.

2. Create a `.env` file in the root directory and add your Hugging Face token:
   ```env
   HUGGINGFACE_TOKEN=your_token_here
   ```

---

## 💻 Running on SCITAS (EPFL)

**Option 1: Interactive job using srun**

To request GPUs on SCITAS for interactive work:

```bash
srun -t 120 -A cs-503 --qos=cs-503 --gres=gpu:1 --mem=16G --pty bash
```

This command launches an interactive Bash shell on a compute node with 1 GPU, 16 GB of RAM, and a 2-hour time limit, using the cs-503 account and QoS settings.

Then activate the environment:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"  # Adjust path if different
conda activate diffusion_arcade
```

Being in an interactive node, you can run the training script using:

``` bash
python run_training.py --config path/to/your/config.yaml --model-name conditioned_diffusion_model --output-dir models --hf-org your_organization
```


**Option 2: Submit as a job via SLURM**

To request a GPU and launch training of the conditioned diffusion model with action run:

```bash
sbatch submit_job.sh <config_file> <your_wandb_key> <num_gpus>
```

> ⚠️ **Important**  
>
> Make sure to follow configuration steps before training the model.

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

#### Inside the diffusion notebook

The diffusion notebook presents the different models we trained to better understand how diffusion models work. The different notebook sections show progress from basic to more advanced settings of a diffusion model:

1. **Unconditional diffusion model:** Trained to generate Pong images from pure Gaussian noise.

2. **Latent diffusion model:**
   - A Variational Autoencoder (VAE) first encodes Pong images into latent representations.
   - A diffusion model is trained on these latents to generate new samples.
   - The generated latents are decoded back into images using the VAE decoder.

> ⚠️ **Important**  
>
> This version produces blurry Pong images. To improve their quality, finetuning the decoder on Pong data could help. However, due to time constraints and because the model trained with images performed well, we focused on training the diffusion model directly on images instead.

3. **Conditioned diffusion model (context frames):** The model receives a sequence of 7 past frames as input and learns to generate the next frame. This introduces consistency into the generation process.

4. **Conditioned diffusion model (context + action):** Builds on the previous version by also conditioning on the player's actions. The model is trained to generate the next frame given both past frames and the corresponding action embeddings.

## 🕹️ Reinforcement Learning with PLE (Pong)

To generate the Pong dataset, follow these steps:

1. Activate the conda environment:
   ```bash
   conda activate diffusion_arcade
   ```

2. Run the training script:
   ```bash
   python3 qlearn_pong.py
   ```

> If you're running this on a different system or cluster, adapt the virtual environment path accordingly.

---

## 👾 Running the game

After training the conditioned diffusion model with actions, you can launch an app to interact with it in real-time.

1. After getting access to a compute node, run the below command **from the compute note**:
```
ssh -f -N -R 7860:localhost:7860 <scitas-username>@izar.epfl.ch
```

2. On your **local machine** run:
```
ssh -f -N -L 7860:localhost:7860 <scitas-username>@izar.epfl.ch
```
This will set up a tunnel from compute node -> cluster -> local machine.

3. Edit the ```game -> model_path``` feature in ```app_config.yaml``` file to point to the correct model path that you want to use.

4. On the compute note, run:
```
cd ~/DiffusionArcade/diffusion_app
python3 app.py
```

5. In your browser on your local machine, navigate to ```localhost:7860``` and you should see the output of the game
