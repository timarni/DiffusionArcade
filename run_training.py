import torch
from diffusion.model import DiffusionModel, load_config


def main():
    config = load_config("config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_cfg = {
        "project": config["wandb"]["project"],
        "entity": config["wandb"]["entity"],
        "name": config["wandb"]["run_name"],
        "config": config
    }

    model = DiffusionModel(
        image_size=config['training']['image_size'],
        in_channels=config['training']['in_channels'],
        out_channels=config['training']['out_channels'],
        device=device,
        timesteps=config['training']['timesteps'],
        beta_schedule=config['training']['beta_schedule'],
        block_out_channels=tuple(config['model']['block_out_channels']),
        layers_per_block=config['model']['layers_per_block'],
        down_block_types=tuple(config['model']['down_block_types']),
        up_block_types=tuple(config['model']['up_block_types']),
        wandb_config=wandb_cfg,
    )

    # TODO: Initialize dataloaders here
    train_dataloader = None
    val_dataloader = None

    model.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=config['training']['epochs'],
        lr=config['training']['learning_rate'],
        wandb_config=config['wandb']
    )


if __name__ == "__main__":
    main()
