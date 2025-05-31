import numpy as np
import matplotlib.pyplot as plt 
import argparse
from pathlib import Path

FILE_PATH = Path(__file__).resolve()

def read_episode_returns(run_stamp):
    np_rewards = np.load(f"{FILE_PATH.parent.parent}/logs/returns_{run_stamp}.npy")
    return np_rewards



def plot_agent_return(episode_returns_list, run_stamps_list):
    # Plot return per episode
    window = 50

    plt.figure(figsize=(8, 4))

    for i in range(len(episode_returns_list)):
        returns = episode_returns_list[i]
        run_stamp = run_stamps_list[i]

        if len(returns) == 0:
            print(f"No data found for run stamp: {run_stamp}")
            continue
        if len(returns.shape) > 1:
            returns = returns.flatten()

        if len(returns) < window:
            print(f"Not enough data points for moving average for run stamp: {run_stamp}")
            continue

        # simple centred moving average (same length as data)
        kernel = np.ones(window) / window
        ma = np.convolve(returns, kernel, mode="same")

        label = "Basic" if i == 0 else "Human-like"
    
        # plt.plot(returns, label="Episode return", alpha=0.3)
        plt.plot(ma, label=f"{label}: {window}-episode moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Learning curve – Pong Q‑learning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FILE_PATH.parent.parent}/logs/return_curve_merged_{run_stamp}.png")
    print(f"Plot return per episode saved to: {FILE_PATH.parent.parent}/logs/return_curve_merged_{run_stamp}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot agent return from log file.")
    parser.add_argument("run_stamp1", type=str, help="Run stamp for saving the plot.")
    parser.add_argument("run_stamp2", type=str, help="Run stamp for saving the plot.")
    args = parser.parse_args()

    episode_returns1 = read_episode_returns(args.run_stamp1)
    episode_returns2 = read_episode_returns(args.run_stamp2)
    plot_agent_return([episode_returns1, episode_returns2], [args.run_stamp1, args.run_stamp2])
