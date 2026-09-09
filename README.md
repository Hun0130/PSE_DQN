# PSE_DQN

**Deep reinforcement learning for scheduling multijob serial production lines.**

Research materials for:

> **Deep Reinforcement Learning-driven Scheduling in Multijob Serial Lines: A Case Study in Automotive Parts Assembly**  
> Sanghoon Lee, Jinyoung Kim, Yongsoon Eun, and Kyung-Joon Park  
> *IEEE Transactions on Industrial Informatics*, February 2024  
> [Read the paper](https://ieeexplore.ieee.org/document/10210628)

This repository contains the production-line simulator, DQN-family implementations, recorded experiment outputs, and model checkpoints used during the research. It preserves the original experiment workflow.

[Researcher: Sanghoon Lee](https://hun0130.github.io/) · [GitHub](https://github.com/Hun0130) · [Questions](https://github.com/Hun0130/PSE_DQN/issues)

## Repository guide

| Location | Contents |
| --- | --- |
| `factory.py` | Production-line simulation and loading of monthly evaluation data |
| `main.py` | Training, checkpoint evaluation, and plotting entry points |
| `DQN.py`, `DDQN.py`, `Duel_DQN.py` | DQN, Double DQN, and Dueling DQN implementations |
| `PDQN.py`, `AllDQN.py` | Additional replay/network experiment variants |
| `DETER.py` | Deterministic scheduling utilities |
| `raw_data_evaluation/` | Product-specific evaluation CSV files for months 05 and 06 |
| `raw_data_stop_time/` | Recorded stop-time data |
| `*_model/` | Saved PyTorch checkpoints |
| `*_data/`, `Deter_data/` | Recorded rewards, production times, losses, and machine-state summaries |
| `Figures/` | Plots saved during development |
| `data_processing/` | Data preparation and analysis scripts |

## Explore the recorded results

You can inspect the included CSV files, text logs, and [saved figures](Figures/) without retraining a model. Files ending in `_production_time.txt` record simulation production times; `_reward.txt` and `_loss.txt` contain the corresponding training series.

The saved outputs include intermediate experiments. A filename alone does not identify a numbered figure or table in the published paper; use the paper's experimental conditions when selecting a run.

## Environment

The source uses **Python 3**, **PyTorch**, **NumPy**, **pandas**, **Matplotlib**, and **SymPy**. The data-processing utilities additionally use **PyArrow**. Because `main.py` imports `turtle`, its Python installation also needs Tk support.

The training and evaluation entry points explicitly select `cuda`. Running them as written requires an NVIDIA GPU and a compatible CUDA-enabled PyTorch installation. Historical dependency versions are not pinned in the repository.

1. Obtain the repository and enter its root directory:

   ```bash
   git clone https://github.com/Hun0130/PSE_DQN.git
   cd PSE_DQN
   ```

2. Use an isolated Python environment. Install a PyTorch build compatible with your GPU using the [official installation instructions](https://pytorch.org/get-started/locally/), along with the packages above.

3. Check that CUDA is available to that environment:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

All experiment paths are relative to the repository root. The checkpoint and data directories are part of the experiment inputs.

## Evaluate the included checkpoint

The active entry point at the end of `main.py` is:

```python
Test(100, 'Dueling_DQN_model/', "model_8542")
```

It loads `Dueling_DQN_model/model_8542.pth`, uses month `06` data, and evaluates the scheduling policy over 100 simulation runs.

After preparing the environment, run:

```bash
python main.py
```

The evaluator writes files named `Deter_data/model_8542_*.txt`, including production-time, downtime, starvation, and blockage outputs. Running it again can replace those files. Evaluation includes exploration and stochastic simulation, so individual results can vary.

## Training and other experiments

Training functions such as `Deep_QN()`, `Double_DQN()`, and `Dueling_DQN()` are defined in `main.py`; their calls in the final `__main__` block are commented out. Select the desired function there before running a training experiment. Hyperparameters are defined in the corresponding algorithm module.

For paper-level replication, match the scheduling scenario, data period, hyperparameters, and evaluation protocol described in the paper. This repository does not include a single command that regenerates every published result.

## Cite the research

```bibtex
@article{lee2024multijob,
  title   = {Deep Reinforcement Learning-driven Scheduling in Multijob Serial Lines: A Case Study in Automotive Parts Assembly},
  author  = {Lee, Sanghoon and Kim, Jinyoung and Eun, Yongsoon and Park, Kyung-Joon},
  journal = {IEEE Transactions on Industrial Informatics},
  year    = {2024},
  url     = {https://ieeexplore.ieee.org/document/10210628}
}
```

## Research team and contact

The paper was authored by **Sanghoon Lee, Jinyoung Kim, Yongsoon Eun, and Kyung-Joon Park**.

For questions about these research materials, contact **[Sanghoon Lee](https://github.com/Hun0130)** at [leesh2913@dgist.ac.kr](mailto:leesh2913@dgist.ac.kr). [More research and tools](https://hun0130.github.io/).
