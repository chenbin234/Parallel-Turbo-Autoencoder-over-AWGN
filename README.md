# Parallel Turbo Autoencoder over AWGN

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-ee4c2c)](https://pytorch.org/) [![W&B](https://img.shields.io/badge/logging-W%26B-ffbe00)](https://wandb.ai/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch reproduction of the parallel turbo autoencoder (TurboAE-continuous) on the AWGN channel from

> **Turbo Autoencoder: Deep learning based channel codes for point-to-point communication channels**<br>
> Yihan Jiang, Hyeji Kim, Himanshu Asnani, Sreeram Kannan, Sewoong Oh, Pramod Viswanath.<br>
> *Advances in Neural Information Processing Systems (NeurIPS)*, 2019. arXiv: [1911.03038](https://arxiv.org/abs/1911.03038)

Note: Refer to the author's [GitHub repository](https://github.com/yihanjiang/turboae) for the original implementation. My implementation is based on the paper and the original code.

## Contents

- [Parallel Turbo Autoencoder over AWGN](#parallel-turbo-autoencoder-over-awgn)
  - [Contents](#contents)
  - [1. Overview](#1-overview)
    - [Pretrained model](#pretrained-model)
    - [Results](#results)
  - [2. Installation](#2-installation)
  - [3. Repository Structure](#3-repository-structure)
  - [4. Evaluating the Pretrained Model](#4-evaluating-the-pretrained-model)
  - [5. Training from Scratch](#5-training-from-scratch)
    - [5.1 Running](#51-running)
    - [5.2 Outputs](#52-outputs)
    - [5.3 Configuration reference](#53-configuration-reference)
  - [6. License](#6-license)

## 1. Overview

TurboAE replaces the component encoders and decoders of a rate-1/3 turbo code with 1-D CNNs and trains them end to end (Figs. 2 and 3 of the paper):

- **Encoder.** Three CNNs map the message $u$ to three code streams; the third one sees the interleaved message $\pi(u)$. The streams are then normalized to zero mean and unit power (soft power constraint).
- **Decoder.** In every iteration, one CNN works in natural order on $(y_1, y_2)$ and a second one in interleaved order on $(\pi(y_1), y_3)$. They exchange $F$ extrinsic features through the (de)interleaver, and a sigmoid on the last output gives the bit estimates.

**Setting.** $K = 100$ information bits, $N = 300$ code symbols, rate $R = 1/3$, 6 decoder iterations with separate weights, $F = 5$, and $\text{SNR} = -10 \log_{10} \sigma^2$ as in the paper. The power constraint uses the statistics of the current batch, in training and in testing.

### Pretrained model

| Checkpoint | Epochs | Interleaver seed | Parameters (encoder / decoder) | Training log |
| --- | ---: | ---: | ---: | --- |
| [Turbo_AutoEncoder_20261002_13_37_35](trained_models/Turbo_AutoEncoder_20261002_13_37_35/) | 600 | 5 | 0.60 M / 2.45 M | [output_parallel_turboae.log](trained_models/Turbo_AutoEncoder_20261002_13_37_35/output_parallel_turboae.log) |

### Results

BER and BLER of the pretrained model on the AWGN channel, measured with $10^7$ packets per SNR point (end of the training log). Compare them with the TurboAE-continuous curve in Fig. 1 of the paper.

| SNR (dB) | −1.5 | −1.0 | −0.5 | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BER | 8.64e-2 | 3.81e-2 | 1.25e-2 | 3.06e-3 | 6.58e-4 | 1.67e-4 | 5.14e-5 | 1.63e-5 | 5.08e-6 |
| BLER | 6.33e-1 | 3.91e-1 | 1.90e-1 | 7.60e-2 | 2.77e-2 | 1.03e-2 | 3.73e-3 | 1.27e-3 | 4.04e-4 |

**Not included:** TurboAE-binary (Sec. 3.2), the non-AWGN channels (Sec. 4.2), the block-length study (Fig. 5), and the reference codes (Turbo, LDPC, Polar, TBCC, CNN-AE).

## 2. Installation

**Requirements**

- Python 3.12 or newer
- A CUDA-capable NVIDIA GPU, strongly recommended. Everything also runs on the CPU, but slowly.
- A free [Weights & Biases](https://wandb.ai/) account. Every run, including evaluation, is logged to W&B (see the note in step 4).

**1. Clone the repository.** The checkpoint (about 12 MB) is part of the repository.

```bash
git clone https://github.com/chenbin234/Parallel-Turbo-Autoencoder-over-AWGN.git
cd Parallel-Turbo-Autoencoder-over-AWGN
```

**2. Create the environment** with [uv](https://docs.astral.sh/uv/), which installs the exact versions pinned in [uv.lock](uv.lock):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # if uv is not installed yet
uv sync                                          # add --no-dev to skip the development tools
source .venv/bin/activate
```

**3. Check the installation.**

```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
```

**4. Log in to Weights & Biases.**

```bash
wandb login
```

> [!NOTE]
> [src/train/train_turbo_ae.py](src/train/train_turbo_ae.py) starts its W&B run with `mode="online"`, which takes precedence over the `WANDB_MODE` environment variable.

## 3. Repository Structure

```text
.
├── src/
│   ├── models/
│   │   ├── parallel_turbo_autoencoder.py  # TurboAE: Encoder, Decoder, interleaver, end-to-end Turbo_AutoEncoder
│   │   └── components/
│   │       └── SameShapeConv1d.py         # Stack of length-preserving 1-D convolutions with ELU
│   ├── channel/
│   │   └── AWGN_Channel.py                # AWGN channel, SNR = -10 log10(sigma^2)
│   ├── train/
│   │   ├── train_turbo_ae.py              # Entry point and TurboAEConfig (all hyperparameters)
│   │   ├── trainer.py                     # Alternating encoder/decoder training, validation, checkpointing
│   │   └── utils.py                       # Losses, DDP setup, seeding
│   ├── inference/
│   │   └── inference_turboae.py           # Monte Carlo BER / BLER evaluation
│   └── features/
│       └── build_pytorch_dataset.py       # InfobitDataset: random information bits
├── trained_models/                        # Pretrained checkpoint and its training log (see the table above)
├── pyproject.toml                         # Package metadata and dependencies
├── uv.lock                                # Locked dependency versions
└── LICENSE
```

## 4. Evaluating the Pretrained Model

[src/train/train_turbo_ae.py](src/train/train_turbo_ae.py) is the single entry point for training and evaluation. All settings are fields of the `TurboAEConfig` dataclass at the top of that file; change a setting by editing its default value. Run every command from the repository root as a module (`python -m src...`), because the code imports the `src` package.

The defaults already evaluate the pretrained model:

```python
# src/train/train_turbo_ae.py, inside TurboAEConfig
pretrained: str | None = "./trained_models/Turbo_AutoEncoder_20261002_13_37_35/Turbo_AutoEncoder_20261002_13_37_35_epoch600.pth"
is_inference: bool = True  # skip training, only evaluate `pretrained`
test_snr_low: float = -1.5  # dB
test_snr_high: float = 2.5  # dB
test_snr_interval: float = 0.5  # dB
test_data_samples: int = 10_000_000  # packets per SNR point, split across all GPUs
```

```bash
python -m src.train.train_turbo_ae --p-array-seed 5
```

The BER and BLER at each SNR are printed and logged to W&B. Pass `--p-array-seed 5`: the checkpoint was trained with interleaver seed 5, not the code default of 3, and with seed 3 the BER at 2 dB rises from 1.6e-5 to 2.5e-5. The full sweep takes about an hour on one RTX 4090. For a quick check, lower `test_data_samples`. To split the packets across several GPUs, use the `torchrun` command from [Section 5.1](#51-running).

## 5. Training from Scratch

### 5.1 Running

Set these fields in `TurboAEConfig`:

```python
pretrained: str | None = None  # start from random weights
is_inference: bool = False  # train, then evaluate
```

Then start training on one GPU:

```bash
python -m src.train.train_turbo_ae --p-array-seed 5  # optional: --seed 2
```

or on several GPUs with DistributedDataParallel, as for the pretrained model:

```bash
torchrun --standalone --nproc_per_node=2 -m src.train.train_turbo_ae --use-ddp --p-array-seed 5
```

Each epoch follows Algorithm 1 of the paper: `num_train_enc` encoder steps at 1 dB, then `num_train_dec` decoder steps with noise levels drawn between −1.5 and 2 dB, each step on a fresh batch of random bits. When the validation loss (at 1 dB) has not improved for 60 epochs, the batch size doubles (up to 4000); after that, the learning rate drops by a factor of 10 (down to $10^{-6}$).

### 5.2 Outputs

- **Checkpoints** are written to `trained_models/Turbo_AutoEncoder_<timestamp>/Turbo_AutoEncoder_<timestamp>_epoch<N>.pth` every `save_every` epochs (default 600, i.e., only the final model).
- **W&B** receives, per epoch, the training and validation loss, the validation BER and BLER at 1 dB, the batch size, and the learning rate. After training, it also receives the BER/BLER-versus-SNR tables. The W&B project name is set in `model_pipeline()`; change it to your own project.

### 5.3 Configuration reference

The `TurboAEConfig` fields behind the hyperparameters in Table 1 of the paper:

| Paper | Field | Setting | Note |
| --- | --- | ---: | --- |
| Block length $K$ | `input_size` | 100 | Information bits per packet; $N = 3K$ |
| Encoder | `enc_num_layer`, `enc_num_unit`, `kernel_size` | 5, 100, 5 | Layers, filters, and kernel size of each $f_{i,\theta}$; the paper uses 2 layers |
| Decoder | `dec_num_layer`, `dec_num_unit`, `kernel_size` | 5, 100, 5 | The same for each $g_{\phi_{i,j}}$ |
| Decoder iterations | `num_iteration` | 6 | |
| Info feature size $F$ | `num_iter_ft` | 5 | Extrinsic features exchanged between the decoders |
| $T_\text{enc}$, $T_\text{dec}$ | `num_train_enc`, `num_train_dec` | 100, 500 | Encoder and decoder steps per epoch |
| Batch size | `batch_size`, `num_samples` | 500 | Initial value; the paper doubles it after 20 stalled epochs, up to 2000 |
| Optimizer | `learning_rate` | 1e-4 | Adam, reduced on plateaus |
| Encoder SNR | `train_enc_snr_low`, `train_enc_snr_high` | 1.0, 1.0 dB | Training SNR of the encoder |
| Decoder SNR | `train_dec_snr_low`, `train_dec_snr_high` | −1.5, 2.0 dB | Training SNR range of the decoder |
| Epochs $M$ | `epochs` | 600 | The paper uses 800 |
| Loss | `loss_function` | BCE | Binary cross-entropy on the bit estimates |

## 6. License

This project is released under the [MIT License](LICENSE).

Questions and bug reports are welcome on the [issue tracker](https://github.com/chenbin234/Parallel-Turbo-Autoencoder-over-AWGN/issues).
