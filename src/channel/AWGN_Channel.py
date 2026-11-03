# This file is to build a AWGN channel with input (transmitter_signal, SNR) and output (output_signal)
# The channel is defined as:
#     output_signal = transmitter_signal + noise
#     noise ~ N(0, 1/SNR)
# The input signal and output signal are both in the form of torch tensor
# The input SNR is a scalar
# The output signal is the same shape as the input signal

import torch


def AWGN_Channel_new(transmitter_signal, SNR_db_low, SNR_db_high):
    """
    Function to add AWGN to the input signal.

    Args:
        transmitter_signal (torch.Tensor): The encoded signal of size (batch_size, k, 3).
        SNR_db_low (float): The lower bound of the signal-to-noise ratio in dB.
        SNR_db_high (float): The upper bound of the signal-to-noise ratio in dB.

    Returns:
        torch.Tensor: The output signal.
    """
    device = transmitter_signal.device

    # Convert SNR from dB to linear scale
    sigma_low = torch.tensor(10 ** (-SNR_db_low * 1.0 / 20)).to(device)
    sigma_high = torch.tensor(10 ** (-SNR_db_high * 1.0 / 20)).to(device)

    # mix the SNR
    this_sigma = (
        torch.rand(transmitter_signal.shape).to(device) * (sigma_low - sigma_high)
        + sigma_high
    )

    noise = this_sigma * torch.randn(transmitter_signal.shape, dtype=torch.float).to(
        device
    )

    output_signal = transmitter_signal + noise

    return output_signal
