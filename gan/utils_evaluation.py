# imports from installed libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
from distutils.version import LooseVersion as Version


def set_all_seeds(seed):
    """
    Set the seed for all relevant RNGs to ensure reproducibility across runs.

    This function sets a fixed seed for random number generators in os, random,
    numpy, and torch, ensuring that the same sequences of random numbers will be
    generated across different program executions when the same seed is used. It is
    particularly useful when trying to reproduce results in machine learning experiments.

    credits: Sebastian Raschka
    """

    # Set the seed for generating random numbers in Python's os module
    os.environ["PL_GLOBAL_SEED"] = str(seed)

    # Set the seed for the default Python RNG
    random.seed(seed)

    # Set the seed for numpy's RNG
    np.random.seed(seed)

    # Set the seed for PyTorch's RNG
    torch.manual_seed(seed)

    # Ensure that CUDA kernels' randomness is also seeded if available
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    """
    Enforces deterministic behavior in PyTorch operations to ensure reproducibility.

    This function configures PyTorch to behave deterministically, especially when running
    on a CUDA (GPU) environment. It disables certain optimizations that introduce non-determinism,
    making sure that the same inputs across different runs produce the same outputs.

    Note: Some PyTorch operations do not support deterministic mode, and using this function
    may have performance implications due to disabled optimizations.

    credits: Sebastian Raschka
    """

    # If CUDA (GPU support) is available, set related options for deterministic behavior
    if torch.cuda.is_available():
        # Disable the auto-tuner that finds the best algorithm for a specific input configuration.
        # This is necessary for reproducibility as different algorithms might produce slightly different results.
        torch.backends.cudnn.benchmark = False

        # Enable CuDNN deterministic mode. This ensures that convolution operations are deterministic.
        torch.backends.cudnn.deterministic = True

    # Set the deterministic flag based on the version of PyTorch.
    # Different versions of PyTorch use different functions to enforce deterministic algorithms.
    if torch.__version__ <= Version("1.7"):
        # For versions 1.7 or older, use `torch.set_deterministic`
        torch.set_deterministic(True)
    else:
        # From version 1.8 forward, use `torch.use_deterministic_algorithms`
        torch.use_deterministic_algorithms(True)


def compare_dfs(df_real, df_fake, table_groupby=[], figsize=3, save=False, path=""):
    """
    Diagnostic function for comparing real and generated (factual) data
    from WGAN models. Prints out comparison tables of means, and standard
    deviations, covariance matrix

    The comapred columns are extract from the common columns names from
    df_real and df_fake

    Args:
        df_real: pandas.DataFrame
            real data
        df_fake: pandas.DataFrame
            data produced by generator
        table_groupby: list
            List of variables to group mean and standard deviation tables by
        save: bool
            Indicate whether to save results to file or print them
        path: string
            Path to save diagnostics for model
    """
    # data prep
    if "source" in list(df_real.columns):
        df_real = df_real.drop("source", axis=1)
    if "source" in list(df_fake.columns):
        df_fake = df_fake.drop("source", axis=1)

    # insert new column at loc=0
    df_real.insert(0, "source", "real"), df_fake.insert(0, "source", "fake")

    common_cols = [c for c in df_real.columns if c in df_fake.columns]
    df_joined = pd.concat(
        [df_real[common_cols], df_fake[common_cols]], axis=0, ignore_index=True
    )
    # df_joined is concatenated PandaDataframe (by concatenating rows)

    df_real, df_fake = df_real.drop("source", axis=1), df_fake.drop("source", axis=1)
    common_cols = [c for c in df_real.columns if c in df_fake.columns]

    # mean and std table

    means = df_joined.groupby(table_groupby + ["source"]).mean().round(2).transpose()
    # groupded by two variables, 1. t; 2. source

    if save:
        means.to_csv(path + "_means.txt", sep=" ")
    else:
        print("-------------comparison of means-------------")
        print(means)

    stds = df_joined.groupby(table_groupby + ["source"]).std().round(2).transpose()

    if save:
        stds.to_csv(path + "_stds.txt", sep=" ")
    else:
        print("-------------comparison of stds-------------")
        print(stds)

    # plots for covariance matrix comparison
    fig1 = plt.figure(figsize=(figsize * 2, figsize * 1))
    s1 = [fig1.add_subplot(1, 2, i) for i in range(1, 3)]
    s1[0].set_xlabel("real")
    s1[1].set_xlabel("fake")
    s1[0].matshow(df_real[common_cols].corr())
    s1[1].matshow(df_fake[common_cols].corr())
