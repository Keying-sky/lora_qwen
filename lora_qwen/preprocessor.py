import h5py
import numpy as np
from transformers import AutoTokenizer

def load_raw(data_path):

    with h5py.File(data_path, "r") as f:
        trajectories = f["trajectories"][:]
        time_points = f["time"][:]
    
    return trajectories, time_points

def preprocess_trajectory(trajectory, alpha=10, decimal=2):
    """
    Preprocess a single trajectory: scale, round and format it into string.

    Params:
        trajectory: The dataset of one system, array of shape (time_points, 2);
        alpha: The scaling factor, default 10;
        decimal: The number of decimal places, default 2.

    Return:
        llmt_strings: list of strings in LLMTIME format
    """
    scaled = trajectory / alpha

    strings = []
    for t in range(scaled.shape[0]):
        prey = f"{scaled[t, 0]:.{decimal}f}"
        predator = f"{scaled[t, 1]:.{decimal}f}"
        strings.append(f"{prey},{predator}")  # the comma here is string
    
    # join with a semicolon
    llmt_strings = ";".join(strings)
    
    return llmt_strings

def examples(data_path, nexamples=2, ntimes=10):
    """
    Get example sequences and output their preprocessed and tokenized results.
    
    Params:
        data_path: Path of dataset;
        nexamples: Number of example sequences (systems) to display, default 2;
        ntimes: Number of time points (per example) to display, default 10.
    
    Returns:
        raws: The raw examples;
        preprocesseds: The preprocessed examples;
        tokenizeds: The tokenized examples.
    """
    np.random.seed(1224)

    trajectories, _ = load_raw(data_path)
    
    # randomly choose index of 2 systems without duplication
    idxs = np.random.choice(trajectories.shape[0], nexamples, replace=False)
    
    raws = []
    preprocesseds = []
    tokenizeds = []

    for idx in idxs:
        # take the first ntimes points of the trajectory
        raw = trajectories[idx, :ntimes, :]
        raws.append(raw)

        # preprocess the examples
        preprocessed = preprocess_trajectory(raw)
        preprocesseds.append(preprocessed)
    
    # tokenize the examples
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

    for preprocessed in preprocesseds:
        tokenize = tokenizer(preprocessed, return_tensors="pt", add_special_tokens=False)
        tokenizeds.append(tokenize.input_ids[0])
    
    return raws, preprocesseds, tokenizeds

    