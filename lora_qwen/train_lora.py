import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from accelerate import Accelerator 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import time

from .preprocessor import load_raw, preprocess_trajectory
from .qwen import load_qwen
from .evaluate import evaluate_model, plot_examples


class LoRALinear(nn.Module):
    """
    LoRA implementation for linear layers, adds a low-rank update to the weight matrix without modifying the original weights.
    
    Params:
        original_linear: The original linear layer to be wrapped
        r: Rank of the low-rank matrices
        alpha: Scaling factor (typically equal to r)
    """
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        super().__init__()
        assert isinstance(original_linear, nn.Linear)
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False  # Freeze original weights
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False  # Freeze bias if present
        
        # Get dimensions from original layer
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.alpha = alpha if alpha else r  # Default alpha to r if not specified
        
        # Create A and B matrices for decomposition
        device = original_linear.weight.device
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device))
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))
        
        # Initialise A with He initialisation for stable training
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")
    
    def forward(self, x):
        """
        Forward pass combining original weights with LoRA adaptation

        Param: Input tensor x
        Return: Combined output from original layer and LoRA adaptation
        """
        base_out = self.original_linear(x) # Original output from frozen weights
        lora_out = (x @ self.A.T) @ self.B.T # LoRA adaptation: x @ A.T @ B.T
        return base_out + lora_out * (self.alpha / self.r) # Combine with scaling factor


def apply_lora_to_model(model, rank=4):
    """
    Apply LoRA to specific layers in the Qwen2.5-Instruct model.
    
    Params:
        model: The Qwen2.5-Instruct model
        rank: Rank for LoRA layers
        
    Return: Modified model with LoRA layers
    """
    # Find all Q and V projection layers in attention blocks and replace with LoRA versions
    for layer in model.model.layers:
        # Replace query projection layer with LoRA
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=rank)
        
        # Replace value projection layer with LoRA
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=rank)

    return model


def process_sequences(texts, tokenizer, max_length=512, stride=256):
    """
    Process text sequences into tokenized chunks with sliding windows.
    
    Params:
        texts: List of text sequences
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        stride: Stride for sliding window
        
    Return: Tensor of token IDs
    """
    all_input_ids = []
    
    for text in texts:
        # tokenize text without special tokens
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]
        
        # create sliding windows
        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i : i + max_length]
            
            # pad short sequences
            if len(chunk) < max_length:
                chunk = torch.cat([
                    chunk, 
                    torch.full((max_length - len(chunk),), tokenizer.pad_token_id)
                ])
            all_input_ids.append(chunk)
    
    return torch.stack(all_input_ids)


def load_and_preprocess_data(data_path, tokenizer, max_ctx_length=512, train_split=0.8):
    """
    Load and preprocess data for training.
    
    Params:
        data_path: Path to the data file
        tokenizer: Tokenizer for the model
        max_ctx_length: Maximum context length
        train_split: Fraction of data to use for training
        
    Returns:
        Training and validation data loaders
    """
    
    np.random.seed(1224)
    torch.manual_seed(1224)
    
    trajectories, _ = load_raw(data_path)
    
    # split trajectories into train and validation sets
    num_trajectories = trajectories.shape[0]
    indices = np.random.permutation(num_trajectories)
    train_size = int(train_split * num_trajectories)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_trajectories = trajectories[train_indices]
    val_trajectories = trajectories[val_indices]
    
    # process trajectories into text sequences
    train_texts = [preprocess_trajectory(traj) for traj in train_trajectories]
    val_texts = [preprocess_trajectory(traj) for traj in val_trajectories]
    
    # ttokenize and create sliding windows
    train_input_ids = process_sequences(
        train_texts, tokenizer, max_ctx_length, stride=max_ctx_length // 2
    )
    val_input_ids = process_sequences(
        val_texts, tokenizer, max_ctx_length, stride=max_ctx_length
    )
    
    # create datasets and data loaders
    train_dataset = TensorDataset(train_input_ids)
    val_dataset = TensorDataset(val_input_ids)
    
    return train_dataset, val_dataset, val_trajectories


def train_lora_model(
    data_path, 
    results_path, 
    lora_rank=4, 
    learning_rate=1e-5, 
    batch_size=3, 
    max_steps=5000, 
    max_ctx_length=512,
    eval_ctx_length=50,
    save_checkpoint_steps=1000,
    eval_steps=1000,
    early_stopping_patience=None,
    resume_from_checkpoint=None
):
    """
    Train a model with LoRA.
    
    Params:
        data_path: Path to the data file
        results_path: Path to save results
        lora_rank: Rank of LoRA matrices
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        max_steps: Maximum number of training steps
        max_ctx_length: Maximum context length
        save_checkpoint_steps: Steps between checkpoints
        eval_steps: Steps between evaluations
        early_stopping_patience: Number of evaluations with no improvement before stopping
        
    Returns: Trained model and training metrics
    """
    np.random.seed(1224)
    torch.manual_seed(1224)

    results_path.mkdir(exist_ok=True)
    
    # log hyperparameters
    hyperparams = {
        'lora_rank': lora_rank,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'max_steps': max_steps,
        'max_ctx_length': max_ctx_length,
        'eval_ctx_length':eval_ctx_length,
        'early_stopping_patience': early_stopping_patience
    }
    
    with open(results_path / "hyperparameters.json", "w") as f:
        json.dump(hyperparams, f, indent=4)

    model, tokenizer = load_qwen()
    
    model = apply_lora_to_model(model, rank=lora_rank)
    
    # count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    

    train_dataset, _, val_trajectories = load_and_preprocess_data(data_path, tokenizer, max_ctx_length=max_ctx_length)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )

    accelerator = Accelerator()
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # training loop
    model.train()
    step = 0
    training_losses = []
    eval_metrics = []
    best_val_mse = float('inf')
    no_improvement_count = 0
    total_training_time = 0

    # resume from checkpoint
    if resume_from_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        
        if 'training_losses' in checkpoint:
            training_losses = checkpoint['training_losses']
        if 'eval_metrics' in checkpoint:
            eval_metrics = checkpoint['eval_metrics']
        if 'best_val_mse' in checkpoint:
            best_val_mse = checkpoint['best_val_mse']
        if 'no_improvement_count' in checkpoint:
            no_improvement_count = checkpoint['no_improvement_count']
        if 'total_training_time' in checkpoint:
            total_training_time = checkpoint['total_training_time']
        print(f"Resuming from step {step}, previous training time: {total_training_time:.2f}s")

    # start training
    start_time = time.time()

    while step < max_steps:
        progress_bar = tqdm(train_loader, desc=f"Step {step}")
        
        for batch in progress_bar:
            batch = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            
            outputs = model(batch[0], labels=batch[0])
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            training_losses.append(loss.item())
            progress_bar.set_postfix(loss=loss.item())
            
            step += 1
            
            # checkpoint
            if step % save_checkpoint_steps == 0:
                current_session_time = time.time() - start_time
                current_total_time = total_training_time + current_session_time

                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'training_losses': training_losses, 
                    'eval_metrics': eval_metrics,
                    'best_val_mse': best_val_mse,
                    'no_improvement_count': no_improvement_count,
                    'total_training_time': current_total_time  
                }, results_path / f"checkpoint_step_{step}.pt")
            
            # evaluate model on validation set
            if step % eval_steps == 0:
                print(f"\nEvaluating at step {step}...")
                eval_model = model.eval()
                
                metrics, _, _ = evaluate_model(
                    eval_model, 
                    tokenizer, 
                    data_path,
                    cxt_len=eval_ctx_length,
                    nsamples=10,
                    use_val_set=True,  
                    val_trajectories=val_trajectories 
                )
                
                # log and save metrics
                print("Validation metrics:")
                for name, value in metrics.items():
                    print(f"  {name}: {value:.4f}")
            
                eval_metrics.append({
                    'step': step,
                    'metrics': metrics,
                })
                
                # check improvement
                if metrics["MSE"] < best_val_mse:
                    best_val_mse = metrics["MSE"]
                    no_improvement_count = 0

                    current_session_time = time.time() - start_time
                    current_total_time = total_training_time + current_session_time

                    # save best model
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metrics,
                        'total_training_time': current_total_time 
                    }, results_path / "best_model.pt")
                    
                else:
                    no_improvement_count += 1
                
                # early stopping
                if early_stopping_patience is not None and no_improvement_count >= early_stopping_patience:
                    print(f"No improvement for {early_stopping_patience} evaluations. Stopping early.")
                    break

                model.train()
            
            if step >= max_steps:
                break

        if early_stopping_patience is not None and no_improvement_count >= early_stopping_patience:
            break

    current_session_time = time.time() - start_time
    total_training_time += current_session_time

    # evaluate final model on the same data
    model.eval()
    final_metrics, final_forecasts, final_actuals = evaluate_model(
        model, tokenizer, data_path,
        cxt_len=eval_ctx_length,
        nsamples=10
    )
    
    # save final metrics and model
    print("\nFinal validation metrics:")
    for name, value in final_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': final_metrics,
    }, results_path / "final_model.pt")
    
    # plot final examples
    plot_examples(final_forecasts, final_actuals, results_path, nexamples=2)
    
    # plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(results_path / "training_loss.png")
    plt.close()
    
    # save results
    actual_steps = len(training_losses)
    
    results = {
        'hyperparameters': hyperparams,
        'training_time': total_training_time,
        'actual_steps': actual_steps,
        'early_stopped': actual_steps < max_steps,
        'final_metrics': final_metrics
    }
    
    with open(results_path / "results.json", "w") as f:
        serializable_results = {k: (float(v) if isinstance(v, np.float32) else v) 
                               for k, v in results.items()}
        json.dump(serializable_results, f, indent=4)

    with open(results_path / "training_history.pkl", "wb") as f:
        pickle.dump({
            'training_losses': training_losses,
            'eval_metrics': eval_metrics,
            'final_metrics': final_metrics,
        }, f)
    
    return model, {
        'training_losses': training_losses,
        'eval_metrics': eval_metrics,
        'final_metrics': final_metrics,
    }
