import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

from .preprocessor import load_raw, preprocess_trajectory


def decode_prediction(text, alpha=10.0):
    """
    Decode the de-tokenized text into numerical array
    
    Params:
        text: De-tokenized text from model output
        alpha: Scaling factor, same as used in preprocessing
    
    Return: Decoded array, shape (timepoints, 2)
    """
    try:
        timepoints = text.split(';') 
        decoded = []
        
        for step in timepoints:  
            if ',' not in step:
                continue
            
            values = step.split(',')
            if len(values) != 2:
                continue
                
            try:
                prey = float(values[0]) * alpha
                predator = float(values[1]) * alpha
                decoded.append([prey, predator])
            except ValueError:
                continue
                
        return np.array(decoded)
    
    except Exception as e:
        print(f"Decoding Error: {e}")
        return np.array([])

def evaluate_model(model, tokenizer, data_path, cxt_len=50, nsamples=10, ntimes=10, 
               use_val_set=False, val_trajectories=None):
    """
    Evaluate the forecasting performance of the model.
    
    Params:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        data_path: Path to raw data (used when use_val_set=False)
        cxt_len: Context length, input to the model
        nsamples: Sample size evaluated (number of systems choosen)
        ntimes: Number of time points to predict
        use_val_set: Whether to use validation set trajectories
        val_trajectories: Validation set trajectories (required when use_val_set=True)
    
    Return: Evaluation metrics
    """
    model.eval()
    
    # Get trajectories
    if use_val_set:
        # use validation set provided
        if val_trajectories is None:
            raise ValueError("val_trajectories must be provided when use_val_set=True")
        trajectories = val_trajectories
        # use different random samples each time by not fixing the seed here
        idxs = np.random.choice(len(trajectories), min(nsamples, len(trajectories)), replace=False)
    else:
        # load raw data and use fixed seed
        np.random.seed(1224)
        torch.manual_seed(1224)
        trajectories, _ = load_raw(data_path)
        # Randomly choose index of (nsamples) systems without duplication
        idxs = np.random.choice(trajectories.shape[0], nsamples, replace=False)
    
    # metrics
    all_mse = []
    all_mae = []
    all_r2 = []
    prey_mse = []
    predator_mse = []
    forecasts = []
    actuals = []
    
    for idx in tqdm(idxs, desc="Evaluating model"):
        trajectory = trajectories[idx]
        
        # use the first cxt_len time step as the context
        context = trajectory[:cxt_len]
        # use the following ntimes as the target.
        target = trajectory[cxt_len:cxt_len + ntimes]
        
        if len(target) < ntimes:
            continue
        
        context_text = preprocess_trajectory(context)
        
        # tokenize the ctx
        input_ids = tokenizer(context_text, return_tensors="pt").input_ids
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            model = model.cuda()
            
        # predict
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=len(tokenizer(preprocess_trajectory(target)).input_ids),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # decode the output token
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # only take the forecasted part
        if context_text in generated_text:
            predicted_text = generated_text[generated_text.index(context_text) + len(context_text):]
            if predicted_text.startswith(';'):
                predicted_text = predicted_text[1:]
        else:
            predicted_text = generated_text
            
        # decode predictions
        predictions = decode_prediction(predicted_text)
        
        # ensure shapes match
        min_length = min(len(predictions), len(target))
        if min_length == 0:
            continue
            
        predictions = predictions[:min_length]
        target_cut = target[:min_length]
        
        # calculate metrics
        mse = mean_squared_error(target_cut, predictions)
        mae = mean_absolute_error(target_cut, predictions)
        
        try:
            r2 = r2_score(target_cut.flatten(), predictions.flatten())
        except:
            r2 = float('nan')
            
        prey_mse_value = mean_squared_error(target_cut[:, 0], predictions[:, 0])
        predator_mse_value = mean_squared_error(target_cut[:, 1], predictions[:, 1])
        
        all_mse.append(mse)
        all_mae.append(mae)
        all_r2.append(r2)
        prey_mse.append(prey_mse_value)
        predator_mse.append(predator_mse_value)
        
        forecasts.append(predictions)
        actuals.append(target_cut)
    
    metrics = {
        "MSE": np.nanmean(all_mse),
        "MAE": np.nanmean(all_mae),
        "R2": np.nanmean([r for r in all_r2 if not np.isnan(r)]),
        "Prey_MSE": np.nanmean(prey_mse),
        "Predator_MSE": np.nanmean(predator_mse)
    }

    return metrics, forecasts, actuals

def plot_examples(forecasts, actuals, results_path, nexamples=2):
    """
    Visualise the predicted results.
    
    Params:
        forecasts: List of predicted values
        actuals: List of true values
        nexamples: Number of examples to display
"""
    for i in range(nexamples):
        pred = forecasts[i]
        actual = actuals[i]
        
        plt.figure(figsize=(8, 5))
        
        plt.plot(actual[:, 0], 'bo-', label='True prey')
        plt.plot(pred[:, 0], 'bs--', label='Predicted prey')
        plt.plot(actual[:, 1], 'ro-', label='True pradetor')
        plt.plot(pred[:, 1], 'rs--', label='Predicted pradetor')

        plt.title('Population Prediction')
        plt.xlabel('time steps')
        plt.ylabel('population')
        plt.legend()

        plt.savefig(results_path/f"example_{i+1}.png", dpi=300)
        plt.close()
