import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  


def load_qwen():  
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)  

    # Freeze all parameters except LM head bias
    for param in model.parameters():
        param.requires_grad = False

    # Add trainable bias to logits
    assert model.lm_head.bias is None 
    model.lm_head.bias = torch.nn.Parameter(
        torch.zeros(model.config.vocab_size, device=model.device)  
    )
    model.lm_head.bias.requires_grad = True  

    return model, tokenizer 