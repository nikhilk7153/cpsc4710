"""
Compare base model vs PPO-trained model on math
"""
import torch
import json
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# Test problems with answers
TEST_PROBLEMS = [
    {"prompt": "What is 15 + 27?", "answer": "42"},
    {"prompt": "Solve: 3x = 21. What is x?", "answer": "7"},
    {"prompt": "What is 8 × 7?", "answer": "56"},
    {"prompt": "If a triangle has angles 30° and 60°, what is the third angle?", "answer": "90"},
    {"prompt": "What is 144 ÷ 12?", "answer": "12"},
    {"prompt": "Solve: x + 5 = 12. What is x?", "answer": "7"},
    {"prompt": "What is 25% of 80?", "answer": "20"},
    {"prompt": "What is the square root of 81?", "answer": "9"},
    {"prompt": "If 2x + 4 = 10, what is x?", "answer": "3"},
    {"prompt": "What is 17 - 9?", "answer": "8"},
    {"prompt": "What is 6²?", "answer": "36"},
    {"prompt": "Solve: 5x = 45. What is x?", "answer": "9"},
    {"prompt": "What is 100 ÷ 4?", "answer": "25"},
    {"prompt": "What is 13 × 4?", "answer": "52"},
    {"prompt": "If a rectangle has length 8 and width 5, what is its area?", "answer": "40"},
    {"prompt": "What is 2³?", "answer": "8"},
    {"prompt": "Solve: x - 7 = 15. What is x?", "answer": "22"},
    {"prompt": "What is 50% of 120?", "answer": "60"},
    {"prompt": "What is 11 × 11?", "answer": "121"},
    {"prompt": "What is the cube root of 27?", "answer": "3"},
]

def extract_number(text):
    patterns = [
        r'(?:=|is|equals?)\s*(\d+)',
        r'(\d+)\s*$',
        r'\*\*(\d+)\*\*',
        r'x\s*=\s*(\d+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches[-1]
    return None

def evaluate_model(model, tokenizer, rm_model, rm_tokenizer, problems, device):
    """Evaluate model"""
    correct = 0
    total_score = 0
    
    for prob in tqdm(problems, desc="Evaluating"):
        messages = [{"role": "user", "content": prob["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # SARM score
        rm_messages = [{"role": "user", "content": prob["prompt"]}, {"role": "assistant", "content": response}]
        rm_text = rm_tokenizer.apply_chat_template(rm_messages, tokenize=False)
        rm_enc = rm_tokenizer(rm_text, return_tensors='pt', truncation=True, max_length=512).to(device)
        score = rm_model(**rm_enc).logits.item()
        total_score += score
        
        # Check correctness
        extracted = extract_number(response)
        if extracted == prob["answer"]:
            correct += 1
    
    return {
        "accuracy": correct / len(problems),
        "avg_sarm_score": total_score / len(problems),
        "correct": correct,
        "total": len(problems)
    }

def main():
    device = "cuda:0"
    
    print("Loading SARM...")
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        'Schrieffer/Llama-SARM-4B',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device
    ).eval()
    rm_tokenizer = AutoTokenizer.from_pretrained('Schrieffer/Llama-SARM-4B')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
    
    # === BASE MODEL ===
    print("\n" + "="*60)
    print("EVALUATING BASE MODEL")
    print("="*60)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map=device
    ).eval()
    
    base_results = evaluate_model(base_model, tokenizer, rm_model, rm_tokenizer, TEST_PROBLEMS, device)
    print(f"\nBase Model Results:")
    print(f"  Accuracy: {base_results['accuracy']:.1%} ({base_results['correct']}/{base_results['total']})")
    print(f"  Avg SARM Score: {base_results['avg_sarm_score']:.4f}")
    
    del base_model
    torch.cuda.empty_cache()
    
    # === FINE-TUNED MODEL ===
    print("\n" + "="*60)
    print("EVALUATING PPO-TRAINED MODEL")
    print("="*60)
    
    ft_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.1-8B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    ft_model = PeftModel.from_pretrained(ft_model, 'outputs/experiments/math_ppo_quick/checkpoint')
    ft_model = ft_model.merge_and_unload()
    ft_model.eval()
    
    ft_results = evaluate_model(ft_model, tokenizer, rm_model, rm_tokenizer, TEST_PROBLEMS, device)
    print(f"\nPPO-Trained Model Results:")
    print(f"  Accuracy: {ft_results['accuracy']:.1%} ({ft_results['correct']}/{ft_results['total']})")
    print(f"  Avg SARM Score: {ft_results['avg_sarm_score']:.4f}")
    
    # === COMPARISON ===
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"                    Base Model    PPO-Trained    Change")
    print(f"  Accuracy:         {base_results['accuracy']:.1%}           {ft_results['accuracy']:.1%}          {ft_results['accuracy'] - base_results['accuracy']:+.1%}")
    print(f"  SARM Score:       {base_results['avg_sarm_score']:.4f}        {ft_results['avg_sarm_score']:.4f}       {ft_results['avg_sarm_score'] - base_results['avg_sarm_score']:+.4f}")
    
    # Save results
    results = {
        "base": base_results,
        "ppo_trained": ft_results,
        "improvement": {
            "accuracy_delta": ft_results['accuracy'] - base_results['accuracy'],
            "sarm_score_delta": ft_results['avg_sarm_score'] - base_results['avg_sarm_score']
        }
    }
    
    Path('outputs/experiments/math_ppo_quick').mkdir(parents=True, exist_ok=True)
    with open('outputs/experiments/math_ppo_quick/comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved comparison to outputs/experiments/math_ppo_quick/comparison.json")

if __name__ == "__main__":
    main()
