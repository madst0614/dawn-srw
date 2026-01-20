#!/usr/bin/env python3
"""
Generation Comparison for Paper
===============================

Generates text samples from DAWN and Baseline models side-by-side for paper.

Usage:
    python scripts/analysis/generate_comparison.py \
        --dawn /path/to/dawn/checkpoint \
        --baseline /path/to/baseline/checkpoint \
        --val_data /path/to/val_data.pt \
        --output paper_generations.txt

Output format:
    Each prompt shows DAWN and Baseline outputs side by side for easy comparison.
"""

import argparse
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from models import create_model_by_version, normalize_version


# Paper-quality prompts organized by category
PAPER_PROMPTS = {
    'factual_knowledge': [
        "The capital of France is",
        "The largest planet in our solar system is",
        "Water boils at",
        "The speed of light is approximately",
        "Albert Einstein was born in",
    ],
    'common_sense': [
        "If you drop a glass on the floor, it will",
        "When it rains, people usually",
        "Fire is hot, but ice is",
        "Birds fly in the sky, fish swim in",
        "At night, the sun is",
    ],
    'narrative_continuation': [
        "Once upon a time, in a small village,",
        "The detective examined the evidence and",
        "She opened the door and saw",
        "After years of hard work, he finally",
        "The storm was approaching, so they",
    ],
    'technical': [
        "In machine learning, gradient descent is",
        "The function of the mitochondria is",
        "HTTP stands for",
        "A neural network consists of",
        "The chemical formula for water is",
    ],
}


def find_checkpoint_file(ckpt_path):
    """Find checkpoint file from path (handles directories)"""
    ckpt_path = Path(ckpt_path)

    if ckpt_path.is_file():
        return ckpt_path

    if ckpt_path.is_dir():
        candidates = ['best_model.pt', 'checkpoint_best.pt', 'model.pt', 'latest.pt']
        for name in candidates:
            if (ckpt_path / name).exists():
                return ckpt_path / name

        pt_files = list(ckpt_path.glob('*.pt'))
        if pt_files:
            return sorted(pt_files, key=lambda f: f.stat().st_mtime)[-1]

    raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    ckpt_file = find_checkpoint_file(checkpoint_path)
    print(f"  Loading: {ckpt_file}")

    ckpt = torch.load(ckpt_file, map_location='cpu')
    model_config = ckpt.get('model_config', ckpt.get('config', {}))

    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = ckpt

    # Auto-detect version
    version = model_config.get('model_version', None)
    if version is None:
        dawn_keys = ['shared_neurons.f_neurons', 'router.neuron_router.neuron_emb']
        if any(k in state_dict for k in dawn_keys):
            version = '17.1'
        else:
            version = 'baseline'

    version = normalize_version(version)
    model = create_model_by_version(version, model_config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Get name from path
    ckpt_path = Path(checkpoint_path)
    name = ckpt_path.name if ckpt_path.is_dir() else ckpt_path.parent.name

    return model, version, name


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=50, device='cuda'):
    """Generate text with top-k sampling"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
    generated = input_ids.clone()
    eos_token_id = tokenizer.sep_token_id

    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(generated, attention_mask=None)
            logits = output[0] if isinstance(output, tuple) else output
            next_token_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def format_output(text, max_len=60):
    """Format output text for display"""
    # Remove prompt from output if it starts with it
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def main():
    parser = argparse.ArgumentParser(description='DAWN vs Baseline Generation Comparison')
    parser.add_argument('--dawn', type=str, required=True, help='Path to DAWN checkpoint')
    parser.add_argument('--baseline', type=str, required=True, help='Path to Baseline checkpoint')
    parser.add_argument('--val_data', type=str, help='Path to validation data (optional)')
    parser.add_argument('--output', type=str, default='paper_generations.txt', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--gen_tokens', type=int, default=50, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--n_samples', type=int, default=1, help='Number of samples per prompt')
    args = parser.parse_args()

    # Device check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load models
    print("\n" + "=" * 80)
    print("Loading DAWN model...")
    dawn_model, dawn_version, dawn_name = load_model(args.dawn, args.device)
    print(f"  Version: {dawn_version}, Name: {dawn_name}")

    print("\nLoading Baseline model...")
    baseline_model, baseline_version, baseline_name = load_model(args.baseline, args.device)
    print(f"  Version: {baseline_version}, Name: {baseline_name}")
    print("=" * 80)

    # Generate and compare
    lines = []
    lines.append("=" * 100)
    lines.append("GENERATION COMPARISON: DAWN vs Baseline")
    lines.append("=" * 100)
    lines.append(f"DAWN Model:     {dawn_name} (v{dawn_version})")
    lines.append(f"Baseline Model: {baseline_name} (v{baseline_version})")
    lines.append(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.gen_tokens}")
    lines.append("=" * 100)

    print("\n" + "\n".join(lines[-6:]))

    for category, prompts in PAPER_PROMPTS.items():
        category_header = f"\n{'='*100}\n[{category.upper().replace('_', ' ')}]\n{'='*100}"
        lines.append(category_header)
        print(category_header)

        for prompt in prompts:
            prompt_header = f"\n{'─'*100}\nPrompt: \"{prompt}\"\n{'─'*100}"
            lines.append(prompt_header)
            print(prompt_header)

            for sample_idx in range(args.n_samples):
                # Generate from both models
                dawn_output = generate_text(
                    dawn_model, tokenizer, prompt,
                    max_new_tokens=args.gen_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=args.device
                )

                baseline_output = generate_text(
                    baseline_model, tokenizer, prompt,
                    max_new_tokens=args.gen_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=args.device
                )

                # Format outputs (remove prompt prefix if present)
                dawn_gen = dawn_output[len(prompt):].strip() if dawn_output.startswith(prompt) else dawn_output
                baseline_gen = baseline_output[len(prompt):].strip() if baseline_output.startswith(prompt) else baseline_output

                if args.n_samples > 1:
                    sample_label = f" (Sample {sample_idx + 1})"
                else:
                    sample_label = ""

                dawn_line = f"  DAWN{sample_label}:     {dawn_gen}"
                baseline_line = f"  Baseline{sample_label}: {baseline_gen}"

                lines.append(dawn_line)
                lines.append(baseline_line)
                print(dawn_line)
                print(baseline_line)

                if args.n_samples > 1 and sample_idx < args.n_samples - 1:
                    lines.append("")
                    print("")

    # Summary
    summary = f"\n{'='*100}\nGeneration complete. {len(PAPER_PROMPTS)} categories, {sum(len(p) for p in PAPER_PROMPTS.values())} prompts.\n{'='*100}"
    lines.append(summary)
    print(summary)

    # Save to file
    output_text = "\n".join(lines)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(output_text)

    print(f"\nResults saved to: {args.output}")

    # Cleanup
    del dawn_model, baseline_model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
