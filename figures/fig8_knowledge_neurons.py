#!/usr/bin/env python3
"""
Figure 8 (Appendix): Knowledge Neurons

Visualizes neuron activation patterns across different factual knowledge prompts.
Shows which neurons are specialized for specific facts vs. shared across facts.

Usage:
    python figures/fig8_knowledge_neurons.py --input routing_analysis/factual
    python figures/fig8_knowledge_neurons.py --demo  # Use demo data
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paper style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

FIGURES_DIR = Path(__file__).parent


def load_json_results(input_dir: Path) -> dict:
    """Load JSON results from routing analysis."""
    results = {}

    # Map filenames to display labels
    label_map = {
        'paris': 'Paris',
        'berlin': 'Berlin',
        'tokyo': 'Tokyo',
        'blue': 'Blue',
        'france': 'Paris',
        'germany': 'Berlin',
        'japan': 'Tokyo',
        'sky': 'Blue',
    }

    for json_file in sorted(input_dir.glob('*.json')):
        name = json_file.stem.lower()

        # Skip comparison files
        if 'comparison' in name:
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            # Determine label from filename
            label = None
            for key, lbl in label_map.items():
                if key in name:
                    label = lbl
                    break

            if label and label not in results:
                results[label] = data
                print(f"  Loaded: {json_file.name} -> {label}")

        except Exception as e:
            print(f"  Error loading {json_file}: {e}")

    return results


def extract_neuron_frequencies(results: dict, top_k: int = 15) -> tuple:
    """Extract neuron activation frequencies from results."""
    # Collect all neuron frequencies across all facts
    all_neurons = defaultdict(float)
    fact_neurons = {}

    for fact, data in results.items():
        fact_neurons[fact] = {}

        # Get neuron_frequencies from the JSON
        neuron_freqs = data.get('neuron_frequencies', [])

        for nf in neuron_freqs:
            if isinstance(nf, dict):
                nid = str(nf.get('neuron', ''))
                pct = nf.get('percentage', 0)
            else:
                continue

            fact_neurons[fact][nid] = pct
            all_neurons[nid] += pct

    # Get top-k neurons by total frequency across all facts
    sorted_neurons = sorted(all_neurons.items(), key=lambda x: x[1], reverse=True)
    top_neuron_ids = [nid for nid, _ in sorted_neurons[:top_k]]

    return top_neuron_ids, fact_neurons


def create_demo_data() -> tuple:
    """Create demo data for testing."""
    facts = ['Paris', 'Berlin', 'Tokyo', 'Blue']

    # Demo neuron IDs
    neuron_ids = ['142', '87', '203', '56', '178',
                  '95', '312', '44', '267', '159',
                  '88', '201', '73', '156', '289']

    # Demo frequencies (percentage)
    data = np.array([
        [92, 88, 85, 15],   # 142 - capital specialist
        [78, 82, 75, 12],   # 87 - capital specialist
        [45, 42, 48, 89],   # 203 - color specialist
        [65, 68, 62, 45],   # 56 - general
        [72, 75, 70, 18],   # 178 - capital leaning
        [55, 58, 52, 65],   # 95 - mixed
        [38, 35, 40, 82],   # 312 - color leaning
        [82, 78, 80, 22],   # 44 - capital specialist
        [48, 52, 45, 75],   # 267 - color leaning
        [62, 65, 60, 55],   # 159 - general
        [70, 72, 68, 25],   # 88 - capital leaning
        [42, 45, 40, 78],   # 201 - color leaning
        [58, 55, 60, 48],   # 73 - general
        [75, 78, 72, 20],   # 156 - capital specialist
        [35, 38, 32, 85],   # 289 - color specialist
    ]).T  # Transpose to [facts, neurons]

    return neuron_ids, facts, data


def create_heatmap(neuron_ids: list, facts: list, data: np.ndarray, output_path: Path):
    """Create the factual knowledge heatmap."""
    fig, ax = plt.subplots(figsize=(8, 3.2))

    # Create heatmap
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    # Set ticks
    ax.set_xticks(np.arange(len(neuron_ids)))
    ax.set_yticks(np.arange(len(facts)))
    ax.set_xticklabels([f'N{n}' for n in neuron_ids], rotation=45, ha='right')
    ax.set_yticklabels(facts)

    # Labels
    ax.set_xlabel('Neuron ID (Top 15 by Frequency)')
    ax.set_ylabel('Target Token')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Activation Frequency (%)', rotation=270, labelpad=15)

    # Add value annotations
    for i in range(len(facts)):
        for j in range(len(neuron_ids)):
            val = data[i, j]
            color = 'white' if val > 60 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                   color=color, fontsize=7)

    # Grid
    ax.set_xticks(np.arange(len(neuron_ids) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(facts) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='minor', size=0)

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), facecolor='white', edgecolor='none')
    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate factual knowledge heatmap')
    parser.add_argument('--input', type=str, default='routing_analysis/factual',
                       help='Input directory with JSON results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: figures/fig8_knowledge_neurons.png)')
    parser.add_argument('--top_k', type=int, default=15,
                       help='Number of top neurons to show')
    parser.add_argument('--demo', action='store_true',
                       help='Use demo data')

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else FIGURES_DIR / 'fig8_knowledge_neurons.png'

    print("=" * 50)
    print("Figure 8: Knowledge Neurons")
    print("=" * 50)

    if args.demo:
        print("\nUsing demo data...")
        neuron_ids, facts, data = create_demo_data()
    else:
        input_dir = Path(args.input)
        if not input_dir.exists():
            print(f"\nInput directory not found: {input_dir}")
            print("Using demo data instead...")
            neuron_ids, facts, data = create_demo_data()
        else:
            print(f"\nLoading from: {input_dir}")
            results = load_json_results(input_dir)

            if len(results) < 2:
                print(f"Not enough results found ({len(results)}). Using demo data...")
                neuron_ids, facts, data = create_demo_data()
            else:
                # Extract frequencies
                top_neurons, fact_neurons = extract_neuron_frequencies(results, args.top_k)

                if not top_neurons:
                    print("No neuron data found. Using demo data...")
                    neuron_ids, facts, data = create_demo_data()
                else:
                    # Order facts consistently
                    fact_order = ['Paris', 'Berlin', 'Tokyo', 'Blue']
                    facts = [f for f in fact_order if f in results]

                    neuron_ids = top_neurons
                    data = np.zeros((len(facts), len(neuron_ids)))

                    for i, fact in enumerate(facts):
                        for j, nid in enumerate(neuron_ids):
                            data[i, j] = fact_neurons.get(fact, {}).get(nid, 0)

    print(f"\nData shape: {data.shape}")
    print(f"Facts: {facts}")
    print(f"Neurons: {len(neuron_ids)}")

    # Create heatmap
    create_heatmap(neuron_ids, facts, data, output_path)

    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
