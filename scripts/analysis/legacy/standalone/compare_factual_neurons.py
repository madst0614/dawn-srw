#!/usr/bin/env python3
"""
Compare neuron activations across multiple factual knowledge prompts.

Analyzes:
1. Which neurons are specific to each fact
2. Which neurons are common across multiple facts (general knowledge neurons)
3. Overlap between different factual categories
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np

try:
    from scripts.analysis.utils import HAS_MATPLOTLIB, plt
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def load_analysis_results(input_dir: str) -> Dict[str, Dict]:
    """Load all JSON analysis results from directory."""
    results = {}
    input_path = Path(input_dir)

    for json_file in input_path.glob("*.json"):
        if json_file.name.startswith("comparison"):
            continue  # Skip comparison output files

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Extract name from filename
                name = json_file.stem
                results[name] = data
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results


def analyze_neuron_overlap(results: Dict[str, Dict], threshold: float = 0.8) -> Dict:
    """
    Analyze neuron overlap across different factual prompts.

    Args:
        results: Dict of analysis results keyed by prompt name
        threshold: Frequency threshold for considering a neuron "active"

    Returns:
        Analysis dictionary
    """
    # Collect neurons from each result (using 80%+ threshold by default)
    neurons_by_fact = {}
    all_neurons = set()

    for name, data in results.items():
        if 'common_neurons_80' in data:
            neurons = set(data['common_neurons_80'])
        elif 'neuron_frequencies' in data:
            # Use threshold to filter
            n_matching = data.get('matching_runs', 1)
            neurons = set(
                nf['neuron'] for nf in data['neuron_frequencies']
                if nf['count'] >= n_matching * threshold
            )
        else:
            neurons = set()

        neurons_by_fact[name] = neurons
        all_neurons.update(neurons)

    # Find neurons that appear in multiple facts
    neuron_fact_count = Counter()
    neuron_to_facts = defaultdict(list)

    for name, neurons in neurons_by_fact.items():
        for n in neurons:
            neuron_fact_count[n] += 1
            neuron_to_facts[n].append(name)

    # Categorize neurons
    n_facts = len(results)

    # Universal neurons (appear in ALL facts)
    universal_neurons = [n for n, count in neuron_fact_count.items() if count == n_facts]

    # Common neurons (appear in >50% of facts)
    common_neurons = [n for n, count in neuron_fact_count.items() if count > n_facts * 0.5]

    # Fact-specific neurons (appear in only 1 fact)
    specific_neurons = {
        name: [n for n in neurons if neuron_fact_count[n] == 1]
        for name, neurons in neurons_by_fact.items()
    }

    # Pairwise overlap
    pairwise_overlap = {}
    fact_names = list(neurons_by_fact.keys())
    for i, name1 in enumerate(fact_names):
        for j, name2 in enumerate(fact_names):
            if i < j:
                overlap = neurons_by_fact[name1] & neurons_by_fact[name2]
                union = neurons_by_fact[name1] | neurons_by_fact[name2]
                jaccard = len(overlap) / len(union) if union else 0
                pairwise_overlap[f"{name1} x {name2}"] = {
                    'overlap_count': len(overlap),
                    'overlap_neurons': sorted(list(overlap)),
                    'jaccard': jaccard,
                }

    return {
        'n_facts': n_facts,
        'total_unique_neurons': len(all_neurons),
        'neurons_by_fact': {k: sorted(list(v)) for k, v in neurons_by_fact.items()},
        'neurons_per_fact_count': {k: len(v) for k, v in neurons_by_fact.items()},
        'universal_neurons': sorted(universal_neurons),
        'common_neurons': sorted(common_neurons),
        'specific_neurons': {k: sorted(v) for k, v in specific_neurons.items()},
        'neuron_to_facts': {str(k): v for k, v in neuron_to_facts.items()},
        'pairwise_overlap': pairwise_overlap,
    }


def print_analysis(analysis: Dict, results: Dict[str, Dict]):
    """Print analysis results in readable format."""
    print("\n" + "=" * 70)
    print("FACTUAL KNOWLEDGE NEURON COMPARISON")
    print("=" * 70)

    print(f"\nAnalyzed {analysis['n_facts']} factual prompts")
    print(f"Total unique neurons (80%+ threshold): {analysis['total_unique_neurons']}")

    # Per-fact summary
    print("\n" + "-" * 50)
    print("NEURONS PER FACT:")
    print("-" * 50)
    for name, count in sorted(analysis['neurons_per_fact_count'].items()):
        match_info = results.get(name, {})
        match_rate = match_info.get('match_rate', 0) * 100
        matching = match_info.get('matching_runs', 0)
        total = match_info.get('total_runs', 0)
        target = match_info.get('target_token', '?')
        print(f"  {name:30s}: {count:3d} neurons | '{target}' in {matching}/{total} ({match_rate:.0f}%)")

    # Universal neurons
    print("\n" + "-" * 50)
    print(f"UNIVERSAL NEURONS (in ALL {analysis['n_facts']} facts): {len(analysis['universal_neurons'])}")
    print("-" * 50)
    if analysis['universal_neurons']:
        print(f"  {analysis['universal_neurons']}")
    else:
        print("  (none)")

    # Common neurons
    print("\n" + "-" * 50)
    print(f"COMMON NEURONS (in >50% facts): {len(analysis['common_neurons'])}")
    print("-" * 50)
    if analysis['common_neurons']:
        print(f"  {analysis['common_neurons'][:30]}")
        if len(analysis['common_neurons']) > 30:
            print(f"  ... and {len(analysis['common_neurons']) - 30} more")

    # Fact-specific neurons
    print("\n" + "-" * 50)
    print("FACT-SPECIFIC NEURONS (unique to one fact):")
    print("-" * 50)
    for name, neurons in sorted(analysis['specific_neurons'].items()):
        if neurons:
            print(f"  {name:30s}: {len(neurons):3d} neurons | {neurons[:10]}...")

    # Pairwise overlap
    print("\n" + "-" * 50)
    print("PAIRWISE OVERLAP (Jaccard similarity):")
    print("-" * 50)
    overlaps = sorted(
        analysis['pairwise_overlap'].items(),
        key=lambda x: x[1]['jaccard'],
        reverse=True
    )
    for pair, info in overlaps[:10]:
        jaccard = info['jaccard']
        count = info['overlap_count']
        bar = '█' * int(jaccard * 20)
        print(f"  {pair:50s}: {jaccard:.2%} ({count:2d} neurons) {bar}")


def plot_comparison(analysis: Dict, output_path: str = None):
    """Generate comparison visualization."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for plotting")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Neurons per fact
    ax = axes[0, 0]
    names = list(analysis['neurons_per_fact_count'].keys())
    counts = list(analysis['neurons_per_fact_count'].values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    bars = ax.barh(range(len(names)), counts, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:25] for n in names], fontsize=8)
    ax.set_xlabel('Number of Neurons (80%+ threshold)')
    ax.set_title('Neurons per Factual Knowledge')
    ax.invert_yaxis()

    # 2. Neuron categories pie chart
    ax = axes[0, 1]
    n_universal = len(analysis['universal_neurons'])
    n_common = len(analysis['common_neurons']) - n_universal
    n_specific = sum(len(v) for v in analysis['specific_neurons'].values())
    n_other = analysis['total_unique_neurons'] - n_universal - n_common - n_specific

    sizes = [n_universal, n_common, n_specific, max(0, n_other)]
    labels = [
        f'Universal\n({n_universal})',
        f'Common\n({n_common})',
        f'Fact-specific\n({n_specific})',
        f'Other\n({max(0, n_other)})'
    ]
    colors_pie = ['#ff6b6b', '#feca57', '#48dbfb', '#c8d6e5']
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax.set_title('Neuron Category Distribution')

    # 3. Overlap heatmap
    ax = axes[1, 0]
    fact_names = sorted(analysis['neurons_by_fact'].keys())
    n = len(fact_names)
    overlap_matrix = np.zeros((n, n))

    for i, name1 in enumerate(fact_names):
        for j, name2 in enumerate(fact_names):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                key = f"{name1} x {name2}" if i < j else f"{name2} x {name1}"
                if key in analysis['pairwise_overlap']:
                    overlap_matrix[i, j] = analysis['pairwise_overlap'][key]['jaccard']

    im = ax.imshow(overlap_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([fn[:12] for fn in fact_names], rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels([fn[:12] for fn in fact_names], fontsize=7)
    ax.set_title('Pairwise Jaccard Similarity')
    plt.colorbar(im, ax=ax)

    # 4. Top shared neurons
    ax = axes[1, 1]
    neuron_to_facts = analysis['neuron_to_facts']
    sorted_neurons = sorted(
        neuron_to_facts.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:15]

    if sorted_neurons:
        neurons = [int(n) for n, _ in sorted_neurons]
        shares = [len(facts) for _, facts in sorted_neurons]
        ax.barh(range(len(neurons)), shares, color='steelblue')
        ax.set_yticks(range(len(neurons)))
        ax.set_yticklabels([f'N{n}' for n in neurons])
        ax.set_xlabel('Number of Facts')
        ax.set_title('Top Shared Neurons')
        ax.invert_yaxis()
        ax.axvline(len(fact_names) * 0.5, color='red', linestyle='--', label='>50% threshold')
        ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare factual knowledge neurons')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing analysis JSON files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for comparison results')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Frequency threshold for neuron selection (default: 0.8)')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output path for comparison plot')
    args = parser.parse_args()

    # Load results
    print(f"Loading analysis results from: {args.input_dir}")
    results = load_analysis_results(args.input_dir)

    if not results:
        print("No analysis results found!")
        return

    print(f"Found {len(results)} analysis files")

    # Run comparison
    analysis = analyze_neuron_overlap(results, threshold=args.threshold)

    # Print results
    print_analysis(analysis, results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Generate plot
    plot_path = args.plot or (args.output.replace('.json', '.png') if args.output else None)
    if plot_path:
        plot_comparison(analysis, plot_path)


if __name__ == '__main__':
    main()
