#!/usr/bin/env python3
"""
Generate All Paper Figures

Single command to generate all figures for the DAWN paper.

Usage:
    # Generate fig1-4 only:
    python figures/generate_all.py

    # Include fig6 with demo data:
    python figures/generate_all.py --demo

    # Include fig6 with checkpoint directories (auto-find training_log.txt):
    python figures/generate_all.py \\
        --dawn path/to/dawn/checkpoints/run_xxx \\
        --vanilla_22m path/to/vanilla_22m/checkpoints/run_xxx \\
        --vanilla_108m path/to/vanilla_108m/checkpoints/run_xxx

    # Generate and zip for download:
    python figures/generate_all.py --demo --zip

Output:
    figures/fig1_architecture.pdf
    figures/fig2_feature_restore_pathway.pdf
    figures/fig3_param_efficiency.pdf
    figures/fig4_attention_knowledge_balance.pdf
    figures/fig6_convergence_comparison.pdf (if --demo or checkpoint paths provided)
    figures/fig8_knowledge_neurons.pdf
"""

import subprocess
import sys
import os
import zipfile
from pathlib import Path

# Get the figures directory
FIGURES_DIR = Path(__file__).parent
PROJECT_ROOT = FIGURES_DIR.parent


def run_script(script_name, extra_args=None):
    """Run a figure generation script."""
    script_path = FIGURES_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    if extra_args:
        print(f"Args: {' '.join(extra_args[:6])}...")  # Truncate long args
    print('='*60)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def create_zip():
    """Create zip file of all generated figures."""
    zip_path = PROJECT_ROOT / 'dawn_figures.zip'

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for ext in ['*.png', '*.pdf']:
            for f in FIGURES_DIR.glob(ext):
                zf.write(f, f.name)

    print(f"\nCreated: {zip_path}")
    return zip_path


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate all paper figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic figures only (fig1-4):
    python figures/generate_all.py

    # All figures with demo convergence curves:
    python figures/generate_all.py --demo

    # All figures with checkpoint directories (40M scale):
    python figures/generate_all.py \\
        --dawn /content/drive/MyDrive/dawn/checkpoints_v17.1_40M_c4_5B/run_xxx \\
        --vanilla_40m /content/drive/MyDrive/dawn/checkpoints_baseline_40M_c4_5B/run_xxx

    # Generate and zip:
    python figures/generate_all.py --demo --zip
        """
    )

    # Fig6 options - checkpoint directories (auto-find training_log.txt)
    parser.add_argument('--dawn', type=str,
                       help='DAWN run directory (will auto-find training_log.txt)')
    parser.add_argument('--vanilla_40m', type=str,
                       help='Vanilla-40M run directory')
    parser.add_argument('--vanilla_22m', type=str,
                       help='Vanilla-22M run directory')
    parser.add_argument('--vanilla_108m', type=str,
                       help='Vanilla-108M run directory')
    parser.add_argument('--demo', action='store_true',
                       help='Use demo data for fig6 convergence comparison')

    # General options
    parser.add_argument('--skip', nargs='+', default=[],
                       help='Skip specific figures (e.g., --skip fig1 fig2)')
    parser.add_argument('--zip', action='store_true',
                       help='Create zip file of all figures after generation')
    parser.add_argument('--show_annotations', action='store_true',
                       help='Show final loss values on fig6')

    args = parser.parse_args()

    # Ensure figures directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = {}

    # Figure 1: Architecture
    if 'fig1' not in args.skip:
        results['fig1'] = run_script('fig1_architecture.py')

    # Figure 2: Feature-Restore Pathway
    if 'fig2' not in args.skip:
        results['fig2'] = run_script('fig2_feature_restore_pathway.py')

    # Figure 3: Parameter Efficiency (standalone)
    if 'fig3' not in args.skip:
        results['fig3'] = run_script('fig3_param_efficiency.py')

    # Figure 6: Convergence Comparison (appendix, optional)
    if 'fig6' not in args.skip:
        fig4_args = []

        if args.demo:
            fig4_args.append('--demo')
        elif args.dawn or args.vanilla_40m or args.vanilla_22m or args.vanilla_108m:
            ckpts = []
            labels = []

            if args.dawn:
                ckpts.append(args.dawn)
                labels.append('DAWN-40M')
            if args.vanilla_40m:
                ckpts.append(args.vanilla_40m)
                labels.append('Vanilla-40M')
            if args.vanilla_22m:
                ckpts.append(args.vanilla_22m)
                labels.append('Vanilla-22M')
            if args.vanilla_108m:
                ckpts.append(args.vanilla_108m)
                labels.append('Vanilla-108M')

            if ckpts:
                fig4_args.extend(['--checkpoints'] + ckpts)
                fig4_args.extend(['--labels'] + labels)

        if args.show_annotations:
            fig4_args.append('--show_annotations')

        if fig4_args:
            results['fig6'] = run_script('fig6_convergence_comparison.py', fig4_args)
        else:
            print("\n[fig6] Skipped: No --demo or checkpoint paths provided")

    # Figure 4: Attention-Knowledge Balance
    if 'fig4' not in args.skip:
        results['fig4'] = run_script('fig4_attention_knowledge_balance.py')

    # Figure 8: Knowledge Neurons (Appendix)
    if 'fig8' not in args.skip:
        results['fig8'] = run_script('fig8_knowledge_neurons.py')

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for fig, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {fig}")

    print(f"\nGenerated {success_count}/{total_count} figures")
    print(f"Output directory: {FIGURES_DIR}")

    # List generated files
    print("\nGenerated files:")
    for f in sorted(FIGURES_DIR.glob('*.pdf')):
        size = f.stat().st_size / 1024
        print(f"  {f.name} ({size:.1f} KB)")

    # Create zip if requested
    if args.zip:
        zip_path = create_zip()
        print(f"\nTo download in Colab:")
        print(f"  from google.colab import files")
        print(f"  files.download('{zip_path}')")

    return 0 if success_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
