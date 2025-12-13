"""Quick verification script to check plotting system is working"""
import os
import pickle

# Check if plot_results.py exists
if os.path.exists('plot_results.py'):
    print('✓ plot_results.py found')
else:
    print('✗ plot_results.py not found')
    exit(1)

# Check if figures directory can be created
os.makedirs('figures', exist_ok=True)
print('✓ figures directory ready')

# Check if sample data exists
sample_files = [
    'logdir/model/regression_sine_fcnn_run0_records.pkl',
    'logdir/model/regression_sine_fcnn_allruns.pkl'
]

found_any = False
for f in sample_files:
    if os.path.exists(f):
        print(f'✓ Sample data found: {f}')
        found_any = True
        
        # Check data structure
        with open(f, 'rb') as file:
            data = pickle.load(file)
        
        if 'runs' in data:
            print(f'  - Multi-run file with {len(data["runs"])} runs')
            for run_id in data['runs']:
                n_iter = len(data['runs'][run_id]['iterations'])
                print(f'    - Run {run_id}: {n_iter} iterations')
        else:
            n_iter = len(data['iterations'])
            run_id = data['metadata'].get('run_id', 'unknown')
            print(f'  - Single-run file (run {run_id}) with {n_iter} iterations')

if not found_any:
    print('! No sample data found. Run training first:')
    print('  bash test_datasets.sh')

# Check existing plots
existing_plots = [f for f in os.listdir('figures') if f.endswith('.png')] if os.path.exists('figures') else []
if existing_plots:
    print(f'\n✓ Found {len(existing_plots)} existing plots in figures/')
    for plot in sorted(existing_plots)[:5]:
        print(f'  - {plot}')
    if len(existing_plots) > 5:
        print(f'  ... and {len(existing_plots) - 5} more')
else:
    print('\n! No plots generated yet. Run:')
    print('  bash plot_latest.sh')

print('\n✓ Plotting system verification complete!')
print('\nTo generate plots:')
print('  bash plot_latest.sh')
print('  python plot_results.py logdir/model/regression_sine_fcnn_run0_records.pkl')
