import subprocess
import os
import json
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

AGENT = 'SmartPacmanAgent'  # 'GhostbusterAgent' para el agente cazafantasmas
REWARD_MODELS = ['model_1', 'model_2', 'model_3']
# REWARD_MODELS = ['ghost_model_1', 'ghost_model_2']  # con AGENT = 'GhostbusterAgent'
HYPERPARAMETER_CONFIGS = ['config_1', 'config_2', 'config_3'] 
TRAINING_EPISODES = 2500
TEST_EPISODES = 100
LAYOUT = 'mediumClassic'
RESULTS_DIR = f'evaluation_results/{datetime.now().strftime("%Y%m%d_%H%M%S")}'

def parse_training_output(output_file):
    episodes = []
    scores = []
    try:
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        pattern = r'Episode (\d+):.*Score = ([-\d.]+)'
        matches = re.findall(pattern, content)
        for match in matches:
            episodes.append(int(match[0]))
            scores.append(float(match[1]))
    except:
        pass
    return episodes, scores

def parse_evaluation_output(output_file):
    results = {'average_score': 0, 'win_rate': 0}
    try:
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        match = re.search(r'Average Score:\s*([-\d.]+)', content)
        if match:
            results['average_score'] = float(match.group(1))
        
        match = re.search(r'Win Rate:.*\(([\d.]+)\)', content)
        if match:
            results['win_rate'] = float(match.group(1)) * 100
    except:
        pass
    return results

def run_training(reward_model, hyperparameter_config, output_file):
    command = [
        'python', 'pacman.py',
        '-p', AGENT,
        '-a', f'reward_model={reward_model},hyperparameter_config={hyperparameter_config}',
        '-x', str(TRAINING_EPISODES),
        '-n', str(TRAINING_EPISODES + 10),
        '-l', LAYOUT,
        '-q'
    ]
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                command, 
                stdout=f, 
                stderr=subprocess.STDOUT,
                text=True, 
                timeout=None
            )
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'Traceback' in content or 'Error' in content or 'Exception' in content:
                print(f"\n{'='*70}")
                print(f"ERROR DETECTED IN: {reward_model}_{hyperparameter_config}")
                print(f"{'='*70}")
                lines = content.split('\n')
                for line in lines[-50:]:
                    print(line)
                print(f"{'='*70}\n")
                return False
        
        return result.returncode == 0
    except Exception as e:
        print(f"Exception during training: {e}")
        return False

def run_evaluation(reward_model, hyperparameter_config, output_file):
    command = [
        'python', 'pacman.py',
        '-p', AGENT,
        '-a', f'reward_model={reward_model},hyperparameter_config={hyperparameter_config}',
        '-n', str(TEST_EPISODES),
        '-l', LAYOUT,
        '-q'
    ]
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                command, 
                stdout=f, 
                stderr=subprocess.STDOUT,
                text=True, 
                timeout=1200
            )
        return result.returncode == 0
    except Exception as e:
        print(f"Exception during evaluation: {e}")
        return False

def smooth_curve(data, window=50):
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(data[start:i+1]))
    return smoothed

def create_graphs(all_training_data, all_results, output_dir):
    
    best_config = max(all_results.items(), key=lambda x: x[1]['average_score'])[0]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    episodes, scores = all_training_data[best_config]
    if len(scores) > 0:
        smoothed_scores = smooth_curve(scores, window=100)
        ax.plot(episodes, smoothed_scores, 'b-', linewidth=2, label='Average Score')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero line')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Average Score (100-episode window)', fontsize=12)
        ax.set_title(f'Performance Curve - Best Configuration: {best_config}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph1_best_performance.png'), dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    configs = list(all_results.keys())
    scores = [all_results[c]['average_score'] for c in configs]
    
    colors = ['green' if c == best_config else 'skyblue' for c in configs]
    bars = ax.bar(range(len(configs)), scores, color=colors, edgecolor='navy', alpha=0.7)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('Comparison: Average Score by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{scores[i]:.1f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph2_score_comparison.png'), dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    win_rates = [all_results[c]['win_rate'] for c in configs]
    
    colors = ['green' if c == best_config else 'lightcoral' for c in configs]
    bars = ax.bar(range(len(configs)), win_rates, color=colors, edgecolor='darkred', alpha=0.7)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Comparison: Win Rate by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50%')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{win_rates[i]:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph3_winrate_comparison.png'), dpi=150)
    plt.close()
    
    return best_config

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"\nEvaluation started - {len(REWARD_MODELS) * len(HYPERPARAMETER_CONFIGS)} configurations")
    print(f"Results: {RESULTS_DIR}\n")
    
    all_training_data = {}
    all_results = {}
    
    for i, model in enumerate(REWARD_MODELS):
        for j, config in enumerate(HYPERPARAMETER_CONFIGS):
            name = f"{model}_{config}"
            print(f"[{i*3+j+1}/9] {name}...", end=' ')
            
            train_file = os.path.join(RESULTS_DIR, f"{name}_train.txt")
            if run_training(model, config, train_file):
                episodes, scores = parse_training_output(train_file)
                all_training_data[name] = (episodes, scores)
                
                test_file = os.path.join(RESULTS_DIR, f"{name}_test.txt")
                if run_evaluation(model, config, test_file):
                    all_results[name] = parse_evaluation_output(test_file)
                    print(f"OK - Score: {all_results[name]['average_score']:.1f}, Win Rate: {all_results[name]['win_rate']:.1f}%")
                else:
                    all_results[name] = {'average_score': 0, 'win_rate': 0}
                    print("Eval failed")
            else:
                all_training_data[name] = ([], [])
                all_results[name] = {'average_score': 0, 'win_rate': 0}
                print("Train failed")
    
    print("\nGenerating graphs...")
    
    if all_results:
        best_config = create_graphs(all_training_data, all_results, RESULTS_DIR)
        
        with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nBest configuration: {best_config}")
        print(f"  Average Score: {all_results[best_config]['average_score']:.2f}")
        print(f"  Win Rate: {all_results[best_config]['win_rate']:.2f}%")
        print(f"\n3 graphs saved to: {RESULTS_DIR}/")
        print("  - graph1_best_performance.png")
        print("  - graph2_score_comparison.png")
        print("  - graph3_winrate_comparison.png")
    else:
        print("\nNo valid results")

if __name__ == '__main__':
    main()