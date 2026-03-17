import matplotlib.pyplot as plt
from pathlib import Path


def subtitle(ax, gpu_type, model, extra):
    """ consistent grey subtitle line """
    ax.text(0.5, 1.03, f'1 x {gpu_type}  |  {model}  |  {extra}',
        transform=ax.transAxes, ha='center', fontsize=8, color='gray')
    
def save_fig(fig, filename, output_dir = "./"):
    output_dir = Path(output_dir)
    fig.savefig(f'{output_dir}/{filename}.png', dpi=300, bbox_inches='tight')
    print(f"saved: {filename}")

def get_prompt_type(row):
    path = str(row.get('dataset-path', '') or '')
    if 'easy' in path:
        return 'easy'
    elif 'hard' in path:
        return 'hard'
    elif str(row.get('dataset-name', '')).lower() == 'random':
        return 'random'
    return None