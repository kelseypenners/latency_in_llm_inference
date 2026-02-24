import matplotlib.pyplot as plt
from pathlib import Path

def save_fig(fig, filename, output_dir = "./"):
    output_dir = Path(output_dir)
    fig.savefig(f'{output_dir}/{filename}.png', dpi=300, bbox_inches='tight')
    print(f"saved: {filename}")