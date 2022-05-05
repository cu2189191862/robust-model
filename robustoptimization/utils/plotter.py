from typing import Dict, List, Union
import matplotlib.pyplot as plt


def generate_evaluations_plot(figure_path: str, comparisons: Dict[str, List[Union[float, None]]], title: str = "evaluations plot"):
    plt.figure(figsize=(10, 5))
    plt.title(f"{title}")
    plt.xlabel("scenarios")
    plt.ylabel("cost")
    plt.scatter(range(len(comparisons["deterministic"])),
                comparisons["deterministic"], label="deterministic", alpha=0.5)
    plt.scatter(range(len(comparisons["robust"])),
                comparisons["robust"], label="robust", alpha=0.5)
    plt.legend()
    plt.savefig(figure_path)
