from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

from run.analysis.utils import get_grouped_results
from run.experiments.experiment import Experiment

to_plot = {
    "Original Prompt": ("original", "no_reduction", "poisoned"),
    "Refined Prompt": ("refined", "no_reduction", "poisoned"),
    "CoT Prompt": ("cot", "no_reduction", "poisoned"),
    "CoT Prompt w/ DE": ("cot", "no_reduction", "poisoned_with_check"),
    "CoT Prompt\nw/ DE + CVE": ("cot", "with_reduction", "poisoned_with_check"),
    # "Refined Prompt\nw/ DC + CR": ("refined", "with_reduction", "poisoned_with_check"),
}

plot_data = defaultdict(lambda: {})
for experiment_name in ["final_35", "final_4"]:
    experiment = Experiment(experiment_name)
    print(experiment_name)

    danger_eval_results = pd.concat(
        {
            "combined prompt": experiment.danger_eval_combined_df,
            "seperate prompt": experiment.danger_eval_separate_df,
        }
    )

    danger_eval_results.index.names = ["Prompt type", "Context type", "qid"]
    danger_eval_results = danger_eval_results.reset_index()

    print("Danger eval results:")
    print(
        danger_eval_results.groupby(["Prompt type", "Context type"]).agg(
            {"dangerous": "mean"}
        )
    )

    group_res = get_grouped_results(experiment)
    print("Main results:")
    print(group_res)

    for k, (prompt_type, context_type, metric) in to_plot.items():
        try:
            plot_data[experiment_name][k] = group_res.loc[(prompt_type, context_type)][
                metric
            ]
        except:
            pass

fig, ax = plt.subplots()

axis_labels = list(to_plot.keys())
series_labels = {"final_35": "gpt 3.5 turbo", "final_4": "gpt 4o"}
n_series = len(plot_data)
wtot = 0.8
width = wtot / n_series
for i, (k, v) in enumerate(plot_data.items()):
    x = [axis_labels.index(k) - wtot / 2 + width * (i + 0.5) for k in v.keys()]
    y = list(v.values())
    ax.bar(x, y, width=width, label=series_labels[k])
ax.set_ylabel("PoisonedRAG Success Rate")
ax.set_xticks(range(len(axis_labels)))
ax.set_xticklabels(axis_labels, rotation=45)
fig.legend()

plt.tight_layout()
plt.savefig("figures/main_plot.jpg")
