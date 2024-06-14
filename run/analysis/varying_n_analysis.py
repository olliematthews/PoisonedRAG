import matplotlib.pyplot as plt

from run.analysis.utils import get_grouped_results
from run.experiments.experiment import Experiment

dangerous_eval_plot = {}
poisoned_plot = {}
for experiment_name in [
    "varying_n_35",
    "varying_n_4",
    "varying_n_4_red",
    "varying_n_35_red",
]:
    print(experiment_name)
    experiment = Experiment(experiment_name)

    print("Danger eval results:")
    danger_eval_grouped = experiment.danger_eval_separate_df.groupby(
        ["Context type"]
    ).agg({"dangerous": "mean"})
    print(danger_eval_grouped)

    group_res = get_grouped_results(experiment)
    print("Main results:")
    print(group_res)

    dangerous_eval_plot[experiment_name] = danger_eval_grouped
    poisoned_plot[experiment_name] = group_res["poisoned"]


# Plot dangerous eval
fig, ax1 = plt.subplots()

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
series_labels = {
    "varying_n_35": ("gpt 3.5 turbo", color_cycle[0], "-"),
    "varying_n_4": ("gpt 4o", color_cycle[1], "-"),
    "varying_n_4_red": ("gpt 4o w/ CVE", color_cycle[0], "--"),
    "varying_n_35_red": ("gpt 3.5 turbo w/ CVE", color_cycle[1], "--"),
}

for experiment_name, plot in dangerous_eval_plot.items():
    x = [int(i) for i in plot.index]
    ax1.plot(
        x,
        plot,
        label=series_labels[experiment_name][0],
        color=series_labels[experiment_name][1],
        linestyle=series_labels[experiment_name][2],
    )

ax1.set_ylabel("Danger Identification Rate")

ax1.set_xlabel("Number of poisoned prompts")
fig.legend()

plt.tight_layout()
plt.savefig("figures/varying_n_dangerous_eval.jpg")


# Plot poisoned
fig, ax1 = plt.subplots()


for experiment_name, plot in poisoned_plot.items():
    x = [int(i[1]) for i in plot.index]
    ax1.plot(
        x,
        plot,
        label=series_labels[experiment_name][0],
        color=series_labels[experiment_name][1],
        linestyle=series_labels[experiment_name][2],
    )

ax1.set_ylabel("PoisonedRAG Success Rate")

ax1.set_xlabel("Number of poisoned prompts")
fig.legend()

plt.tight_layout()
plt.savefig("figures/varying_n_poisoned.jpg")
