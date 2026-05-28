results = [
    {
        "eval_loss": "1.081",
        "eval_accuracy": {"accuracy": 0.5279576221306651},
        "eval_precision": [0.3311897106109325, 0.7808510638297872, 0.13392857142857142],
        "eval_recall": [0.27837837837837837, 0.6157718120805369, 0.43795620437956206],
        "eval_f1": [0.302496328928047, 0.6885553470919324, 0.20512820512820512],
        "eval_runtime": "13.85",
        "eval_samples_per_second": "122.6",
        "eval_steps_per_second": "7.724",
        "epoch": "1",
    },
    {
        "eval_loss": "0.9468",
        "eval_accuracy": {"accuracy": 0.6156562683931724},
        "eval_precision": [0.3786407766990291, 0.7780725022104332, 0.1891891891891892],
        "eval_recall": [0.3162162162162162, 0.738255033557047, 0.35766423357664234],
        "eval_f1": [0.3446244477172312, 0.7576409814894532, 0.2474747474747475],
        "eval_runtime": "15.38",
        "eval_samples_per_second": "110.5",
        "eval_steps_per_second": "6.958",
        "epoch": "2",
    },
    {
        "eval_loss": "1.083",
        "eval_accuracy": {"accuracy": 0.6209535020600353},
        "eval_precision": [
            0.39184952978056425,
            0.7821080602302923,
            0.18725099601593626,
        ],
        "eval_recall": [0.33783783783783783, 0.7407718120805369, 0.34306569343065696],
        "eval_f1": [0.36284470246734396, 0.7608789314950453, 0.2422680412371134],
        "eval_runtime": "16.42",
        "eval_samples_per_second": "103.5",
        "eval_steps_per_second": "6.517",
        "epoch": "3",
    },
]


def format_number(f: float):
    return f"{f:.2f}"


def format_list(l: list[float]):
    return f"[{', '.join(map(format_number, l))}]"


print("""\\begin{center}
\\begin{tabular}{|c|c|c|c|c|}
    \\hline
    & Accuracy & Precision & Recall & F1 \\\\
    \\hline \\hline""")
for epoch in results:

    print(
        (
            f"    Epoch {epoch['epoch']} & "
            + f"{format_number(epoch['eval_accuracy']['accuracy'])} & "
            + f"{format_list(epoch['eval_precision'])} & "
            + f"{format_list(epoch['eval_recall'])} & "
            + f"{format_list(epoch['eval_f1'])} \\\\\n    \\hline"
        )
    )
print("\\end{tabular}")
print("\\end{center}")
