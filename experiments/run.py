"""Run the reproducible benchmark on all registered datasets."""

from experiments.run_tabular_repro import ALL_DATASETS, main as run_benchmark


def main():
    run_benchmark(dataset_scope=ALL_DATASETS)


if __name__ == "__main__":
    main()
