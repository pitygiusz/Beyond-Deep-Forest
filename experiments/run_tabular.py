"""Run the reproducible benchmark on tabular datasets only."""

from experiments.run_tabular_repro import TABULAR_DATASET_ORDER, main as run_benchmark


def main():
    run_benchmark(dataset_scope=TABULAR_DATASET_ORDER)


if __name__ == "__main__":
    main()
