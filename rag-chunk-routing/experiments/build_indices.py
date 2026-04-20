# TODO: Orchestrate chunking and index construction for all configured chunk sizes.

from __future__ import annotations

import argparse

from rag_cr import Config, load_config, set_seed


def main(config: Config) -> None:
    set_seed(config.project.seed)
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
