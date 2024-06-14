import argparse

from poisoned_rag_defence.logger import logger


def experiment_name_parse_args():
    parser = argparse.ArgumentParser("Run experiment step")

    parser.add_argument("--experiment_name", type=str, required=True)

    args = parser.parse_args()
    logger.debug(str(args))

    return args


def experiment_name__parse_args():
    parser = argparse.ArgumentParser("Run experiment step")

    parser.add_argument("--experiment_name", type=str, required=True)

    args = parser.parse_args()
    print(args)

    return args
