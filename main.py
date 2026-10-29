import time

from Train.trainer import OptimizedExperimentRunner
from Utils.config import get_logger


def main(config):
    start_time = time.time()
    logger = get_logger(config["model"])
    logger.info(f"Starting training for model: {config['model']} | dataset: {config['dataset']}")
    OptimizedExperimentRunner(config, logger).run_experiment()
    logger.info(f"Finished. Elapsed time: {(time.time() - start_time) / 60.0:.2f} minutes\n")
