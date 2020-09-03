
import pickle
from pathlib import Path

def write_results(video_name, app_name, results, logger):

    logger.info(f'Writing inference results of application {app_name} on video {video_name}.')
    results_folder = Path(f'results/{app_name}')
    results_folder.mkdir(parents=True, exist_ok=True)
    with open(results_folder / Path(video_name).stem, 'wb') as f:
        pickle.dump(results, f)

def read_results(video_name, app_name, logger):

    logger.info(f'Reading inference results of application {app_name} on video {video_name}.')
    results_folder = Path(f'results/{app_name}')
    with open(results_folder / Path(video_name).stem, 'rb') as f:
        return pickle.load(f)
