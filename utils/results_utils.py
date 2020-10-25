
import pickle
from pathlib import Path

def write_results(video_name, app_name, results, logger):

    logger.info(f'Writing inference results of application {app_name} on video {video_name}.')
    results_file = Path(f'results/{app_name}/{video_name}')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

def read_results(video_name, app_name, logger):

    logger.info(f'Reading inference results of application {app_name} on video {video_name}.')
    results_file = Path(f'results/{app_name}/{video_name}')
    with open(results_file,  'rb') as f:
        return pickle.load(f)
