import torch
from config import Config
from dsl import DSL
from logger.stdout_logger import StdoutLogger
from vae.models import load_model
from search.latent_search import LatentSearch
from tasks import get_task_cls
import sys


if __name__ == '__main__':

    log_file = open('output.log', 'w')

    sys.stdout = log_file
    sys.stderr = log_file
    
    dsl = DSL.init_default_karel()

    if Config.disable_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(Config.model_name, dsl, device)
    
    task_cls = get_task_cls(Config.env_task)
    
    params = torch.load(Config.model_params_path, map_location=device)
    model.load_state_dict(params, strict=False)
    
    searcher = LatentSearch(model, task_cls, dsl)
    
    StdoutLogger.log('Main', f'Starting Latent Search with model {Config.model_name} for task {Config.env_task}')
    
    best_program, converged, num_evaluations = searcher.search(use_latent_only=Config.use_latent_only)
    
    StdoutLogger.log('Main', f'Converged: {converged}')
    StdoutLogger.log('Main', f'Final program: {best_program}')
    StdoutLogger.log('Main', f'Number of evaluations: {num_evaluations}')
