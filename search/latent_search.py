from __future__ import annotations
from functools import partial
from multiprocessing import Pool
import os
import time
import torch

from dsl import DSL
from dsl.base import Program
from search.top_down import TopDownSearch
from vae.models.base_vae import BaseVAE
from logger.stdout_logger import StdoutLogger
from tasks.task import Task
from config import Config
from vae.models.sketch_vae import SketchVAE
import numpy as np

import copy 


def execute_program(program_str: str, task_envs: list[Task],
                    dsl: DSL) -> tuple[Program, int, float]:
    try:
        program = dsl.parse_str_to_node(program_str)
    except AssertionError: # In case of invalid program (e.g. does not have an ending token)
        return Program(), 0, -float('inf')
    # If program is a sketch
    if not program.is_complete():
        # Let TDS complete and evaluate programs
        tds = TopDownSearch()
        tds_result = tds.synthesize(program, dsl, task_envs, Config.datagen_sketch_iterations)
        program, num_evaluations, mean_reward = tds_result
        if program is None: # Failsafe for TDS result
            return program, num_evaluations, -float('inf')
        if not program.is_complete(): # TDS failed to complete program
            return program, num_evaluations, -float('inf')
    # If program is a complete program
    else:
        # Evaluate single program
        mean_reward = 0.
        for task_env in task_envs:
            mean_reward += task_env.evaluate_program(program)
        num_evaluations = 1
        mean_reward /= len(task_envs)
    return program, num_evaluations, mean_reward


class LatentSearch:
    """Implements the CEM method from LEAPS paper.
    """
    def __init__(self, model: BaseVAE, task_cls: type[Task], dsl: DSL):
        self.model = model
        if issubclass(type(self.model), SketchVAE):
            self.dsl = dsl.extend_dsl()
        else:
            self.dsl = dsl
        self.device = self.model.device
        self.population_size = Config.search_population_size
        self.elitism_rate = Config.search_elitism_rate
        self.n_elite = int(Config.search_elitism_rate * self.population_size)
        self.number_executions = Config.search_number_executions
        self.number_iterations = Config.search_number_iterations
        self.sigma = Config.search_sigma
        self.model_hidden_size = Config.model_hidden_size
        self.task_envs = [task_cls(i) for i in range(self.number_executions)]
        self.original_task_envs = [copy.deepcopy(env) for env in self.task_envs]
        # self.task_envs = [task_cls(0)]
        self.program_filler = TopDownSearch()
        self.filler_iterations = Config.search_topdown_iterations
        output_dir = os.path.join('output', Config.experiment_name, 'latent_search')
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, f'seed_{Config.model_seed}.csv')
        self.trace_file = os.path.join(output_dir, f'seed_{Config.model_seed}.gif')
        self.restart_timeout = Config.search_restart_timeout


    def init_population(self) -> torch.Tensor:
        """Initializes the CEM population from a normal distribution.

        Returns:
            torch.Tensor: Initial population as a tensor.
        """
        return torch.stack([
            torch.randn(self.model_hidden_size, device=self.device) for _ in range(self.population_size)
        ])
    
    def reset_task_envs(self):
        self.task_envs = [copy.deepcopy(env) for env in self.original_task_envs]
        
    def execute_population(self, population: torch.Tensor, use_latent_only) -> tuple[list[str], torch.Tensor, int]:
        """Runs the given population in the environment and returns a list of mean rewards, after
        `Config.search_number_executions` executions.

        Args:
            population (torch.Tensor): Current population as a tensor.

        Returns:
            tuple[list[str], int, torch.Tensor]: List of programs as strings, list of mean rewards
            as tensor and number of evaluations as int.
        """
        # use_latent_only = True
        population_size = population.shape[0]
        rewards = torch.zeros(population_size, device=self.device)
        eval_curve = []
        current_best = -float('inf')

        if not use_latent_only:
            programs_tokens = self.model.decode_vector(population)
            programs_str = [self.dsl.parse_int_to_str(prog_tokens) for prog_tokens in programs_tokens]
            
            if Config.multiprocessing_active:
                with Pool() as pool:
                    fn = partial(execute_program, task_envs=self.task_envs, dsl=self.dsl)
                    results = pool.map(fn, programs_str)
            else:
                results = [execute_program(p, self.task_envs, self.dsl) for p in programs_str]
            
            for idx, (p, num_eval, r) in enumerate(results):
                program_str = self.dsl.parse_node_to_str(p)
                rewards[idx] = r
                self.num_evaluations += num_eval
                current_best = max(current_best, r)
                eval_curve.append((current_best, self.num_evaluations))
                if r > self.best_reward:
                    self.best_reward = r
                    self.best_program = program_str
                    StdoutLogger.log('Latent Search',f'New best reward: {self.best_reward}')
                    StdoutLogger.log('Latent Search',f'New best program: {self.best_program}')
                    StdoutLogger.log('Latent Search',f'Number of evaluations: {self.num_evaluations}')
                    with open(self.output_file, mode='a') as f:
                        t = time.time() - self.start_time
                        f.write(f'{t},{self.num_evaluations},{self.best_reward},{self.best_program}\n')
                if self.best_reward >= 1.0:
                    self.converged = True
                    break  

            return rewards, eval_curve  
        
        else:
            StdoutLogger.log("Latent Search", f"BEFORE Number of evaluations: {self.num_evaluations}")
            sequence_length = Config.data_max_demo_length
            all_actions = []
            
            self.reset_task_envs()

            num_envs = len(self.task_envs)

            for env_idx in range(num_envs):
                env = self.task_envs[env_idx]
                state_tensor = torch.tensor(env.get_state().get_state(), dtype=torch.float32, device=self.model.device)
                state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0) 
                s_h = state_tensor.repeat(population_size, 1, 1, 1, 1)  
                s_h = s_h.unsqueeze(2).repeat(1, 1, sequence_length, 1, 1, 1) 

                # Dummy action history 
                a_h = torch.full(
                    (population_size, 1, sequence_length),
                    fill_value=0,
                    dtype=torch.long,
                    device=self.model.device
                )
                a_h_masks = torch.zeros_like(a_h, dtype=torch.bool, device=self.model.device)

                with torch.no_grad():
                    pred_actions, _, _ = self.model.policy_executor(
                        population, 
                        s_h=s_h,
                        a_h=a_h,
                        a_h_mask=a_h_masks,
                        a_h_teacher_enforcing=False
                    )

                actions = pred_actions.cpu().numpy() 
                
                all_actions.append(actions.copy())
                
            env_rewards = torch.zeros(population_size, len(self.task_envs), device=self.device)
            self.reset_task_envs()
            original_envs = [copy.deepcopy(env) for env in self.task_envs]

            for i in range(population_size):
                total_reward_sum = 0.0

                for env_idx in range(num_envs):
                    env = copy.deepcopy(original_envs[env_idx])

                    total_reward = 0.0
                    terminated = False

                    for action in all_actions[env_idx][i]:

                        if action >= self.model.num_agent_actions - 1:
                            # penlize NOP actions to try and avoid them
                            total_reward -= 0.001*np.count_nonzero(all_actions[env_idx][i] == 5)
                            break

                        env.state.run_action(action)
                        terminated, r = env.get_reward(env.get_state())   

                        total_reward += r
                        if terminated or env.state.is_crashed():
                            break
        
                    env_rewards[i, env_idx] = total_reward
                    total_reward_sum += total_reward

                mean_reward = total_reward_sum / num_envs
                rewards[i] = mean_reward
                self.num_evaluations += 1

                current_best = max(current_best, mean_reward)
                eval_curve.append((current_best, self.num_evaluations))

                if mean_reward > self.best_reward:
                    self.best_reward = mean_reward
                    self.best_program = population[i].cpu().tolist()
                    
                    self.best_actions = [actions[i].tolist() for actions in all_actions]
                    self.best_env_rewards = env_rewards[i].tolist()

                    StdoutLogger.log("Latent Search", f"New best reward: {self.best_reward}")
                    StdoutLogger.log("Latent Search", f"Number of evaluations: {self.num_evaluations}")
                    for env_idx, (env_actions, env_reward) in enumerate(zip(self.best_actions, self.best_env_rewards)):
                        StdoutLogger.log(
                        "Latent Search",
                        f"Env {env_idx} - Actions: {env_actions}, Reward: {env_reward}"
                    )

                    with open(self.output_file, mode='a') as f:
                        t = time.time() - self.start_time
                        f.write(f"{t},{self.num_evaluations},{self.best_reward},{self.best_program}\n")
                
                if self.best_reward >= 1.0:
                    self.converged = True
                    StdoutLogger.log("Latent Search", f"Early stopping at individual {i}: mean reward >= 1.0")
                    break

            StdoutLogger.log("Latent Search", f"AFTER Number of evaluations: {self.num_evaluations}")
            return rewards, eval_curve

    
    def search(self, use_latent_only = True, return_curve = False) -> tuple[str, bool, int]:
        """Main search method. Searches for a program using the specified DSL that yields the
        highest reward at the specified task.

        Returns:
            tuple[str, bool]: Best program in string format and a boolean value indicating
            if the search has converged.
        """
        population = self.init_population()
        self.converged = False
        self.num_evaluations = 0
        counter_for_restart = 0
        self.best_reward = -float('inf')
        self.best_program = None
        prev_mean_elite_reward = -float('inf')
        self.start_time = time.time()

        num_restarts = 0

        reward_curve = []

        with open(self.output_file, mode='w') as f:
            f.write('time,num_evaluations,best_reward,best_program\n')

        for iteration in range(1, self.number_iterations + 1):
            StdoutLogger.log('Latent Search',f'Executing population')
            rewards, eval_curve = self.execute_population(population, use_latent_only)

            reward_curve.extend(eval_curve)

            current_best = rewards.max().item()
            StdoutLogger.log('Latent Search', f'Iteration {iteration}: Best reward = {current_best}, Evaluations = {self.num_evaluations}')
            
            if self.converged:
                break
            
            best_indices = torch.topk(rewards, self.n_elite).indices
            elite_population = population[best_indices]
            mean_elite_reward = torch.mean(rewards[best_indices])

            StdoutLogger.log('Latent Search',f'Iteration {iteration} mean elite reward: {mean_elite_reward}')
            
            if mean_elite_reward.cpu().numpy() == prev_mean_elite_reward:
                counter_for_restart += 1
            else:
                counter_for_restart = 0
            if counter_for_restart >= self.restart_timeout and self.restart_timeout > 0:
                population = self.init_population()
                counter_for_restart = 0

                num_restarts += 1
                StdoutLogger.log('Latent Search','Restarted population.')
            else:
                new_indices = torch.ones(elite_population.size(0), device=self.device).multinomial(
                    self.population_size, replacement=True)
                if Config.search_reduce_to_mean:
                    elite_population = torch.mean(elite_population, dim=0).repeat(self.n_elite, 1)
                new_population = []
                for index in new_indices:
                    sample = elite_population[index]
                    new_population.append(
                        sample + self.sigma * torch.randn_like(sample, device=self.device)
                    )
                population = torch.stack(new_population)
            prev_mean_elite_reward = mean_elite_reward.cpu().numpy()
        
        if not self.converged:
            with open(self.output_file, mode='a') as f:
                t = time.time() - self.start_time
                f.write(f'{t},{self.num_evaluations},{self.best_reward},{self.best_program}\n')
        
        if return_curve:
            return self.best_program, self.converged, self.num_evaluations, reward_curve, num_restarts
        else:
            return self.best_program, self.converged, self.num_evaluations
