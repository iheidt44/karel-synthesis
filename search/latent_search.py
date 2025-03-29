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
        
        
    def execute_population(self, population: torch.Tensor) -> tuple[list[str], torch.Tensor, int]:
        """Runs the given population in the environment and returns a list of mean rewards, after
        `Config.search_number_executions` executions.

        Args:
            population (torch.Tensor): Current population as a tensor.

        Returns:
            tuple[list[str], int, torch.Tensor]: List of programs as strings, list of mean rewards
            as tensor and number of evaluations as int.
        """
        # programs_tokens = self.model.decode_vector(population)
        # programs_str = [self.dsl.parse_int_to_str(prog_tokens) for prog_tokens in programs_tokens]
        
        # if Config.multiprocessing_active:
        #     with Pool() as pool:
        #         fn = partial(execute_program, task_envs=self.task_envs, dsl=self.dsl)
        #         results = pool.map(fn, programs_str)
        # else:
        #     results = [execute_program(p, self.task_envs, self.dsl) for p in programs_str]
        
        # rewards = []
        # for p, num_eval, r in results:
        #     program_str = self.dsl.parse_node_to_str(p)
        #     rewards.append(r)
        #     self.num_evaluations += num_eval
        #     if r > self.best_reward:
        #         self.best_reward = r
        #         self.best_program = program_str
        #         StdoutLogger.log('Latent Search',f'New best reward: {self.best_reward}')
        #         StdoutLogger.log('Latent Search',f'New best program: {self.best_program}')
        #         StdoutLogger.log('Latent Search',f'Number of evaluations: {self.num_evaluations}')
        #         with open(self.output_file, mode='a') as f:
        #             t = time.time() - self.start_time
        #             f.write(f'{t},{self.num_evaluations},{self.best_reward},{self.best_program}\n')
        #     if self.best_reward >= 1.0:
        #         self.converged = True
        #         break     

        # rewards = np.zeros((len(population), len(self.task_envs)))
        # rewards = []
        # self.num_evaluations = 0

        # for z in population:

        #     total_reward = 0.0

        #     for env in self.task_envs:
        #         env_state = env.generate_state()
        #         env.reset_state()

        #         # Convert environment state to tensor format expected by policy_executor
        #         state_tensor = torch.tensor(env_state.get_state(), dtype=torch.float32, device=self.model.device)
        #         state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        #         s_h = state_tensor.unsqueeze(0).repeat(1, 1, self.model.max_demo_length, 1, 1, 1)

        #         a_h = torch.full((1, 1, self.model.max_demo_length),
        #                         fill_value=self.model.num_agent_actions - 1,
        #                         dtype=torch.long, device=self.model.device)
        #         a_h_masks = torch.zeros_like(a_h)

        #         with torch.no_grad():
        #             pred_actions, _, _ = self.model.policy_executor(
        #                 z.unsqueeze(0),
        #                 s_h=s_h,
        #                 a_h=a_h,
        #                 a_h_mask=a_h_masks,
        #                 a_h_teacher_enforcing=False
        #             )

        #         # Evaluate action sequence in the environment
        #         actions = pred_actions[0].cpu().numpy()
        #         #print(actions)
        #         # reward = 0.0
        #         for action in actions:
        #             if action >= 5:
        #                 continue
        #             env.state.run_action(action)
        #             terminated, r = env.get_reward(env.get_state())
        #             total_reward += r
        #             if terminated or env.state.is_crashed():
        #                 break
        #         #print('total',total_reward)
            
        #     mean_reward = total_reward / len(self.task_envs)
        #     #print('mean', mean_reward)
        #     rewards.append(mean_reward)
        #     self.num_evaluations += 1

        #     if mean_reward > self.best_reward:
        #         self.best_reward = mean_reward
        #         self.best_program = z
        #         StdoutLogger.log("Latent Search", f"New best reward: {mean_reward}")
            
        #     if self.best_reward >= 1.0:
        #         self.converged = True
        #         break

        population_size = population.shape[0]
        rewards = torch.zeros(population_size, device=self.device)
        self.num_evaluations = 0

        # Batch process all environments for each z
        for env_idx, env in enumerate(self.task_envs):
            env_state = env.generate_state()
            env.reset_state()

            # Convert environment state to tensor format expected by policy_executor
            state_tensor = torch.tensor(env_state.get_state(), dtype=torch.float32, device=self.model.device)
            state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            # Repeat for population_size and add demos_per_program dimension
            s_h = state_tensor.repeat(population_size, 1, 1, 1, 1)  # [population_size, 1, C, H, W]
            s_h = s_h.unsqueeze(2).repeat(1, 1, self.model.max_demo_length, 1, 1, 1)  # [population_size, 1, max_demo_length, C, H, W]

            # Dummy action history (NOP-filled) and mask
            a_h = torch.full(
                (population_size, 1, self.model.max_demo_length),
                fill_value=self.model.num_agent_actions - 1,
                dtype=torch.long,
                device=self.model.device
            )
            a_h_masks = torch.zeros_like(a_h, dtype=torch.bool, device=self.model.device)

            with torch.no_grad():
                pred_actions, _, _ = self.model.policy_executor(
                    population,  # [population_size, hidden_size]
                    s_h=s_h,
                    a_h=a_h,
                    a_h_mask=a_h_masks,
                    a_h_teacher_enforcing=False
                )

            # Evaluate action sequences in the environment
            actions = pred_actions.cpu().numpy()  # [population_size, max_demo_length]
            for i in range(population_size):
                env.reset_state()  # Reset for each individual
                total_reward = 0.0
                terminated = False

                for action in actions[i]:
                    if action >= self.model.num_agent_actions - 1:  # NOP or invalid
                        break
                    env.state.run_action(action)
                    terminated, r = env.get_reward(env.get_state())
                    total_reward += r
                    if terminated or env.state.is_crashed():
                        break

                rewards[i] += total_reward

        # Compute mean reward across environments
        rewards /= len(self.task_envs)
        self.num_evaluations += population_size

        # Update best reward and "program" (latent vector)
        for i, reward in enumerate(rewards):
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_program = population[i].cpu().tolist()  # Store z as list for logging
                StdoutLogger.log("Latent Search", f"New best reward: {self.best_reward}")
                StdoutLogger.log("Latent Search", f"New best latent: {self.best_program}")
                StdoutLogger.log("Latent Search", f"Number of evaluations: {self.num_evaluations}")
                with open(self.output_file, mode='a') as f:
                    t = time.time() - self.start_time
                    f.write(f"{t},{self.num_evaluations},{self.best_reward},{self.best_program}\n")
            if self.best_reward >= 1.0:
                self.converged = True
                break


        # return torch.tensor(rewards, device=self.device)
        return rewards

    
    def search(self) -> tuple[str, bool, int]:
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
        with open(self.output_file, mode='w') as f:
            f.write('time,num_evaluations,best_reward,best_program\n')

        for iteration in range(1, self.number_iterations + 1):
            StdoutLogger.log('Latent Search',f'Executing population')
            rewards = self.execute_population(population)
            
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
        
        # best_program_nodes = self.dsl.parse_str_to_node(self.best_program)
        # self.task_envs[0].trace_program(best_program_nodes, self.trace_file, 1000)
        
        if not self.converged:
            with open(self.output_file, mode='a') as f:
                t = time.time() - self.start_time
                f.write(f'{t},{self.num_evaluations},{self.best_reward},{self.best_program}\n')
        
        return self.best_program, self.converged, self.num_evaluations
