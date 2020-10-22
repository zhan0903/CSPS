import random
from utils.memory import ray_get_and_free,ReplayBuffer
from agent import cem,ppo,sac
import torch
import gym,ray,time
import numpy as np


CEM = "cem"
PPO = "ppo"
SAC = "sac"


class EngineCSPC:
    def __init__(self,args,global_mem,local_mem,logger):
        self.args = args
        self.agent_key = None
        self.gap = args.gap
        self.global_mem = global_mem
        self.local_mem = local_mem
        
        self.cem = cem.CEM.remote(args)
        self.sac = sac.SAC(args)
        self.ppo = ppo.PPO.remote(args)
        self.all_steps = 0
        self.agent_score = {}
        self.agent_steps = {SAC:0,PPO:0,CEM:0}
        self.logger = logger
        self.sac_ppo = False
        self.sac_cem = False
        self.ppo_cem = False
    
    def policies_transfer(self):
        ## policy transfer.
        sac_params_net, sac_params_mu = self.sac.get_actor_params()
        sac_value_params = self.sac.get_value_params()
        ppo_params_net, ppo_params_mu = ray_get_and_free(self.ppo.get_actor_params.remote())

        if (self.agent_score[PPO] - self.agent_score[CEM]) > self.gap:
            print("ppo -> cem")
            self.ppo_cem = True
            assert ppo_params_net != None
            ray_get_and_free(self.cem.transfer_ppo.remote(ppo_params_net,ppo_params_mu))

        if (self.agent_score[SAC] - self.agent_score[CEM]) > self.gap:
            print("sac -> cem")
            self.sac_cem = True
            ray_get_and_free(self.cem.transfer_sac.remote(sac_params_net,sac_params_mu))

        if (self.agent_score[SAC] - self.agent_score[PPO]) > self.gap:
            print("sac -> ppo")
            self.sac_ppo = True
            ray_get_and_free(self.ppo.transfer.remote(sac_params_net,sac_params_mu,sac_value_params))


    def train(self,max_steps):
        all_steps = 0

        time_start = time.time()
        return_gap = -np.inf
        MaxTestScore = -np.inf

        cem_iteration = self.args.iteration_steps
        ppo_iteration = self.args.iteration_steps
        sac_iteration = self.args.iteration_steps

        # init train sac agent, the sac_init_steps should > start_steps
        sac_timesteps = self.sac.train(self.args.sac_init_training,self.global_mem,self.local_mem,return_gap)
        all_steps += sac_timesteps
        self.agent_steps[SAC] += sac_timesteps

        ## test policy
        test_cem_id = self.cem.test_policy.remote()
        test_ppo_id = self.ppo.test_policy.remote()
        cem_score = ray_get_and_free(test_cem_id)
        ppo_score = ray_get_and_free(test_ppo_id)
        sac_score = self.sac.test_policy()

        self.agent_score[CEM] = cem_score
        self.agent_score[PPO] = ppo_score
        self.agent_score[SAC] = sac_score

        return_gap = min(self.agent_score.values())
        self.policies_transfer()

        while all_steps < max_steps:
            cem_id = self.cem.train.remote(cem_iteration,return_gap,self.sac_cem,self.ppo_cem)
            cem_timesteps,global_mem_temp,local_mem_temp = ray_get_and_free(cem_id)
            self.global_mem.add_epsodes(global_mem_temp)
            self.local_mem.add_epsodes(local_mem_temp)
            
            all_steps += cem_timesteps
            self.agent_steps[CEM] += cem_timesteps

            ppo_iteration = cem_timesteps
            sac_iteration = cem_timesteps

            if cem_timesteps > self.args.iteration_steps:
                cem_iteration = min(10000,cem_timesteps)

            ppo_id = self.ppo.train.remote(ppo_iteration,return_gap,self.sac_ppo)
            ppo_timesteps,global_mem_temp,local_mem_temp = ray_get_and_free(ppo_id)

            self.global_mem.add_epsodes(global_mem_temp)
            self.local_mem.add_epsodes(local_mem_temp)
            
            all_steps += ppo_timesteps
            self.agent_steps[PPO] += ppo_timesteps
            self.sac.update(cem_timesteps+ppo_timesteps,self.global_mem,self.local_mem)
            
            sac_timesteps = self.sac.train(sac_iteration,self.global_mem,self.local_mem,return_gap)
            all_steps += sac_timesteps
            self.agent_steps[SAC] += sac_timesteps

            ## test policy
            test_cem_id = self.cem.test_policy.remote()
            test_ppo_id = self.ppo.test_policy.remote()
            cem_score = ray_get_and_free(test_cem_id)
            ppo_score = ray_get_and_free(test_ppo_id)
            sac_score = self.sac.test_policy()

            self.agent_score[CEM] = cem_score
            self.agent_score[PPO] = ppo_score
            self.agent_score[SAC] = sac_score

            Test_score = max(self.agent_score.values())
            return_gap = min(self.agent_score.values())

            if Test_score > MaxTestScore: 
                MaxTestScore = Test_score

            agent_score = {}
            for k,v in self.agent_score.items():
                agent_score[k] = int(v)

            ## log
            self.logger.log_tabular('TotalEnvInteracts',int(all_steps))
            self.logger.log_tabular('TestScore',Test_score)
            self.logger.log_tabular('AgentsScore',agent_score)
            self.logger.log_tabular('MaxTestScore',MaxTestScore)
            self.logger.log_tabular('Time',int(time.time()-time_start))
            self.logger.dump_tabular()

            self.policies_transfer()