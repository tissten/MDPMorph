import logging
import time
import os
import numpy as np
import csv

from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)


def evaluate_policy_network(
    env, agent1, agent2, Model_name, config: TrainingConfig, record=None, total_steps=0, normalisation=True
):
    
    ###########################################################
    file_names = [f"./Metamorphic/test suite/T{i}.txt" for i in range(40, 50)]

    for file_name in file_names:
        print(f"Running RL experiment with {file_name} in process {os.getpid()}...")
        file_path = os.path.join("Test_suite", file_name)
        with open(file_path, "r") as file:
            initial_noises = [list(map(float, line.split())) for line in file]

        MR = 0
        tot_num = len(initial_noises)
        for initial_noise in initial_noises:
            # for _ in range(5):
            
#####################################################################################################33
            state = env.reset()
            state1 = state + initial_noise
            state2 = state + initial_noise
            # print(state)

            if record is not None:
                frame = env.grab_frame()
                record.start_video(total_steps + 1, frame)

            number_eval_episodes = int(config.number_eval_episodes)

            for eval_episode_counter in range(number_eval_episodes):
                episode_timesteps = 0
                episode_reward1 = 0
                episode_reward2 = 0
                episode_num = 0
                done1 = False
                done2 = False
                # truncated = False
                total_reward1 = []
                total_reward2 = []
                max_time = 5
                start_time = time.time()

                while True:
                    episode_timesteps += 1
                    # print("1")
                    if not done1:
                        
                        normalised_action1 = agent1.select_action_from_policy(state1, evaluation=True)
                        denormalised_action1 = (
                            hlp.denormalize(
                                normalised_action1, env.max_action_value, env.min_action_value
                            )
                            if normalisation
                            else normalised_action1
                        )
                        next_state1, reward1, done1, _ = env.step(denormalised_action1)
                        total_reward1.append(reward1)
                        state1 = next_state1
                        episode_reward1 += reward1

                    if not done2:
                        normalised_action2 = agent2.select_action_from_policy(state2, evaluation=True)
                        denormalised_action2 = (
                            hlp.denormalize(
                                normalised_action2, env.max_action_value, env.min_action_value
                            )
                            if normalisation
                            else normalised_action2
                        )
                        next_state2, reward2, done2, _ = env.step(denormalised_action2)
                        total_reward2.append(reward2)
                        state2 = next_state2
                        episode_reward2 += reward2
                    
                    
                    

                    if eval_episode_counter == 0 and record is not None:
                        frame = env.grab_frame()
                        record.log_video(frame)

                    if time.time()-start_time > max_time:
                        print("Exceeded maximum allowed time. Exiting.")
                        if record is not None:
                            record.log_eval(
                                total_steps=total_steps + 1,
                                episode=eval_episode_counter + 1,
                                episode_reward=episode_reward1,
                                display=True,
                            )

                        # Reset environment
                        state = env.reset()
                        episode_reward1 = 0
                        episode_reward2 = 0
                        episode_timesteps = 0
                        episode_num += 1
                        break
                    

                    if done1 or done2:
                        if record is not None:
                            record.log_eval(
                                total_steps=total_steps + 1,
                                episode=eval_episode_counter + 1,
                                episode_reward=episode_reward1,
                                display=True,
                            )

                        # Reset environment
                        state = env.reset()
                        episode_reward1 = 0
                        episode_reward2 = 0
                        episode_timesteps = 0
                        episode_num += 1
                        break
                
                if len(total_reward1) != len(total_reward2):
                    len1 = len(total_reward1)
                    len2 = len(total_reward2)
                    min_len = min(len1, len2)
                    total_reward1 = total_reward1[:min_len]
                    # print(sub_state1_aligned)
                    total_reward2 = total_reward2[:min_len]
                
                
                distance = np.linalg.norm(np.array(total_reward2) - np.array(total_reward1))
                # print(distance)

                if distance > 1.4:
                    MR += 1

                record.stop_video()
        

        output_file = os.path.join("./Metamorphic/Test_outcome", f"MR5_output_{Model_name + 1}.csv")

        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            

            if os.path.getsize(output_file) == 0:
                writer.writerow(["Violation rate", "Input"])

            writer.writerow([MR / tot_num, tot_num])


def policy_based_train(
    env,
    env_eval,
    agent,
    memory,
    record,
    train_config: TrainingConfig,
    alg_config: AlgorithmConfig,
    display=False,
    normalisation=True,
):
    start_time = time.time()

    max_steps_training = alg_config.max_steps_training
    max_steps_exploration = alg_config.max_steps_exploration
    number_steps_per_evaluation = train_config.number_steps_per_evaluation
    number_steps_per_train_policy = alg_config.number_steps_per_train_policy

    # Algorthm specific attributes - e.g. NaSA-TD3 dd
    intrinsic_on = (
        bool(alg_config.intrinsic_on) if hasattr(alg_config, "intrinsic_on") else False
    )

    min_noise = alg_config.min_noise if hasattr(alg_config, "min_noise") else 0
    noise_decay = alg_config.noise_decay if hasattr(alg_config, "noise_decay") else 1.0
    noise_scale = alg_config.noise_scale if hasattr(alg_config, "noise_scale") else 0.1

    logging.info(
        f"Training {max_steps_training} Exploration {max_steps_exploration} Evaluation {number_steps_per_evaluation}"
    )

    batch_size = alg_config.batch_size
    G = alg_config.G

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    state = env.reset()

    episode_start = time.time()
    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            logging.info(
                f"Running Exploration Steps {total_step_counter + 1}/{max_steps_exploration}"
            )

            denormalised_action = env.sample_action()

            # algorithm range [-1, 1] - note for DMCS this is redudenant but required for openai
            if normalisation:
                normalised_action = hlp.normalize(
                    denormalised_action, env.max_action_value, env.min_action_value
                )
            else:
                normalised_action = denormalised_action
        else:
            noise_scale *= noise_decay
            noise_scale = max(min_noise, noise_scale)

            # algorithm range [-1, 1]
            normalised_action = agent.select_action_from_policy(
                state, noise_scale=noise_scale
            )
            # mapping to env range [e.g. -2 , 2 for pendulum] - note for DMCS this is redudenant but required for openai
            if normalisation:
                denormalised_action = hlp.denormalize(
                    normalised_action, env.max_action_value, env.min_action_value
                )
            else:
                denormalised_action = normalised_action

        next_state, reward_extrinsic, done, truncated = env.step(denormalised_action)
        if display:
            env.render()

        intrinsic_reward = 0
        if intrinsic_on and total_step_counter > max_steps_exploration:
            intrinsic_reward = agent.get_intrinsic_reward(
                state, normalised_action, next_state
            )

        total_reward = reward_extrinsic + intrinsic_reward

        memory.add(
            state,
            normalised_action,
            total_reward,
            next_state,
            done,
        )

        state = next_state
        episode_reward += reward_extrinsic  # Note we only track the extrinsic reward for the episode for proper comparison

        info = {}
        if (
            total_step_counter >= max_steps_exploration
            and total_step_counter % number_steps_per_train_policy == 0
        ):
            for _ in range(G):
                info = agent.train_policy(memory, batch_size)

        if intrinsic_on:
            info["intrinsic_reward"] = intrinsic_reward

        if (total_step_counter + 1) % number_steps_per_evaluation == 0:
            logging.info("*************--Evaluation Loop--*************")
            evaluate_policy_network(
                env_eval,
                agent,
                train_config,
                record=record,
                total_steps=total_step_counter,
                normalisation=normalisation,
            )
            logging.info("--------------------------------------------")

        if done or truncated:
            episode_time = time.time() - episode_start
            record.log_train(
                total_steps=total_step_counter + 1,
                episode=episode_num + 1,
                episode_steps=episode_timesteps,
                episode_reward=episode_reward,
                episode_time=episode_time,
                **info,
                display=True,
            )

            # Reset environment
            state = env.reset()
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
            episode_start = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
