"""
This script is used to train reinforcement learning agents in DMCS/OpenAI/pyboy.
The main function parses command-line arguments, creates the environment, network,
and memory instances, and then trains the agent using the specified algorithm.
"""

import logging
import sys
from pathlib import Path
import os
import sys
import importlib.util
import argparse
import os
sys.path.append(os.path.abspath(r"./cares_reinforcement_learning"))


# import train_loops.policy_loop_MR2_Test as pbe
import train_loops.ppo_loop as ppe
import train_loops.value_loop as vbe
import yaml
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.util import NetworkFactory, Record, RLParser
from cares_reinforcement_learning.util import helpers as hlp
from environments.environment_factory import EnvironmentFactory
from natsort import natsorted
from util.configurations import GymEnvironmentConfig

logging.basicConfig(level=logging.INFO)


def evaluate(data_path, training_config, seed, alg_config, env, agent1, agent2, record, pbe):

    model_path = Path(f"{data_path}/{seed}/models/")
    folders = list(model_path.glob("*"))

    folders = natsorted(folders)

    for folder in folders[:-2]:
        agent1.load_models(folder, f"{alg_config.algorithm}")
        agent2.load_models(folder, f"{alg_config.algorithm}")

        total_steps = int(folder.name.split("_")[-1]) - 1
        Model_name = total_steps

        # if alg_config.algorithm == "PPO":
        #     ppe.evaluate_ppo_network(
        #         env,
        #         agent1,
        #         agent2,
        #         training_config,
        #         record=record,
        #         total_steps=total_steps,
        #         # display=env_config.display,
        #     )

        # elif agent.type == "policy":
        pbe.evaluate_policy_network(
            env,
            agent1,
            agent2,
            Model_name,
            training_config,
            record=record,
            total_steps=total_steps,
            normalisation=True,
            # display=env_config.display,
        )

        # if agent.type == "discrete_policy":
        #     pbe.evaluate_policy_network(
        #         env,
        #         agent,
        #         training_config,
        #         record=record,
        #         total_steps=total_steps,
        #         normalisation=False,
        #         # display=env_config.display,
        #     )
        # elif agent.type == "value":
        #     vbe.evaluate_value_network(
        #         env,
        #         agent,
        #         training_config,
        #         alg_config,
        #         record=record,
        #         total_steps=total_steps,
        #         # display=env_config.display,
        #     )
        # else:
        #     raise ValueError(f"Agent type is unknown: {agent.type}")


def train(
    env_config, training_config, alg_config, env, env_eval, agent, memory, record, pbe
):
    if alg_config.algorithm == "PPO":
        ppe.ppo_train(
            env,
            env_eval,
            agent,
            record,
            training_config,
            alg_config,
            display=env_config.display,
        )
    elif agent.type == "policy":
        pbe.policy_based_train(
            env,
            env_eval,
            agent,
            memory,
            record,
            training_config,
            alg_config,
            display=env_config.display,
            normalisation=True,
        )
    elif agent.type == "discrete_policy":
        pbe.policy_based_train(
            env,
            env_eval,
            agent,
            memory,
            record,
            training_config,
            alg_config,
            display=env_config.display,
            normalisation=False,
        )
    elif agent.type == "value":
        vbe.value_based_train(
            env,
            env_eval,
            agent,
            memory,
            record,
            training_config,
            alg_config,
            display=env_config.display,
        )
    else:
        raise ValueError(f"Agent type is unknown: {agent.type}")


def main():
    """
    The main function that orchestrates the training process.
    """
    parser = RLParser(GymEnvironmentConfig)
    # configurations = parser.parse_args(argv=remaining_args)
    configurations = parser.parse_args()
    run_config = configurations["run_config"]
    env_config = configurations["env_config"]
    training_config = configurations["train_config"]
    alg_config = configurations["alg_config"]

    env_factory = EnvironmentFactory()
    network_factory = NetworkFactory()
    memory_factory = MemoryFactory()

    logging.info(
        "\n---------------------------------------------------\n"
        "ENVIRONMENT CONFIG\n"
        "---------------------------------------------------"
    )

    logging.info(f"\n{yaml.dump(dict(env_config), default_flow_style=False)}")

    logging.info(
        "\n---------------------------------------------------\n"
        "ALGORITHM CONFIG\n"
        "---------------------------------------------------"
    )

    logging.info(f"\n{yaml.dump(dict(alg_config), default_flow_style=False)}")

    logging.info(
        "\n---------------------------------------------------\n"
        "TRAINING CONFIG\n"
        "---------------------------------------------------"
    )

    logging.info(f"\n{yaml.dump(dict(training_config), default_flow_style=False)}")

    device = hlp.get_device()
    logging.info(f"Device: {device}")

    # run_name = input(
    #     "Double check your experiment configurations :) Press ENTER to continue. (Optional - Enter a name for this run)\n"
    # )

    if device.type == "cpu":
        no_gpu_answer = input(
            "Device being set as CPU - No cuda or mps detected. Do you still want to continue? Note: Training will be slower on cpu only. [y/n]"
        )

        if no_gpu_answer not in ["y", "Y"]:
            logging.info(
                "Terminating Experiment - check CUDA or mps is installed correctly."
            )
            sys.exit()

    logging.info(f"Command: {run_config.command}")

    logging.info(f"Data Path: {run_config.data_path}")

    base_log_dir = Record.create_base_directory(
        domain=env_config.domain,
        task=env_config.task,
        gym=env_config.gym,
        algorithm=alg_config.algorithm,
        base_dir = '/home/jil824/CARES_NEW/Metamorphic/model_trash/',
        run_name="",
    )

    logging.info(f"Base Log Directory: {base_log_dir}")

    record = Record(
        base_directory=f"{base_log_dir}",
        algorithm=alg_config.algorithm,
        task=env_config.task,
        agent=None,
        record_video=training_config.record_eval_video,
    )

    record.save_configurations(configurations)

    # Split the evaluation and training loop setup
    for iteration, seed in enumerate(training_config.seeds):
        logging.info(
            f"Iteration {iteration+1}/{len(training_config.seeds)} with Seed: {seed}"
        )
        # This line should be here for seed consistency issues
        env = env_factory.create_environment(env_config, alg_config.image_observation)
        env_eval = env_factory.create_environment(
            env_config, alg_config.image_observation
        )
        hlp.set_seed(seed)
        env.set_seed(seed)
        env_eval.set_seed(seed)

        logging.info(f"Algorithm: {alg_config.algorithm}")
        agent1 = network_factory.create_network(
            env.observation_space, env.action_num, alg_config
        )
        agent2 = network_factory.create_network(
            env.observation_space, env.action_num, alg_config
        )

        if agent1 is None:
            raise ValueError(
                f"Unknown agent for default algorithms {alg_config.algorithm}"
            )
        if agent2 is None:
            raise ValueError(
                f"Unknown agent for default algorithms {alg_config.algorithm}"
            )

        memory = memory_factory.create_memory(alg_config)

        # create the record class - standardised results tracking
        record.set_agent(agent1)
        record.set_agent(agent2)
        record.set_sub_directory(f"{seed}")

        if run_config.command == "train":
            # Train the policy or value based approach
            train(
                env_config,
                training_config,
                alg_config,
                env,
                env_eval,
                agent1,
                memory,
                record,
                pbe,
            )
        elif run_config.command == "evaluate":
            # Evaluate the policy or value based approach
            evaluate(
                run_config.data_path,
                training_config,
                seed,
                alg_config,
                env_eval,
                agent1,
                agent2,
                record,
                pbe,
            )

        record.save()


if __name__ == "__main__":
    
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument('--relation_path', type=str, required=True,
                            help='Path to the Python file containing the metamorphic relation')
    args, remaining_args = base_parser.parse_known_args()

    
    relation_path = args.relation_path
    module_name = os.path.splitext(os.path.basename(relation_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, relation_path)
    pbe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pbe)
    sys.argv = [sys.argv[0]] + remaining_args
    main()
