from datetime import datetime
import argparse
import yaml
import json
import dotenv
import networkx
import random
from pathlib import Path
import numpy as np

from twon_lss.simulations.wp3_simulation import (
    RandomRanker,
    SemanticSimilarityRanker,
    RankerArgs,
    ChronologicalRanker
)

from twon_lss.simulations.wp3_simulation import (
    Simulation,
    SimulationArgs,
    WP3Agent,
    AgentInstructions,
    WP3LLM,
    agent_parameter_estimation,
)

from twon_lss.schemas import Post, User, Feed, Network
from twon_lss.utility import LLM, Message, Chat, Noise, EmbeddingModelInterface, LocalEmbeddingModel, APIEmbeddingModel


def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)

def simulation_main(args):
    # Set ENV
    ENV = dotenv.dotenv_values("../" * 3 + ".env")

    # Set seeds for reproducibility of user configuration || First simulation step will remove seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parameters
    LENGTH_AGENT_MEMORY: int = args.agent_memory_length
    PERSISTENCE: int = args.persistence 
    STEPS: int = args.num_steps
    NUM_AGENTS: int = args.num_agents


    # Initialize ranker
    if args.ranker == "RandomRanker":
        RANKER = RandomRanker(
            args=RankerArgs(persistence=PERSISTENCE, noise=Noise(low=1.0, high=1.0))
        )
    elif args.ranker == "SemanticSimilarityRanker":
        RANKER = SemanticSimilarityRanker(
            args=RankerArgs(persistence=PERSISTENCE, noise=Noise(low=1.0, high=1.0))
        )
    elif args.ranker == "ChronologicalRanker":
        RANKER = ChronologicalRanker(
            args=RankerArgs(persistence=PERSISTENCE, noise=Noise(low=1.0, high=1.0))
        )
    else:
        raise ValueError(f"Unknown ranker type: {args.ranker}. Choose from 'RandomRanker', 'SemanticSimilarityRanker', 'ChronologicalRanker'.")


    # Load LLM
    AGENT_LLM = WP3LLM(
        api_key=ENV["RUNPOD_TOKEN"],
        url=args.runpod_url,
        enforce_disabled_reasoning=False if "TWON-Agent" not in args.model_name else True,
    )

    # Set up agents and users and individuals
    # Agents
    AGENTS_INSTRUCTIONS_CFG = json.load(open("../data/agents.instructions.json"))
    AGENTS_PERSONAS_CFG = json.load(open("../data/agents.personas_covid.json"))
    AGENTS_PERSONAS_CFG = [persona for persona in AGENTS_PERSONAS_CFG if persona["covid_flag"] == True]
    AGENTS_PERSONAS_CFG = random.sample(AGENTS_PERSONAS_CFG, k=NUM_AGENTS)
    AGENTS_PERSONAS_CFG = sorted(AGENTS_PERSONAS_CFG, key=lambda x: x["posts_per_day"], reverse=True)

    # Change Prompts in user history to newest version of instructions
    for persona in AGENTS_PERSONAS_CFG:
        for message in persona["history"]:
            if message.get("role") == "user":
                message["content"] = AGENTS_INSTRUCTIONS_CFG["actions"]["post_prompt"]

    # Users
    usernames = [user.get("screen_name", f"User{i}") for i, user in enumerate(AGENTS_PERSONAS_CFG)]    
    USERS = [User(id=username) for username in usernames]

    # Individuals
    INDIVIDUALS = {
        user: WP3Agent(
            llm=AGENT_LLM,
            instructions=AgentInstructions(**AGENTS_INSTRUCTIONS_CFG["actions"]),
            dynamic_cognition=args.dynamic_cognition,
            authentic_posts=persona["covid_tweets"] if args.covid_scenario else [message.get("content") for message in persona["history"] if message.get("role") == "assistant"],
            cognition=persona["cognition"] if args.use_auth_profile else "",
            bio=persona["bio"] if args.use_auth_profile else "You are a user commenting on online content. Keep your comments consistent with your previous writing style and the perspectives you have expressed earlier.",
            memory=persona["history"][-LENGTH_AGENT_MEMORY*2:] if args.use_auth_history else [],
            memory_length=LENGTH_AGENT_MEMORY*2,
            activation_probability=agent_parameter_estimation(posts_per_day=persona["posts_per_day"] if args.use_auth_activation else 2.6, seed=i)["activation_probability"],
            posting_probability=agent_parameter_estimation(posts_per_day=persona["posts_per_day"] if args.use_auth_activation else 2.6, seed=i)["posting_probability"],
            read_amount=agent_parameter_estimation(posts_per_day=persona["posts_per_day"] if args.use_auth_activation else 2.6, seed=i)["read_amount"],
        )
        for i, (user, persona) in enumerate(zip(
            USERS, AGENTS_PERSONAS_CFG, strict=False
        ))
    }


    # Set up network
    if args.network_type == "complete":
        NETWORK = Network.from_graph(networkx.complete_graph(n = len(USERS)), USERS)
    elif args.network_type == "barabasi_albert":
        NETWORK = Network.from_graph(networkx.barabasi_albert_graph(n = len(USERS), m=args.m), USERS)


    # Set up initial feed
    if args.covid_scenario:
        init_tweets = [
            [post for post in personas["covid_tweets"]]
            for personas in AGENTS_PERSONAS_CFG
        ]
    else:
        init_tweets = [
            [message.get("content") for message in personas["history"] if message.get("role") == "assistant"]
            for personas in AGENTS_PERSONAS_CFG
        ]

    FEED = Feed([
        Post(user=user, content=post)
        for user, history in zip(USERS, init_tweets, strict=False)
        for post in history[-2:]
    ])

    
    # Initialize simulation
    simulation = Simulation(
        args=SimulationArgs(num_steps=STEPS, persistence=PERSISTENCE),
        persistence=PERSISTENCE,
        embedding_model= LocalEmbeddingModel(),
        ranker=RANKER,
        individuals=INDIVIDUALS,
        network=NETWORK,
        feed=FEED,
        output_path=Path(f"runs/{args.name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}/").mkdir(exist_ok=True, parents=True) or Path(f"runs/{args.name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}/")
    )
    
    # Save configuration
    with open(Path(simulation.output_path) / "simulation_config.yaml", "w") as f:
        args.version = "1.2.0" # Increase if script changes!!!
        yaml.dump(vars(args), f)

    # Start simulation
    simulation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file to override arguments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--agent_memory_length", type=int, default=15, help="Length of agent memory in actions (1 action = 1 prompt-completion pair)")
    parser.add_argument("--num_agents", type=int, default=1028, help="Number of agents in the simulation")
    parser.add_argument("--num_steps", type=int, default=288, help="Number of simulation steps")
    parser.add_argument("--network_type", type=str, default="complete", help="Type of network to use (e.g., complete, barabasi_albert)")
    parser.add_argument("--m", type=int, default=40, help="Barabasi-Albert model parameter m (number of edges to attach from a new node to existing nodes)")
    parser.add_argument("--persistence", type=int, default=144, help="Ranker persistence in steps")
    parser.add_argument("--covid_scenario", type=bool, default=False, help="Whether to use the COVID-19 data-driven scenario")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--runpod_url", type=str)
    parser.add_argument("--dynamic_cognition", type=bool, default=True, help="Whether to use dynamic agent cognition")
    parser.add_argument("--use_auth_profile", type=bool, default=True, help="Whether to use agent bio and cognition")
    parser.add_argument("--use_auth_history", type=bool, default=True, help="Whether to use agent history")
    parser.add_argument("--use_auth_activation", type=bool, default=True, help="Whether to use agent-specific activation parameters")
    parser.add_argument("--name", type=str, default="data-driven-cond", help="Name of the simulation run")
    parser.add_argument("--ranker", type=str, default="SemanticSimilarityRanker", help="Type of ranker to use (e.g., RandomRanker, SemanticSimilarityRanker, ChronologicalRanker)")
    args = parser.parse_args()

    # Override with YAML if provided
    if args.config:
        yaml_args = load_config(args.config)
        # Merge: CLI args override YAML
        for key, value in vars(yaml_args).items():
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    
    simulation_main(args)