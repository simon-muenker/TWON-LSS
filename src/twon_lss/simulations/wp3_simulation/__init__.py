import logging
import multiprocessing
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import random
import numpy as np
import os
import time
from rich.progress import track

from twon_lss.interfaces import (
    AgentInterface,
    AgentActions,
    SimulationInterface,
    SimulationInterfaceArgs,
)
from twon_lss.schemas import Feed, User, Post


from twon_lss.simulations.wp3_simulation.agent import WP3Agent, AgentInstructions
from twon_lss.simulations.wp3_simulation.ranker import RankerArgs, SemanticSimilarityRanker, RandomRanker, ChronologicalRanker

from twon_lss.simulations.wp3_simulation.utility import WP3LLM, agent_parameter_estimation, simulation_load_estimator
from twon_lss.utility.llm import EmbeddingModelInterface

__all__ = [
    "Simulation",
    "SimulationArgs",
    "Agent",
    "AgentInstructions",
    "Ranker",
    "RandomRanker",
    "SemanticSimilarityRanker",
    "RankerArgs",
    "WP3LLM",
    "agent_parameter_estimation",
    "simulation_load_estimator",
    "ChronologicalRanker",
]


class SimulationArgs(SimulationInterfaceArgs):
    num_steps: int = 288
    persistence: int = 144


class Simulation(SimulationInterface):

    embedding_model: typing.Union[EmbeddingModelInterface, None] = None

    def model_post_init(self, __context: typing.Any):

        # Ensure there is no seed
        random.seed() 
        np.random.seed()
        logging.debug(">f seeds removed for simulation run")
        
        # Generate initial embeddings if required
        if hasattr(self, "embedding_model") and self.embedding_model is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = "false" # Must be set before importing transformers/tokenizers
            logging.debug(f">f generating embeddings for {len(self.feed)} feed posts")
            embeddings = self.embedding_model.extract(
                [post.content for post in self.feed]
            )
            for post, embedding in zip(self.feed, embeddings):
                post.embedding = embedding

        # Add posts to agents for proper ranking in the first step
        for post in self.feed:
            agent = self.individuals.get(post.user)
            if agent:
                agent.posts.append(post)

        # Start simulation
        logging.debug(">f init simulation")
        self.output_path.mkdir(exist_ok=True)
        (self.output_path / "rankings").mkdir(exist_ok=True)

    def _wrapper_step_agent(
        self,
        post_scores: typing.Dict[typing.Tuple[User, Post], float],
        user: User,
        agent: AgentInterface,
    ):
        user_feed = self._filter_posts_by_user(post_scores, user)
        user_feed.sort(key=lambda x: x[1], reverse=True)

        return self._step_agent(user, agent, Feed([post for post, _ in user_feed]))


    def _step_agent(self, user: User, agent: WP3Agent, feed: Feed):

        posts: typing.List[Post] = []

        # Session activates
        agent.activations += 1

        # Determine how many posts the users reads
        user_feed_top = feed[:agent.read_amount]
        logging.debug(f">i number of feed items {len(user_feed_top)} for user {user.id}")

        # Read posts in the feed
        agent.consume_feed(user_feed_top, user)

        # Post new content
        agent_posting_probability = agent.posting_probability
        while agent_posting_probability > 1.0:
            posts.append(Post(user=user, content=agent.post()))
            agent_posting_probability -= 1.0
        if random.random() <= agent.posting_probability:
            posts.append(Post(user=user, content=agent.post()))
        agent.posts.extend(posts)


        if agent.dynamic_cognition:
            agent.cognition_update()

        return user, agent, posts
    

    def _step(self, n: int = 0) -> None:
        # Strip feed to only recent posts for efficiency
        stripped_feed = self.feed.filter_by_timestamp(n, self.args.persistence)
        
        # Calculate post scores
        active_individuals = {user: agent for user, agent in self.individuals.items() if random.random() <= agent.activation_probability}
        logging.debug(f">i calculating post scores for {len(active_individuals)} active individuals")
        if len(active_individuals) == 0:
            logging.warning(">w no active individuals this step ,this may be due to low activation probabilities -> consider adjusting them.")
            return

        post_scores: typing.Dict[typing.Tuple[User, Post], float] = self.ranker(
            individuals=active_individuals, feed=stripped_feed, network=self.network
        )

        self._rankings_to_json(path = self.output_path / "rankings" / f"step_{n}_ranking.json", rankings=post_scores)
        
        # For I/O bound: more workers; for CPU bound: stick to cpu_count
        logging.debug(f">i stepping through {len(active_individuals)} active individuals")
        max_workers = min(len(active_individuals), multiprocessing.cpu_count() * 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._wrapper_step_agent, post_scores, user, agent): user 
                for user, agent in active_individuals.items()
            }

            responses = []
            for future in as_completed(futures):
                responses.append(future.result())

        # Add timestamp to posts
        posts = [post for _, _, agent_posts in responses for post in agent_posts]
        for post in posts:
            post.timestamp = n

        # Generate embeddings if required
        if hasattr(self, "embedding_model") and self.embedding_model is not None:
            logging.debug(f">f generating embeddings for {len(posts)} posts")
            embeddings = self.embedding_model.extract([post.content for post in posts])

            for post, embedding in zip(posts, embeddings):
                post.embedding = embedding

        self.feed.extend(posts)