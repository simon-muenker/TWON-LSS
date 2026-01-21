from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import statistics
import logging
import time
import random
import typing

from twon_lss.interfaces import RankerInterface, RankerArgsInterface

from twon_lss.schemas import User, Post, Feed
from twon_lss.schemas.network import Network
from twon_lss.simulations.wp3_simulation.agent import WP3Agent
from twon_lss.utility import LLM, EmbeddingModelInterface, LocalEmbeddingModel

from sklearn.metrics.pairwise import cosine_similarity


__all__ = ["Ranker", "RankerArgs"]


class RankerArgs(RankerArgsInterface):
    pass


class RandomRanker(RankerInterface):
    args: RankerArgs = RankerArgs()

    def __call__(
        self, individuals: typing.List[User], feed: Feed, network: Network
        ) -> typing.Dict[typing.Tuple[User, Post], float]:
        # WP3 adapted call version because simulation step passes individuals instead of users at ranker call
        # For this ranker we dont need individuals, so we drop them when passing to extract
        return super().__call__(list(individuals.keys()), feed, network)

    def _compute_network(self, post: Post) -> float:
        return 0.0

    def _compute_individual(self, agent: WP3Agent, post: Post, feed: Feed) -> float:
        return random.random()


class ChronologicalRanker(RankerInterface):
    args: RankerArgs = RankerArgs()

    def __call__(
        self, individuals: typing.List[User], feed: Feed, network: Network
        ) -> typing.Dict[typing.Tuple[User, Post], float]:
        # WP3 adapted call version because simulation step passes individuals instead of users at ranker call
        # For this ranker we dont need individuals, so we drop them when passing to extract
        return super().__call__(list(individuals.keys()), feed, network)
    
    def _compute_network(self, post: Post) -> float:
        return post.timestamp

    def _compute_individual(self, agent: WP3Agent, post: Post, feed: Feed) -> float:
        return random.random() # So that we do not cut in generation order but order by timestamp only and shuffle within same timestamp posts

    
class SemanticSimilarityRanker(RankerInterface):
    args: RankerArgs = RankerArgs()

    def __call__(
        self, individuals: typing.List[User], feed: Feed, network: Network
        ) -> typing.Dict[typing.Tuple[User, Post], float]:

        logging.debug(f"{len(feed)=}")

        # compute global scores
        global_scores: typing.Dict[str, float] = {}
        for post in feed:
            global_scores[post.id] = self._compute_network(post)

        # parallelize user processing
        with ProcessPoolExecutor() as executor:
            user_results = executor.map(
                self._process_user,
                [(individuals[individual], individual, feed, network, global_scores) for individual in individuals.keys()]
            )

        # create lookup dicts for user.id to user and post.id to post
        user_lookup = {user.id: user for user in individuals.keys()}
        post_lookup = {post.id: post for post in feed}

        # merge results
        final_scores = {}
        for user_score_dict in user_results:

            # map back to (User, Post) keys
            mapped_dict = {
                (user_lookup[user_id], post_lookup[post_id]): score
                for (user_id, post_id), score in user_score_dict.items()
            }

            final_scores.update(mapped_dict)

        return final_scores
    

    def _process_user(
        self, args: typing.Tuple[WP3Agent, User, Feed, Network, typing.Dict[str, float]]
    ) -> typing.Dict[typing.Tuple[User, Post], float]:
        agent, user, feed, network, global_scores = args
        scores = {}

        for post in self.get_individual_posts(user, feed, network):
            individual_score = self._compute_individual(agent, post, feed)
            global_score = global_scores[post.id]

            combined_score = (
                self.args.weights.individual * individual_score
                + self.args.weights.network * global_score
            )

            scores[(user.id, post.id)] = combined_score # * self.args.noise() 

        return scores
    

    def _compute_network(self, post: Post) -> float:
        return 0.0

    def _compute_individual(self, agent: WP3Agent, post: Post, feed: Feed) -> float:
        try:
            return(statistics.mean(
                cosine_similarity([post.embedding], [item.embedding for item in agent.posts[-10:]])[0]
            ))
        except Exception as e:
            logging.error(f"Failed to compute semantic similarity: {e}")
            return 0.0