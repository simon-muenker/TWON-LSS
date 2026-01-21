import abc
import typing
import logging

import pydantic

from twon_lss.utility import Noise
from twon_lss.schemas import User, Post, Feed, Network

from concurrent.futures import ProcessPoolExecutor


class RankerInterfaceWeights(pydantic.BaseModel):
    network: float = 1.0
    individual: float = 1.0


class RankerArgsInterface(abc.ABC, pydantic.BaseModel):
    weights: RankerInterfaceWeights = pydantic.Field(
        default_factory=RankerInterfaceWeights
    )
    noise: Noise = pydantic.Field(default_factory=Noise)


class RankerInterface(abc.ABC, pydantic.BaseModel):
    args: RankerArgsInterface = pydantic.Field(default_factory=RankerArgsInterface)

    def __call__(
        self, users: typing.List[User], feed: Feed, network: Network
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
                [(user, feed, network, global_scores) for user in users]
            )

        user_lookup = {user.id: user for user in users}
        post_lookup = {post.id: post for post in feed}

        # merge results
        final_scores = {}
        for user_score_dict in user_results:
            mapped_dict = {
                (user_lookup[user_id], post_lookup[post_id]): score
                for (user_id, post_id), score in user_score_dict.items()
            }

            final_scores.update(mapped_dict)

        return final_scores

    def _process_user(
        self, args: typing.Tuple[User, Feed, Network, typing.Dict[str, float]]
    ) -> typing.Dict[typing.Tuple[User, Post], float]:
        user, feed, network, global_scores = args
        scores = {}

        for post in self.get_individual_posts(user, feed, network):
            individual_score = self._compute_individual(user, post, feed)
            global_score = global_scores[post.id]

            combined_score = (
                self.args.weights.individual * individual_score
                + self.args.weights.network * global_score
            )

            scores[(user.id, post.id)] = self.args.noise() * combined_score

        return scores

    def get_individual_posts(self, user: User, feed: Feed, network: Network):

        return [
            post
            for neighbor in network.get_neighbors(user)
            for post in feed.get_items_by_user(neighbor).get_unread_items_by_user(user)
        ]

    @abc.abstractmethod
    def _compute_network(self, post: Post) -> float:
        pass

    @abc.abstractmethod
    def _compute_individual(self, user: User, post: Post, feed: Feed) -> float:
        pass
