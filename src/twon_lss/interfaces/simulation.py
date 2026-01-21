import abc
import typing
import logging
import pathlib
import json
import itertools
import time

import pydantic

from rich.progress import track

from twon_lss.interfaces import AgentInterface, RankerInterface
from twon_lss.schemas import User, Network, Feed, Post
from twon_lss.utility.llm import EmbeddingModelInterface


class SimulationInterfaceArgs(pydantic.BaseModel):
    num_steps: int = 100
    num_posts_to_interact_with: int = 5
    persistence: int = 50


class SimulationInterface(abc.ABC, pydantic.BaseModel):
    args: SimulationInterfaceArgs = pydantic.Field(
        default_factory=SimulationInterfaceArgs
    )

    ranker: RankerInterface
    individuals: typing.Dict[User, AgentInterface]

    network: Network
    feed: Feed

    output_path: pydantic.DirectoryPath = pydantic.Field(
        default_factory=lambda: pathlib.Path.cwd() / "output/"
    )

    def model_post_init(self, __context: typing.Any):

        logging.debug(">f init simulation")
        self.output_path.mkdir(exist_ok=True)
        (self.output_path / "rankings").mkdir(exist_ok=True)

    def __call__(self) -> None:
        for n in track(range(self.args.num_steps)):
            time_start = time.time()
            logging.debug(f">f simulate step {n=}")
            
            self._step(n)

            time_end = time.time()
            logging.debug(f">f step {n=} done in {time_end - time_start:.2f}s")
            with open(self.output_path / "time_per_step.log", "a") as f:
                f.write(f"{n},{time_end - time_start:.2f}\n")

            if n % 10 == 0:
                self.network.to_json(self.output_path / "network.json")
                self.feed.to_json(self.output_path / "feed.json")
                self._individuals_to_json(self.output_path / "individuals.json")

    def _step(self, n: int = 0) -> None:
        post_scores: typing.Dict[typing.Tuple[User, Post], float] = self.ranker(
            users=self.individuals.keys(), feed=self.feed, network=self.network
        )

        self._rankings_to_json(path = self.output_path / "rankings" / f"step_{n}_ranking.json", rankings=post_scores)

        responses: typing.List[
            typing.Tuple[User, AgentInterface, typing.List[Post]]
        ] = list(
            itertools.starmap(
                self._wrapper_step_agent,
                [
                    (post_scores, user, agent)
                    for user, agent in self.individuals.items()
                ],
            )
        )

        # Add ID to posts
        posts = [post for _, _, agent_posts in responses for post in agent_posts]
        for post in posts:
            post.timestamp = n
            
        self.individuals = {user: agent for user, agent, _ in list(responses)}
        self.feed.extend(posts)

    def _wrapper_step_agent(
        self,
        post_scores: typing.Dict[typing.Tuple[User, Post], float],
        user: User,
        agent: AgentInterface,
    ) -> typing.Tuple[User, AgentInterface]:
        user_feed = self._filter_posts_by_user(post_scores, user)
        user_feed.sort(key=lambda x: x[1], reverse=True)
        user_feed_top = user_feed[: self.args.num_posts_to_interact_with]

        logging.debug(f">i number of feed items {len(user_feed)} for user {user.id}")
        return self._step_agent(user, agent, Feed([post for post, _ in user_feed_top]))

    @abc.abstractmethod
    def _step_agent(
        self, user: User, agent: AgentInterface, feed: Feed
    ) -> typing.Tuple[User, AgentInterface, typing.List[Post]]:
        pass

    def _individuals_to_json(self, path: str):
        json.dump(
            {
                user.id: agent.model_dump(mode="json", exclude=["llm"])
                for user, agent in self.individuals.items()
            },
            open(path, "w"),
            indent=4,
        )

    def _rankings_to_json(
        self, rankings: typing.Dict[typing.Tuple[User, Post], float], path: str
    ):
        json.dump(
            [
                {"user": user.id, "post": post.id, "score": score}
                for (user, post), score in rankings.items()
            ],
            open(path, "w"),
            indent=4,
        )

    @staticmethod
    def _filter_posts_by_user(
        posts_scores: typing.Dict[typing.Tuple[User, Post], float], user: User
    ) -> typing.List[typing.Tuple[Post, float]]:
        return [
            (post_content, post_score)
            for (post_user, post_content), post_score in posts_scores.items()
            if user == post_user
        ]
