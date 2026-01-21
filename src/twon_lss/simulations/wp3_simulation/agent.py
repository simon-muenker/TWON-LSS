import typing
import re
import enum
import logging

import pydantic

from twon_lss.interfaces import AgentInterface

from twon_lss.schemas import Post
from twon_lss.utility import LLM, Chat, Message
from twon_lss.schemas import Feed, User, Post

import numpy as np


__all__ = ["Agent", "AgentInstructions"]


class AgentInstructions(pydantic.BaseModel):
    read_prompt: str
    post_prompt: str
    feed_placeholder: str
    cognition_update: str
    profile_format: str


class WP3Agent(AgentInterface):
    
    # LLM backbone
    llm: LLM 

    # Static variables
    authentic_posts: list[str] = pydantic.Field(default_factory=list)
    dynamic_cognition: bool = pydantic.Field(default=False)
    memory_length: int = pydantic.Field(default=4, ge=0, le=50)
    instructions: AgentInstructions
    bio: str = pydantic.Field(default="")
    activation_probability: float
    posting_probability: float
    read_amount: int

    # Dynamic variables
    cognition: str = pydantic.Field(default="")
    memory: typing.List[Message] = pydantic.Field(default_factory=list)
    posts: typing.List[Post] = pydantic.Field(default_factory=list)
    activations: int = 0
    
    def select_actions(self, post: Post):
        pass

    def _append_to_memory(self, content: str, role="assistant") -> None:
        self.memory.append(Message(role=role, content=content))

    def _inference(self, prompt: str) -> Chat:
        return self.llm.generate(
            Chat(
                [
                    Message(role="system", content=self._profile()),
                    *self.memory[-self.memory_length:],
                    Message(role="user", content=prompt),
                ]
            )
        )
    
    # Actions
    def _profile(self) -> str:
        return self.instructions.profile_format.format(bio=self.bio, cognition=self.cognition)
    
    def cognition_update(self) -> None:
        """
        Currently not used in the simulation, but can be called to update the agent's cognition based on its memory
        """
        response: str = self._inference(self.instructions.cognition_update + "\nCurrent cognition: " + self.cognition)
        logging.debug(f"Agent response: {response}")
        self.cognition = response
        

    def _like(self, post: Post, user: User) -> None:
        """
        Currently a simple probabilistic like function. The likes arent used in the simulation yet.
        """
        if np.random.rand() <= 0.25:
            post.likes.add(user)
            return True
        return False
    
    def _read(self, feed_str) -> str:
        response: str = self._inference(self.instructions.read_prompt.format(feed=feed_str))
        logging.debug(f"Agent response: {response}")

        self._append_to_memory(self.instructions.feed_placeholder, role="user")
        self._append_to_memory(response)

    def consume_feed(self, posts: list[Post], user:User) -> str:
        feed_str = ""
        for post in posts:
            post.reads.add(user)
            if self._like(post, user):
                feed_str += f">{post.user.id}: {post.content}\n"   # (You like this post) 
            else:
                feed_str += f">{post.user.id}: {post.content}\n"
        self._read(feed_str)
        
    def post(self) -> str:     
        prompt = self.instructions.post_prompt
        response: str = self._inference(prompt)
        self._append_to_memory(prompt, role="user")
        self._append_to_memory(response)
        logging.debug(f"Agent response: {response}")

        return response
