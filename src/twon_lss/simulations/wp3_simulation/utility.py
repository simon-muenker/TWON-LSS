import logging
import random
import requests
import typing
import time
import numpy as np
import pydantic

from twon_lss.utility import LLM, Chat, Message


class WP3LLM(LLM):
    api_key: str
    url: str
    enforce_disabled_reasoning: bool = True
    missed_responses: int = 0
    max_missed_responses: int = 5

    def _query(self, payload):
        headers: dict = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.url, headers=headers, json=payload)
        return response.json()
    
    def generate(self, chat: Chat, max_retries: int = 5) -> str:

        try:
            payload = {
                "input": {
                    "messages": chat.model_dump(),
                    "sampling_params": {"max_tokens": 400}
                }
            }

            if self.enforce_disabled_reasoning:
                for message in payload["input"]["messages"]:
                    if message["role"] == "user":
                        message["content"] += " /no_think"
                    
            response: str = self._query(payload)
            logging.info(f"LLM response: {response}")

            response = response["output"][0]["choices"][0]["tokens"][0]
            if self.enforce_disabled_reasoning:
                response = response.split("</think>")[-1].strip()

        except Exception as e:
            logging.error(f"Failed to query LLM: {e}")
            if max_retries > 0:
                time.sleep(60)
                return self.generate(chat, max_retries - 1)
            
            if self.missed_responses < self.max_missed_responses:
                self.missed_responses += 1
                logging.error(f"Missed responses: {self.missed_responses}")
                return ""
            
            raise RuntimeError(f"Failed to generate response from LLM after retries\n\n{chat.model_dump()}\n\n") from e
        
        return response





def power_law_sample(min_val, max_val, a=2.5):
    """
    Sample from power law using np.random.power, with min_val being most likely.
    
    Parameters:
    - min_val: minimum value (most likely, heavy side)
    - max_val: maximum value (least likely)
    - a: shape parameter (higher = more concentration at min_val)
    
    Returns: float in [min_val, max_val]
    """
    # np.random.power(a) gives values in [0,1] biased toward 1
    # We invert it so high values map to min_val
    sample = 1 - np.random.power(a)
    
    # Scale to [min_val, max_val]
    return min_val + (max_val - min_val) * sample



def agent_parameter_estimation(posts_per_day, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    # Higher posting frequency → higher activation (correlated)
    a = max(1, 15 - posts_per_day)  # More posts → lower a → higher activation

    activation_probability = power_law_sample(0.007, 0.5, a=a)
    # Reads per day: correlated with activation probability
    # More active users read more per day
    min_reads = activation_probability * 144 * 3
    max_reads = min(activation_probability * 144 * 20, 750) # Cap at 750 reads/day
    reads_per_day = power_law_sample(min_reads, max_reads, a=3.0)
    
    # Calculate reads per activation
    activations_per_day = activation_probability * 144
    read_amount = int(reads_per_day / activations_per_day)
    read_amount = np.clip(read_amount, 3, 100)
    
    # Calculate posting probability
    posting_probability = posts_per_day / (activation_probability * 144)
    
    return {
        "activation_probability": activation_probability,
        "read_amount": read_amount,
        "posting_probability": posting_probability
    }




def simulation_load_estimator(agents_config_list = list[dict[str, float]], user_confirmation: bool = False):
    
    # Estimate the complexity of the simulation based on agent parameters
    num_agents = len(agents_config_list)

    # Average number of activated agents per step
    avg_activation = sum(agent["activation_probability"] for agent in agents_config_list) / num_agents

    # Average number of posts per round (accounts also for activation)
    avg_posts_per_round = sum(
        agent["posting_probability"] * agent["activation_probability"] for agent in agents_config_list
    ) / num_agents


    # Max read value per step
    max_reads_per_step = max(agent["read_amount"] for agent in agents_config_list)
    min_reads_per_step = min(agent["read_amount"] for agent in agents_config_list)

    # Max reads per day (assuming 6 steps per hour, 24 hours)
    max_reads_per_day = max(agent["read_amount"] * 24 * 6 * agent["activation_probability"] for agent in agents_config_list)
    min_reads_per_day = min(agent["read_amount"] * 24 * 6 * agent["activation_probability"] for agent in agents_config_list)


    print("\n\n=== Simulation Load Estimation ===")
    print(f"Number of agents: {num_agents}")

    # Read statistics
    print("\n--- Read Statistics ---")
    print(f"Minimum reads per agent per day: {min_reads_per_day}")
    print(f"Maximum reads per agent per day: {max_reads_per_day}")
    print(f"Minimum reads per agent per step: {min_reads_per_step}")
    print(f"Maximum reads per agent per step: {max_reads_per_step}")

    # Activation and posting statistics
    print("\n--- Activation Statistics ---")
    print(f"Estimated average activated agents per step: {avg_activation * num_agents:.2f}")
    
    # Posting statistics
    print("\n--- Posting Statistics ---")
    print(f"Estimated average new posts per step: {avg_posts_per_round * num_agents:.2f}")
    print(f"Tweets generated per day: {avg_posts_per_round * num_agents * 24 * 6:.2f}")
    print("===================================\n\n")

    # Warning if tweets generated per day is not at least double the number of max reads per day, warn the user
    if (avg_posts_per_round * num_agents * 24 * 6) < (2 * max_reads_per_day):
        print("!!! WARNING: The number of generated posts per day is less than double the maximum reads per day. This may lead to agents not having enough content to read/renders ranking useless. Consider increasing posting probabilities or number of agents. !!!")
        user_confirmation = True

    # User confirmation
    if user_confirmation:
        confirmation = input("Do you want to proceed with these settings? (y/n): ")
        if confirmation.lower() != 'y':
            raise RuntimeError("Simulation aborted by user.")
