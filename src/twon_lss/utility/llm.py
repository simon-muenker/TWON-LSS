from abc import abstractmethod
import logging
import requests
import typing
import time

import pydantic


class Message(pydantic.BaseModel):
    role: typing.Literal["system", "user", "assistant"]
    content: str

class Chat(pydantic.RootModel):
    root: typing.List[Message]


# LLM Implementation
class LLM(pydantic.BaseModel):
    api_key: str

    model: str = "Qwen/Qwen3-4B-Instruct-2507:nscale"
    url: str = "https://router.huggingface.co/v1/chat/completions"

    def _query(self, payload):
        headers: dict = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.url, headers=headers, json=payload)
        return response.json()
    
    def generate(self, chat: Chat, max_retries: int = 5) -> str:
        try:
            response: str = self._query(
                {
                    "messages": chat.model_dump(),
                    "model": self.model,
                }
            )["choices"][0]["message"]["content"]
        
        except Exception as e:
            logging.error(f"Failed to query LLM: {e}")
            if max_retries > 0:
                time.sleep(60)
                return self.generate(chat, max_retries - 1)
            raise RuntimeError("Failed to generate response from LLM after retries") from e
        
        return response



# Embedding Model Interface and Implementations
class EmbeddingModelInterface(pydantic.BaseModel):
    @abstractmethod
    def extract(self, text: typing.Optional[typing.Union[str, list]], max_retries: int = 3):
        pass

class LocalEmbeddingModel(EmbeddingModelInterface):
    model_name: str = "mixedbread-ai/mxbai-embed-large-v1"
    batch_size: int = 16
    model: typing.Any = None
    
    def model_post_init(self, context):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

    def extract(self, text: typing.Optional[typing.Union[str, list]]):
        if isinstance(text, str):
            return self.model.encode([text])[0].tolist()
        elif isinstance(text, list):
            embeddings = []
            for i in range(0, len(text), self.batch_size):
                batch = text[i:i + self.batch_size]
                emb_batch = self.model.encode(batch)
                embeddings.extend([emb.tolist() for emb in emb_batch])
            return embeddings
        
        
class APIEmbeddingModel(EmbeddingModelInterface):
    api_key: str
    model: str = "Qwen/Qwen3-4B-Instruct-2507:nscale"
    url: str = "https://router.huggingface.co/v1/chat/completions"

    def _query(self, payload):
        headers: dict = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.url, headers=headers, json=payload)
        return response.json()
    

    def extract(self, text: typing.Optional[typing.Union[str, list]], max_retries: int = 3):
        """
        Returns embeddings for either text or list of texts.
        """

        if self.url == "https://router.huggingface.co/v1/chat/completions":
            raise ValueError("Extract endpoint not supported for chat completions API. Use HF-Inference URL that includs endpoint and model for extract")
        

        if len(text) < 100:
            try:
                return self._query({
                    "inputs": text,
                })
            except Exception as e:
                logging.error(f"Failed to extract embeddings: {e}")
                if max_retries > 0:
                    time.sleep(5)
                    return self.extract(text, max_retries - 1)
                raise RuntimeError("Failed to extract embeddings after retries") from e
            
        else:
            # Chunking for long texts
            CHUNK_SIZE = 100
            chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
            embeddings = []
            for chunk in chunks:
                try:
                    emb_chunk = self._query({
                        "inputs": chunk,
                    })
                    embeddings.extend(emb_chunk)
                except Exception as e:
                    logging.error(f"Failed to extract embeddings for chunk: {e}")
                    if max_retries > 0:
                        time.sleep(5)
                        emb_chunk = self.extract(chunk, max_retries - 1)
                        embeddings.extend(emb_chunk)
                    else:
                        raise RuntimeError("Failed to extract embeddings after retries") from e
            return embeddings
