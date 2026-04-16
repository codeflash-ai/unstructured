from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Optional

import numpy as np
from numba import njit
from pydantic import Field, SecretStr

from unstructured.documents.elements import Element
from unstructured.embed.interfaces import BaseEmbeddingEncoder, EmbeddingConfig
from unstructured.utils import requires_dependencies

if TYPE_CHECKING:
    from voyageai import Client

# Token limits for different VoyageAI models
VOYAGE_TOTAL_TOKEN_LIMITS = {
    "voyage-context-3": 32_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-3.5": 320_000,
    "voyage-2": 320_000,
    "voyage-02": 320_000,
    "voyage-3-large": 120_000,
    "voyage-code-3": 120_000,
    "voyage-large-2-instruct": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-multilingual-2": 120_000,
    "voyage-law-2": 120_000,
    "voyage-large-2": 120_000,
    "voyage-3": 120_000,
    "voyage-3-lite": 120_000,
    "voyage-code-2": 120_000,
    "voyage-3-m-exp": 120_000,
    "voyage-multimodal-3": 120_000,
}

# Batch size for embedding requests (max documents per batch)
MAX_BATCH_SIZE = 1000


class VoyageAIEmbeddingConfig(EmbeddingConfig):
    api_key: SecretStr
    model_name: str
    show_progress_bar: bool = False
    batch_size: Optional[int] = Field(default=None)
    truncation: Optional[bool] = Field(default=None)
    output_dimension: Optional[int] = Field(default=None)

    @requires_dependencies(
        ["voyageai"],
        extras="embed-voyageai",
    )
    def get_client(self) -> "Client":
        """Creates a VoyageAI python client to embed elements."""
        from voyageai import Client

        return Client(
            api_key=self.api_key.get_secret_value(),
        )

    def get_token_limit(self) -> int:
        """Get the token limit for the current model."""
        return VOYAGE_TOTAL_TOKEN_LIMITS.get(self.model_name, 120_000)


@dataclass
class VoyageAIEmbeddingEncoder(BaseEmbeddingEncoder):
    config: VoyageAIEmbeddingConfig

    def get_exemplary_embedding(self) -> List[float]:
        return self.embed_query(query="A sample query.")

    def initialize(self):
        pass

    @property
    def num_of_dimensions(self) -> tuple[int, ...]:
        exemplary_embedding = self.get_exemplary_embedding()
        return np.shape(exemplary_embedding)

    @property
    def is_unit_vector(self) -> bool:
        exemplary_embedding = self.get_exemplary_embedding()
        return np.isclose(np.linalg.norm(exemplary_embedding), 1.0)

    def _is_context_model(self) -> bool:
        """Check if the model is a contextualized embedding model."""
        return "context" in self.config.model_name

    def _build_batches(self, texts: List[str], client: "Client") -> Iterable[List[str]]:
        """
        Generate batches of texts based on token limits.

        Args:
            texts: List of texts to batch.
            client: VoyageAI client instance to use for tokenization.

        Yields:
            Batches of texts as lists.
        """
        if not texts:
            return

        max_tokens_per_batch = self.config.get_token_limit()

        # Tokenize all texts in one API call
        all_token_lists = client.tokenize(texts, model=self.config.model_name)
        token_counts = np.array([len(tokens) for tokens in all_token_lists], dtype=np.int64)
        # Compute batch boundaries using numba for performance
        batch_indices = _fast_batch_indices(token_counts, max_tokens_per_batch)

        for i in range(len(batch_indices) - 1):
            yield texts[batch_indices[i] : batch_indices[i + 1]]

    def _embed_batch(
        self, batch: List[str], client: "Client", input_type: str = "document"
    ) -> List[List[float]]:
        """
        Embed a batch of texts using the appropriate method for the model.

        Args:
            batch: List of texts to embed.
            client: VoyageAI client instance to use for embedding.
            input_type: Type of input ("document" or "query").

        Returns:
            List of embedding vectors.
        """
        if self._is_context_model():
            result = client.contextualized_embed(
                inputs=[batch],
                model=self.config.model_name,
                input_type=input_type,
                output_dimension=self.config.output_dimension,
            )
            return [list(emb) for emb in result.results[0].embeddings]
        else:
            result = client.embed(
                texts=batch,
                model=self.config.model_name,
                input_type=input_type,
                truncation=self.config.truncation,
                output_dimension=self.config.output_dimension,
            )
            return [list(emb) for emb in result.embeddings]

    def embed_documents(self, elements: List[Element]) -> List[Element]:
        """
        Embed documents with automatic batching based on token limits.

        Args:
            elements: List of elements to embed.

        Returns:
            List of elements with embeddings added.
        """
        if not elements:
            return []

        client = self.config.get_client()
        texts = [str(e) for e in elements]
        all_embeddings: List[List[float]] = []

        # Process each batch
        batches = list(self._build_batches(texts, client))

        if self.config.show_progress_bar:
            try:
                from tqdm.auto import tqdm  # type: ignore

                batches = tqdm(batches, desc="Embedding batches")
            except ImportError as e:
                raise ImportError(
                    "Must have tqdm installed if `show_progress_bar` is set to True. "
                    "Please install with `pip install tqdm`."
                ) from e

        for batch in batches:
            batch_embeddings = self._embed_batch(batch, client, input_type="document")
            all_embeddings.extend(batch_embeddings)

        return self._add_embeddings_to_elements(elements, all_embeddings)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            query: Query string to embed.

        Returns:
            Embedding vector.
        """
        client = self.config.get_client()
        batch_embeddings = self._embed_batch([query], client, input_type="query")
        return batch_embeddings[0]

    def count_tokens(self, texts: List[str]) -> List[int]:
        """
        Count tokens for the given texts.

        Args:
            texts: List of texts to count tokens for.

        Returns:
            List of token counts for each text.
        """
        if not texts:
            return []

        client = self.config.get_client()
        token_lists = client.tokenize(texts, model=self.config.model_name)
        return [len(token_list) for token_list in token_lists]

    @staticmethod
    def _add_embeddings_to_elements(elements, embeddings) -> List[Element]:
        assert len(elements) == len(embeddings)
        elements_w_embedding = []
        for i, element in enumerate(elements):
            element.embeddings = embeddings[i]
            elements_w_embedding.append(element)
        return elements


@njit(cache=True)
def _fast_batch_indices(token_counts: np.ndarray, max_tokens_per_batch: int) -> np.ndarray:
    """
    Compute start indices for each batch of texts using token limits, with Numba JIT.

    Args:
        token_counts: 1-d np.ndarray with the token count for each text.
        max_tokens_per_batch: Maximum allowed tokens in a batch.

    Returns:
        Indices (np.ndarray) of "start" positions for each batch (including 0).
        The last index is always equal to len(token_counts).
    """
    n = token_counts.shape[0]
    if n == 0:
        return np.array([0], dtype=np.int64)

    starts = np.empty(n + 1, dtype=np.int64)
    s_idx = 0
    starts[s_idx] = 0
    s_idx += 1
    batch_tokens = token_counts[0]
    batch_size = 1
    for i in range(1, n):
        tc = token_counts[i]
        if (batch_size >= MAX_BATCH_SIZE) or (batch_tokens + tc > max_tokens_per_batch):
            starts[s_idx] = i
            s_idx += 1
            batch_tokens = tc
            batch_size = 1
        else:
            batch_tokens += tc
            batch_size += 1
    starts[s_idx] = n
    return starts[: s_idx + 1]
