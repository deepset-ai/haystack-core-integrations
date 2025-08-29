import asyncio
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component
from tqdm import tqdm

from ollama import AsyncClient, Client


@component
class OllamaDocumentEmbedder:
    """
    Computes the embeddings of a list of Documents and stores the obtained vectors in the embedding field of each
    Document. It uses embedding models compatible with the Ollama Library.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

    doc = Document(content="What do llamas say once you have thanked them? No probllama!")
    document_embedder = OllamaDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)
    ```
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        url: str = "http://localhost:11434",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        keep_alive: Optional[Union[float, str]] = None,
        prefix: str = "",
        suffix: str = "",
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        batch_size: int = 32,
    ):
        """
        :param model:
            The name of the model to use. The model should be available in the running Ollama instance.
        :param url:
            The URL of a running Ollama instance.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature, top_p, and others.
            See the available arguments in
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param timeout:
            The number of seconds before throwing a timeout error from the Ollama API.
        :param keep_alive:
            The option that controls how long the model will stay loaded into memory following the request.
            If not set, it will use the default value from the Ollama (5 minutes).
            The value can be set to:
            - a duration string (such as "10m" or "24h")
            - a number in seconds (such as 3600)
            - any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
            - '0' which will unload the model immediately after generating a response.
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        :param progress_bar:
            If `True`, shows a progress bar when running.
        :param meta_fields_to_embed:
            List of metadata fields to embed along with the document text.
        :param embedding_separator:
            Separator used to concatenate the metadata fields to the document text.
        :param batch_size:
            Number of documents to process at once.
        """
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.generation_kwargs = generation_kwargs or {}
        self.url = url
        self.model = model
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed
        self.embedding_separator = embedding_separator
        self.suffix = suffix
        self.prefix = prefix

        self._client = Client(host=self.url, timeout=self.timeout)
        self._async_client = AsyncClient(host=self.url, timeout=self.timeout)

    def _prepare_input(self, documents: List[Document]) -> List[Document]:
        """
        Prepares the list of documents to embed by appropriate validation.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "OllamaDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a list of strings, please use the OllamaTextEmbedder."
            )
            raise TypeError(msg)

        return documents

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepares the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            if self.meta_fields_to_embed is not None:
                meta_values_to_embed = [
                    str(doc.meta[key])
                    for key in self.meta_fields_to_embed
                    if key in doc.meta and doc.meta[key] is not None
                ]
            else:
                meta_values_to_embed = []

            text_to_embed = (
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            ).replace("\n", " ")

            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _embed_batch(
        self, texts_to_embed: List[str], batch_size: int, generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[List[float]]:
        """
        Internal method to embed a batch of texts.
        """

        all_embeddings = []

        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]
            result = self._client.embed(
                model=self.model,
                input=batch,
                options=generation_kwargs,
                keep_alive=self.keep_alive,
            )
            all_embeddings.extend(result["embeddings"])

        return all_embeddings

    async def _embed_batch_async(
        self, texts_to_embed: List[str], batch_size: int, generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[List[float]]:
        """
        Internal method to embed a batch of texts asynchronously.
        """
        all_embeddings = []

        batches = [texts_to_embed[i : i + batch_size] for i in range(0, len(texts_to_embed), batch_size)]

        tasks = [
            self._async_client.embed(
                model=self.model,
                input=batch,
                options=generation_kwargs,
                keep_alive=self.keep_alive,
            )
            for batch in batches
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, res in enumerate(results):
            if isinstance(res, BaseException):
                err_msg = f"Embedding batch {idx} raised an exception."
                raise RuntimeError(err_msg)
            all_embeddings.extend(res["embeddings"])

        return all_embeddings

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(
        self, documents: List[Document], generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[List[Document], Dict[str, Any]]]:
        """
        Runs an Ollama Model to compute embeddings of the provided documents.

        :param documents:
            Documents to be converted to an embedding.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, etc. See the
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :returns: A dictionary with the following keys:
            - `documents`: Documents with embedding information attached
            - `meta`: The metadata collected during the embedding process
        """
        documents = self._prepare_input(documents=documents)

        if not documents:
            # return early if we were passed an empty list
            return {"documents": [], "meta": {}}

        generation_kwargs = generation_kwargs or self.generation_kwargs

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)
        embeddings = self._embed_batch(
            texts_to_embed=texts_to_embed, batch_size=self.batch_size, generation_kwargs=generation_kwargs
        )

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": {"model": self.model}}

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    async def run_async(
        self, documents: List[Document], generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[List[Document], Dict[str, Any]]]:
        """
        Asynchronously run an Ollama Model to compute embeddings of the provided documents.

        :param documents:
            Documents to be converted to an embedding.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, etc. See the
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :returns: A dictionary with the following keys:
            - `documents`: Documents with embedding information attached
            - `meta`: The metadata collected during the embedding process
        """

        documents = self._prepare_input(documents=documents)

        if not documents:
            # return early if we were passed an empty list
            return {"documents": [], "meta": {}}

        generation_kwargs = generation_kwargs or self.generation_kwargs

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)
        embeddings = await self._embed_batch_async(
            texts_to_embed=texts_to_embed, batch_size=self.batch_size, generation_kwargs=generation_kwargs
        )

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": {"model": self.model}}
