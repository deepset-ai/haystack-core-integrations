from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_TYPE_EMBEDDING: _ClassVar[ModelType]
    MODEL_TYPE_CLASSIFIER: _ClassVar[ModelType]
    MODEL_TYPE_RERANKER: _ClassVar[ModelType]

class TruncationDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRUNCATION_DIRECTION_RIGHT: _ClassVar[TruncationDirection]
    TRUNCATION_DIRECTION_LEFT: _ClassVar[TruncationDirection]
MODEL_TYPE_EMBEDDING: ModelType
MODEL_TYPE_CLASSIFIER: ModelType
MODEL_TYPE_RERANKER: ModelType
TRUNCATION_DIRECTION_RIGHT: TruncationDirection
TRUNCATION_DIRECTION_LEFT: TruncationDirection

class InfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class InfoResponse(_message.Message):
    __slots__ = ("version", "sha", "docker_label", "model_id", "model_sha", "model_dtype", "model_type", "max_concurrent_requests", "max_input_length", "max_batch_tokens", "max_batch_requests", "max_client_batch_size", "tokenization_workers")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    SHA_FIELD_NUMBER: _ClassVar[int]
    DOCKER_LABEL_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_SHA_FIELD_NUMBER: _ClassVar[int]
    MODEL_DTYPE_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENT_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    MAX_INPUT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    MAX_CLIENT_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    TOKENIZATION_WORKERS_FIELD_NUMBER: _ClassVar[int]
    version: str
    sha: str
    docker_label: str
    model_id: str
    model_sha: str
    model_dtype: str
    model_type: ModelType
    max_concurrent_requests: int
    max_input_length: int
    max_batch_tokens: int
    max_batch_requests: int
    max_client_batch_size: int
    tokenization_workers: int
    def __init__(self, version: _Optional[str] = ..., sha: _Optional[str] = ..., docker_label: _Optional[str] = ..., model_id: _Optional[str] = ..., model_sha: _Optional[str] = ..., model_dtype: _Optional[str] = ..., model_type: _Optional[_Union[ModelType, str]] = ..., max_concurrent_requests: _Optional[int] = ..., max_input_length: _Optional[int] = ..., max_batch_tokens: _Optional[int] = ..., max_batch_requests: _Optional[int] = ..., max_client_batch_size: _Optional[int] = ..., tokenization_workers: _Optional[int] = ...) -> None: ...

class Metadata(_message.Message):
    __slots__ = ("compute_chars", "compute_tokens", "total_time_ns", "tokenization_time_ns", "queue_time_ns", "inference_time_ns")
    COMPUTE_CHARS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    TOKENIZATION_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_TIME_NS_FIELD_NUMBER: _ClassVar[int]
    compute_chars: int
    compute_tokens: int
    total_time_ns: int
    tokenization_time_ns: int
    queue_time_ns: int
    inference_time_ns: int
    def __init__(self, compute_chars: _Optional[int] = ..., compute_tokens: _Optional[int] = ..., total_time_ns: _Optional[int] = ..., tokenization_time_ns: _Optional[int] = ..., queue_time_ns: _Optional[int] = ..., inference_time_ns: _Optional[int] = ...) -> None: ...

class EmbedRequest(_message.Message):
    __slots__ = ("inputs", "truncate", "normalize", "truncation_direction", "prompt_name", "dimensions")
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    TRUNCATE_FIELD_NUMBER: _ClassVar[int]
    NORMALIZE_FIELD_NUMBER: _ClassVar[int]
    TRUNCATION_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    PROMPT_NAME_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    inputs: str
    truncate: bool
    normalize: bool
    truncation_direction: TruncationDirection
    prompt_name: str
    dimensions: int
    def __init__(self, inputs: _Optional[str] = ..., truncate: _Optional[bool] = ..., normalize: _Optional[bool] = ..., truncation_direction: _Optional[_Union[TruncationDirection, str]] = ..., prompt_name: _Optional[str] = ..., dimensions: _Optional[int] = ...) -> None: ...

class EmbedResponse(_message.Message):
    __slots__ = ("embeddings", "metadata")
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    embeddings: _containers.RepeatedScalarFieldContainer[float]
    metadata: Metadata
    def __init__(self, embeddings: _Optional[_Iterable[float]] = ..., metadata: _Optional[_Union[Metadata, _Mapping]] = ...) -> None: ...

class EmbedSparseRequest(_message.Message):
    __slots__ = ("inputs", "truncate", "truncation_direction", "prompt_name")
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    TRUNCATE_FIELD_NUMBER: _ClassVar[int]
    TRUNCATION_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    PROMPT_NAME_FIELD_NUMBER: _ClassVar[int]
    inputs: str
    truncate: bool
    truncation_direction: TruncationDirection
    prompt_name: str
    def __init__(self, inputs: _Optional[str] = ..., truncate: _Optional[bool] = ..., truncation_direction: _Optional[_Union[TruncationDirection, str]] = ..., prompt_name: _Optional[str] = ...) -> None: ...

class SparseValue(_message.Message):
    __slots__ = ("index", "value")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    index: int
    value: float
    def __init__(self, index: _Optional[int] = ..., value: _Optional[float] = ...) -> None: ...

class EmbedSparseResponse(_message.Message):
    __slots__ = ("sparse_embeddings", "metadata")
    SPARSE_EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    sparse_embeddings: _containers.RepeatedCompositeFieldContainer[SparseValue]
    metadata: Metadata
    def __init__(self, sparse_embeddings: _Optional[_Iterable[_Union[SparseValue, _Mapping]]] = ..., metadata: _Optional[_Union[Metadata, _Mapping]] = ...) -> None: ...

class EmbedAllRequest(_message.Message):
    __slots__ = ("inputs", "truncate", "truncation_direction", "prompt_name")
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    TRUNCATE_FIELD_NUMBER: _ClassVar[int]
    TRUNCATION_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    PROMPT_NAME_FIELD_NUMBER: _ClassVar[int]
    inputs: str
    truncate: bool
    truncation_direction: TruncationDirection
    prompt_name: str
    def __init__(self, inputs: _Optional[str] = ..., truncate: _Optional[bool] = ..., truncation_direction: _Optional[_Union[TruncationDirection, str]] = ..., prompt_name: _Optional[str] = ...) -> None: ...

class TokenEmbedding(_message.Message):
    __slots__ = ("embeddings",)
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    embeddings: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, embeddings: _Optional[_Iterable[float]] = ...) -> None: ...

class EmbedAllResponse(_message.Message):
    __slots__ = ("token_embeddings", "metadata")
    TOKEN_EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    token_embeddings: _containers.RepeatedCompositeFieldContainer[TokenEmbedding]
    metadata: Metadata
    def __init__(self, token_embeddings: _Optional[_Iterable[_Union[TokenEmbedding, _Mapping]]] = ..., metadata: _Optional[_Union[Metadata, _Mapping]] = ...) -> None: ...

class PredictRequest(_message.Message):
    __slots__ = ("inputs", "truncate", "raw_scores", "truncation_direction")
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    TRUNCATE_FIELD_NUMBER: _ClassVar[int]
    RAW_SCORES_FIELD_NUMBER: _ClassVar[int]
    TRUNCATION_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    inputs: str
    truncate: bool
    raw_scores: bool
    truncation_direction: TruncationDirection
    def __init__(self, inputs: _Optional[str] = ..., truncate: _Optional[bool] = ..., raw_scores: _Optional[bool] = ..., truncation_direction: _Optional[_Union[TruncationDirection, str]] = ...) -> None: ...

class PredictPairRequest(_message.Message):
    __slots__ = ("inputs", "truncate", "raw_scores", "truncation_direction")
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    TRUNCATE_FIELD_NUMBER: _ClassVar[int]
    RAW_SCORES_FIELD_NUMBER: _ClassVar[int]
    TRUNCATION_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    inputs: _containers.RepeatedScalarFieldContainer[str]
    truncate: bool
    raw_scores: bool
    truncation_direction: TruncationDirection
    def __init__(self, inputs: _Optional[_Iterable[str]] = ..., truncate: _Optional[bool] = ..., raw_scores: _Optional[bool] = ..., truncation_direction: _Optional[_Union[TruncationDirection, str]] = ...) -> None: ...

class Prediction(_message.Message):
    __slots__ = ("score", "label")
    SCORE_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    score: float
    label: str
    def __init__(self, score: _Optional[float] = ..., label: _Optional[str] = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ("predictions", "metadata")
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    predictions: _containers.RepeatedCompositeFieldContainer[Prediction]
    metadata: Metadata
    def __init__(self, predictions: _Optional[_Iterable[_Union[Prediction, _Mapping]]] = ..., metadata: _Optional[_Union[Metadata, _Mapping]] = ...) -> None: ...

class RerankRequest(_message.Message):
    __slots__ = ("query", "texts", "truncate", "raw_scores", "return_text", "truncation_direction")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TEXTS_FIELD_NUMBER: _ClassVar[int]
    TRUNCATE_FIELD_NUMBER: _ClassVar[int]
    RAW_SCORES_FIELD_NUMBER: _ClassVar[int]
    RETURN_TEXT_FIELD_NUMBER: _ClassVar[int]
    TRUNCATION_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    query: str
    texts: _containers.RepeatedScalarFieldContainer[str]
    truncate: bool
    raw_scores: bool
    return_text: bool
    truncation_direction: TruncationDirection
    def __init__(self, query: _Optional[str] = ..., texts: _Optional[_Iterable[str]] = ..., truncate: _Optional[bool] = ..., raw_scores: _Optional[bool] = ..., return_text: _Optional[bool] = ..., truncation_direction: _Optional[_Union[TruncationDirection, str]] = ...) -> None: ...

class RerankStreamRequest(_message.Message):
    __slots__ = ("query", "text", "truncate", "raw_scores", "return_text", "truncation_direction")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TRUNCATE_FIELD_NUMBER: _ClassVar[int]
    RAW_SCORES_FIELD_NUMBER: _ClassVar[int]
    RETURN_TEXT_FIELD_NUMBER: _ClassVar[int]
    TRUNCATION_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    query: str
    text: str
    truncate: bool
    raw_scores: bool
    return_text: bool
    truncation_direction: TruncationDirection
    def __init__(self, query: _Optional[str] = ..., text: _Optional[str] = ..., truncate: _Optional[bool] = ..., raw_scores: _Optional[bool] = ..., return_text: _Optional[bool] = ..., truncation_direction: _Optional[_Union[TruncationDirection, str]] = ...) -> None: ...

class Rank(_message.Message):
    __slots__ = ("index", "text", "score")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    index: int
    text: str
    score: float
    def __init__(self, index: _Optional[int] = ..., text: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...

class RerankResponse(_message.Message):
    __slots__ = ("ranks", "metadata")
    RANKS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ranks: _containers.RepeatedCompositeFieldContainer[Rank]
    metadata: Metadata
    def __init__(self, ranks: _Optional[_Iterable[_Union[Rank, _Mapping]]] = ..., metadata: _Optional[_Union[Metadata, _Mapping]] = ...) -> None: ...

class EncodeRequest(_message.Message):
    __slots__ = ("inputs", "add_special_tokens", "prompt_name")
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    ADD_SPECIAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_NAME_FIELD_NUMBER: _ClassVar[int]
    inputs: str
    add_special_tokens: bool
    prompt_name: str
    def __init__(self, inputs: _Optional[str] = ..., add_special_tokens: _Optional[bool] = ..., prompt_name: _Optional[str] = ...) -> None: ...

class SimpleToken(_message.Message):
    __slots__ = ("id", "text", "special", "start", "stop")
    ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SPECIAL_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    id: int
    text: str
    special: bool
    start: int
    stop: int
    def __init__(self, id: _Optional[int] = ..., text: _Optional[str] = ..., special: _Optional[bool] = ..., start: _Optional[int] = ..., stop: _Optional[int] = ...) -> None: ...

class EncodeResponse(_message.Message):
    __slots__ = ("tokens",)
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    tokens: _containers.RepeatedCompositeFieldContainer[SimpleToken]
    def __init__(self, tokens: _Optional[_Iterable[_Union[SimpleToken, _Mapping]]] = ...) -> None: ...

class DecodeRequest(_message.Message):
    __slots__ = ("ids", "skip_special_tokens")
    IDS_FIELD_NUMBER: _ClassVar[int]
    SKIP_SPECIAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[int]
    skip_special_tokens: bool
    def __init__(self, ids: _Optional[_Iterable[int]] = ..., skip_special_tokens: _Optional[bool] = ...) -> None: ...

class DecodeResponse(_message.Message):
    __slots__ = ("text",)
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...
