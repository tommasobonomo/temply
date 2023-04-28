from dataclasses import asdict, dataclass, field


@dataclass
class TokenSpan:
    # Span that refers to tokens within a tokenized sentence
    start: int = -1
    end: int = -1


@dataclass
class CharSpan:
    # Span that refers to characters in a non-tokenized sentence
    start_char: int = -1
    end_char: int = -1


@dataclass
class TextSpan(CharSpan, TokenSpan):
    text_string: str = ""


@dataclass
class EventTrigger(TextSpan):
    trigger_id: str = ""
    trigger_type: str = ""


@dataclass
class EventArgument(TextSpan):
    argument_id: str = ""
    argument_role: str = ""


@dataclass
class Event:
    event_id: str
    trigger: EventTrigger
    arguments: list[EventArgument]
    tokens: list[str] = field(default_factory=list)


@dataclass
class Document:
    document_id: str
    events: list[Event]
    text: str

    def to_dict(self) -> dict:
        """Returns dict representation of document and its events"""
        return asdict(self)


@dataclass
class Seq2seqSample:
    id: int
    unique_id: str
    predicate_indices: list[int]
    predicate_type: str
    source_sequence: str
    target_sequence: str
    dataset_id: str
    compositional_tokens: list[str]
    tokens: list[str]

    def to_dict(self) -> dict:
        """Returns dict representation of sequence2sequence sample"""
        return asdict(self)
