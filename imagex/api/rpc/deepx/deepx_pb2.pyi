from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class InferRequest(_message.Message):
    __slots__ = ["camera_id", "capture_config", "capture_index", "image_index", "mask_index", "timestamp"]
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    CAPTURE_CONFIG_FIELD_NUMBER: _ClassVar[int]
    CAPTURE_INDEX_FIELD_NUMBER: _ClassVar[int]
    IMAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    MASK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    camera_id: str
    capture_config: str
    capture_index: int
    image_index: int
    mask_index: int
    timestamp: int
    def __init__(self, image_index: _Optional[int] = ..., mask_index: _Optional[int] = ..., timestamp: _Optional[int] = ..., camera_id: _Optional[str] = ..., capture_config: _Optional[str] = ..., capture_index: _Optional[int] = ...) -> None: ...

class PingReply(_message.Message):
    __slots__ = ["ok"]
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: bool = ...) -> None: ...

class SuccessReply(_message.Message):
    __slots__ = ["ok"]
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: bool = ...) -> None: ...
