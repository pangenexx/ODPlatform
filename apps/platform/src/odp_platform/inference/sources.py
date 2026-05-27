from pathlib import Path
from typing import Iterator, Optional, List, Union, Tuple
import numpy as np

from .frame_source import (
    ImageSource as FS_ImageSource,
    ImageFolderSource,
    VideoSource,
    CameraSource,
    create_frame_source,
    create_threaded_source,
    CameraConfig,
    SourceType,
    Frame,
    draw_hud,
)


class ImageSource:
    def __init__(self, source: Union[str, Path, int]):
        self.source = source
        self._inner = create_frame_source(str(source) if not isinstance(source, int) else source)
        self._inner.open()
        self._fps_timer = 0.0
        self._frame_count = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        frame = self._inner.read()
        if frame is None:
            raise StopIteration
        self._frame_count = frame.info.frame_index
        return frame.image

    def get_total_frames(self) -> Optional[int]:
        if isinstance(self._inner, (VideoSource,)):
            return self._inner._total_frames
        if isinstance(self._inner, ImageFolderSource):
            return len(self._inner._image_paths)
        return None

    def get_fps(self) -> Optional[float]:
        try:
            return self._inner._fps
        except AttributeError:
            pass
        if isinstance(self._inner, CameraSource):
            return self._inner._config.fps
        return None

    def release(self):
        self._inner.close()

    @property
    def cap(self):
        return getattr(self._inner, '_cap', None)

    @property
    def source_type(self) -> SourceType:
        return self._inner.get_source_type()


class VideoWriter:
    def __init__(self, output_path: str, fps: float = 30.0, codec: str = 'mp4v'):
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.width = 0
        self.height = 0

    def write(self, frame: np.ndarray):
        import cv2
        h, w = frame.shape[:2]
        if self.writer is None:
            self.width = w
            self.height = h
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr_frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def list_images(directory: Union[str, Path]) -> List[str]:
    path = Path(directory)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return sorted([str(p) for p in path.glob('*') if p.suffix.lower() in image_extensions])