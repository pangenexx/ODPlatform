import cv2
import numpy as np
import requests
from pathlib import Path
from typing import Iterator, Optional, List, Union
from io import BytesIO


class ImageSource:
    def __init__(self, source: Union[str, Path, int]):
        self.source = source
        self.cap = None
        self.image_list = []
        self.current_index = 0
        
        self._init_source()
    
    def _init_source(self):
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开摄像头: {self.source}")
        
        elif isinstance(self.source, str) and (self.source.startswith('http://') or self.source.startswith('https://')):
            self.image_list = [self.source]
        
        elif Path(self.source).exists() and Path(self.source).is_file():
            path = Path(self.source)
            ext = path.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                self.image_list = [str(path)]
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                self.cap = cv2.VideoCapture(str(path))
                if not self.cap.isOpened():
                    raise ValueError(f"无法打开视频文件: {self.source}")
            else:
                raise ValueError(f"不支持的文件格式: {ext}")
        
        elif Path(self.source).exists() and Path(self.source).is_dir():
            path = Path(self.source)
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            self.image_list = sorted([str(p) for p in path.glob('*') if p.suffix.lower() in image_extensions])
            if not self.image_list:
                raise ValueError(f"目录中没有找到图片文件: {self.source}")
        
        else:
            raise ValueError(f"文件或目录不存在: {self.source}\n当前工作目录: {Path.cwd()}")
    
    def __iter__(self) -> Iterator[np.ndarray]:
        return self
    
    def __next__(self) -> np.ndarray:
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.current_index < len(self.image_list):
            image_path = self.image_list[self.current_index]
            self.current_index += 1
            
            if image_path.startswith('http://') or image_path.startswith('https://'):
                response = requests.get(image_path)
                response.raise_for_status()
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        raise StopIteration
    
    def get_total_frames(self) -> Optional[int]:
        if self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return len(self.image_list)
    
    def get_fps(self) -> Optional[float]:
        if self.cap is not None:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return None
    
    def release(self):
        if self.cap is not None:
            self.cap.release()


class VideoWriter:
    def __init__(self, output_path: str, fps: float = 30.0, codec: str = 'mp4v'):
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.width = 0
        self.height = 0
    
    def write(self, frame: np.ndarray):
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
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def list_images(directory: Union[str, Path]) -> List[str]:
    path = Path(directory)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return sorted([str(p) for p in path.glob('*') if p.suffix.lower() in image_extensions])
