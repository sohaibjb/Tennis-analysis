"""Video sources (file, webcam, RTSP/HTTP stream) and the output writer."""
from __future__ import annotations

import threading
import time

import cv2


def parse_source(src: str):
    return int(src) if str(src).isdigit() else src


def is_live(src) -> bool:
    return isinstance(src, int) or str(src).lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://"))


class VideoSource:
    """Iterates (frame, timestamp_s). For live sources timestamps come from the wall clock.

    realtime=True runs a reader thread that always hands out the newest frame, so a
    pipeline slower than the camera drops frames instead of falling further behind.
    """

    def __init__(self, src, realtime: bool = False, fps_override: float | None = None):
        self.src = parse_source(src)
        self.live = is_live(self.src)
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {src}")
        fps = fps_override or self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 < fps < 241 else 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total = n if n > 0 and not self.live else None
        self.realtime = realtime and self.live
        self._latest = None
        self._stop = False
        self._t0 = None
        if self.realtime:
            self._cond = threading.Condition()
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self):
        while not self._stop:
            ok, frame = self.cap.read()
            with self._cond:
                self._latest = (frame, time.monotonic()) if ok else None
                if not ok:
                    self._stop = True
                self._cond.notify_all()

    def __iter__(self):
        i = 0
        last_seen = None
        while True:
            if self.realtime:
                with self._cond:
                    self._cond.wait_for(lambda: self._stop or (self._latest is not None and self._latest is not last_seen))
                    if self._latest is None:
                        return
                    last_seen = self._latest
                    frame, ts = self._latest
            else:
                ok, frame = self.cap.read()
                if not ok:
                    return
                ts = time.monotonic() if self.live else i / self.fps
            if self._t0 is None:
                self._t0 = ts
            yield frame, (ts - self._t0) if self.live else ts
            i += 1

    def release(self):
        self._stop = True
        self.cap.release()


class VideoSink:
    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.w = cv2.VideoWriter(path, fourcc, fps, size)
        if not self.w.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")

    def write(self, frame):
        self.w.write(frame)

    def release(self):
        self.w.release()
