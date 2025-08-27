from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np

@dataclass
class Detection:
    bbox: Tuple[int,int,int,int]
    score: float

@dataclass
class FaceDetector:
    backend: str = "haar"
    dnn_prototxt: Optional[str] = None
    dnn_model: Optional[str] = None
    dnn_conf_thresh: float = 0.5

    def _load_haar(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        return cv2.CascadeClassifier(cascade_path)

    def detect(self, image: np.ndarray) -> List[Detection]:
        if self.backend == "haar":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detector = self._load_haar()
            rects = detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (30,30))
            return [Detection((int(x),int(y),int(w),int(h)), 1.0) for (x,y,w,h) in rects]
        elif self.backend == "dnn":
            if not self.dnn_model or not self.dnn_prototxt:
                raise ValueError("Provide dnn_prototxt and dnn_model for backend='dnn'")
            net = cv2.dnn.readNetFromCaffe(self.dnn_prototxt, self.dnn_model)
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            out: List[Detection] = []
            for i in range(0, detections.shape[2]):
                conf = detections[0,0,i,2]
                if conf < self.dnn_conf_thresh:
                    continue
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (sx, sy, ex, ey) = box.astype("int")
                x,y = max(0,int(sx)), max(0,int(sy))
                ex,ey = min(w-1,int(ex)), min(h-1,int(ey))
                out.append(Detection((x,y,ex-x,ey-y), float(conf)))
            return out
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

def draw_detections(image: np.ndarray, dets: List[Detection]) -> np.ndarray:
    out = image.copy()
    for d in dets:
        x,y,w,h = d.bbox
        cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(out, f"{d.score:.2f}", (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return out
