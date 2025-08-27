import argparse, os, cv2
from opencv_face_detector_pro import FaceDetector, draw_detections

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--backend", default="haar", choices=["haar","dnn"])
    p.add_argument("--prototxt", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--save", default="outputs/face_result.jpg")
    args = p.parse_args()

    img = cv2.imread(args.image)
    det = FaceDetector(backend=args.backend, dnn_prototxt=args.prototxt, dnn_model=args.model)
    boxes = det.detect(img)
    vis = draw_detections(img, boxes)
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(args.save, vis)
    print(f"Saved: {args.save} ({len(boxes)} faces)")

if __name__ == "__main__":
    main()
