# Training the three models

Use a GPU (your own, or free Google Colab: Runtime → Change runtime type → GPU).
On a CPU, training takes hours instead of minutes.

## 1. Ball detector (YOLO26)

Dataset: any tennis-ball dataset in YOLO format, e.g. Roboflow Universe "tennis-ball-detection"
(viren-dhanwani, version 6, CC BY 4.0, 578 images). On Roboflow pick *Download dataset → YOLOv8* format.
More frames from your own footage, labelled the same way, give the biggest improvement.

```bash
python training/train_yolo.py --data datasets/tennis-ball-detection-6/data.yaml \
    --out weights/ball_yolo26.pt --model yolo26s.pt --imgsz 1280 --epochs 100
```

## 2. Player detector (YOLO26)

The public COCO model already finds people well, so this is optional. Fine-tune if you need it to
separate players from ball kids / umpire, or for unusual camera angles. Any "tennis player" YOLO
dataset works (Roboflow Universe has several), or label ~300 frames of your own footage.

```bash
python training/train_yolo.py --data datasets/tennis-players/data.yaml \
    --out weights/players_yolo26.pt --model yolo26s.pt --imgsz 640 --epochs 60
```

## 3. Court keypoints (ResNet50, 14 points)

Dataset: TennisCourtDetector (github.com/yastrebksv/TennisCourtDetector): 8,841 broadcast frames,
1280×720, 14 points each. Download it from the link in that repo's README and unzip it so you have:

```
data/images/*.png
data/data_train.json
data/data_val.json
```

```bash
python training/train_keypoints.py --data data --epochs 40 --batch 32
```

The script saves the best epoch to `weights/keypoints_resnet50.pth` in exactly the format `config.yaml`
expects (input 224, output pixels, ImageNet normalisation). It mirrors images with the correct
left/right point swap and varies court colours so the model generalises across tournaments.
Validation prints the mean keypoint error in original-image pixels; under ~5 px is good.

## Colab in four cells

```python
!git clone <your repo or upload the zip> && cd tennis_analysis && pip install -r requirements.txt
!python training/train_yolo.py --data /content/ball/data.yaml --out weights/ball_yolo26.pt --imgsz 1280
!python training/train_keypoints.py --data /content/data --epochs 40 --batch 32
from google.colab import files; files.download('weights/ball_yolo26.pt'); files.download('weights/keypoints_resnet50.pth')
```
