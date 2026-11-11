Put your trained weights here:

  players_yolo26.pt        Ultralytics YOLO26 player detector
  ball_yolo26.pt           Ultralytics YOLO26 ball detector
  keypoints_resnet50.pth   ResNet50 court keypoints (fc -> 28 outputs, 224x224 input)

(or change the paths in config.yaml)

You can run the project before you have all three:

- No player / ball weights: the public COCO YOLO26 model (yolo26s.pt) is used automatically
  (classes "person" and "sports ball"). Players work well; the ball is found less often.

- No keypoint weights:
    a) Fixed camera: click the 4 court corners once
         python calibrate_court.py --source input_videos\match.mp4
         python main.py --source input_videos\match.mp4 --court runs\match_court.json
    b) Use a public ResNet50 court model with the same format, e.g. the "tennis court key point
       model" linked in the README of github.com/abdullahtarek/tennis_analysis. Save it here as
       keypoints_resnet50.pth; the default config settings (224, pixels, ImageNet) match it.

- Train your own: see training/README.md
