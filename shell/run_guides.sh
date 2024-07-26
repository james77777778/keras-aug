export KERAS_BACKEND=tensorflow
export TF_CPP_MIN_LOG_LEVEL=3
python3 -m guides.voc_yolov8_aug && echo "Finished guides.voc_yolov8_aug"
python3 -m guides.oxford_yolov8_aug && echo "Finished guides.oxford_yolov8_aug"
rm output_* && echo "All passed!"
