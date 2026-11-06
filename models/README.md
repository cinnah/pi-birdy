# Models

`mobilenet_v2_1.0_224_quant.tflite` and `imagenet_labels.txt` are from the
TensorFlow hosted models (Apache 2.0, see `LICENSE`). Run `./download.sh` to
fetch them. They are committed so the Pi needs no network at install time.

The model has no softmax layer: its output dequantizes to logits, and the
classifier applies softmax itself.

`imagenet_bird_classes.txt` lists which output indices are birds; the
classifier sums their probabilities to get P(bird).

`mobilenet_v2_1.0_224_inat_bird_quant.tflite` + `inat_bird_labels.txt` are the
Coral iNaturalist bird classifier (965 classes incl. `background`, MobileNetV2
uint8, output already softmax-scaled 1/256) from
github.com/google-coral/test_data, Apache-2.0. Used for species recognition.

`efficientdet_lite0_320_ptq.tflite` + `coco_labels.txt` are the Coral COCO object
detector (EfficientDet-Lite0, uint8 320×320, outputs post-processed: boxes as
`ymin,xmin,ymax,xmax` fractions, classes, scores, count) from
github.com/google-coral/test_data, Apache-2.0. Used as the second bird/not-bird
signal: a `bird` detection (label line 16, class id 15 — line index = id) at or
above `detector.threshold` makes a photo a bird even when the ImageNet gate says no.
