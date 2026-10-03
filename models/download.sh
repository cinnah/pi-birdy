#!/usr/bin/env bash
# Downloads the MobileNetV2 ImageNet TFLite model and labels into this directory.
set -euo pipefail
cd "$(dirname "$0")"
tmp="$(mktemp -d)"
curl -sSL -o "$tmp/model.tgz" \
  https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
tar -xzf "$tmp/model.tgz" -C "$tmp"
cp "$tmp/mobilenet_v2_1.0_224_quant.tflite" .
curl -sSL -o imagenet_labels.txt \
  https://raw.githubusercontent.com/tensorflow/tensorflow/v2.15.0/tensorflow/lite/java/demo/app/src/main/assets/labels_mobilenet_quant_v1_224.txt
rm -rf "$tmp"
echo "model: $(du -h mobilenet_v2_1.0_224_quant.tflite | cut -f1), labels: $(wc -l < imagenet_labels.txt) lines"
