# KerasAug Benchmark

Updated: 2023/07/03

## Installation

- CUDA 11.8.0
- CUDNN 8.6.0
- Python 3.8.10
- Tensorflow 2.12.0
- KerasCV 0.5.0 (f05494c1057c95cbf44abac3238afcf262a50431)
- KerasAug 0.5.6

```bash
pip install tensorflow==2.12.0 keras-aug==0.5.6 git+https://github.com/keras-team/keras-cv.git tensorflow-datasets tqdm
```

## Usage

```bash
cd benchmarks

# run with GPU (KerasAug vs. KerasCV)
TF_CPP_MIN_LOG_LEVEL=2 python run_gpu_benchmark.py
```

## Setup

- Intel i7-7700K
- NVIDIA GTX 1080 8GB
- Unit: FPS (frames per second)
- Metric: The mean of the fastest 80% of the 20 trials
- Image size: (640, 640, 3), float32
- Batch size: 128
- Graph mode (`@tf.function`)

## Results

KerasAug is generally faster than KerasCV.

| Type           | Layer                      | KerasAug | KerasCV   |        |
|----------------|----------------------------|----------|-----------|--------|
| Geometry       | RandomHFlip                | 2123     | 1956      | fair   |
|                | RandomVFlip                | 1871     | 1767      | fair   |
|                | RandomRotate               | 1703     | 1723      | fair   |
|                | RandomAffine               | 2578     | 2355      | fair   |
|                | RandomCropAndResize        | 2664     | 213       | +1150% |
|                | Resize (224, 224)          | 2480     | 222       | +1017% |
| Intensity      | RandomBrightness           | 3052     | 2768      | fair   |
|                | RandomContrast\*           | 3099     | 2976      | fair   |
|                | RandomBrightnessContrast\* | 2881     | 609       | +373%  |
|                | RandomColorJitter\*        | 2013     | 597       | +237%  |
|                | RandomGaussianBlur         | 2345     | 203       | +1055% |
|                | Invert                     | 2691     | X         |        |
|                | Grayscale                  | 2917     | 3116      | fair   |
|                | Equalize                   | 196      | 139       | +41%   |
|                | AutoContrast               | 3095     | 3025      | fair   |
|                | Posterize                  | 3033     | 2144      | fair   |
|                | Solarize                   | 3133     | 2972      | fair   |
|                | Sharpness                  | 2982     | 2833      | fair   |
| Regularization | RandomCutout               | 2994     | 2795      | fair   |
|                | RandomGridMask             | 918      | 196       | +368%  |
| Mix            | CutMix                     | 2967     | 2957      | fair   |
|                | MixUp                      | 1897     | 1861      | fair   |
| Auto           | AugMix                     | 79       | X (Error) |        |
|                | RandAugment                | 301      | 246       | +22%   |

\*: The implementation of contrast adjustment in KerasCV differs from that of KerasAug.
