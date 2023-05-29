# KerasAug Benchmark

Updated: 2023/05/16

## Installation

- CUDA 11.8.0
- CUDNN 8.6.0
- Python 3.8.10
- Tensorflow 2.12.0
- Tensorflow Probability 0.20.0
- KerasCV 0.5.0
- KerasAug 0.5.1

```bash
pip install keras-aug==0.5.1 keras-cv==0.5.0 tensorflow==2.12.0 --upgrade
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

| Type           | Layer                     | KerasAug | KerasCV   |      |
|----------------|---------------------------|----------|-----------|------|
| Geometry       | RandomHFlip               | 2148     | 1859      |+15%  |
|                | RandomVFlip               | 2182     | 2075      |+5%   |
|                | RandomRotate              | 2451     | 1829      |+34%  |
|                | RandomAffine              | 2141     | 1240      |+73%  |
|                | RandomCropAndResize       | 3014     | 209       |+1342%|
|                | Resize (224, 224)         | 2853     | 213       |+1239%|
| Intensity      | RandomBrightness          | 3028     | 3097      |close |
|                | RandomContrast\*          | 2806     | 2645      |+6%   |
|                | RandomBrightnessContrast\*| 3068     | 612       |+401% |
|                | RandomColorJitter\*       | 1932     | 1221      |+58%  |
|                | RandomGaussianBlur        | 2758     | 207       |+1232%|
|                | Invert                    | 2992     | X         |X     |
|                | Grayscale                 | 2841     | 2872      |close |
|                | Equalize                  | 206      | 139       |+48%  |
|                | AutoContrast              | 3116     | 2991      |+4%   |
|                | Posterize                 | 2917     | 2445      |+19%  |
|                | Solarize                  | 3025     | 2882      |+5%   |
|                | Sharpness                 | 2969     | 2915      |close |
| Regularization | RandomCutout              | 3222     | 3268      |close |
|                | RandomGridMask            | 947      | 197       |+381% |
| Mix            | CutMix                    | 2671     | 2445      |+9%   |
|                | MixUp                     | 2593     | 1996      |+29%  |
| Auto           | AugMix                    | 83       | X (Error) |X     |
|                | RandAugment               | 282      | 249       |+13%  |

\*: The implementation of contrast adjustment in KerasCV differs from that of KerasAug.
