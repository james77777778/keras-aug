# KerasAug Benchmark

Updated: 2023/05/09

## Installation

- CUDA 11.8.0
- CUDNN 8.6.0
- Python 3.8.10
- Tensorflow 2.12.0
- Tensorflow Probability 0.19.0
- KerasCV nightly
- KerasAug nightly
- torch 2.0.0
- torchvision 0.15.1

```bash
pip install git+https://github.com/keras-team/keras-cv.git tensorflow==2.12.0 tensorflow_probability==0.19.0 --upgrade
pip install git+https://github.com/james77777778/keras-aug.git
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

## Usage

```bash
cd benchmarks

# run with GPU (KerasAug vs. KerasCV)
TF_CPP_MIN_LOG_LEVEL=2 python run_gpu_benchmark.py

# run with CPU only (KerasAug vs. KerasCV vs. torchvision)
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 python run_cpu_benchmark.py
```

## Setup

- Intel i7-7700K
- NVIDIA GTX 1080 8GB
- Unit: FPS (frames per second)
- Metric: The median of the fastest 80% of the 20 trials
- Image size: (640, 640, 3), float32
- Batch size: 128
- Graph mode (`@tf.function`)

## Results

### GPU Result

KerasAug is generally faster than KerasCV.

| Type           | Layer                     | KerasAug | KerasCV   |
|----------------|---------------------------|----------|-----------|
| Geometry       | RandomHFlip               | 2325     | 1769      |
|                | RandomVFlip               | 2012     | 1923      |
|                | RandomRotate              | 1896     | 1782      |
|                | RandomAffine              | 1901     | 818       |
|                | RandomCropAndResize       | 2480     | 210       |
|                | Resize (224, 224)         | 2550     | 213       |
| Intensity      | RandomBrightness          | 3054     | 2925      |
|                | RandomContrast\*          | 2941     | 3086      |
|                | RandomBrighnessContrast\* | 3009     | 629       |
|                | RandomColorJitter\*       | 2201     | 1120      |
|                | RandomGaussianBlur        | 2632     | 196       |
|                | Invert                    | 2933     | X         |
|                | Grayscale                 | 3072     | 2762      |
|                | Equalize                  | 204      | 140       |
|                | AutoContrast              | 2873     | 2744      |
|                | Posterize                 | 3081     | 2929      |
|                | Solarize                  | 2828     | 2560      |
|                | Sharpness                 | 2554     | 2560      |
| Regularization | RandomCutout              | 2995     | 2978      |
|                | RandomGridMask            | 904      | 202       |
| Mix            | CutMix                    | 2352     | 2780      |
|                | MixUp                     | 2596     | 2962      |
| Auto           | AugMix                    | 80       | X (Error) |
|                | RandAugment               | 283      | 253       |

\*: The implementation of contrast adjustment in KerasCV differs from that of KerasAug.

### CPU Result

I'm not sure why, but when run on CPU, TensorFlow may be slower than torchvision.

| Type      |                           | KerasAug | KerasCV | torchvision |
|-----------|---------------------------|----------|---------|-------------|
| Geometry  | RandomHFlip               | 301      | 254     | 2376        |
|           | RandomVFlip               | 334      | 275     | 2339        |
|           | RandomRotate              | 178      | 160     | 93          |
|           | RandomAffine              | 181      | 50      | 109         |
|           | RandomCropAndResize       | 849      | 534     | 1730        |
|           | Resize (224, 224)         | 764      | 539     | 2182        |
| Intensity | RandomBrightness          | 635      | 640     | 894         |
|           | RandomContrast\*          | 268      | 327     | 649         |
|           | RandomBrighnessContrast\* | 248      | 40      | 475         |
|           | RandomColorJitter\*       | 59       | 40      | 44          |
|           | RandomGaussianBlur        | 121      | 53      | 55          |
|           | Grayscale                 | 338      | 338     | 2983        |
|           | Equalize                  | 124      | 147     | 226         |
|           | AutoContrast              | 316      | 188     | 705         |
|           | Posterize                 | 308      | 460     | 899         |
|           | Sharpness                 | 74       | 74      | 167         |

\*: The implementation of contrast adjustment in KerasCV differs from that of KerasAug and torchvision.
