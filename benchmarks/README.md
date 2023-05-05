# KerasAug Benchmark

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
pip install keras-aug
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
```

## Usage

```bash
cd benchmarks

# run with GPU (KerasAug vs. KerasCV)
TF_CPP_MIN_LOG_LEVEL=2 python run_gpu_benchmark.py

# run with CPU only (KerasAug vs. KerasCV vs. torchvision)
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1 python run_cpu_benchmark.py
```

## Results

- Intel i7-7700K
- NVIDIA GTX 1080 8GB
- Unit: FPS (frame per second)
- Metric: Median of 20 trials
- Image size: (640, 640, 3), float32
- Batch size: 128

### GPU Result

Overall, KerasAug runs faster than KerasCV.

||KerasAug|KerasCV|
|-|-|-|
|HorizontalFlip|2325|1769|
|VerticalFlip|2012|1923|
|RandomRotate|1896|1782|
|RandomAffine|1901|818|
|RandomBrightness|3054|2925|
|RandomContrast\*|2941|3086|
|RandomBrighnessContrast\*|3009|629|
|RandomColorJitter\*|2201|1120|
|Grayscale|3072|2762|
|Resize (224, 224)|2593|210|
|RandomCropAndResize|2480|210|
|Equalize|204|140|
|RandomGaussianBlur|2632|196|
|CutMix|2352|2780|
|MixUp|2596|2962|
|AutoContrast|2873|2744|
|Posterize|3081|2929|
|Solarize|2828|2560|
|Sharpness|2554|2560|
|Invert|2933|X|

### CPU Result

Tensorflow might be slower than torchvision when run on CPU. (don't know why...)

||KerasAug|KerasCV|torchvision|
|-|-|-|-|
|HorizontalFlip|301|254|2376|
|VerticalFlip|334|275|2339|
|RandomRotate|178|160|93|
|RandomAffine|181|50|109|
|RandomBrightness|635|640|894|
|RandomContrast\*|268|327|649|
|RandomBrighnessContrast\*|248|40|475|
|RandomColorJitter\*|59|40|44|
|Grayscale|338|338|2983|
|Resize (224, 224)|764|539|2182|
|RandomCropAndResize|849|534|1730|
|Equalize|124|147|226|
|RandomGaussianBlur|121|53|55|
|CutMix|358|359|X|
|MixUp|359|352|X|
|AutoContrast|316|188|705|
|Posterize|308|460|899|
|Solarize|284|285|X|
|Sharpness|74|74|167|
|Invert|870|74|2024|

\*: The contrast adjustment implementation in KerasCV is different from KerasAug & torchvision
