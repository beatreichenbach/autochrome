# Autochrome

https://en.wikipedia.org/wiki/Autochrome_Lumi%C3%A8re

## Links

[Advanced Emulsion](https://www.youtube.com/watch?v=I4_7tW-cx1I)

[Realistic Film Grain Rendering](https://www.ipol.im/pub/art/2017/192/article_lr.pdf)

[YouTube: Film Grain Rendering](https://cg.ivd.kit.edu/publications/2015/spectrum/paper-preprint.pdf)

[film_grain_rendering_gpu](https://github.com/alasdairnewson/film_grain_rendering_gpu)


## Spectral Reconstruction

### Textures (Reflective Spectra)

[Rgb to Spectrum Conversion](http://sv-journal.org/2015-4/03/en/index.php?lang=en)

[Physically Meaningful Rendering using Tristimulus Colours](https://cg.ivd.kit.edu/publications/2015/spectrum/paper-preprint.pdf)

[A Low-Dimensional Function Space for Efficient Spectral Upsampling](https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Jakob2019Spectral_3.pdf)

### Image

https://github.com/caiyuanhao1998/MST-plus-plus
https://github.com/caiyuanhao1998/BiSCI
https://github.com/hustvl/SAUNet
https://github.com/Deep-imagelab/AWAN
https://github.com/dsabarinathan/LightWeightModel
https://github.com/divyakraman/Single-Image-Unsupervised-Hyperspectral-Reconstruction


https://github.com/AmusementClub/vs-fgrain-cuda
https://github.com/alasdairnewson/film_grain_rendering_gpu

```python
import numpy as np
mu = 0.1
sigma = 0.0
u = 0.5

cell_size = 1 / np.ceil(1 / mu)
area = np.pi * ((mu * mu) + (sigma * sigma))
lambda_u = -((cell_size * cell_size) / area) * np.log(1 - u)

print(lambda_u)
print(np.exp(-lambda_u))
```

https://github.com/caiyuanhao1998/MST-plus-plus

https://github.com/boazarad/NTIRE2022_spectral


https://imaging.kodakalaris.com/sites/default/files/files/products/e4040_portra_800.pdf
https://github.com/alasdairnewson/film_grain_rendering_gpu/blob/5695ff2abea762c90740a0ae309336dc9be4cd95/src/film_grain_rendering.cu
https://github.com/mitsuba-renderer/rgb2spec
https://github.com/nbvdkamp/rgb2spec-rs


https://www.photrio.com/forum/threads/an-alternative-to-negative-lab-pro-and-lr-has-to-exist-c-41-reversal-and-orange-mask-removal.166696/post-2170267