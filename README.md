# DipWizards Package

The `DipWizards` package is developed by the AIWizards team. It provides a collection of functions and classes for generating noise and applying filters to grayscale and color images. With this package, you can apply various noise types and spatial/frequency filters to your images.

## Features

- **Noise Functions**: The package includes functions for generating different types of noise, such as Gaussian noise, Poisson noise, and salt-and-pepper noise. These functions allow you to create images with various random noise patterns.

- **Filter Functions**: The package includes functions for applying spatial and frequency filters to images. You can use these functions to apply filters like mean filter, median filter, Gaussian filter, sharpening filter, and more.

## Installation

To install the `DipWizards` package, use the following command:
pip install dipwizards

## Usage Example

Below is an example demonstrating how you can use the functions and classes of the `DipWizards` package to generate noise and apply different filters to images:

```python
import Core.utils as utl

# Create Gaussian noise
noise_generator = utl.ImageNoiseRGB(image)
noisy_image = noise_generator..add_noise('gauss', var=0.4)

# Apply mean filter
filter = utl.SpatialFilterRGB(img_gauss)
filtered_image = filter..apply_filter('mean',   3, 0)
```
## Summary

The DipWizards package is a powerful collection of functions and classes for generating noise and applying filters to images. With this package, you can enhance your images and adjust them with various transformations.

## License

This project is licensed under the MIT License.
