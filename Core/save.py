import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import requests
from io import BytesIO
import sys

def show_mult_img(rows, columns, img_names, vmin=0, vmax=255):
    fig = plt.figure(figsize=(15, 17), dpi=100)
    for i in range(len(img_names)):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img_names[i], cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title('img_' + str(i))

def display_hist_plt(img, bins=256, range=(0, 256)):
    plt.figure(figsize=(4, 2), dpi=100)
    plt.hist(img.flat, bins=bins, range=range)
    plt.show()

def show_img_plt(img, c_map ='gray', dpi=100, fig_hight=8, fig_width=6, vmin=0, vmax=255):
    plt.figure(figsize=(fig_hight, fig_width), dpi=dpi)
    plt.imshow(img, cmap=c_map, vmin=vmin, vmax=vmax)
    
def show_img_cv(img_title, img):
    cv2.imshow(img_title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_img(img_title, img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(img_title)
    plt.axis('off')
    plt.show()    
    
class ImageNoise:
    """
    Class to add different types of noise to an image. The types of noise that can be added are:
    - Gaussian
    - Salt and Pepper
    - Poisson
    - Speckle

    The image is passed in when creating an object of the class.

    Parameters
    ----------
    image : ndarray
        The input image.
    """
    def __init__(self, image):
        self.image = image

    def add_gauss_noise(self, **kwargs):
        """
        Add Gaussian noise to the image.

        Parameters
        ----------
        mean : float, optional
            Mean of the Gaussian distribution to generate noise (default is 0).
        var : float, optional
            Variance of the Gaussian distribution to generate noise (default is 0.1).
        """
        mean = kwargs.get('mean', 0)
        var = kwargs.get('var', 0.1)
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, self.image.shape)
        return self.image + self.image * gauss
    
    def add_sp_noise(self, **kwargs):
        """
        Add Salt & Pepper noise to the image.
    
        Parameters
        ----------
        s_vs_p : float, optional
            Ratio of salt to pepper (default is 0.5).
        amount : float, optional
            Overall proportion of image pixels to replace with noise (default is 0.004).
        """
        s_vs_p = kwargs.get('s_vs_p', 0.5)
        amount = kwargs.get('amount', 0.004)
        out = np.copy(self.image)
    
        # Salt mode
        num_salt = np.ceil(amount * self.image.size * s_vs_p).astype(int)
        coords = tuple(map(lambda dim: np.random.randint(0, dim, num_salt), self.image.shape))
        out[coords] = 255
    
        # Pepper mode
        num_pepper = np.ceil(amount * self.image.size * (1. - s_vs_p)).astype(int)
        coords = tuple(map(lambda dim: np.random.randint(0, dim, num_pepper), self.image.shape))
        out[coords] = 0
    
        return out

    def add_poisson_noise(self, **kwargs):
        """
        Add Poisson noise to the image.
    
        The noise is added as per a Poisson distribution. This function does not take any additional parameters.
        """
        # Convert the image to double data type
        image = self.image.astype(np.float64)
    
        # Scale the image to the range of 0-1
        image /= np.max(image)
    
        # Convert the image to represent counts in the range of 0-255
        image *= 255
    
        # Apply the Poisson noise
        noisy = np.random.poisson(image)
    
        # Normalize the noisy image
        noisy = noisy / np.max(noisy)
        
        noisy *= 255.
        noisy = noisy.astype(np.uint8) 
    
        return noisy

    def add_speckle_noise(self, **kwargs):
        """
        Add Speckle noise to the image.

        Speckle noise is a multiplicative noise. This function does not take any additional parameters.
        """
        gauss = np.random.randn(*self.image.shape)
        return self.image + self.image * gauss

    def add_noise(self, noise_typ, **kwargs):
        """
        Add noise to the image.

        Parameters
        ----------
        noise_typ : str
            Type of noise to add. Options are 'gauss', 's&p', 'poisson', or 'speckle'.
        **kwargs :
            Additional parameters for the noise functions. These depend on the type of noise.
        """
        match noise_typ:
            case "gauss":
                return self.add_gauss_noise(**kwargs)
            case "s&p":
                return self.add_sp_noise(**kwargs)
            case "poisson":
                return self.add_poisson_noise(**kwargs)
            case "speckle":
                return self.add_speckle_noise(**kwargs)
            case _:
                raise ValueError(f"Noise type '{noise_typ}' is not supported")
class ImageNoiseRGB:
    """
    Class to add different types of noise to an RGB image. The types of noise that can be added are:
    - Gaussian
    - Salt and Pepper
    - Poisson
    - Speckle

    The image is passed in when creating an object of the class.

    Parameters
    ----------
    image : ndarray
        The input image.
    """
    def __init__(self, image):
        self.image = image

    def add_gauss_noise(self, **kwargs):
        """
        Add Gaussian noise to the RGB image.

        Parameters
        ----------
        mean : float, optional
            Mean of the Gaussian distribution to generate noise (default is 0).
        var : float, optional
            Variance of the Gaussian distribution to generate noise (default is 0.1).
        """
        mean = kwargs.get('mean', 0)
        var = kwargs.get('var', 0.1)
        sigma = var**0.5
        x , y ,chal = self.image.shape
        # gauss = np.random.normal(mean, sigma, self.image.shape)
        gauss = np.random.normal(mean,sigma,self.image.size)
        gauss = gauss.reshape(x , y ,chal)
        gauss_resized = np.copy(gauss).astype('uint8')

        # Add the Gaussian noise to the image
        # result= np.clip(self.image + gauss_resized, 0, 255)
        return self.image + gauss_resized

    def add_sp_noise(self, **kwargs):
        """
        Add Salt & Pepper noise to the image.

        Parameters
        ----------
        s_vs_p : float, optional
            Ratio of salt to pepper (default is 0.5).
        amount : float, optional
            Overall proportion of image pixels to replace with noise (default is 0.004).
        """
        s_vs_p = kwargs.get('s_vs_p', 0.5)
        amount = kwargs.get('amount', 0.004)
        out = np.copy(self.image)

        # Salt mode
        num_salt = np.ceil(amount * self.image.size * s_vs_p).astype(int)
        coords = tuple(map(lambda dim: np.random.randint(0, dim, num_salt), self.image.shape))
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * self.image.size * (1. - s_vs_p)).astype(int)
        coords = tuple(map(lambda dim: np.random.randint(0, dim, num_pepper), self.image.shape))
        out[coords] = 0

        return out

    def add_poisson_noise(self, **kwargs):
        '''
        Add Poisson noise to the RGB image.
    
        The noise is added as per a Poisson distribution. This function does not take any additional parameters.
        '''
        # Convert the image to double data type
        image = self.image.astype(np.float64)
    
        # Scale the image to the range of 0-1
        image /= np.max(image)
    
        # Convert the image to represent counts in the range of 0-255
        image *= 255
    
        # Apply the Poisson noise
        noisy = np.random.poisson(image)
    
        # Normalize the noisy image
        noisy = noisy / np.max(noisy)
        
        noisy *= 255.
        noisy = noisy.astype(np.uint8) 
    
        return noisy

    def add_speckle_noise(self, **kwargs):
        """
        Add Speckle noise to the image.

        Speckle noise is a multiplicative noise. This function does not take any additional parameters.
        """
        gauss = np.random.randn(*self.image.shape)
        noisy = self.image + self.image * gauss
        return np.clip(noisy, 0, 1)

    def add_noise(self, noise_typ, **kwargs):
        """
        Add noise to theتصویر.

        Parameters
        ----------
        noise_typ : str
            Type of noise to add. Options are 'gauss', 's&p', 'poisson', or 'speckle'.
        **kwargs :
            Additional parameters for the noise functions. These depend on the type of noise.
        """
        if self.image.ndim == 2:
            self.image = np.expand_dims(self.image, axis=-1)
        if self.image.max() > 1:
            self.image = self.image.astype(np.float32) / 255.

        match noise_typ:
            case "gauss":
                return self.add_gauss_noise(**kwargs)
            case "s&p":
                return self.add_sp_noise(**kwargs)
            case "poisson":
                return self.add_poisson_noise(**kwargs)
            case "speckle":
                return self.add_speckle_noise(**kwargs)
            case _:
                raise ValueError(f"Noise type '{noise_typ}' is not supported")


class SpatialFilter:
    def __init__(self, image):
        self.image = image

    def mean_filter(self, filter_size, padding=0):
        padded_image = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        return cv2.blur(padded_image, (filter_size, filter_size))

    def median_filter(self, filter_size, padding=0):
        padded_image = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        return cv2.medianBlur(padded_image, filter_size)

    def adaptive_filter(self, filter_size, padding=0):
        padded_image = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        return cv2.adaptiveThreshold(padded_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, filter_size, 2)
    
    def apply_filter(self, filter_name, filter_size, padding=0):
        
        match filter_name:
            case 'mean':
                return self.mean_filter(filter_size, padding)
            case 'median':
                return self.median_filter(filter_size, padding)
            case 'adaptive':
                return self.adaptive_filter(filter_size, padding)
            case _:
                raise ValueError(f"Filter '{filter_name}' is not supported")

# class SpatialFilterRGB:
#     def __init__(self, image):
#         self.image = image

#     def mean_filter(self, filter_size, padding=0):
#         padded_image = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
#         filtered_image = cv2.blur(padded_image, (filter_size, filter_size))
#         return filtered_image

#     def median_filter(self, filter_size, padding=0):
#         padded_image = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
#         filtered_image = cv2.medianBlur(padded_image, filter_size)
#         return filtered_image


#     def adaptive_filter(self, filter_size, padding=0,adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType = cv2.THRESH_BINARY,my_constant = 0):
        
#         # adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C   # Fixed
#         # thresholdType = cv2.THRESH_BINARY
#         # my_constant = 2
#         # adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
#         # thresholdType = cv2.THRESH_BINARY
        
        
#         B, G, R = cv2.split(self.image)

#         padded_imageR = cv2.copyMakeBorder(R, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
#         padded_imageG = cv2.copyMakeBorder(G, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
#         padded_imageB = cv2.copyMakeBorder(B, padding, padding, padding, padding, cv2.BORDER_CONSTANT)


#         thresholdR = cv2.adaptiveThreshold(padded_imageR, 255, adaptiveMethod, thresholdType, filter_size, my_constant)
#         thresholdG = cv2.adaptiveThreshold(padded_imageG, 255, adaptiveMethod, thresholdType, filter_size, my_constant)
#         thresholdB = cv2.adaptiveThreshold(padded_imageB, 255, adaptiveMethod, thresholdType, filter_size, my_constant)

#         final = cv2.merge([thresholdB, thresholdG, thresholdR])
#         return final
        
#     def apply_filter(self, filter_name, filter_size, padding=0,adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType = cv2.THRESH_BINARY,my_constant = 0):

#         # Apply the selected filter on the Grayscale image
#         match filter_name:
#             case 'mean':
#                 filtered_image = self.mean_filter(filter_size, padding)
#             case 'median':
#                 filtered_image = self.median_filter(filter_size, padding)
#                 filtered_image = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#             case 'adaptive':
#                 filtered_image = self.adaptive_filter(filter_size, padding,adaptiveMethod,thresholdType)
#                 filtered_image = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#             case _:
#                 raise ValueError(f"Filter '{filter_name}' is not supported")
        
#         # Convert the pixel values of the output image to a range of 0-255
#         # filtered_image = cv2.convertScaleAbs(filtered_image, alpha=(255.0/filtered_image.max()))
#             # Normalize pixel values to the range of 0-255
        
        
#         return filtered_image


class SpatialFilterRGB:
    def __init__(self, image):
        self.BGR_image = image

    def mean_filter(self, filter_size, padding=0):
        padded_image = cv2.copyMakeBorder(self.BGR_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        kernel_size = (filter_size, filter_size)
        return cv2.blur(padded_image, kernel_size)

    def median_filter(self, filter_size, padding=0):
        padded_image = cv2.copyMakeBorder(self.BGR_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        return cv2.medianBlur(padded_image, filter_size)

    def adaptive_filter(self, filter_size = 3, padding=0, maxVal = 255, \
                        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                        thresholdType = cv2.THRESH_BINARY, \
                        borderType = cv2.BORDER_CONSTANT, \
                        C = 1):
        
        imgB, imgG, imgR = cv2.split(self.BGR_image)
        padded_imageB = cv2.copyMakeBorder(imgB, padding, padding, padding, padding, borderType)
        f_b = cv2.adaptiveThreshold(padded_imageB, maxVal, adaptiveMethod, thresholdType, filter_size, C)
        
        padded_imageG = cv2.copyMakeBorder(imgG, padding, padding, padding, padding, borderType)
        f_g = cv2.adaptiveThreshold(padded_imageG, maxVal, adaptiveMethod, thresholdType, filter_size, C)
        
        padded_imageR = cv2.copyMakeBorder(imgR, padding, padding, padding, padding, borderType)
        f_r = cv2.adaptiveThreshold(padded_imageR, maxVal, adaptiveMethod, thresholdType, filter_size, C)
        
        return cv2.merge([f_b, f_g, f_r])
        # padded_image = cv2.copyMakeBorder(self.BGR_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        # return cv2.adaptiveThreshold(padded_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, filter_size, 2)
    
    
    FILTER_TYPE_MEAN = 'mean'
    FILTER_TYPE_MEDIAN = 'median'
    FILTER_TYPE_ADAPTIVE = 'adaptive'
    
    def apply_filter(self, filter_name, filter_size, padding=0):
        
        match filter_name:
            case self.FILTER_TYPE_MEAN:
                return self.mean_filter(filter_size, padding)
            case self.FILTER_TYPE_MEDIAN:
                return self.median_filter(filter_size, padding)
            case self.FILTER_TYPE_ADAPTIVE:
                return self.adaptive_filter(filter_size, padding)
            case _:
                raise ValueError(f"Filter '{filter_name}' is not supported")


class FrequencyFilter:
    def __init__(self, image):
        self.image = image

    def fftshift(self):
        f = np.fft.fft2(self.image)
        fshift = np.fft.fftshift(f)
        return fshift

    def ifftshift(self, fshift):
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        return img_back

    def create_mask(self, radius):
        base = np.zeros(self.image.shape[:2])
        cv2.circle(base, (self.image.shape[1]//2, self.image.shape[0]//2), int(radius), (1, 1, 1), -1, 8, 0)
        return base

    def low_pass_filter(self, radius):
        fshift = self.fftshift()
        mask = self.create_mask(radius)
        fshift = fshift * mask
        return self.ifftshift(fshift)

    def high_pass_filter(self, radius):
        fshift = self.fftshift()
        mask = self.create_mask(radius)
        fshift = fshift * (1 - mask)
        return self.ifftshift(fshift)

    def band_pass_filter(self, min_radius, max_radius):
        fshift = self.fftshift()
        min_mask = self.create_mask(min_radius)
        max_mask = self.create_mask(max_radius)
        band_mask = min_mask - max_mask
        fshift = fshift * band_mask
        return self.ifftshift(fshift)

    def band_reject_filter(self, min_radius, max_radius):
        fshift = self.fftshift()
        min_mask = self.create_mask(min_radius)
        max_mask = self.create_mask(max_radius)
        band_mask = min_mask - max_mask
        fshift = fshift * (1 - band_mask)
        return self.ifftshift(fshift)
class FrequencyFilterRGB:
    def __init__(self, image):
        # self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image = image

    def fftshift(self):
        # Split the image into its separate color channels
        b, g, r = cv2.split(self.image)

        # Apply FFT to each color channel
        b_fft = np.fft.fft2(b)
        g_fft = np.fft.fft2(g)
        r_fft = np.fft.fft2(r)

        # Shift the FFT output for each color channel
        b_fft_shifted = np.fft.fftshift(b_fft)
        g_fft_shifted = np.fft.fftshift(g_fft)
        r_fft_shifted = np.fft.fftshift(r_fft)

        return b_fft_shifted, g_fft_shifted, r_fft_shifted

    def ifftshift(self, b_fft_shifted,g_fft_shifted,r_fft_shifted):
        f_ishiftb = np.fft.ifftshift(b_fft_shifted)
        f_ishiftg = np.fft.ifftshift(g_fft_shifted)
        f_ishiftr = np.fft.ifftshift(r_fft_shifted)

        img_backb = np.fft.ifft2(f_ishiftb)
        img_backg = np.fft.ifft2(f_ishiftg)
        img_backr = np.fft.ifft2(f_ishiftr)

        img_backb = np.abs(img_backb)
        img_backg = np.abs(img_backg)
        img_backr = np.abs(img_backr)

        img_back = cv2.merge([img_backb, img_backg, img_backr]).astype(int)
        
        return img_back

    def create_mask(self, radius):
        base = np.zeros(self.image.shape[:2])
        cv2.circle(base, (self.image.shape[1]//2, self.image.shape[0]//2), int(radius), (1, 1, 1), -1, 8, 0)
        return base


    def low_pass_filter(self, radius):

        b_fft_shifted,g_fft_shifted,r_fft_shifted = self.fftshift()

        # Create mask for low-pass filtering
        mask = self.create_mask(radius)

        # Apply the mask to the frequency domain for each color channel
        b_fft_filtered = b_fft_shifted * mask
        g_fft_filtered = g_fft_shifted * mask
        r_fft_filtered = r_fft_shifted * mask

        # Shift the filtered FFT output back for each color channel
        b_fft_filtered_shifted = np.fft.ifftshift(b_fft_filtered)
        g_fft_filtered_shifted = np.fft.ifftshift(g_fft_filtered)
        r_fft_filtered_shifted = np.fft.ifftshift(r_fft_filtered)

        # Apply IFFT to get back to the spatial domain for each color channel
        b_filtered = np.fft.ifft2(b_fft_filtered_shifted).real
        g_filtered = np.fft.ifft2(g_fft_filtered_shifted).real
        r_filtered = np.fft.ifft2(r_fft_filtered_shifted).real

        # Merge the color channels back into an RGB image
        img_filtered = cv2.merge([b_filtered, g_filtered, r_filtered])

        # Normalize the result and return as uint8 image
        filtered_image = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return filtered_image

    def high_pass_filter(self, radius):
        b_fft_shifted,g_fft_shifted,r_fft_shifted = self.fftshift()

        mask = self.create_mask(radius)

        b_fft_shifted = b_fft_shifted * (1 - mask)
        g_fft_shifted = g_fft_shifted * (1 - mask)
        r_fft_shifted = r_fft_shifted * (1 - mask)
        ifshift = self.ifftshift(b_fft_shifted, g_fft_shifted, r_fft_shifted)
        filtered_image = cv2.normalize(ifshift, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
        return filtered_image


    def band_pass_filter(self, min_radius, max_radius):
        b_fft_shifted,g_fft_shifted,r_fft_shifted = self.fftshift()

        min_mask = self.create_mask(min_radius)
        max_mask = self.create_mask(max_radius)
        band_mask = min_mask - max_mask

        b_fft_shifted = b_fft_shifted * band_mask
        g_fft_shifted = g_fft_shifted * band_mask
        r_fft_shifted = r_fft_shifted * band_mask
        ifshift = self.ifftshift(b_fft_shifted, g_fft_shifted, r_fft_shifted)
        filtered_image = cv2.normalize(ifshift, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return filtered_image

    def band_reject_filter(self, min_radius, max_radius):
        b_fft_shifted,g_fft_shifted,r_fft_shifted = self.fftshift()

        min_mask = self.create_mask(min_radius)
        max_mask = self.create_mask(max_radius)
        band_mask = min_mask - max_mask

        b_fft_shifted = b_fft_shifted * (1 - band_mask)
        g_fft_shifted = g_fft_shifted * (1 - band_mask)
        r_fft_shifted = r_fft_shifted * (1 - band_mask)
        ifshift = self.ifftshift(b_fft_shifted, g_fft_shifted, r_fft_shifted)
        filtered_image = cv2.normalize(ifshift, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return filtered_image


class PointProcessing:
    def __init__(self, image):
        self.image = image

    def apply_threshold(self, threshold_value):
        """
        Applies a binary threshold to the image.
        :param threshold_value: Threshold value for binarization.
        """
        _, binary_image = cv2.threshold(self.image, threshold_value, 255, cv2.THRESH_BINARY)
        return binary_image

    def apply_gamma_correction(self, gamma):
        """
        Applies gamma correction to the image.
        :param gamma: Gamma value for correction.
        """
        gamma_corrected = np.array(255 * (self.image / 255) ** gamma, dtype='uint8')
        return gamma_corrected

    def apply_negative(self):
        """
        Applies negative transformation to the image.
        """
        return 255 - self.image

    def apply_log_transform(self, c=1):
        """
        Applies logarithmic transformation to the image.
        :param c: Constant for scaling the transformation.
        """
        log_transformed = c * np.log(1 + self.image)
        return np.array(log_transformed, dtype='uint8')

    def apply_contrast_stretching(self, r_min=0, r_max=255):
        """
        Applies contrast stretching to the image.
        :param r_min: Minimum intensity value after stretching.
        :param r_max: Maximum intensity value after stretching.
        """
        stretched = np.interp(self.image, [np.min(self.image), np.max(self.image)], [r_min, r_max])
        return np.array(stretched, dtype='uint8')
class PointProcessingRGB:
    def __init__(self, image):
        self.image = image

    def apply_threshold(self, threshold_value):
        """
        Applies a binary threshold to the image.
        :param threshold_value: Threshold value for binarization.
        """
        if len(self.image.shape) == 3: # if the image is color
            b, g, r = cv2.split(self.image)
            _, b = cv2.threshold(b, threshold_value, 255, cv2.THRESH_BINARY)
            _, g = cv2.threshold(g, threshold_value, 255, cv2.THRESH_BINARY)
            _, r = cv2.threshold(r, threshold_value, 255, cv2.THRESH_BINARY)
            binary_image = cv2.merge((b, g, r))
        else: # if the image is grayscale
            _, binary_image = cv2.threshold(self.image, threshold_value, 255, cv2.THRESH_BINARY)
        return binary_image

    def apply_gamma_correction(self, gamma):
        """
        Applies gamma correction to the image.
        :param gamma: Gamma value for correction.
        """
        if len(self.image.shape) == 3: # if the image is color
            b, g, r = cv2.split(self.image)
            b_corrected = np.array(255 * (b / 255) ** gamma, dtype='uint8')
            g_corrected = np.array(255 * (g / 255) ** gamma, dtype='uint8')
            r_corrected = np.array(255 * (r / 255) ** gamma, dtype='uint8')
            gamma_corrected = cv2.merge((b_corrected, g_corrected, r_corrected))
        else: # if the image is grayscale
            gamma_corrected = np.array(255 * (self.image / 255) ** gamma, dtype='uint8')
        return gamma_corrected

    def apply_negative(self):
        """
        Applies negative transformation to the image.
        """
        if len(self.image.shape) == 3: # if the image is color
            b, g, r = cv2.split(self.image)
            b_neg = 255 - b
            g_neg = 255 - g
            r_neg = 255 - r
            negative = cv2.merge((b_neg, g_neg, r_neg))
        else: # if the image is grayscale
            negative = 255 - self.image
        return negative

    def apply_log_transform(self, c=1):
        """
        Applies logarithmic transformation to the image.
        :param c: Constant for scaling the transformation.
        """
        if len(self.image.shape) == 3: # if the image is color
            b, g, r = cv2.split(self.image)
            b_transformed = c * np.log(1 + b)
            g_transformed = c * np.log(1 + g)
            r_transformed = c * np.log(1 + r)
            log_transformed = cv2.merge((b_transformed, g_transformed, r_transformed))
        else: # if the image is grayscale
            log_transformed = c * np.log(1 + self.image)
        return np.array(log_transformed, dtype='uint8')

    def apply_contrast_stretching(self, r_min=0, r_max=255):
        """
        Applies contrast stretching to the image.
        :param r_min: Minimum intensity value after stretching.
        :param r_max: Maximum intensity value after stretching.
        """
        if len(self.image.shape) == 3: # if the image is color
            b, g, r = cv2.split(self.image)
            b_stretched = np.interp(b, [np.min(b), np.max(b)], [r_min, r_max])
            g_stretched = np.interp(g, [np.min(g), np.max(g)], [r_min, r_max])
            r_stretched = np.interp(r, [np.min(r), np.max(r)], [r_min, r_max])
            stretched = cv2.merge((b_stretched, g_stretched, r_stretched))
        else: # if the image is grayscale
            stretched = np.interp(self.image, [np.min(self.image), np.max(self.image)], [r_min, r_max])
        return np.array(stretched, dtype='uint8')


class HistogramEnhancement:
    def __init__(self, image):
        self.image = image

    def apply_histogram_equalization(self):
        """
        Applies histogram equalization to the image.
        """            
        return cv2.equalizeHist(self.image)

    def match_histogram(self, target_image):
        """
        Matches the histogram of the image to the histogram of the target image.
        :param target_image: Image whose histogram will be used as a reference.
        """        
        pass
class HistogramEnhancementRGB:
    def __init__(self, image):
        self.image = image

    def apply_histogram_equalization(self):
        """
        Applies histogram equalization to the BGR image.
        """
        if len(self.image.shape) == 3: # if the image is color
            b, g, r = cv2.split(self.image)
            b_eq = cv2.equalizeHist(b)
            g_eq = cv2.equalizeHist(g)
            r_eq = cv2.equalizeHist(r)
            equalized = cv2.merge((b_eq, g_eq, r_eq))
        else: # if the image is grayscale
            equalized = cv2.equalizeHist(self.image)
        return equalized

    def match_histogram(self, target_image):
        """
        Matches the histogram of the image to the histogram of the target image.
        :param target_image: Image whose histogram will be used as a reference.
        """
        if len(self.image.shape) == 3: # if the image is color
            b, g, r = cv2.split(self.image)
            b_target, g_target, r_target = cv2.split(target_image)
            b_matched = cv2.equalizeHist(b, b_target)
            g_matched = cv2.equalizeHist(g, g_target)
            r_matched = cv2.equalizeHist(r, r_target)
            matched = cv2.merge((b_matched, g_matched, r_matched))
        else: # if the image is grayscale
            matched = cv2.equalizeHist(self.image, target_image)
        return matched


class ImageCompression:
    def __init__(self, image_path):        
        self.image = image_path

    def compress_using_jpeg(self, quality):
        """
        Compresses the image using JPEG compression.
        :param quality: Compression quality (0-100), higher value means better quality.
        """
        encoded_image, compressed_image = cv2.imencode('.jpg', self.image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return compressed_image

    def compress_using_png(self, compression_level):
        """
        Compresses the image using PNG compression.
        :param compression_level: Compression level (0-9), higher value means higher compression.
        """
        encoded_image, compressed_image = cv2.imencode('.png', self.image, [int(cv2.IMWRITE_PNG_COMPRESSION), compression_level])
        return compressed_image
    def compress_using_huffman(self):
        pass
class ImageCompressionRGB:
    def __init__(self, image_path):        
        self.image = image

    def compress_using_jpeg(self, quality):
        """
        Compresses the image using JPEG compression.
        :param quality: Compression quality (0-100), higher value means better quality.
        """
        encoded_image, compressed_image = cv2.imencode('.jpg', self.image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return compressed_image

    def compress_using_png(self, compression_level):
        """
        Compresses the image using PNG compression.
        :param compression_level: Compression level (0-9), higher value means higher compression.
        """
        encoded_image, compressed_image = cv2.imencode('.png', self.image, [int(cv2.IMWRITE_PNG_COMPRESSION), compression_level])
        return compressed_image
    def compress_using_huffman(self):
        pass

def ifftshift(fshift):
    fishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(fishift)
    img_back = np.abs(img_back)
    return img_back

def fftshiftBGR(image):
    # freq_images = []
    # channels = cv2.split(image)
    # for channel in channels:
    #     freq_image = np.fft.fftshift(np.fft.fft2(channel))
    #     freq_images.append(freq_image)
    # merged_freq_images = np.concatenate(freq_images, axis=-1)
    # shifted = cv2.merge([merged_freq_images[..., i] for i in range(merged_freq_images.shape[-1])])


    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
 
    f = np.fft.fftn(img) 
 
    fshiftRGB = np.fft.fftshift(f)

    img_back = 20* np.log(np.abs(fshiftRGB)).astype(np.uint8) 
    # img_back = cv2.normalize(fshiftRGB, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) 

    return img_back ,fshiftRGB   

def ifftshiftBGR(fshiftRGB):
    f_ishift = np.fft.ifftshift(fshiftRGB)
    img_back = np.real(np.fft.ifftn(f_ishift))

    # در صورت نیاز، تبدیل نوع داده تصویر به uint8
    # img_back = np.uint8(img_back)
    img_back = cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return img_back


# class SpatialFilterRGB:
#     def __init__(self, image):
#         self.image = image

#     def mean_filter(self, filter_size, padding=0):
#         padded_image = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
#         filtered_image = cv2.blur(padded_image, (filter_size, filter_size))
#         return filtered_image

#     def median_filter(self, filter_size, padding=0):
#         padded_image = cv2.copyMakeBorder(self.image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
#         filtered_image = cv2.medianBlur(padded_image, filter_size)
#         return filtered_image


#     def adaptive_filter(self, filter_size , padding=0):
        
#         adaptiveMethod = cv2.cv2.ADAPTIVE_THRESH_MEAN_C   #ADAPTIVE_THRESH_GAUSSIAN_C  
#         thresholdType = cv2.THRESH_BINARY
#         my_constant = 2
        
#         R, G, B = cv2.split(self.image)
        
#         padded_imageR = cv2.copyMakeBorder(R, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
#         padded_imageG = cv2.copyMakeBorder(G, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
#         padded_imageB = cv2.copyMakeBorder(B, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
                
    
#         thresholdR = cv2.adaptiveThreshold(padded_imageR, 255, adaptiveMethod, thresholdType, filter_size, my_constant)
#         thresholdG = cv2.adaptiveThreshold(padded_imageG, 255, adaptiveMethod, thresholdType, filter_size, my_constant)
#         thresholdB = cv2.adaptiveThreshold(padded_imageB, 255, adaptiveMethod, thresholdType, filter_size, my_constant)
        
#         final = cv2.merge([thresholdR, thresholdG, thresholdB])
#         return final
        
#     def apply_filter(self, filter_name, filter_size, padding=0):

#         # Apply the selected filter on the Grayscale image
#         match filter_name:
#             case 'mean':
#                 filtered_image = self.mean_filter(filter_size, padding)
#             case 'median':
#                 filtered_image = self.median_filter(filter_size, padding)
#             case 'adaptive':
#                 filtered_image = self.adaptive_filter(filter_size, padding)
#             case _:
#                 raise ValueError(f"Filter '{filter_name}' is not supported")
        
#         filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)    
#         return filtered_image




