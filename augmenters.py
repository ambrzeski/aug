from __future__ import absolute_import
import random
import itertools
import numpy as np
import cv2 as cv
from aug import helper


# ----------------------------------------------------------------------------------------------------------------------
# Abstract augmenter
# ----------------------------------------------------------------------------------------------------------------------

class Augmenter(object):

    def __init__(self):
        self.name = None
        self.params = None
        self.params_random = None

    @staticmethod
    def transform(img, *arg):
        pass

    def augment_random(self, *images):
        params = [p.get() for p in self.params_random]
        name = "{}[{}]".format(self.name, helper.to_string(params))
        results = []
        for img in images:
            dst = self.transform(img, *params)
            results.append(dst)
        return results, name

    def augment_fixed(self, image):
        for params in itertools.product(*self.params):
            dst = self.transform(image, *params)
            name = "{}[{}]".format(self.name, helper.to_string(params))
            yield (dst, name)


# ----------------------------------------------------------------------------------------------------------------------
# Augmenters
# ----------------------------------------------------------------------------------------------------------------------

class CopyParam(object):
    @staticmethod
    def get():
        return "{}"


class CopyAugmenter(Augmenter):

    def __init__(self):
        super(CopyAugmenter, self).__init__()
        self.param = "{}"
        self.params_random = [CopyParam()]
        self.name = 'co'

    @staticmethod
    def transform(img, param):
        return img


class RotateAugmenter(Augmenter):

    def __init__(self, degrees=(0, 90, 180, 270)):
        super(RotateAugmenter, self).__init__()
        self.params = [degrees]
        self.params_random = [RandomElement(degrees)]
        self.name = 'ro'

    @staticmethod
    def transform(img, degree):
        if degree == 0:
            return img
        if degree not in [90, 180, 270]:
            print("Warning: unsupported rotation degree. Ignoring")
            return None
        if degree == 90:
            dst = cv.transpose(img)
            return cv.flip(dst, 1)
        if degree == 180:
            return cv.flip(img, -1)
        if degree == 270:
            dst = cv.transpose(img)
            return cv.flip(dst, 0)


class RotateAngleAugmenter(Augmenter):

    def __init__(self, degrees=(-10, 0, 10)):
        super(RotateAngleAugmenter, self).__init__()
        self.params = [degrees]
        self.params_random = [RandomFromRange(min(degrees), max(degrees))]
        self.name = 'ra'

    @staticmethod
    def transform(img, degree):
        if degree == 0:
            return img
        h, w = img.shape[:2]
        m = cv.getRotationMatrix2D((w / 2, h / 2), degree, 1)
        return cv.warpAffine(img, m, (w, h))


class FlipAugmenter(Augmenter):

    def __init__(self, flips=(None, 0, 1)):
        super(FlipAugmenter, self).__init__()
        self.params = [flips]
        self.params_random = [RandomElement(flips)]
        self.name = 'fl'

    @staticmethod
    def transform(img, flip):
        if flip is None:
            return img
        dst = cv.flip(img, flip)
        return dst


class HueDistortionAugmenter(Augmenter):

    def __init__(self, factors=(-3, 3, 0, 7, -7)):
        super(HueDistortionAugmenter, self).__init__()
        self.params = [factors]
        self.params_random = [RandomFromRange(min(factors), max(factors))]
        self.name = 'hu'

    @staticmethod
    def transform(img, factor):

        if factor == 0 or helper.is_mask(img):
            return img

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)     # H: 0 - 180, S: 0 - 255, V: 0 - 255
        hsv_max = 180

        h = np.uint8(h + factor)
        if factor > 0:
            h[h > hsv_max] -= hsv_max
        if factor < 0:
            h[h > hsv_max] -= 75

        dst = cv.merge([h, s, v])
        dst = cv.cvtColor(dst, cv.COLOR_HSV2BGR)
        return dst


class SaturationDistortionAugmenter(Augmenter):

    def __init__(self, factors=(-10, 10, 0, 20, -20)):
        super(SaturationDistortionAugmenter, self).__init__()
        self.params = [factors]
        self.params_random = [RandomFromRange(min(factors), max(factors))]
        self.name = 'sa'

    @staticmethod
    def transform(img, factor):

        if factor == 0 or helper.is_mask(img):
            return img

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)     # H: 0 - 180, S: 0 - 255, V: 0 - 255

        s = cv.add(s, factor)
        s[s == factor] = 0

        dst = cv.merge([h, s, v])
        dst = cv.cvtColor(dst, cv.COLOR_HSV2BGR)
        return dst


class BrightnessAugmenter(Augmenter):

    def __init__(self, ratios=(0.9, 1.0, 1.1)):
        super(BrightnessAugmenter, self).__init__()
        self.params = [ratios]
        self.params_random = [RandomFromRange(min(ratios), max(ratios))]
        self.name = 'br'

    @staticmethod
    def transform(img, ratio):

        if ratio == 1.0 or helper.is_mask(img):
            return img

        img = img * ratio
        img[img > 255] = 255
        img = np.uint8(img)
        return img


class BlurAugmenter(Augmenter):

    def __init__(self, kernels=(1, 7)):
        super(BlurAugmenter, self).__init__()
        self.params = [kernels]
        self.params_random = [RandomNormalFromRange(min(kernels), max(kernels), step=2)]
        self.name = 'bl'

    @staticmethod
    def transform(img, kernel):

        if kernel <= 1 or helper.is_mask(img):
            return img

        dst = cv.GaussianBlur(img, (kernel, kernel), 0)
        return dst


class NoiseAugmenter(Augmenter):

    def __init__(self, variances=(0, 7, 15)):
        super(NoiseAugmenter, self).__init__()
        self.params = [variances]
        self.params_random = [RandomNormalFromRange(min(variances), max(variances))]
        self.name = 'no'

    @staticmethod
    def transform(img, variance):

        if variance == 0 or helper.is_mask(img):
            return img

        mean = 0
        sigma = variance**0.5
        gauss = np.random.normal(mean, sigma, img.shape)
        # gauss = gauss.reshape(image.shape)
        dst = img + gauss
        dst[dst < 0] = 0
        dst[dst > 255] = 255
        dst = np.uint8(dst)
        return dst


class PerspectiveAugmenter(Augmenter):

    def __init__(self, direction=(0, 1, 2, 3), ratio=(0.0, 0.3)):
        super(PerspectiveAugmenter, self).__init__()
        self.params = [direction, ratio]
        self.params_random = [RandomElement(direction), RandomFromRange(min(ratio), max(ratio))]
        self.name = 'pe'

    @staticmethod
    def transform(img, direction, ratio):

        if ratio == 0.0:
            return img

        height = img.shape[0]
        width = img.shape[1]
        offset = int((ratio / 2) * (height + width) / 2)

        if direction == 0:
            vectors = np.float32([[offset, 0], [-offset, 0], [0, 0], [0, 0]])
        elif direction == 1:
            vectors = np.float32([[0, 0], [0, offset], [0, -offset], [0, 0]])
        elif direction == 2:
            vectors = np.float32([[0, 0], [0, 0], [-offset, 0], [offset, 0]])
        else:
            vectors = np.float32([[0, offset], [0, 0], [0, 0], [0, -offset]])

        orig_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        m = cv.getPerspectiveTransform(orig_pts - vectors, orig_pts)
        return cv.warpPerspective(img, m, dsize=img.shape[:2][::-1])


class SkewAugmenter(Augmenter):

    def __init__(self, direction=(0, 1, 2, 3), ratio=(0.0, 0.3)):
        super(SkewAugmenter, self).__init__()
        self.params = [direction, ratio]
        self.params_random = [RandomElement(direction), RandomFromRange(min(ratio), max(ratio))]
        self.name = 'sk'

    @staticmethod
    def transform(img, direction, ratio):

        if ratio == 0.0:
            return img

        height = img.shape[0]
        width = img.shape[1]
        offset = int((ratio / 2) * (height + width) / 2)

        if direction == 0:
            vectors = np.float32([[offset, 0], [0, 0], [-offset, 0]])
            orig_pts = np.float32([[0, 0], [width, 0], [width, height]])
        elif direction == 1:
            vectors = np.float32([[0, offset], [0, 0], [0, -offset]])
            orig_pts = np.float32([[width, 0], [width, height], [0, height]])
        elif direction == 2:
            vectors = np.float32([[-offset, 0], [0, 0], [offset, 0]])
            orig_pts = np.float32([[width, 0], [width, height], [0, height]])
        else:
            vectors = np.float32([[0, offset], [0, 0], [0, -offset]])
            orig_pts = np.float32([[0, 0], [width, 0], [width, height]])

        m = cv.getAffineTransform(orig_pts - vectors, orig_pts)
        return cv.warpAffine(img, m, dsize=img.shape[:2][::-1])


class CircleAugmenter(Augmenter):

    def __init__(self, shrink_ratios=(0.8, 0.9)):
        super(CircleAugmenter, self).__init__()
        self.params = [shrink_ratios]
        self.params_random = [RandomFromRange(min(shrink_ratios), max(shrink_ratios))]
        self.name = 'ci'

    @staticmethod
    def transform(img, shrink_ratio):
        center = (int(img.shape[1] / 2), int(img.shape[0] / 2))
        radius = int(min(img.shape[0], img.shape[1]) * 0.5 * shrink_ratio)
        circle = img * 0
        cv.circle(circle, center, radius, (1, 1, 1, 1), -1)
        img = img * circle
        crop = img[center[1] - radius:center[1] + radius, center[0] - radius:center[0] + radius]
        return crop


class PCAColorAugmenter(Augmenter):

    def __init__(self, ratio=(-0.1, -0.05, 0, 0.05, 0.1), eigen_vecs=None, eigen_vals=None):
        """
        PCA color augmentation introduced in AlexNet. Default eigen vals and vecs are taken from:
        https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua and they represent values
        calculated for a subset of ImageNet assuming BGR format.
        """
        super(PCAColorAugmenter, self).__init__()
        self.params = [ratio]
        self.params_random = [RandomNormal(0, max(ratio))]
        self.name = 'pc'

        self.eigen_vecs = eigen_vecs
        if eigen_vecs is None:
            self.eigen_vecs = np.array(
                [[ 0.4203, -0.6948, -0.5836],
                 [-0.8140, -0.0045, -0.5808],
                 [ 0.4009,  0.7192, -0.5675]]
            )

        self.eigen_vals = eigen_vals
        if eigen_vals is None:
            self.eigen_vals = np.array(
                [0.0045, 0.0188, 0.2175]
            )

    def transform(self, img, ratio):

        if ratio == 0 or helper.is_mask(img):
            return img

        update = np.dot(self.eigen_vecs, np.expand_dims(np.multiply(self.eigen_vals, ratio), axis=1))
        update = np.int8(update.flatten() * 255)
        val = tuple(update.tolist()) + (0,)
        return cv.add(img, val)


class CropAugmenter(Augmenter):

    def __init__(self, scales=(0.5, 0.8, 1.0), x=(.0, 1.0), y=(.0, 1.0), rectangular=False):
        super(CropAugmenter, self).__init__()
        self.rectangular = rectangular
        self.name = 'cr'
        self.params = [scales, x, y]
        self.params_random = [RandomElement(scales),
                              RandomFromRange(min(x), max(x)),
                              RandomFromRange(min(y), max(y))]

    def transform(self, img, scale, x, y):
        crop_width = int(img.shape[1] * scale)
        crop_height = int(img.shape[0] * scale)
        if self.rectangular:
            crop_width = crop_height = min(crop_width, crop_height)
        crop_x = int((img.shape[1] - crop_width) * x)
        crop_y = int((img.shape[0] - crop_height) * y)
        crop = img[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width].copy()
        return crop


# ----------------------------------------------------------------------------------------------------------------------
# MultiAugmenter
# ----------------------------------------------------------------------------------------------------------------------

class MultiAugmenter(object):

    def __init__(self, augmenters):
        self.augmenters = augmenters

    def augment_fixed(self, *images):
        return self.augment(images, 'augment_fixed')

    def augment_random(self, *images):
        res = self.augment(images, 'augment_random')
        return res[0][0], res[0][1]

    def augment(self, image, method):
        result = []
        for augmenter in self.augmenters:
            augmented = []
            input_ = result
            if not input_:
                input_ = [(image, '')]
            for img, name in input_:
                res = getattr(augmenter, method)(*img)
                for nimg, nname in [res]:
                    augmented.append((nimg, name + ("_" if name else "") + nname))
            result = augmented

        if not self.augmenters:
            result.append((image, ""))

        return result


# ----------------------------------------------------------------------------------------------------------------------
# Random params
# ----------------------------------------------------------------------------------------------------------------------

class RandomElement(object):

    def __init__(self, elements):
        self.elements = elements

    def get(self):
        return random.choice(self.elements)


class RandomFromRange(object):

    def __init__(self, a, b, step=1):
        self.a = a
        self.b = b
        self.step = step
        if step != 1:
            elements = self.valid_elements()
            self.random_element = RandomElement(elements)
            self.get = self.random_element.get

    def get(self):
        if isinstance(self.a, int):
            return random.randint(self.a, self.b)
        else:
            return random.uniform(self.a, self.b)

    def valid_elements(self):
        elements = []
        value = self.a
        while value <= self.b:
            elements.append(value)
            value += self.step
        return elements


class RandomNormal(object):

    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def get(self):
        return random.normalvariate(self.mean, self.sigma)


class RandomNormalFromRange(RandomFromRange):

    def __init__(self, a, b, step=1):
        super(RandomNormalFromRange, self).__init__(a, b, step)
        self.a = a
        self.b = b
        self.step = step
        # 3 sigma (covers 99.7% of samples)
        self.sigma = abs(b - a) / 3.0

    def get(self):
        val = random.normalvariate(self.a, self.sigma)
        if val < self.a:
            val = 2 * self.a - val
        if isinstance(self.a, int):
            val = int(val)
            if self.step != 1:
                elements = self.valid_elements()
                idx = np.abs(np.array(elements)-val).argmin()
                val = elements[idx]
        return val


def get_by_name(name):
    d = {
        "copy": CopyAugmenter,
        "rotate": RotateAugmenter,
        "rotateangle": RotateAngleAugmenter,
        "flip": FlipAugmenter,
        "huedistortion": HueDistortionAugmenter,
        "saturationdistortion": SaturationDistortionAugmenter,
        "brightness": BrightnessAugmenter,
        "blur": BlurAugmenter,
        "noise": NoiseAugmenter,
        "perspective": PerspectiveAugmenter,
        "skew": SkewAugmenter,
        "circle": CircleAugmenter,
        "pcacolor": PCAColorAugmenter,
        "crop": CropAugmenter,
    }
    return d[name.lower()]
