import random

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms.functional as tvisf
import torchvision
import torch.nn.functional as F


class Transform:
    """
    A set of transformations, used for e.g. data augmentation.

    Args:
        transforms: An arbitrary number of transformations, derived from the TransformBase class.
                    They are applied in the order they are given.

    The Transform object can jointly transform images, bounding boxes and segmentation masks.
    This is done by calling the object with the following key-word arguments (all are optional).

    The following arguments are inputs to be transformed. They are either supplied as a single instance, or a list of instances.
        image: Image.
        coords: 2xN dimensional Tensor of 2D image coordinates [y, x].
        bbox: Bounding box on the form [x, y, w, h].
        mask: Segmentation mask with discrete classes.

    The following parameters can be supplied with calling the transform object:
        joint (bool): If True then transform all images/coords/bbox/mask in the list jointly using the same transformation.
                      Otherwise each tuple (images, coords, bbox, mask) will be transformed independently using
                      different random rolls. Default: True.
        new_roll (bool): If False, then no new random roll is performed, and the saved result from the previous roll
                         is used instead. Default: True.

    Check the AIATRACKProcessing class for examples.
    """

    def __init__(self, *transforms):
        if len(transforms) == 1 and isinstance(transforms[0], (list, tuple)):
            transforms = transforms[0]
        self.transforms = transforms
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['joint', 'new_roll']
        self._valid_all = self._valid_inputs + self._valid_args

    def __call__(self, **inputs):
        # Checks if the input names are actually valid
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        for v in inputs.keys():
            if v not in self._valid_all:
                raise ValueError(
                    "ERROR: incorrect input \'{}\' to transform, only supports inputs {} and arguments {}".format(v,
                                                                                                                  self._valid_inputs,
                                                                                                                  self._valid_args))
        # Turn on joint transform
        joint_mode = inputs.get('joint', True)
        # Start a new roll for probabilities
        new_roll = inputs.get('new_roll', True)

        if not joint_mode:
            out = zip(*[self(**inp) for inp in self._split_inputs(inputs)])
            return tuple(list(o) for o in out)

        out = {k: v for k, v in inputs.items() if k in self._valid_inputs}

        for t in self.transforms:
            out = t(**out, joint=joint_mode, new_roll=new_roll)
        if len(var_names) == 1:
            return out[var_names[0]]
        # Make sure order is correct
        return tuple(out[v] for v in var_names)

    def _split_inputs(self, inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        split_inputs = [{k: v for k, v in zip(var_names, vals)} for vals in zip(*[inputs[vn] for vn in var_names])]

        for arg_name, arg_val in list(filter(lambda it: it[0] != 'joint' and it[0] in self._valid_args, inputs.items())):
            if isinstance(arg_val, list):
                for inp, av in zip(split_inputs, arg_val):
                    inp[arg_name] = av
            else:
                for inp in split_inputs:
                    inp[arg_name] = arg_val
        return split_inputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class TransformBase:
    """
    Base class for transformation objects. See the Transform class for details.
    """

    def __init__(self):
        # Add 'att' to valid inputs
        self._valid_inputs = ['image', 'coords', 'bbox', 'mask', 'att']
        self._valid_args = ['new_roll']
        self._valid_all = self._valid_inputs + self._valid_args
        self._rand_params = None

    def __call__(self, **inputs):
        # Split input
        input_vars = {k: v for k, v in inputs.items() if k in self._valid_inputs}
        input_args = {k: v for k, v in inputs.items() if k in self._valid_args}

        # Roll random parameters for the transform
        if input_args.get('new_roll', True):
            rand_params = self.roll()
            if rand_params is None:
                rand_params = ()
            elif not isinstance(rand_params, tuple):
                rand_params = (rand_params,)
            self._rand_params = rand_params

        outputs = dict()
        for var_name, var in input_vars.items():
            if var is not None:
                transform_func = getattr(self, 'transform_' + var_name)
                if var_name in ['coords', 'bbox']:
                    params = (self._get_image_size(input_vars),) + self._rand_params
                else:
                    params = self._rand_params
                if isinstance(var, (list, tuple)):
                    outputs[var_name] = [transform_func(x, *params) for x in var]
                else:
                    outputs[var_name] = transform_func(var, *params)
        return outputs

    def _get_image_size(self, inputs):
        im = None
        for var_name in ['image', 'mask']:
            if inputs.get(var_name) is not None:
                im = inputs[var_name]
                break
        if im is None:
            return None
        if isinstance(im, (list, tuple)):
            im = im[0]
        if isinstance(im, np.ndarray):
            return im.shape[:2]
        if torch.is_tensor(im):
            return (im.shape[-2], im.shape[-1])
        raise Exception('ERROR: unknown image type')

    def roll(self):
        return None

    def transform_image(self, image, *rand_params):
        """
        Must be deterministic.
        """

        return image

    def transform_coords(self, coords, image_shape, *rand_params):
        """
        Must be deterministic.
        """

        return coords

    def transform_bbox(self, bbox, image_shape, *rand_params):
        # Assumes [x, y, w, h]
        # Check if not overloaded
        if self.transform_coords.__code__ == TransformBase.transform_coords.__code__:
            return bbox

        coord = bbox.clone().view(-1, 2).t().flip(0)

        x1 = coord[1, 0]
        x2 = coord[1, 0] + coord[1, 1]

        y1 = coord[0, 0]
        y2 = coord[0, 0] + coord[0, 1]

        coord_all = torch.tensor([[y1, y1, y2, y2], [x1, x2, x2, x1]])

        coord_transf = self.transform_coords(coord_all, image_shape, *rand_params).flip(0)
        tl = torch.min(coord_transf, dim=1)[0]
        sz = torch.max(coord_transf, dim=1)[0] - tl
        bbox_out = torch.cat((tl, sz), dim=-1).reshape(bbox.shape)
        return bbox_out

    def transform_mask(self, mask, *rand_params):
        """
        Must be deterministic.
        """

        return mask

    def transform_att(self, att, *rand_params):
        """
        Added to deal with attention masks.
        """

        return att


class ToTensor(TransformBase):
    """
    Convert to a Tensor.
    """

    def transform_image(self, image):
        # Handle numpy array
        if image.ndim == 2:
            image = image[:, :, None]

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        # Backward compatibility
        if isinstance(image, torch.ByteTensor):
            return image.float().div(255)
        else:
            return image

    def transfrom_mask(self, mask):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)

    def transform_att(self, att):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        else:
            raise ValueError('ERROR: dtype must be np.ndarray or torch.Tensor')


class ToTensorAndJitter(TransformBase):
    """
    Convert to a Tensor and jitter brightness.
    """

    def __init__(self, brightness_jitter=0.0, normalize=True):
        super().__init__()
        self.brightness_jitter = brightness_jitter
        self.normalize = normalize

    def roll(self):
        return np.random.uniform(max(0, 1 - self.brightness_jitter), 1 + self.brightness_jitter)

    def transform_image(self, image, brightness_factor):
        # Handle numpy array
        image = torch.from_numpy(image.transpose((2, 0, 1)))

        # Backward compatibility
        if self.normalize:
            return image.float().mul(brightness_factor / 255.0).clamp(0.0, 1.0)
        else:
            return image.float().mul(brightness_factor).clamp(0.0, 255.0)

    def transform_mask(self, mask, brightness_factor):
        if isinstance(mask, np.ndarray):
            return torch.from_numpy(mask)
        else:
            return mask

    def transform_att(self, att, brightness_factor):
        if isinstance(att, np.ndarray):
            return torch.from_numpy(att).to(torch.bool)
        elif isinstance(att, torch.Tensor):
            return att.to(torch.bool)
        else:
            raise ValueError('ERROR: dtype must be np.ndarray or torch.Tensor')


class Normalize(TransformBase):
    """
    Normalize image.
    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def transform_image(self, image):
        return tvisf.normalize(image, self.mean, self.std, self.inplace)


class ToGrayscale(TransformBase):
    """
    Converts image to grayscale with probability.
    """

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability
        self.color_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_grayscale):
        if do_grayscale:
            if torch.is_tensor(image):
                raise NotImplementedError('ERROR: implement torch variant')
            img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            return np.stack([img_gray, img_gray, img_gray], axis=2)
            # return np.repeat(np.sum(img * self.color_weights, axis=2, keepdims=True).astype(np.uint8), 3, axis=2)
        return image


class ToBGR(TransformBase):
    """
    Converts image to BGR.
    """

    def transform_image(self, image):
        if torch.is_tensor(image):
            raise NotImplementedError('ERROR: implement torch variant')
        img_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return img_bgr


class RandomHorizontalFlip(TransformBase):
    """
    Horizontally flip image randomly with a probability p.
    """

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_flip):
        if do_flip:
            if torch.is_tensor(image):
                return image.flip((2,))
            return np.fliplr(image).copy()
        return image

    def transform_coords(self, coords, image_shape, do_flip):
        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1, :] = (image_shape[1] - 1) - coords[1, :]
            return coords_flip
        return coords

    def transform_mask(self, mask, do_flip):
        if do_flip:
            if torch.is_tensor(mask):
                return mask.flip((-1,))
            return np.fliplr(mask).copy()
        return mask

    def transform_att(self, att, do_flip):
        if do_flip:
            if torch.is_tensor(att):
                if att.type() == 'torch.BoolTensor':
                    return att.to(torch.int).flip((-1,)).to(torch.bool)
                else:
                    return att.flip((-1,))
            return np.fliplr(att).copy()
        return att


class RandomHorizontalFlip_Norm(RandomHorizontalFlip):
    """
    Horizontally flip image randomly with a probability p.
    The difference is that the coord is normalized to [0,1].
    """

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability

    def transform_coords(self, coords, image_shape, do_flip):
        """
        We should use 1 rather than image_shape.
        """

        if do_flip:
            coords_flip = coords.clone()
            coords_flip[1, :] = 1 - coords[1, :]
            return coords_flip
        return coords

class RandomCropping(TransformBase):
    """
    Crop image randomly with a probability p.
    """

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability
        #self.cropper = torchvision.transforms.RandomCrop(size=(200, 200))
        self.size1 = np.random.randint(150, 200)
        self.size2 = np.random.randint(150, 200)
        self.cropper = torchvision.transforms.CenterCrop((self.size1,self.size2))
    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_crop):

        if do_crop:
             # We may want to randomize the size
            if torch.is_tensor(image):
                out = self.cropper(image.unsqueeze(0))
                out = F.interpolate(out,size=(320,320), mode='bilinear', align_corners=False).squeeze(0)
                return out
        return image

    def transform_coords(self, coords, image_shape, do_crop):
        # Transforming the bbox
        pass

    def transform_mask(self, mask, do_crop):
        # Transforming the mask
        if do_crop:
             # We may want to randomize the size
            if torch.is_tensor(mask):
                out = self.cropper(mask)
                out = F.interpolate(out,size=(320,320), mode='bilinear', align_corners=False)
                return out
        return mask

    def transform_att(self, att, do_crop):
        # Transforming the attention mask
        if do_crop:
             # We may want to randomize the size
            if torch.is_tensor(att):
                out = self.cropper(att.unsqueeze(0).unsqueeze(0).float())
                out = F.interpolate(out, size=(320, 320), mode='bilinear', align_corners=False).bool().squeeze(0).squeeze(0)
                return out
        return att

class RandomRotation(TransformBase):

    """Applies rotation to the image with a probability of 0.5"""

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability
        #self.rotator = torchvision.transforms.RandomRotation(degrees=(0,180))
        self.angle = np.random.randint(0,180)

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_rot):
        if do_rot:
            if torch.is_tensor(image):
                return tvisf.rotate(img=image, angle=self.angle)
        return image

    def transform_coords(self, coords, image_shape, do_rot):
        pass

    def transform_mask(self, mask, do_rot):
        if do_rot:
            if torch.is_tensor(mask):
                return tvisf.rotate(img=mask, angle=self.angle)
        return mask

    def transform_att(self, att, do_rot):
        if do_rot:
            if torch.is_tensor(att):
                att = att.unsqueeze(0)
                return tvisf.rotate(img=att, angle=self.angle).squeeze(0)
        return att

class Gaussian_Blur(TransformBase):

    """
    Applies gaussian blur to the system
    """

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, kernel_size=5, sigma=(0.1,5), do_blur=False):

        if do_blur:
            image = tvisf.gaussian_blur(image,kernel_size,sigma)

        return image

class Salt_and_pepper(TransformBase):

    """
    Applies salt and pepper noise to the system
    """

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, salt_prob=0.05, pepper_prob=0.05, do_sprinkle=False):

        if do_sprinkle:
            if torch.is_tensor(image):
                noisy_image_tensor = image.clone()

                # Get the dimensions of the image
                _, height, width = noisy_image_tensor.shape

                # Add salt noise
                salt_pixels = torch.rand(image.shape[1:], device=image.device) < salt_prob
                noisy_image_tensor[:, salt_pixels] = 1.0  # Set salt noise to 1.0 (white)

                # Add pepper noise
                pepper_pixels = torch.rand(image.shape[1:], device=image.device) < pepper_prob
                noisy_image_tensor[:, pepper_pixels] = 0.0  # Set pepper noise to 0.0 (black)

        return image