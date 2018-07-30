import random

import cv2 as cv
import math


def get_random_positive_crops(img_shape, bounding_rects, crop_sizes):
    """
    Returns random crops for each given bounding rects. For each bounding rect a set of crops are generated as defined
    in crop_sizes. Therefore, the number of returned crops equals to len(bounding_rects) * len(crop_sizes). If some crop
    size is to small to fully cover a bounding rect or is too big in comparison to the bounding_rect, the crops size
    will be ignored and replaced with another one (randomly picked from appropriate crop sizes).
    """

    crops = []

    # For each positive bounding box in the image
    for x, y, w, h in bounding_rects:

        # Find crop sizes are large enough to cover bounding rect, but not too big
        rect_size = max(w, h)
        min_size = rect_size
        max_size = 3 * rect_size
        abs_crops_sizes = get_abs_crop_sizes(img_shape, crop_sizes)
        valid_crops_sizes = [cs for cs in abs_crops_sizes if min_size <= cs <= max_size]
        if not valid_crops_sizes and max_size < min(abs_crops_sizes):
            valid_crops_sizes.append(min(abs_crops_sizes))

        # Assert that there was at least one large enough crop size
        if len(valid_crops_sizes) == 0:
            raise ValueError("None of crop sizes ({}) is proper to cover the bounding rect (size: {})"
                             .format(abs_crops_sizes, min_size))

        # If some crops had wrong size, add more crops of proper size
        valid_crops_sizes_copy = valid_crops_sizes[:]
        while len(valid_crops_sizes) < len(crop_sizes):
            cs = random.choice(valid_crops_sizes_copy)
            valid_crops_sizes.append(cs)

        # Generate random crops
        for crop_size in valid_crops_sizes:

            # Randomize position of a crop to fully containing the bounding rect
            max_offset_x = min(crop_size - w, x)
            max_offset_y = min(crop_size - h, y)
            min_offset_x = max(0, x + crop_size - img_shape[1])
            min_offset_y = max(0, y + crop_size - img_shape[0])
            offset_x = random.randrange(min_offset_x, max_offset_x + 1)
            offset_y = random.randrange(min_offset_y, max_offset_y + 1)
            crop = (x - offset_x, y - offset_y, crop_size, crop_size)
            crops.append(crop)

    return crops


def get_random_negative_crops(img, bounding_rects, crop_sizes, rand):
    """
    Randomizes a given number of positive and valid crops.
    """
    crops = []
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    abs_crop_sizes = get_abs_crop_sizes(img.shape, crop_sizes)

    for _ in range(rand):

        # Pick one crop size
        abs_crop_size = random.choice(abs_crop_sizes)

        # Proper ranges for start (x, y)
        x_range = img.shape[1] - abs_crop_size
        y_range = img.shape[0] - abs_crop_size

        # Attempt to randomize a valid negative crop
        attempts = 20
        for __ in range(attempts):

            # Randomize crop
            x = random.randrange(x_range)
            y = random.randrange(y_range)
            crop = (x, y, abs_crop_size, abs_crop_size)

            # Validate
            if is_crop_negative(bounding_rects, crop) and is_crop_valid(img_gray, crop):
                crops.append(crop)
                break

    return crops


def get_all_negative_crops(img, bounding_rects, crop_sizes, crop_overlaps):
    """
    Returns all possible crops from full grids defined be crop_sizes and crop_overlaps, such that they do not
    contain any area marked positive in the provided mask.
    """
    crops = []
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # For each grid
    abs_crops_sizes = get_abs_crop_sizes(img.shape, crop_sizes)
    for crop_size, crop_overlap in zip(abs_crops_sizes, crop_overlaps):
        grid = CropGrid(img.shape, crop_size, crop_overlap)
        crops_grid = grid.get()

        # For each possible crop
        for crop in crops_grid:
            if is_crop_negative(bounding_rects, crop) and is_crop_valid(img_gray, crop):
                crops.append(crop)

    return crops


def get_all_crops(img_shape, crop_sizes, crop_overlaps, leftover_handling=None):
    """
    Prepares a grid of rectangular crops for an image of given shape in form of a list of (x, y, w, h) tuples.
    The size of the rectangular crops will be equal to he minimum of side length multiplied the crop size.

    Example usage:

    crops = get_all_crops(img, crop_size=[0.2, 0.5], crop_overlaps=[0.5, 0.5])
    for x, y, w, h in crops:
        crop = img[y:y + h, x:x + w]

    :param img_shape: shape of the image
    :param crop_sizes: list of crop sizes as ratios of shorter image dimension
    :param crop_overlaps: overlaps size as ratios of the crops size (the larger overlap, the crops are generated)
    :param leftover_handling: method for handling leftover space at the end of the image (right and bottome sides).
           Can be: None - leftovers are ignored; "fill" - additional crops will placed to fill the leftovers resulting
           in non-standard overlaps in this area; "center" - crop grid will be centered to make leftover area equal for
           all sides of the image.
    :return: list of (x, y, w, h) tuples
    """
    crops = []

    # For each grid
    abs_crops_sizes = get_abs_crop_sizes(img_shape, crop_sizes)
    for crop_size, crop_overlap in zip(abs_crops_sizes, crop_overlaps):
        grid = CropGrid(img_shape, crop_size, crop_overlap, leftover_handling=leftover_handling)
        crops_grid = grid.get()
        crops.extend(crops_grid)

    return crops


class CropGrid(object):
    """
    Prepares a full grid of crops of given size and overlap.
    """

    def __init__(self, img_shape, crop_size, crop_overlap, leftover_handling=None):
        self.img_shape = img_shape
        self.size = crop_size
        self.overlap = crop_overlap
        self.leftover_handling = leftover_handling

    def get(self):
        height = self.img_shape[0]
        width = self.img_shape[1]
        xs = self.get_crop_start_coordinates(self.size, width)
        ys = self.get_crop_start_coordinates(self.size, height)
        crops = []
        for x in xs:
            for y in ys:
                crop = (x, y, self.size, self.size)
                crops.append(crop)
        return crops

    def get_crop_start_coordinates(self, crop_size, max_value):
        overlap_size = int(math.floor(self.overlap * crop_size))
        coords = []

        current_value = 0
        if self.leftover_handling == "center":
            current_value = ((max_value - crop_size) % (crop_size-overlap_size)) / 2

        while current_value + crop_size <= max_value:
            coords.append(current_value)
            current_value += crop_size
            if current_value < max_value:
                current_value -= overlap_size

        if self.leftover_handling == "fill" and current_value < max_value:
            coords.append(max_value - crop_size)

        return coords


def is_crop_negative(bounding_rects, roi):
    """
    Checks if the crop DOES NOT contain any part of the mask by comparing bounding rects.
    """
    for rect in bounding_rects:
        if overlap(rect, roi):
            return False
    return True


def is_crop_negative_by_mask(mask, roi):
    """
    Checks if the crop DOES NOT contain any part of the mask.
    """
    x, y, w, h = roi
    crop = mask[y:y+h, x:x+w]
    return cv.countNonZero(crop) == 0


def is_crop_valid(img_gray, roi):
    """
    Check if the crop doesn't lay too much off the real image area.
    """
    x, y, w, h = roi
    crop = img_gray[y:y + h, x:x + w]
    non_black = cv.countNonZero(crop)
    return 1.0 * non_black / (w * h) > 0.70


def get_abs_crop_sizes(img_shape, crop_sizes):
    img_size = min(img_shape[0], img_shape[1])
    abs_crops_sizes = [int(img_size * cs) for cs in crop_sizes]
    return abs_crops_sizes


def get_bounding_rects(mask):
    ret, thresh = cv.threshold(mask, 20, 255, 0)
    im, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bounding_rects = [cv.boundingRect(cnt) for cnt in contours]
    return bounding_rects


def overlap(r1, r2):
    """
    Overlapping rectangles overlap both horizontally & vertically
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    # return range_overlap(r1.left, r1.right, r2.left, r2.right) and range_overlap(r1.bottom, r1.top, r2.bottom, r2.top)
    return range_overlap(x1, x1+w1, x2, x2+w2) and range_overlap(y1, y1+h1, y2, y2+h2)


def range_overlap(a_min, a_max, b_min, b_max):
    """
    Neither range is completely greater than the other
    """
    return (a_min <= b_max) and (b_min <= a_max)
