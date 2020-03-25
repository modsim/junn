from math import sqrt, atan

import cv2
import numpy as np

from tunable import Selectable


class ROIDetector(Selectable):
    def get_rois(self, image):
        threshold = 0.5
        image = image > threshold
        image = (image * 255).astype(np.uint8)

        # this problem is non-trivial unfortunately:
        # for matplotlib, pixels are -0.5 ... +0.5 wide, with integer coordinates at the center
        # for fiji, pixels are 0 ... 1 wide, with a +0.5 coordinate at the center

        # another problem: fiji expects single pixel regions to be surrounded by four points,
        # but opencv will reduce single pixel contours to a single coordinate

        # the solution is to upscale the image by 2x and work with that ...

        scaling_factor = 2
        repeated = np.repeat(np.repeat(image, scaling_factor, axis=0), scaling_factor, axis=1)
        contours, hierarchy = cv2.findContours(repeated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            # convert to float to allow for sub-pixel precision
            contour = contour.astype(np.float32)[:, 0, :]
            # divide by our scaling factor (2)
            contour /= float(scaling_factor)

            # round (upwards)
            contour = np.ceil(contour)
            # convert back to integer
            contour = contour.astype(np.int32)

            # --> contour is correct like this for ImageJ

            # for matplotlib, it would need to be shifted by 0.5!
            # contour -= 0.5

            yield contour


class RawROIDetector(ROIDetector, ROIDetector.Default):
    pass


# TODO: refactor. or remove?
class FilteringROIDetector(ROIDetector):
    calibration = 1.0

    @staticmethod
    def calculate_properties(contour, calibration=1.0):
        m = cv2.moments(contour)

        size = m['m00'] * (calibration * calibration)

        (center_x, center_y), (side_a, side_b), angle = cv2.minAreaRect(contour)
        width = min(side_a, side_b)
        length = max(side_a, side_b)

        point_test = cv2.pointPolygonTest(contour, (center_x, center_y), measureDist=False)

        solidity = cv2.contourArea(contour) / cv2.contourArea(cv2.convexHull(contour))

        # eccentricity
        try:
            epsilon = ((m['mu20'] - m['mu02']) * (m['mu20'] - m['mu02']) - 4.0 * (m['mu11'] * m['mu11'])) / (
                    (m['mu20'] + m['mu02']) * (m['mu20'] + m['mu02']))
        except ZeroDivisionError:  # …
            epsilon = float('nan')

        i_frag = sqrt(4.0 * m['mu11'] * m['mu11'] + (m['mu20'] - m['mu02']) * (m['mu20'] - m['mu02']))
        i_1 = 0.5 * ((m['mu20'] + m['mu02']) + i_frag)
        i_2 = 0.5 * ((m['mu20'] + m['mu02']) - i_frag)
        try:
            a = 2.0 * sqrt(i_1 / m['m00'])
        except ZeroDivisionError:
            a = float('nan')

        try:
            b = 2.0 * sqrt(i_2 / m['m00'])
        except ZeroDivisionError:
            b = float('nan')

        try:
            theta = 0.5 * atan((2 * m['mu11']) / (m['mu20'] - m['mu02']))
        except ZeroDivisionError:
            theta = float('nan')

        try:
            spread = (i_1 + i_2) / (m['m00'] * m['m00'])
        except ZeroDivisionError:
            spread = float('nan')
        try:
            elongation = (i_2 - i_1) / (i_1 + i_2)
        except ZeroDivisionError:
            elongation = float('nan')
        try:
            epsilon2 = sqrt((a * a) - (b * b)) / a
        except ZeroDivisionError:
            epsilon2 = float('nan')

        return dict(
            size=size,
            center_x=center_x,
            center_y=center_y,
            width=width,
            length=length,
            aspect=length / width,
            point_test=point_test,
            solidity=solidity,
            epsilon=epsilon,
            theta=theta,
            spread=spread,
            elongation=elongation,
            epsilon2=epsilon2
        )

    @staticmethod
    def find_mismatches(properties, ranges):
        mismatches = []
        for what, (low, high) in ranges.items():
            value = properties[what]

            if value < low or value > high:
                mismatches.append(what)
        return mismatches

    def get_rois(self, image):
        debug = False
        # debug = True

        if debug:
            def debug_out(msg):
                print(msg, end='')
        else:
            def debug_out(msg):
                pass

        float_contours = True
        smoothing_window, smoothing_times = 3, 3

        filtering = True

        filter_dict = dict(
            aspect=(0, 10),
            size=(0.25, 100),
            solidity=(0.75, 1.0),
            epsilon=(-float('inf'), 0.8)
        )

        for contour in super().get_rois(image):
            if filtering:
                properties = self.calculate_properties(contour, calibration=self.calibration)
                mismatches = self.find_mismatches(properties, filter_dict)
                if mismatches:
                    debug_out(''.join(mismatches))
                    continue

            if float_contours:
                contour = contour.astype(np.float32)

            if smoothing_window:
                for _ in range(smoothing_times):
                    from scipy.ndimage import uniform_filter1d
                    contour = uniform_filter1d(contour, smoothing_window, axis=0, mode='wrap')

            yield contour
