"""Tunables to configure the augmentation."""
from tunable import Tunable


class BlurProbability(Tunable):
    """Probability to blur the image."""

    default = 0.25


class BlurMinimumSigma(Tunable):
    """Blurring minimum sigma value."""

    default = 0.8


class BlurMaximumSigma(Tunable):
    """Blurring maximum sigma value."""

    default = 2.5


class BlurSteps(Tunable):
    """Blurring levels between minimum and maxmimum."""

    default = 25


class ReflectProbability(Tunable):
    """Probability to apply any reflction."""

    default = 1.0


class ReflectXProbability(Tunable):
    """Probability to reflect the image horizontally."""

    default = 0.5


class ReflectYProbability(Tunable):
    """Probability to reflect the image vertically."""

    default = 0.5


class RotateProbability(Tunable):
    """Probability to rotate the image."""

    default = 0.25


class RotateMinimum(Tunable):
    """Minimum rotation angle."""

    default = -360.0


class RotateMaximum(Tunable):
    """Maximum rotation angle."""

    default = 360.0


class ShearProbability(Tunable):
    """Probability to apply any shear transformation."""

    default = 0.25


class ShearXAngleMinimum(Tunable):
    """Minimum horizontal shear angle."""

    default = -45.0


class ShearXAngleMaximum(Tunable):
    """Maximum horizontal shear angle."""

    default = 45.0


class ShearYAngleMinimum(Tunable):
    """Minimum vertical shear angle."""

    default = -45.5


class ShearYAngleMaximum(Tunable):
    """Maximum vertical shear angle."""

    default = 45.0


class ScaleProbability(Tunable):
    """Probability to rescale the image."""

    default = 0.25


class ScaleMinimum(Tunable):
    """Minimum rescaling factor."""

    default = 0.75


class ScaleMaximum(Tunable):
    """Maximum rescaleing factor."""

    default = 1.5
