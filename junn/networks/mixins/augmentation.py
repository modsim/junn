from tunable import Tunable


class BlurProbability(Tunable):
    default = 0.25


class BlurMinimumSigma(Tunable):
    default = 0.8


class BlurMaximumSigma(Tunable):
    default = 2.5


class BlurSteps(Tunable):
    default = 25


class ReflectProbability(Tunable):
    default = 1.0


class ReflectXProbability(Tunable):
    default = 0.5


class ReflectYProbability(Tunable):
    default = 0.5


class RotateProbability(Tunable):
    default = 0.25


class RotateMinimum(Tunable):
    default = -360.0


class RotateMaximum(Tunable):
    default = 360.0


class ShearProbability(Tunable):
    default = 0.25


class ShearXAngleMinimum(Tunable):
    default = -45.0


class ShearXAngleMaximum(Tunable):
    default = 45.0


class ShearYAngleMinimum(Tunable):
    default = -45.5


class ShearYAngleMaximum(Tunable):
    default = 45.0


class ScaleProbability(Tunable):
    default = 0.25


class ScaleMinimum(Tunable):
    default = 0.75


class ScaleMaximum(Tunable):
    default = 1.5
