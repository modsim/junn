"""Losses as mixins."""
from ...common.losses import dice_loss


class DiceLoss:
    """Mixin to set a NeuralNetwork loss to the Dice loss."""

    # noinspection PyMethodMayBeStatic
    def get_loss(self):  # noqa: D102
        return dice_loss
