from ...common.losses import dice_loss


class DiceLoss:
    # noinspection PyMethodMayBeStatic
    def get_loss(self):
        return dice_loss
