from .detr import DETR


def build_model(backbone, transformer, num_classes):
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=50,
        aux_loss=False,
    )

    return model