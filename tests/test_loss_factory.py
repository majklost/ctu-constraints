import torch

from constraints.datatools.label_schema import LabelSchema
from constraints.factories.losses import available_loss_presets, create_loss_computer
from constraints.types import FieldParams, LossInput, TransformSpec

LABEL_SCHEMA = LabelSchema.from_lists(
    ["background", "boundary", "lumen"],
    [(0.0, 0.0, 0.0), (0.9, 0.1, 0.1), (0.1, 0.7, 0.1)],
)


def test_every_loss_preset_creates_a_named_composite() -> None:
    for preset in available_loss_presets():
        computer = create_loss_computer(preset, LABEL_SCHEMA)
        assert computer.preset_name == preset
        assert len(computer.terms) == 2


def test_factory_adds_regularization_as_an_explicit_weighted_term() -> None:
    computer = create_loss_computer(
        "bce_one_side_sdf_squared",
        LABEL_SCHEMA,
        field_regularization_weight=0.25,
    )

    assert computer.weights == [20.0, 1.0, 0.25]
    assert computer.terms[-1].name == "registration/deformation_gradient"

    field = torch.zeros((1, 2, 4, 4), requires_grad=True)
    with torch.no_grad():
        field[:, 0] = torch.arange(4).view(1, 4, 1)
    loss = computer.terms[-1](
        LossInput(transform_spec=TransformSpec(field=FieldParams(field)))
    )
    assert loss > 0
