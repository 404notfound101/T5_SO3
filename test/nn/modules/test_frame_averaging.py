import pytest
import torch
from GraphT5_3D.nn.modules.frame_averaging import (
    create_frames,
    apply_frames,
    invert_frames,
)


def random_rotation_matrix_torch():
    # Create a random 3x3 matrix
    random_matrix = torch.rand(3, 3)

    # Perform QR decomposition
    q, _ = torch.linalg.qr(random_matrix)

    # Ensure the determinant is 1
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]

    return q


@pytest.fixture
def mock_input():
    return {
        "x": torch.randn(2, 3, 3, 3),
        "common_token_mask": torch.ones(2, 3),
    }


@pytest.fixture
def mock_MLP():
    return torch.nn.Sequential(
        torch.nn.Linear(9, 256), torch.nn.ReLU(), torch.nn.Linear(256, 9)
    )


@pytest.fixture
def random_rotation_matrix():
    return random_rotation_matrix_torch()


def test_create_frames(mock_input):
    input_tensor = mock_input["x"]
    input_mask = mock_input["common_token_mask"]
    frame, center = create_frames(input_tensor, input_mask)
    padded_input_tensor = torch.cat([input_tensor, torch.zeros(2, 1, 3, 3)], dim=1)
    padded_input_mask = torch.cat([input_mask, torch.tensor([[0.0], [1.0]])], dim=1)
    padded_frame, padded_center = create_frames(padded_input_tensor, padded_input_mask)
    assert torch.allclose(frame[0], padded_frame[0])
    assert torch.allclose(center[0], padded_center[0])
    assert not torch.allclose(frame[1], padded_frame[1])
    assert not torch.allclose(center[1], padded_center[1])


def test_equivariance(mock_input, mock_MLP, random_rotation_matrix):
    input_tensor = mock_input["x"]
    input_mask = mock_input["common_token_mask"]
    frame, center = create_frames(input_tensor, input_mask)
    framed_input = apply_frames(input_tensor, input_mask, frame, center)
    output = mock_MLP(framed_input)
    output_reverted = invert_frames(output, input_mask, frame, center).reshape(
        input_tensor.shape
    )

    rotated_input = input_tensor @ random_rotation_matrix
    frame_rotated, center_rotated = create_frames(rotated_input, input_mask)
    rotated_framed_input = apply_frames(
        rotated_input, input_mask, frame_rotated, center_rotated
    )
    output_rotated = mock_MLP(rotated_framed_input)
    output_rotated_reverted = invert_frames(
        output_rotated, input_mask, frame_rotated, center_rotated
    ).reshape(input_tensor.shape)

    assert torch.allclose(
        output_reverted @ random_rotation_matrix, output_rotated_reverted, atol=1e-5
    )
    assert torch.allclose(
        output.reshape(2, 4, -1).mean(dim=1),
        output_rotated.reshape(2, 4, -1).mean(dim=1),
    )
