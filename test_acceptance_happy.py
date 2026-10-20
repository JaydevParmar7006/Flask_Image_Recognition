
"""Acceptance tests for successful image uploads."""

from io import BytesIO

from PIL import Image

import app as app_module


def create_test_image(width=64, height=64, use_noise=False):
    """Create a valid JPEG image in memory."""
    if use_noise:
        image = Image.effect_noise(
            (width, height), 100
        ).convert("RGB")
    else:
        image = Image.new(
            "RGB", (width, height), color="white"
        )

    image_data = BytesIO()
    image.save(image_data, format="JPEG", quality=90)
    image_data.seek(0)
    return image_data


def test_acceptance_successful_upload(client, monkeypatch):
    """Verify a valid JPEG upload returns a prediction."""
    monkeypatch.setattr(
        app_module, "predict_result", lambda _image: 3
    )

    response = client.post(
        "/prediction",
        data={
            "file": (create_test_image(), "test_image.jpg")
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b">3</h2>" in response.data
    assert b"File cannot be processed." not in response.data


def test_acceptance_valid_large_image(client, monkeypatch):
    """Verify a valid, relatively large JPEG can be processed."""
    monkeypatch.setattr(
        app_module, "predict_result", lambda _image: 5
    )

    image_data = create_test_image(
        width=1600, height=1600, use_noise=True
    )

    assert len(image_data.getvalue()) > 100_000

    response = client.post(
        "/prediction",
        data={"file": (image_data, "large_image.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b">5</h2>" in response.data
    assert b"File cannot be processed." not in response.data


def test_acceptance_valid_image_size_upload(client, monkeypatch):
    """Verify a 1280 by 720 image is accepted."""
    monkeypatch.setattr(
        app_module, "predict_result", lambda _image: 7
    )

    image_data = create_test_image(
        width=1280, height=720
    )

    response = client.post(
        "/prediction",
        data={"file": (image_data, "resolution_test.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b">7</h2>" in response.data
    assert b"File cannot be processed." not in response.data
