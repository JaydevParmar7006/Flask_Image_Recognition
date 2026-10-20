
"""Regression tests for the Flask image recognition application."""

from io import BytesIO

from PIL import Image

import app as app_module


def create_valid_image():
    """Create a valid JPEG image for testing."""
    image = Image.new("RGB", (64, 64), color="white")
    image_data = BytesIO()
    image.save(image_data, format="JPEG")
    image_data.seek(0)
    return image_data


def test_home_page_accessible(client):
    """Verify that the application home page loads."""
    response = client.get("/")

    assert response.status_code == 200
    assert b"Hand Sign Digit Language Detection" in response.data


def test_valid_image_prediction(client, monkeypatch):
    """Verify that a valid upload displays a predicted digit."""
    monkeypatch.setattr(
        app_module, "predict_result", lambda _image: 4
    )

    response = client.post(
        "/prediction",
        data={"file": (create_valid_image(), "hand.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b">4</h2>" in response.data
    assert b"File cannot be processed." not in response.data


def test_missing_file_returns_client_error(client):
    """Verify that missing uploads receive HTTP 400."""
    response = client.post(
        "/prediction",
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"File cannot be processed." in response.data


def test_invalid_image_returns_client_error(client):
    """Verify that invalid image content is rejected."""
    invalid_image = BytesIO(b"fake_image_data")

    response = client.post(
        "/prediction",
        data={"file": (invalid_image, "invalid.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"File cannot be processed." in response.data


def test_prediction_requires_post(client):
    """Verify that GET requests cannot submit predictions."""
    response = client.get("/prediction")

    assert response.status_code == 405
