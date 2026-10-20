
"""Integration tests for successful image predictions."""

from io import BytesIO

from PIL import Image

import app as app_module


def test_successful_prediction(client, monkeypatch):
    """Verify upload, preprocessing, and response rendering."""
    image = Image.new(
        "RGB", (80, 60), color="white"
    )

    image_data = BytesIO()
    image.save(image_data, format="JPEG")
    image_data.seek(0)

    def mock_prediction(processed_image):
        """Verify the image was preprocessed correctly."""
        assert processed_image.shape == (1, 224, 224, 3)
        assert processed_image.min() >= 0
        assert processed_image.max() <= 1
        return 6

    monkeypatch.setattr(
        app_module, "predict_result", mock_prediction
    )

    response = client.post(
        "/prediction",
        data={"file": (image_data, "test.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b">6</h2>" in response.data
    assert b"File cannot be processed." not in response.data
