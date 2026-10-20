
"""Acceptance tests for image upload edge cases."""

from io import BytesIO

from PIL import Image

import app as app_module


def create_test_image(width=64, height=64, noisy=False):
    """Generate valid JPEG image data."""
    if noisy:
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


def upload_image(client, image_data, filename, headers=None):
    """Submit an image using the Flask test client."""
    return client.post(
        "/prediction",
        data={"file": (image_data, filename)},
        content_type="multipart/form-data",
        headers=headers,
    )


def assert_successful_prediction(response, digit):
    """Verify that the response displays the expected digit."""
    assert response.status_code == 200
    assert f">{digit}</h2>".encode() in response.data
    assert b"File cannot be processed." not in response.data


def test_edge_case_large_image_upload(client, monkeypatch):
    """Verify successful processing of a large valid JPEG."""
    monkeypatch.setattr(
        app_module, "predict_result", lambda _image: 2
    )

    image_data = create_test_image(
        width=1400, height=1400, noisy=True
    )

    assert len(image_data.getvalue()) > 100_000

    response = upload_image(
        client, image_data, "large_image.jpg"
    )

    assert_successful_prediction(response, 2)


def test_edge_case_invalid_metadata(client, monkeypatch):
    """Verify a valid image without EXIF metadata."""
    monkeypatch.setattr(
        app_module, "predict_result", lambda _image: 3
    )

    image_data = create_test_image()

    with Image.open(BytesIO(image_data.getvalue())) as image:
        assert len(image.getexif()) == 0

    response = upload_image(
        client, image_data, "no_metadata.jpg"
    )

    assert_successful_prediction(response, 3)


def test_edge_case_non_standard_image_extensions(
    client, monkeypatch
):
    """Verify valid JPEG data with an unusual extension."""
    monkeypatch.setattr(
        app_module, "predict_result", lambda _image: 4
    )

    response = upload_image(
        client,
        create_test_image(),
        "hand_sign.unknown",
    )

    assert_successful_prediction(response, 4)


def test_edge_case_sequential_image_uploads(
    client, monkeypatch
):
    """Verify two consecutive uploads return distinct results."""
    predictions = iter([5, 6])

    monkeypatch.setattr(
        app_module,
        "predict_result",
        lambda _image: next(predictions),
    )

    response1 = upload_image(
        client, create_test_image(), "first.jpg"
    )

    response2 = upload_image(
        client, create_test_image(), "second.jpg"
    )

    assert_successful_prediction(response1, 5)
    assert_successful_prediction(response2, 6)


def test_edge_case_unexpected_headers(client, monkeypatch):
    """Verify valid uploads with an additional HTTP header."""
    monkeypatch.setattr(
        app_module, "predict_result", lambda _image: 7
    )

    response = upload_image(
        client,
        create_test_image(),
        "header_test.jpg",
        headers={"X-Unexpected-Header": "value"},
    )

    assert_successful_prediction(response, 7)


def test_edge_case_empty_file_upload(client):
    """Verify that an empty image file is rejected."""
    response = upload_image(
        client, BytesIO(b""), "empty.jpg"
    )

    assert response.status_code == 400
    assert b"File cannot be processed." in response.data
