from src.data_loader import get_transform

def test-transforms_returns_transform():
    transform = get_transforms(image_size=150)
    assert callable(transform)