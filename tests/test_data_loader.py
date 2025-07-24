from src.data_loader import get_transforms

def test_transforms_returns_transform():
    transform = get_transforms(image_size=150)
    assert callable(transform)