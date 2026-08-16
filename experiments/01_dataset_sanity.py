from utils.datasets import get_cifar10_loader


def main():
    loader = get_cifar10_loader(
        batch_size=64,
        train=False,
    )

    images, labels = next(iter(loader))

    print("Images:", images.shape)
    print("Labels:", labels.shape)


if __name__ == "__main__":
    main()