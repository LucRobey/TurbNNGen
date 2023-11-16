import torch

if __name__ == "__main__":
    print("== Test Cuda ==")
    print("Cuda is available ?", torch.cuda.is_available())
    print("Device name :", torch.cuda.get_device_name(0))