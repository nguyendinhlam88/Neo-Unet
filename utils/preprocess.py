from PIL import Image
import torchvision.transforms as T

def preprocess(img_path, input_size):
    img = Image.open(img_path)
    img = img.resize((input_size[1], input_size[0]), Image.BICUBIC)
    img = T.ToTensor()(img)
    img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    return img