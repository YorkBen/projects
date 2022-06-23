from PIL import Image
from torchvision import transforms
import os

input_dir = r'imgs0'
output_dir = r'imgs0_trans4'

# torchvision.transforms.ColorJitter(brightness=0.5)
# torchvision.transforms.ColorJitter(hue=0.5)
# torchvision.transforms.ColorJitter(contrast=0.5)
# torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# torchvision.transforms.Pad(padding=10, fill=0)
# torchvision.transforms.GaussianBlur(11, sigma=(0.1, 2.0))
# torchvision.transforms.RandomApply(transforms, p=0.5)
# torchvision.transforms.RandomChoice(transforms)

trans = transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(hue=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ColorJitter(saturation=0.5),
        transforms.GaussianBlur(11, sigma=(0.1, 2.0))

        # # transforms.Pad(padding=(4, 0)),
        # # transforms.Pad(padding=(0, 2)),
        # transforms.Pad(padding=(0, 0))
])

for f in os.listdir(input_dir):
    if f.split('.')[1] in ['jpg', 'tif']:
        img = Image.open(os.path.join(input_dir, f))
        img_trans = trans(img)
        img_trans.save(os.path.join(output_dir, f.replace('jpg', 'tif')))
