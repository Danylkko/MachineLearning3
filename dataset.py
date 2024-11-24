import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from transformers import GPT2TokenizerFast
import PIL.Image

class ImageCaptionDataset(Dataset):
    def __init__(self, images: np.array, captions: np.array, data_folder):
        self.images = images
        self.captions = captions
        self.data_folder = data_folder
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.bos_token = self.tokenizer.eos_token
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = f"{self.data_folder}/{self.images[idx]}"
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption = self.tokenizer.bos_token + self.captions[idx] + self.tokenizer.eos_token

        tokens = self.tokenizer(caption, return_tensors='pt', padding='max_length', max_length=256, truncation=True,
                                add_special_tokens=False)
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        return image, input_ids, attention_mask
