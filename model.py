
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

classes_ja = ["Tシャツ/トップ","ズボン","プルオーバー","ドレス","コート","サンダル","ワイシャツ","スニーカー","バッグ","アンクルブーツ"]
classes_en = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
n_class = len(classes_ja)
img_size = 28

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.cnn = nn.Sequential(
        nn.Conv2d(1,16,3, padding='same'),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16,32,3, padding='same'),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding='same'),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        nn.Flatten(1,-1),
        nn.Linear(3136, 128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128,10),
    )

  def forward(self, x):
    return self.cnn(x)

  def get_loss(self, x, t):
    y = self.forward(x)
    return F.cross_entropy(y, t)

# These lines should be outside the class definition
model = Model()

model.load_state_dict(
  torch.load("model_cnn.pth", map_location=torch.device("cpu")
))

def predict(img):
  img = img.convert("L")
  img = img.resize((img_size, img_size))
  transform = transforms.Compose([
      transforms.ToTensor(), transforms.Normalize((0.0),(1.0))
  ])

  img = transform(img)
  x = img.reshape(1,1,img_size,img_size)

  model.eval()
  y = model(x)

  y_prob = F.softmax(torch.squeeze(y))
  sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)

  return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
