# Creating a Custom Dataset for your files

- Custom Dataset은 (\_\_init\_\_, \_\_len\_\_, \_\_getitem\_\_)이 세가지 함수를 구현해야한다.

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

### \_\_init\_\_

- \_\_init\_\_함수는 Dataset 객체가 생성(instantiate)될 때 한번만 실행된다. 여기서 이미지와 주석 파일(annotation_file)이 포함된 디렉토리와 두가지 변형(transform)을 초기화한다.

> labels.csv 파일은 다음과 같다.

```python
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

```python
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
```

### \_\_len\_\_

- **len**함수는 데이터셋의 샘플 개수를 반환한다.

```python
def __len__(self):
    return len(self.img_labels)
```

### \_\_getitem\_\_

- \_\_getitem\_\_함수는 주어진 인덱스 idx에 해당하는 샘플을 데이터셋에서 불러오고 반환한다. 인덱스 기반으로, 디스크에서 이미지의 위치를 식별하고, read_image를 사용하여 이미지를 테서로 변환하고, self.img_labels의 csv데이터로부터 해당하는 정답(label)을 가져오고, (해당하는 경우)변형(transform) 함수들을 호출한 뒤, 텐서 이미지와 라벨을 Python dict형으로 변환한다.

```python
def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    sample = {"image": image, "label": label}
    return sample
```
