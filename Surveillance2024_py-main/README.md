# 居家安全好幫手: 危險檢測，社交人臉識別，表情識別

項目簡介。

## 開始之前

請確保您已經安裝了 [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) 和 [Poetry](https://python-poetry.org/docs/#installation)。本項目使用 Python 3.9 和 Poetry 來管理依賴。
或用conda安裝poetry

## 環境設置

### 創建 Conda 虛擬環境

首先，使用 Conda 創建一個新的虛擬環境，運行以下命令：

```bash
conda create --name your_env_name python=3.9
```
激活您的虛擬環境：
```bash
conda activate your_env_name
```
### 安裝poetry
```bash
conda install conda-forge::poetry
```
到項目目錄
```bash
poetry install
```
### 運行項目
```bash
poetry run python main.py
```
### 創建自己家人的數據集
範例:
```markdown
datasets
└── NewJeans
    ├── Hyein
    │   ├── Hyein0.jpg
    │   ├── Hyein1.jpg
    │   ├── Hyein2.jpg
    │   ├── Hyein3.jpg
    │   └── Hyein4.jpg
    ├── Minji
    │   ├── Minji0.jpg
    │   ├── Minji1.jpg
    │   ├── Minji2.jpg
    │   ├── Minji3.jpg
    │   └── Minji4.jpg
    ├── Hanni
    │   ├── Hanni0.jpg
    │   ├── Hanni1.jpg
    │   ├── Hanni2.jpg
    │   ├── Hanni3.jpg
    │   └── Hanni4.jpg
    ├── Danielle
    │   ├── NewJeansDanielle0.jpg
    │   ├── NewJeansDanielle1.jpg
    │   ├── NewJeansDanielle2.jpg
    │   ├── NewJeansDanielle3.jpg
    │   └── NewJeansDanielle4.jpg
    └── Haerin
        ├── Haerin0.jpg
        ├── Haerin1.jpg
        ├── Haerin2.jpg
        ├── Haerin3.jpg
        └── Haerin4.jpg

```
```markdown
datasets
└── family
    ├── family_member1
    │   ├── family_member1_0.jpg
    │   ├── family_member11.jpg
    │   ├── family_member12.jpg
    │   ├── family_member13.jpg
    │   └── family_member14.jpg
    ├── family_member2
    │   ├── family_member20.jpg
    │   ├── family_member21.jpg
    │   ├── family_member22.jpg
    │   ├── family_member23.jpg
    │   └── family_member24.jpg


```

將自己家人的圖片按照上面方式放到資料夾裡(不用編號，資料夾名對照家人名稱就行):

再來，修改 face__recognition.py中
```python
def encode_faces(base_path="datasets/NewJeans/"):
    ...
```
的base_path，將的base_path改成你的family的資料夾的路徑
如:

```python
def encode_faces(base_path="datasets/family/"):
    ...
```
註: 5到15張都可，盡量少而精，質量好的照片，增加數量會增加運算效能

