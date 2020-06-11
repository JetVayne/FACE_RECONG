# 1. clone
```
git clone https://github.com/JetVayne/FACE_RECONG.git
```

# 移到此資料夾
```
cd Face-Feat-Tool-Glance
```

# 照著 environment.txt 安裝環境與套件

# 執行 test.py
```python=
from pprint import pprint as pp
from glance.face_analyst import FaceAnalyst

img_path = 'puff.jpg'

fa = FaceAnalyst(img_path, gen_ff_img=True)
fa.analyze()

if fa.result:
    pp(fa.data)
    print(fa.data.keys())

```

# 若有成功印出 dictionary 即成功

# 之後直接移動整個 glance 資料夾， import 即可