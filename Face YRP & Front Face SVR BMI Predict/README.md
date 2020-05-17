---
tags: Face Recong, Head Pose, Img-Nrlz, Face YRP
---

# 2020 / 05 / 14 - ML Method Document

## Result Display

### Face YRP
![](https://i.imgur.com/B2Mzhg8.jpg)

### FF1 - CJWR
#### cheekbone to jaw width - refer to paper
![](https://i.imgur.com/TVmhLe8.jpg)

### FF2 - WHR
#### jaw width to upper facial height ratio - refer to paper
![](https://i.imgur.com/kdzdvr4.jpg)

### FF3 - PAR
#### perimeter to area ratio - refer to paper
![](https://i.imgur.com/jJY1GzE.jpg)

### FF4 - ES
#### eye size - refer to paper
![](https://i.imgur.com/MiJ193M.jpg)

### FF5 - FWR
#### face width to lower face height ratio - refer to paper
![](https://i.imgur.com/wsxHv3k.jpg)

### FF6 - MEH
#### mean of eyebrow height - refer to paper
![](https://i.imgur.com/bRUsLAF.jpg)

### FF7 - CFR
#### cheek fat ratio - add by Jet
#### ( (Face Area - Face Pyramid) / Face Area ) / Face Height Ratio
![](https://i.imgur.com/abfl4G2.jpg)

## Main Point
- 若現在直接用網路上提供的方法將 2D 照片中的臉轉正的話，可能會改變不少臉部特徵的相對關係

- 主要是因為一些臉部深度上的特徵，無法簡單的處理，EX: 硬是將臉轉正的話，鼻子可以能會歪掉

- 所以目前的原則是盡量不改變原本的照片資料，利用 Yaw, Roll, Pitch (YRP 偏航軸) 計算 DataSet-A 中每一張臉的 YRP 角度

- 由於 Roll 是比較好處理的，所以只針對 Pitch, Yaw 設 threshold, 只取 ± 10° ，這樣我們就能先將DataSet-A 中，偏轉幅度較小的臉先抓出來

- 透過這個方法，我們從 DataSet-A 的 759 張照片中，得到了 256 張正臉，約為整個資料集的 33%

- 下一步，參考[論文](https://hackmd.io/i2RKj4fhR8OJO0qv55Wa-w) 使用了7個臉部特徵並用 SVR 去預測照片中人臉的 BMI 值

- 由於論文中的 LMK 取得方法不是 DLib，我有刪除一個特徵，並改為一個我自己新定義的特徵

- 平均誤差為 9% - 13%，而有些資料單筆的誤差會達到 20% 這是目前遇到的問題，而且還要趕快找出將偏轉角度大的照片正規畫的方法

<br>

## Next
1. 探討使用的臉部特徵是否合理

2. obj model 的優化

3. 是否考慮直接使用 3D 開源套件

<br>
