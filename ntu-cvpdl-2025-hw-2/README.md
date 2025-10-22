# CVPDL HW2 - YOLO Object Detection

這是一個完整的 YOLO 物件檢測流程，將 CVPDL_hw2 數據集轉換為 YOLO 格式並訓練模型。

## 📁 檔案結構

```
ntu-cvpdl-2025-hw-2/
├── CVPDL_hw2/
│   └── CVPDL_hw2/
│       ├── train/          # 訓練數據（圖片+標註）
│       │   ├── img0001.png
│       │   ├── img0001.txt # 格式: <class>,<x>,<y>,<w>,<h>
│       │   └── ...
│       └── test/           # 測試數據（僅圖片）
│           ├── img0001.png
│           └── ...
├── convert_to_yolo.py      # 數據轉換腳本
├── generate_submission.py  # 生成提交文件腳本
├── run_pipeline.sh         # 完整流程執行腳本
└── sample_submission.csv   # 提交格式範例
```

## 🚀 快速開始

### 方法 1：一鍵執行完整流程（推薦）

```bash
bash run_pipeline.sh
```

這個腳本會自動完成：
1. 轉換數據集為 YOLO 格式
2. 訓練 YOLO 模型
3. 對測試集進行預測
4. 生成 submission.csv

### 方法 2：分步執行

#### 步驟 1：轉換數據集

```bash
python convert_to_yolo.py
```

這會創建 `yolo_dataset/` 目錄，包含：
- `train/images` 和 `train/labels` - 訓練集（90%）
- `val/images` 和 `val/labels` - 驗證集（10%）
- `test/images` - 測試集
- `dataset.yaml` - YOLO 配置文件

#### 步驟 2：訓練模型

```bash
cd /home/young_wu/2025-CVPDL/ntu-cvpdl-2025-hw-2

# 基礎訓練
yolo detect train \
    model=yolo11n.pt \
    data=yolo_dataset/dataset.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16

# 進階訓練（更高精度）
yolo detect train \
    model=yolo11m.pt \
    data=yolo_dataset/dataset.yaml \
    epochs=200 \
    imgsz=640 \
    batch=8 \
    patience=30 \
    optimizer=AdamW \
    lr0=0.001
```

訓練完成後，最佳模型會保存在 `runs/detect/train/weights/best.pt`

#### 步驟 3：預測測試集

```bash
# 使用訓練好的模型進行預測
yolo predict \
    model=runs/detect/train/weights/best.pt \
    source=CVPDL_hw2/CVPDL_hw2/test \
    save_txt=True \
    save_conf=True \
    conf=0.25 \
    project=runs/detect \
    name=predict \
    exist_ok=True
```

#### 步驟 4：生成提交文件

```bash
python generate_submission.py
```

這會生成 `submission.csv`，格式為：
```csv
Image_ID,PredictionString
1,<conf_1> <bb_left_1> <bb_top_1> <bb_width_1> <bb_height_1> <class_1> <conf_2> ...
2,...
```

## 📊 數據格式說明

### 輸入格式（原始數據）
```
# img0001.txt
0,779,276,26,60
0,680,261,35,52
```
- 格式：`<class>,<x>,<y>,<width>,<height>`
- 座標：絕對像素值，左上角為原點

### YOLO 格式（轉換後）
```
# img0001.txt
0 0.415625 0.275000 0.013542 0.062500
0 0.354167 0.270833 0.018229 0.054167
```
- 格式：`<class> <x_center> <y_center> <width> <height>`
- 座標：歸一化到 [0, 1]，中心點座標

### 提交格式（最終輸出）
```csv
Image_ID,PredictionString
1,0.95 100.5 200.3 50.2 80.1 0 0.88 300.1 150.5 60.3 90.2 0
```
- 格式：`<conf> <bb_left> <bb_top> <bb_width> <bb_height> <class>`
- 多個檢測框用空格分隔

## ⚙️ 訓練參數調整

### 模型選擇
- `yolo11n.pt` - Nano（最快，準確度較低）
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium（推薦，平衡性能）
- `yolo11l.pt` - Large
- `yolo11x.pt` - XLarge（最慢，準確度最高）

### 關鍵參數
```bash
epochs=100          # 訓練輪數（建議 100-200）
imgsz=640          # 圖片大小（640 或 1280）
batch=16           # 批次大小（根據 GPU 記憶體調整）
conf=0.25          # 信心閾值（0.1-0.5，越低檢測越多）
iou=0.45           # IoU 閾值用於 NMS
patience=20        # Early stopping patience
lr0=0.01           # 初始學習率
```

### 性能優化建議

1. **提高準確度**：
   - 使用更大的模型（yolo11m 或 yolo11l）
   - 增加訓練輪數（200-300 epochs）
   - 調整信心閾值（嘗試 0.15-0.35）
   - 使用更大的圖片尺寸（1280）

2. **加快訓練**：
   - 使用較小的模型（yolo11n 或 yolo11s）
   - 減少批次大小
   - 使用混合精度訓練：`amp=True`

3. **數據增強**（自動啟用）：
   - 隨機翻轉、旋轉
   - 顏色抖動
   - Mosaic 增強

## 📈 監控訓練

訓練過程會生成以下文件：
- `runs/detect/train/weights/best.pt` - 最佳模型
- `runs/detect/train/weights/last.pt` - 最後一個 epoch 的模型
- `runs/detect/train/results.png` - 訓練曲線圖
- `runs/detect/train/confusion_matrix.png` - 混淆矩陣

使用 TensorBoard 查看訓練過程：
```bash
tensorboard --logdir runs/detect/train
```

## 🔍 驗證預測結果

檢查生成的標籤：
```bash
ls runs/detect/predict/labels/
head runs/detect/predict/labels/img0001.txt
```

可視化預測結果（會保存在 runs/detect/predict/）：
```bash
yolo predict \
    model=runs/detect/train/weights/best.pt \
    source=CVPDL_hw2/CVPDL_hw2/test/img0001.png \
    save=True \
    conf=0.25
```

## 🐛 常見問題

### 1. CUDA 記憶體不足
```bash
# 減少批次大小
batch=8  # 或更小

# 或使用 CPU
device=cpu
```

### 2. 訓練時間太長
```bash
# 使用更小的模型
model=yolo11n.pt

# 減少訓練輪數
epochs=50
```

### 3. 檢測結果太少/太多
```bash
# 調整信心閾值
conf=0.15  # 更多檢測
conf=0.40  # 更少檢測
```

### 4. 驗證集準確度低
- 檢查數據標註是否正確
- 增加訓練輪數
- 使用更大的模型
- 調整學習率

## 📝 注意事項

1. **類別數量**：當前設定為單類別檢測（class=0），如果有多個類別需要修改
2. **數據驗證**：訓練前建議檢查轉換後的 YOLO 標註是否正確
3. **測試集預測**：測試集沒有標註文件是正常的
4. **提交格式**：確保 submission.csv 格式與 sample_submission.csv 一致

## 🎯 進階技巧

### 使用預訓練模型微調
```bash
yolo detect train \
    model=runs/detect/train/weights/best.pt \
    data=yolo_dataset/dataset.yaml \
    epochs=50 \
    lr0=0.0001  # 較小的學習率
```

### 集成多個模型
訓練多個模型並對結果進行平均或投票可以提高準確度。

### 超參數搜索
```bash
yolo detect train \
    model=yolo11n.pt \
    data=yolo_dataset/dataset.yaml \
    epochs=100 \
    optimizer=auto \
    auto_augment=auto
```

## 📊 預期結果

- 訓練集圖片：~855 張（90%）
- 驗證集圖片：~95 張（10%）
- 測試集圖片：550 張
- 訓練時間：約 20-60 分鐘（取決於硬體和參數）

訓練完成後的 mAP（mean Average Precision）應該能達到 0.7+ 以上。

## 🔗 相關資源

- [Ultralytics YOLO 官方文檔](https://docs.ultralytics.com/)
- [YOLO 訓練教程](https://docs.ultralytics.com/modes/train/)
- [YOLO 預測教程](https://docs.ultralytics.com/modes/predict/)

Good luck! 🚀
