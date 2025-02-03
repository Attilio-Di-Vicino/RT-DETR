# RT-DETR R50VD 6x COCO Model

## 📌 Model Overview
This repository uses **RT-DETR R50VD 6x COCO** as the primary model for real-time object detection. RT-DETR is a transformer-based object detector designed for high-speed inference without Non-Maximum Suppression (NMS), outperforming YOLO models in both speed and accuracy.

The chosen model:  
**`rtdetr_r50vd_6x_coco_from_paddle.pth`**  
- **Backbone:** ResNet-50VD
- **Training Epochs:** 6x (~72 epochs)
- **Dataset:** COCO
- **Converted from:** PaddlePaddle

This model provides a **balance between speed and accuracy**, making it ideal for real-time applications.

---

## 🔍 Why This Model?
Among multiple RT-DETR models, `rtdetr_r50vd_6x_coco_from_paddle.pth` was chosen for the following reasons:

- **Accuracy:** Higher precision than smaller backbone models (e.g., ResNet-18/34VD) while maintaining efficiency.
- **Speed:** Faster inference than larger models (e.g., ResNet-101VD) while still achieving competitive AP (Average Precision).
- **Generalization:** Trained on the **COCO dataset**, which provides a diverse set of object categories and real-world scenarios.
- **Stable Performance:** Trained for 6x epochs, ensuring better convergence compared to 1x or 2x models.

---

## 🔄 Alternative Model Choices
If different requirements arise, here are other RT-DETR models and their key differences:

| Model | Size | Speed | Accuracy | Best Use Case |
|--------|------|-------|----------|--------------|
| `rtdetr_r18vd_6x_coco.pth` | 77 MB | ⚡ Fastest | 🔹 Lower AP | Embedded systems, low-power devices |
| `rtdetr_r34vd_6x_coco_from_paddle.pth` | 120 MB | ⚡ Fast | 🔹 Medium AP | Balanced efficiency |
| **`rtdetr_r50vd_6x_coco_from_paddle.pth`** | **164 MB** | 🏆 Best Balance | 🏆 High AP | **Real-time detection with good accuracy** |
| `rtdetr_r101vd_6x_coco_from_paddle.pth` | 293 MB | 🐢 Slower | 🚀 Higher AP | High-precision applications |
| `rtdetr_r50vd_m_6x_coco_from_paddle.pth` | 140 MB | ⚡ Optimized | 🔹 Slightly lower AP | Lightweight version of R50VD |

- **Choose R18VD if** you need **speed over accuracy**.
- **Choose R101VD if** you need **maximum accuracy** and can tolerate slower inference.
- **Choose R50VD (this model)** if you need the **best balance** of speed and accuracy.

---

## 🚀 Conclusion
The **RT-DETR R50VD 6x COCO model** is an excellent choice for real-time object detection, providing a strong balance between inference speed and accuracy. It is particularly well-suited for applications requiring real-time detection without the complexity of Non-Maximum Suppression (NMS).