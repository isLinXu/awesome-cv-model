# awesome-cv-model

awesome cv model

# 计算机视觉-算法整理列表

---

基于对算法研究和应用的长期实践经验,总结了一份深度学习时代计算机视觉炼丹师不可错过的算法列表。

我们按照不同方向，为大家推荐了每个计算机视觉方向必须了解的典型算法。

此列表对于深度学习算法，从学术、科研、生态、产业应用等角度，综合考量了各个方面，归纳为了两个评价维度：即**算法影响力**和**算法性能**。基于算法的综合得分，我们会划分 P0、P1 和 P2 算法，作为最终的列表。其中，第一版分级列表涉及的计算机视觉方向包括：**图像分类**、**目标检测**、**实例分割**、**语义分割**、**动作识别**、**2D 姿态估计**和 **OCR** ，共7个主流方向。

这份算法列表用途举例如下：

| 算法列表用途         |
| -------------------- |
| - 业务算法参考       |
| - 训练框架适配和评测 |
| - 推理引擎适配和评测 |
| - 训练芯片适配和评测 |
| - 集群环境评测       |

![image](https://github.com/isLinXu/issues/assets/59380685/ebeb9542-268e-4e75-afb2-75a278c5dee5)

## 一、分级推荐列表

### 1、图像分类

| 方向         | 算法             | 论文名                                                       | 推荐优先级 |
| :----------- | :--------------- | :----------------------------------------------------------- | :--------- |
| **图像分类** | ResNet           | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | P0         |
|              | MobileNet V2     | [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) | P0         |
|              | ViT              | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | P0         |
|              | Swin Transformer | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | P0         |
|              | EfficientNet     | [Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | P1         |
|              | DenseNet         | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) | P1         |
|              | SENets           | [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) | P1         |
|              | RepVGG           | [Repvgg: Making vgg-style convnets great again](https://arxiv.org/abs/2101.03697) | P1         |
|              | InceptionV3      | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | P2         |
|              | VGGNet           | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | P2         |
|              | ShuffleNetV2     | [Shufflenet v2: Practical guidelines for efficient cnn architecture design](https://arxiv.org/abs/1807.11164) | P2         |

### 2、目标检测

|   **任务**   |   **算法**    | **论文名**                                                   | 推荐优先级 |
| :----------: | :-----------: | :----------------------------------------------------------- | :--------- |
| **目标检测** |      SSD      | [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) | P0         |
|              |    YOLOV3     | [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) | P0         |
|              |   CenterNet   | [Objects as Points](https://arxiv.org/abs/1904.07850)        | P0         |
|              | Faster R-CNN  | [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) | P0         |
|              |   RetinaNet   | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) | P0         |
|              |     FCOS      | [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) | P1         |
|              |     ATSS      | [Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection](https://arxiv.org/abs/1912.02424) | P1         |
|              | Cascade R-CNN | [Cascade R-CNN: High Quality Object Detection and Instance Segmentation](https://arxiv.org/abs/1906.09756) | P2         |
|              |     DETR      | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | P2         |

### 3、实例分割

|     任务     |    算法    | 论文名                                                       | 推荐优先级 |
| :----------: | :--------: | :----------------------------------------------------------- | :--------- |
| **实例分割** | Mask R-CNN | [Mask R-CNN](https://arxiv.org/abs/1703.06870)               | P0         |
|              |    SOLO    | [SOLO: Segmenting Objects by Locations](https://arxiv.org/abs/1912.04488) | P1         |

### 4、语义分割

|   **任务**   |   **算法**    | **论文名**                                                   | **推荐优先级** |
| :----------: | :-----------: | :----------------------------------------------------------- | :------------- |
| **语义分割** |      FCN      | [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) | P0             |
|              | DeepLabv3Plus | [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) | P0             |
|              |     UNet      | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | P0             |
|              |    PSPNet     | [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105) | P1             |
|              |   DeepLabv3   | [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) | P1             |
|              |    UperNet    | [Unified Perceptual Parsing for Scene Understanding](https://arxiv.org/abs/1807.10221) | P2             |

### 5、动作识别

|     任务     |   算法   | 论文名                                                       | 推荐优先级 |
| :----------: | :------: | :----------------------------------------------------------- | :--------- |
| **动作识别** | SlowFast | [Slowfast networks for video recognition](https://arxiv.org/abs/1812.03982) | P0         |
|              |   TSN    | [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859) | P1         |
|              |  ST-GCN  | [Spatial temporal graph convolutional networks for skeleton-based action recognition](https://arxiv.org/abs/1801.07455) | P1         |

### 6、2D姿态估计

|      任务       |   算法   | 论文名                                                       | 推荐优先级 |
| :-------------: | :------: | :----------------------------------------------------------- | :--------- |
| **2D 姿态估计** |  HRNet   | [Deep high-resolution representation learning for human pose estimation](https://arxiv.org/abs/1902.09212) | P0         |
|                 | DeepPose | [Deeppose: Human pose estimation via deep neural networks](https://arxiv.org/abs/1312.4659) | P2         |

### 7、OCR识别

|  任务   | 算法  | 论文名                                                       | 推荐优先级 |
| :-----: | :---: | :----------------------------------------------------------- | :--------- |
| **OCR** | DBNet | [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947) | P0         |
|         | CRNN  | [An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/abs/1507.05717) | P0         |
|         |  SAR  | [Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/abs/1811.00751) | P2         |

## 二、影响力数据

---

### 1、图像分类

| 方向         | 算法                     | 论文名                                                       | 影响力评分 | Google Scholar 引用数 | Google Search 结果数 | GitHub repository Search 结果数 | GitHub code Search 结果数 |
| :----------- | :----------------------- | :----------------------------------------------------------- | :--------- | :-------------------- | :------------------- | :------------------------------ | :------------------------ |
| **图像分类** | ResNet (2015）           | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | 5          | 123,773               | 2,480,000            | 6,631                           | 823,551                   |
|              | MobileNet V2 (2018）     | [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) | 3          | 10,007                | 48,100               | 865                             | 98,276                    |
|              | ViT (2020）              | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | 4          | 4,985                 | 85,000               | 1,891                           | 183,629                   |
|              | EfficientNet (2019）     | [Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | 4          | 7,195                 | 132,000              | 861                             | 72,186                    |
|              | InceptionV3 (2015）      | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | 3          | 20,421                | 156,000              | 526                             | 169,504                   |
|              | DenseNet (2016）         | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) | 3          | 25,988                | 268,000              | 1,174                           | 64,907                    |
|              | SENets (2017）           | [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) | 2          | 13,565                | 51,100               | 167                             | 14,368                    |
|              | Swin Transformer (2021） | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | 3          | 1,805                 | 33,900               | 189                             | 24,178                    |
|              | VGGNet (2014）           | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | 4          | 81,277                | 847,000              | 4,386                           | 400,117                   |
|              | ShuffleNetV2 (2018）     | [Shufflenet v2: Practical guidelines for efficient cnn architecture design](https://arxiv.org/abs/1807.11164) | 2          | 2,371                 | 14,800               | 24                              | 5,891                     |
|              | RepVGG (2021）           | [Repvgg: Making vgg-style convnets great again](https://arxiv.org/abs/2101.03697) | 2          | 192                   | 5,500                | 37                              | 2,604                     |

### 2、目标检测

| 方向         | 算法                  | 论文名                                                       | 影响力评分 | Google Scholar 引用数 | Google Search 结果数 | GitHub repository Search 结果数 | GitHub code Search 结果数 |
| :----------- | :-------------------- | :----------------------------------------------------------- | :--------- | :-------------------- | :------------------- | :------------------------------ | :------------------------ |
| **目标检测** | SSD (2015）           | [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) | 4          | 22,196                | 85,900               | 3961                            | 484552                    |
|              | YOLOV3 (2018）        | [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) | 5          | 13,740                | 248,000              | 4909                            | 112,082                   |
|              | Cascade R-CNN (2017） | [Cascade R-CNN: High Quality Object Detection and Instance Segmentation](https://arxiv.org/abs/1906.09756) | 2          | 438                   | 27,500               | 41                              | 55                        |
|              | CenterNet (2019）     | [Objects as Points](https://arxiv.org/abs/1904.07850)        | 2          | 1,638                 | 20,100               | 339                             | 26,024                    |
|              | DETR (2020）          | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | 4          | 2,579                 | 48,000               | 592                             | 38,909                    |
|              | Faster R-CNN (2015）  | [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) | 4          | 44,477                | 389,000              | 2,175                           | 7,822                     |
|              | FCOS (2019）          | [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) | 3          | 2,139                 | 45,600               | 167                             | 49,960                    |
|              | RetinaNet (2017）     | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) | 4          | 14,103                | 117,000              | 766                             | 136,616                   |
|              | ATSS (2019）          | [Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection](https://arxiv.org/abs/1912.02424) | 2          | 518                   | 8,760                | 22                              | 20,023                    |

### 3、实例分割

| 方向         | 算法               | 论文名                                                       | 影响力评分 | Google Scholar 引用数 | Google Search 结果数 | GitHub repository Search 结果数 | GitHub code Search 结果数 |
| :----------- | :----------------- | :----------------------------------------------------------- | :--------- | :-------------------- | :------------------- | :------------------------------ | :------------------------ |
| **实例分割** | Mask R-CNN (2017） | [Mask R-CNN](https://arxiv.org/abs/1703.06870)               | 4          | 19,529                | 101,000              | 2,229                           | 9,627                     |
|              | SOLO (2019）       | [SOLO: Segmenting Objects by Locations](https://arxiv.org/abs/1912.04488) | 2          | 285                   | 10,200               | 9                               | 254,495                   |

### 4、语义分割

| 方向         | 算法                  | 论文名                                                       | 影响力评分 | Google Scholar 引用数 | Google Search 结果数 | GitHub repository Search 结果数 | GitHub code Search 结果数 |
| :----------- | :-------------------- | :----------------------------------------------------------- | :--------- | :-------------------- | :------------------- | :------------------------------ | :------------------------ |
| **语义分割** | PSPNet (2016）        | [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105) | 3          | 7519                  | 53,900               | 218                             | 21,377                    |
|              | DeepLabv3 (2017）     | [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) | 3          | 5055                  | 52,900               | 487                             | 32901                     |
|              | FCN (2014）           | [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) | 5          | 31794                 | 158,000              | 1,846                           | 271,636                   |
|              | DeepLabv3Plus (2018） | [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) | 4          | 6815                  | 55,400               | 134                             | 16,118                    |
|              | UNet (2015）          | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | 4          | 44256                 | 132,000              | 9,445                           | 31,920                    |
|              | UperNet (2018）       | [Unified Perceptual Parsing for Scene Understanding](https://arxiv.org/abs/1807.10221) | 2          | 475                   | 3,200                | 5                               | 9,208                     |

### 5、动作识别

| 方向         | 算法             | 论文名                                                       | 影响力评分 | Google Scholar 引用数 | Google Search 结果数 | GitHub repository Search 结果数 | GitHub code Search 结果数 |
| :----------- | :--------------- | :----------------------------------------------------------- | :--------- | :-------------------- | :------------------- | :------------------------------ | :------------------------ |
| **动作识别** | TSN (2016）      | [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859) | 4          | 2,929                 | 18,800               | 179                             | 131,000                   |
|              | SlowFast (2018） | [Slowfast networks for video recognition](https://arxiv.org/abs/1812.03982) | 3          | 1,335                 | 10,700               | 78                              | 9,917                     |
|              | ST-GCN (2018）   | [Spatial temporal graph convolutional networks for skeleton-based action recognition](https://arxiv.org/abs/1801.07455) | 3          | 2,037                 | 9,680                | 69                              | 2,565                     |

###  6、2D姿态估计

| 方向            | 算法             | 论文名                                                       | 影响力评分 | Google Scholar 引用数 | Google Search 结果数 | GitHub repository Search 结果数 | GitHub code Search 结果数 |
| :-------------- | :--------------- | :----------------------------------------------------------- | :--------- | :-------------------- | :------------------- | :------------------------------ | :------------------------ |
| **2D 姿态估计** | HRNet (2019）    | [Deep high-resolution representation learning for human pose estimation](https://arxiv.org/abs/1902.09212) | 5          | 1,868                 | 25,100               | 218                             | 39,000                    |
|                 | DeepPose (2019） | [Deeppose: Human pose estimation via deep neural networks](https://arxiv.org/abs/1312.4659) | 2          | 2,625                 | 14,700               | 37                              | 945                       |

### 7、OCR

| 方向    | 算法          | 论文名                                                       | 影响力评分 | Google Scholar 引用数 | Google Search 结果数 | GitHub repository Search 结果数 | GitHub code Search 结果数 |
| :------ | :------------ | :----------------------------------------------------------- | :--------- | :-------------------- | :------------------- | :------------------------------ | :------------------------ |
| **OCR** | DBNet (2019） | [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947) | 3          | 222                   | 12,100               | 53                              | 2,289                     |
|         | CRNN (2015）  | [An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/abs/1507.05717) | 4          | 1785                  | 129,000              | 765                             | 21,271                    |
|         | SAR (2018）   | [Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/abs/1811.00751) | 3          | 166                   | 1,040                | 3                               | 23,031                    |

## 三、性能数据

---

### 1、图像分类

|              |                  | 小模型（参数量： ~20M） |               |              |       |        |        | 中模型（参数量： 20M ~ 40M） |               |              |       |         |        | 大模型（参数量：40M ~ ） |               |              |       |         |         |
| :----------- | :--------------- | :---------------------- | :------------ | :----------- | :---- | :----- | :----- | :--------------------------- | :------------ | :----------- | :---- | :------ | :----- | :----------------------- | ------------- | ------------ | ----- | ------- | ------- |
| 方向         | 算法名称         | 名称                    | FPS (PyTorch) | FPS(T4-FP16) | 精度  | 参数量 | 计算量 | 名称                         | FPS (PyTorch) | FPS(T4-FP16) | 精度  | 参数量  | 计算量 | 名称                     | FPS (PyTorch) | FPS(T4-FP16) | 精度  | 参数量  | 计算量  |
| **图像分类** | VGGNet           | /                       | /             | /            | /     | /      | /      | /                            | /             | /            | /     | /       | /      | VGG-16                   | 354.5         | 4.41         | 71.62 | 0.138G  | 15.538G |
|              | ResNet           | R18                     | 356.1         | 1.46         | 71.71 | 11.69M | 1.827G | R50                          | 178.9         | 2.95         | 80.12 | 25.557M | 4.145G | R101                     | 96.6          | 5.11         | 81.35 | 44.549M | 7.883G  |
|              | MobileNet V2     | MobileNet V2            | 185.3         | 1            | 71.86 | 3.505M | 0.334G | /                            | /             | /            | /     | /       | /      | /                        | /             | /            | /     | /       | /       |
|              | ViT              | /                       | /             | /            | /     | /      | /      | /                            | /             | /            | /     | /       | /      | base                     | 155.1         | 9.53         | 82.36 | 91.234M | 17.587G |
|              | ShuffleNetV2     | ShuffleNetV2            | 139.8         | 1.24         | 69.55 | 2.279M | 0.155G | /                            | /             | /            | /     | /       | /      | /                        | /             | /            | /     | /       | /       |
|              | RepVGG           | RepVGG-A0               | 413.9         | 1.27         | 72.41 | 9.109M | 1.535G | /                            | /             | /            | /     | /       | /      | RepVGG-B3                | 282.8         | 7.33         | 80.52 | 0.123G  | 29.224G |
|              | InceptionV3      | /                       | /             | /            | /     | /      | /      | InceptionV3                  | 95.4          |              | 77.57 | 23.835M | 2.861G |                          |               | 4.53         | /     | /       | /       |
|              | SENets           | /                       | /             | /            | /     | /      | /      | R50                          | 116.1         | 5.44         | 77.74 | 28.088M | 4.153G | R101                     | 59.5          | 7.55         | 78.26 | 49.327M | 7.896G  |
|              | EfficientNet     | B0                      | 106.1         | 3.47         | 77.53 | 5.289M | 0.422G | B4                           | 58.7          | 5.55         | 83.25 | 19.342M | 1.596G | B8                       | 26.8          | 8.93         | 85.38 | 87.413M | 7.239G  |
|              | Swin Transformer | /                       | /             | /            | /     | /      | /      | /                            | /             | /            | /     | /       | /      | base                     | 47.8          | 8.05         | 83.36 | 87.768M | 15.468G |
|              | DenseNet         | DenseNet-121            | 63.9          | 5.84         | 74.96 | 7.979M | 2.913G | DenseNet-161                 | 47.8          | 6.86         | 77.61 | 28.681M | 7.875G | /                        | /             | /            | /     | /       | /       |
|              |                  |                         |               |              |       |        |        |                              |               |              |       |         |        |                          |               |              |       |         |         |

### 2、目标检测

|          |               | 小模型（参数量： ~20M） |               |              |      |         |        | 中模型（参数量： 20M ~ 40M） |               |              |      |         |         | 大模型（参数量：40M ~ ） |               |              |      |         |         |
| :------- | :------------ | :---------------------- | :------------ | :----------- | :--- | :------ | :----- | :--------------------------- | :------------ | :----------- | :--- | :------ | :------ | :----------------------- | ------------- | ------------ | ---- | ------- | ------- |
| 方向     | 算法名称      | 名称                    | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量  | 计算量 | 名称                         | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量  | 计算量  | 名称                     | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量  | 计算量  |
| 目标检测 | YOLOV3        | MobileNet v2-320        | 123.7         | 6.73         | 22.2 | 3.738M  | 0.824G | /                            | /             | /            | /    | /       | /       | DarkNet53-608            | 90.2          | 12.38        | 33.7 | 61.949M | 9.604G  |
|          | SSD           | MobileNet v2-320        | 102.3         | 9.64         | 21.3 | /       | /      | VGG-300                      | 121.9         | 33.77        | 25.5 | /       | /       | VGG-512                  | 87.8          | 33.16        | 29.5 | /       | /       |
|          | CenterNet     | R18-FPN                 | 68.8          | 8.86         | 42.5 | 19.273M | 7.649G | R50                          | 52.8          | /            | 46.3 | 32.293M | 10.089G | R101                     | 39.9          | /            | 48.1 | 51.285M | 13.827G |
|          | RetinaNet     | R18-FPN                 | 72.7          | 19.7         | 39.8 | 21.41M  | 9.272G | R50                          | 53.1          | 25.58        | 43.2 | 37.969M | 11.769G | R101                     | 41            | 32.2         | 44.2 | 56.961M | 15.507G |
|          | ATSS          | R18-FPN                 | 66            | /            | 42   | 19.276M | 7.651G | R50                          | 50.4          | 25.62        | 46.3 | 32.295M | 10.092G | R101                     | 38.8          | 30.67        | 48.2 | 51.287M | 13.83G  |
|          | FCOS          | R18-FPN                 | 63.7          | /            | 42   | 19.278M | 7.651G | R50                          | 50.1          | 25.26        | 45.7 | 32.297M | 10.092G | R101                     | 38.1          | 31.41        | 47.2 | 51.289M | 13.83G  |
|          | Faster R-CNN  | /                       | /             | /            | /    | /       | /      | R18-FPN                      | 63.6          | /            | 40   | 28.684M | 0.14T   | R101                     | 37.2          | 27.18        | 46.3 | 60.745M | 0.255T  |
|          | DETR          | /                       | /             | /            | /    | /       | /      | R18                          | 59.8          | 22.07        | 38   | 28.85M  | 43.269G | R101                     | 34.5          | /            | 43   | 60.567M | 0.153T  |
|          | Cascade R-CNN | /                       | /             | /            | /    | /       | /      | R18-FPN                      | 48.7          | /            | 43.3 | 56.326M | 0.168T  | R101                     | 31.4          | 33.51        | 48.3 | 88.387M | 0.283T  |
|          |               |                         |               |              |      |         |        |                              |               |              |      |         |         |                          |               |              |      |         |         |

### 3、实例分割

|              |            | 小模型（参数量： ~20M） |               |              |      |         |         | 中模型（参数量： 20M ~ 40M） |               |              |      |         |         | 大模型（参数量：40M ~ ） |               |              |      |         |         |
| :----------- | :--------- | :---------------------- | :------------ | :----------- | :--- | :------ | :------ | :--------------------------- | :------------ | :----------- | :--- | :------ | :------ | :----------------------- | ------------- | ------------ | ---- | ------- | ------- |
| 方向         | 算法名称   | 名称                    | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量  | 计算量  | 名称                         | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量  | 计算量  | 名称                     | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量  | 计算量  |
| **实例分割** | Mask R-CNN | /                       | /             | /            | /    | /       | /       | R18-FPN                      | 54.5          | /            | 37.2 | 31.328M | 0.193T  | R101                     | 41.7          | 31.38        | 42   | 63.388M | 0.308T  |
|              | SOLO       | R18-FPN                 | 38.6          | /            | 34.8 | 23.232M | 31.296G | R50                          | 31.6          | 347.61       | 38   | 36.301M | 33.891G | R101                     | 27.2          | /            | 39.4 | 55.293M | 37.629G |
|              |            |                         |               |              |      |         |         |                              |               |              |      |         |         |                          |               |              |      |         |         |

### 4、语义分割

|              |               | 小模型（参数量： ~20M） |               |              |       |         |        | 中模型（参数量： 20M ~ 40M） |               |              |       |         |        | 大模型（参数量：40M ~ ） |               |              |       |         |        |
| :----------- | :------------ | :---------------------- | :------------ | :----------- | :---- | :------ | :----- | :--------------------------- | :------------ | :----------- | :---- | :------ | :----- | :----------------------- | ------------- | ------------ | ----- | ------- | ------ |
| 方向         | 算法名称      | 名称                    | FPS (PyTorch) | FPS(T4-FP16) | 精度  | 参数量  | 计算量 | 名称                         | FPS (PyTorch) | FPS(T4-FP16) | 精度  | 参数量  | 计算量 | 名称                     | FPS (PyTorch) | FPS(T4-FP16) | 精度  | 参数量  | 计算量 |
| **语义分割** | PSPNet        | R18                     | 45.9          | 7.6          | 74.87 | 12.641M | 0.434T | /                            | /             | /            | /     | /       | /      | R101                     | 10.5          | 38.07        | 79.76 | 65.603M | 2.051T |
|              | FCN           | R18                     | 45.5          | 7.56         | 71.11 | 12.674M | 0.444T | /                            | /             | /            | /     | /       | /      | R101                     | 10.8          | 39.86        | 75.13 | 66.125M | 2.205T |
|              | DeepLabv3     | R18                     | 42.6          | 7.83         | 76.7  | 13.838M | 0.48T  | /                            | /             | /            | /     | /       | /      | R101                     | 9.2           | 49.13        | 80.2  | 84.74M  | 2.781T |
|              | DeepLabv3Plus | R18                     | 39.2          | 9.33         | 76.89 | 12.32M  | 0.434T | R50                          | 13.5          | 32.57        | 80.09 | 41.225M | 1.413T | R101                     | 9.5           | 45.55        | 80.97 | 60.217M | 2.035T |
|              | UperNet       | /                       | /             | /            | /     | /       | /      | R18                          | 17.7          | 30.99        | 76.02 | 40.807M | 1.764T | R101                     | 12.2          | 38.24        | 79.4  | 83.043M | 2.052T |
|              | UNet          | /                       | /             | /            | /     | /       | /      | UNet-FCN                     | 11            | 39.57        | 69.1  | 28.988M | 1.626T | /                        | /             | /            | /     | /       | /      |

### 5、动作识别

|              |          | 小模型（参数量： ~20M） |               |              |       |        |        | 中模型（参数量： 20M ~ 40M） |               |              |      |        |        | 大模型（参数量：40M ~ ） |               |              |      |        |        |
| :----------- | :------- | :---------------------- | :------------ | :----------- | :---- | :----- | :----- | :--------------------------- | :------------ | :----------- | :--- | :----- | :----- | :----------------------- | ------------- | ------------ | ---- | ------ | ------ |
| 方向         | 算法名称 | 名称                    | FPS (PyTorch) | FPS(T4-FP16) | 精度  | 参数量 | 计算量 | 名称                         | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量 | 计算量 | 名称                     | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量 | 计算量 |
| **动作识别** | ST-GCN   | ST-GCN                  | 168.3         | /            | 87.58 | 3.1M   | 5.7G   | None                         | /             | /            | /    | /      | /      | /                        | /             | /            | /    | /      | /      |
|              | TSN      | /                       | /             | /            | /     | /      | /      | R50                          | 125.1         | 123.02       | 70.6 | 24.3M  | 102.7G | R101                     | 71.4          | 191.97       | 74.1 | 43.3M  | 195.8G |
|              | SlowFast | /                       | /             | /            | /     | /      | /      | R50                          | 43.1          | 399.03       | 76   | 34.6M  | 66.1G  | R101                     | 28.8          |              | 77.9 | 62.9M  | 126G   |

### 6、2D 姿态估计

|                 |          | 小模型（参数量： ~20M） |               |              |       |        |        | 中模型（参数量： 20M ~ 40M） |               |              |       |        |        | 大模型（参数量：40M ~ ） |               |              |       |        |        |
| :-------------- | :------- | :---------------------- | :------------ | :----------- | :---- | :----- | :----- | :--------------------------- | :------------ | :----------- | :---- | :----- | :----- | :----------------------- | ------------- | ------------ | ----- | ------ | ------ |
| 方向            | 算法名称 | 名称                    | FPS (PyTorch) | FPS(T4-FP16) | 精度  | 参数量 | 计算量 | 名称                         | FPS (PyTorch) | FPS(T4-FP16) | 精度  | 参数量 | 计算量 | 名称                     | FPS (PyTorch) | FPS(T4-FP16) | 精度  | 参数量 | 计算量 |
| **2D 姿态估计** | DeepPose | Resnet50_256x192        | 78.6          | 2.65         | 0.528 | /      | /      | Resnet101_256x192            | 40.1          | 4.54         | 0.562 | /      | /      | Resnet152_256x192        | 28.5          | 6.35         | 0.584 | /      | /      |
|                 | HRNet    | HRNet_w32_256x192       | 16.3          | 7.87         | 0.746 | /      | /      | /                            | /             | 10.02        | /     | /      | /      | HRNet_w48_256x192        | 16.1          | /            | 0.756 | /      | /      |

### 7、文本识别

|              |          | 小模型（参数量： ~20M） |               |              |      |        |        | 中模型（参数量： 20M ~ 40M） |               |              |      |        |        | 大模型（参数量：40M ~ ） |               |              |      |         |         |
| :----------- | :------- | :---------------------- | :------------ | :----------- | :--- | :----- | :----- | :--------------------------- | :------------ | :----------- | :--- | :----- | :----- | :----------------------- | ------------- | ------------ | ---- | ------- | ------- |
| 方向         | 算法名称 | 名称                    | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量 | 计算量 | 名称                         | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量 | 计算量 | 名称                     | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量  | 计算量  |
| **文本识别** | CRNN     | MobileNet-v2            | 163.4         | 1.06         | 61.2 | 6.577M | 0.194G | /                            | /             | /            | /    | /      | /      | /                        | /             | /            | /    | /       | /       |
|              | SAR      | /                       | /             | /            | /    | /      | /      | /                            | /             | /            | /    | /      | /      | R31                      | 19.3          | /            | 87.3 | 57.458M | 17.414G |
|              |          |                         |               |              |      |        |        |                              |               |              |      |        |        |                          |               |              |      |         |         |

### 8、文本检测

|          |          | 小模型（参数量： ~20M） |               |              |      |         |         | 中模型（参数量： 20M ~ 40M） |               |              |      |         |         | 大模型（参数量：40M ~ ） |               |              |      |         |         |
| :------- | :------- | :---------------------- | :------------ | :----------- | :--- | :------ | :------ | :--------------------------- | :------------ | :----------- | :--- | :------ | :------ | :----------------------- | ------------- | ------------ | ---- | ------- | ------- |
| 方向     | 算法名称 | 名称                    | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量  | 计算量  | 名称                         | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量  | 计算量  | 名称                     | FPS (PyTorch) | FPS(T4-FP16) | 精度 | 参数量  | 计算量  |
| 文本检测 | DBNet    | R18                     | 56.6          | 6.45         | 69.3 | 12.341M | 24.897G | R50                          | 43.4          | /            | 71.2 | 26.281M | 35.232G | R101                     | 34.2          | /            | 72.8 | 46.331M | 51.268G |
|          |          |                         |               |              |      |         |         |                              |               |              |      |         |         |                          |               |              |      |         |         |

## 四、PaddleX List

---

### 1、图像分类


| 模型类型 | 模型                                    | Top1 Acc(%) | GPU 推理耗时(ms) | CPU 推理耗时(ms) | 模型存储大小(M) | 详情                                                         |
| :------- | :-------------------------------------- | :---------- | :--------------- | :--------------- | :-------------- | :----------------------------------------------------------- |
| 图像分类 | CLIP_vit_base_patch16_224               | 85.39       | 4.68             | 67.35            | 331             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6797877) |
|          | SwinTransformer_base_patch4_window7_224 | 83.37       | 6.55             | 838.15           | 342             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6798186) |
|          | PP-HGNet_small                          | 81.51       | 2.43             | 24.01            | 94              | [info](https://aistudio.baidu.com/projectdetail/paddlex/6798078) |
|          | ResNet50                                | 76.50       | 2.15             | 10.83            | 98              | [info](https://aistudio.baidu.com/projectdetail/paddlex/6797043) |
|          | PP-LCNet_x1_0                           | 71.32       | 0.47             | 1.63             | 12              | [info](https://aistudio.baidu.com/projectdetail/paddlex/6797638) |
|          | MobileNetV3_small_x1_0                  | 68.24       | 0.83             | 1.79             | 12              | [info](https://aistudio.baidu.com/projectdetail/paddlex/6798005) |

### 2、图像分割

| 模型类型 | 模型       | mIoU (%) | GPU 推理耗时(ms) | CPU 推理耗时(ms) | 模型存储大小(M) | 详情                                                         |
| :------- | :--------- | :------- | :--------------- | :--------------- | :-------------- | :----------------------------------------------------------- |
| 图像分割 | OCRNet     | 82.15    | 158.60           | 3011.93          | 270             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6806846) |
|          | PP-LiteSeg | 77.04    | 7.55             | 208.67           | 31              | [info](https://aistudio.baidu.com/projectdetail/paddlex/6806492) |

### 3、目标检测

| 模型类型 | 模型                | mAP(%) | GPU 推理耗时(ms) | CPU 推理耗时(ms) | 模型存储大小(M) | 详情                                                         |
| :------- | :------------------ | :----- | :--------------- | :--------------- | :-------------- | :----------------------------------------------------------- |
| 目标检测 | rt_detr_hgnetv2_l   | 53.0   | 12.34            | 228.53           | 125             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6791833) |
|          | ppyoloe_plus_l      | 52.9   | 11.57            | 256.29           | 200             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6798256) |
|          | picodet_s_320_lcnet | 29.1   | 9.46             | 6.27             | 5               | [info](https://aistudio.baidu.com/projectdetail/paddlex/6798284) |

### 4、OCR

| 模型类型 | 模型            | 检测 mAP(%) | 识别 Top1 Acc(%) | GPU 推理耗时(ms) | CPU 推理耗时(ms) | 模型存储大小(M) | 详情                                                         |
| :------- | :-------------- | :---------- | :--------------- | :--------------- | :--------------- | :-------------- | :----------------------------------------------------------- |
| OCR      | PP-OCRv4-server | 82.69       | 79.20            | 32.59            | 423.91           | 198             | [文本检测](https://aistudio.baidu.com/projectdetail/paddlex/6792800)/[文本识别](https://aistudio.baidu.com/projectdetail/paddlex/6793806) |
|          | PP-OCRv4-mobile | 77.79       | 78.20            | 3.12             | 35.47            | 15              | [文本检测](https://aistudio.baidu.com/projectdetail/paddlex/6792883)/[文本识别](https://aistudio.baidu.com/projectdetail/paddlex/6793861) |

### 5、PP-ChatOCR

| 模型类型   | 模型       | 关键信息抽取准确率                | 启动训练                                                     |
| :--------- | :--------- | :-------------------------------- | :----------------------------------------------------------- |
| PP-ChatOCR | PP-ChatOCR | 61%（百度自有数据集，复杂度较大） | [info](https://aistudio.baidu.com/projectdetail/paddlex/6796372) |

### 6、PDF转Word

| 模型类型  | 算法     | 模型              | 精度        | GPU 推理耗时(ms) | CPU 推理耗时(ms) | 模型存储大小(M) | 详情                                                         |
| :-------- | :------- | :---------------- | :---------- | :--------------- | :--------------- | :-------------- | :----------------------------------------------------------- |
| PDF转Word | 版面分析 | picodet_layout_1x | 86.80       | 4.99             | 76.41            | 9.7             | [版面分析](https://aistudio.baidu.com/projectdetail/paddlex/6793486) |
|           | 文本检测 | ch_PP-OCRv4_det   | 77.79/82.69 | 3.63/77.74       | -/2244.96        | 4.7/111         | [mobile](https://aistudio.baidu.com/projectdetail/paddlex/6792883)/[server](https://aistudio.baidu.com/projectdetail/paddlex/6792800) |
|           | 文本识别 | ch_PP-OCRv4_rec   | 78.20/79.20 | 1.46/6.55        | -/134.59         | 11/89           | [mobile](https://aistudio.baidu.com/projectdetail/paddlex/6793861)/[server](https://aistudio.baidu.com/projectdetail/paddlex/6793806) |
|           | 表格识别 | SLANet            | 76.31       | 868.23           | 395.39           | 9.3             | [SLANet](https://aistudio.baidu.com/projectdetail/paddlex/6796154) |

### 7、时序预测


| 模型类型 | 模型          | mse   | mae   | 模型存储大小(M) | 详情                                                         |
| :------- | :------------ | :---- | :---- | :-------------- | :----------------------------------------------------------- |
| 时序预测 | DLinear       | 0.386 | 0.445 | 80k             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6792900) |
|          | RLinear       | 0.408 | 0.456 | 44k             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6793480) |
|          | Nlinear       | 0.411 | 0.459 | 44k             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6793506) |
|          | PatchTST      | 0.291 | 0.380 | 2.2M            | [info](https://aistudio.baidu.com/projectdetail/paddlex/6792846) |
|          | TimesNet      | 0.284 | 0.386 | 5.2M            | [info](https://aistudio.baidu.com/projectdetail/paddlex/6793437) |
|          | TiDE          | 0.376 | 0.441 | 35M             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6793534) |
|          | Nonstationary | 0.385 | 0.463 | 61M             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6793558) |
|          | XGBoost       | 0.426 | 0.470 | 15M             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6792901) |
|          | PP-TS         | 0.210 | 0.318 | 63M             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6792747) |

### 8、图像识别系统

| 模型类型     | 模型       | 主体检测模型 mAP(%) | Aliproduct数据集recall@1(%) | Aliproduct数据集mAP(%) | SOP数据集recall@1(%) | SOP数据集mAP(%) | 详情                                                         |
| :----------- | :--------- | :------------------ | :-------------------------- | :--------------------- | :------------------- | :-------------- | :----------------------------------------------------------- |
| 图像识别系统 | PP-ShiTuV2 | 41.5                | 84.2                        | 83.3                   | 77.6                 | 55.3            | [主体检测](https://aistudio.baidu.com/projectdetail/paddlex/6807004)/[特征提取](https://aistudio.baidu.com/projectdetail/paddlex/6807332) |

### 9、3D目标检测

| 模型类型   | 模型        | 精度               | NDS (%) | GPU 推理耗时(ms) | CPU 推理耗时(ms) | 模型存储大小(M) | 详情                                                         |
| :--------- | :---------- | :----------------- | :------ | :--------------- | :--------------- | :-------------- | :----------------------------------------------------------- |
| 3D目标检测 | CaDDN       | 7.86%（3DmAP Mod） | -       | 182.4            | -                | 121             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6807096) |
|            | CenterPoint | 50.79% (3D mAP)    | 61.30   | 34.0             | -                | 24.5            | [info](https://aistudio.baidu.com/projectdetail/paddlex/6807282) |
|            | PETRv1      | 38.35 (3D mAP)     | 43.52   | 262.5            | 22341.2          | 345             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6807328) |
|            | PETRv2      | 41.05 (3D mAP)     | 49.86   | 515.3            | 42693.8          | 121             | [info](https://aistudio.baidu.com/projectdetail/paddlex/6807340) |





## 五、附录

---

1.影响力评价目前共有四个维度，即Google Scholar 引用数、Google Search 结果数、GitHub repository 数、GitHub code 数，分别定义为变量A、B、C、D .
2.计算各维度对应 Arxiv 发表至今的月均结果，即 ：$a$,$b$,$c$,$d$。
$$
a=\frac A{(Year_{now}-Year_{arxiv})*12+Month_{now}-Month_{arxiv}}
$$

$b,c,d$计算方式亦然。

注：$Year_{now}$与$Month_{now}$分别表示当前年、月份，$Year_{arxiv}$与$Month_{arxiv}$分别表示改算法对应的论文在$Arxiv$上发表的年、月份。

3.计算各维度的细项评分，即$S_{a}$，$S_{b}$，$S_{c}$，$S_{d}$，以$S_{a}$为例，如下公式中$a_{max}$，$a_{avg}$，$a_{min}$，分别表示该方向（如目标检测）下所有算法中$a$的最大值，算数平均值与最小值。
$$
\left.S_a=\left\{\begin{array}{ll}4,\quad&a>\frac{amax+a_avg}2\\3,\quad&\frac{amax+a_avg}2>a>a_{avg}\\2,\quad&a_{avg}>a>\frac{a_{min}+a_{avg}}2\\1,\quad&\frac{a_{min}+a_{avg}}2>a\end{array}\right.\right.
$$

4.计算该模型的影响力总体得分，即$S$：

$S=Roundup(\frac{S_a+S_b+S_c+S_d}4)$，其中$Roundup$为向上舍入取整函数。





1.影响力评价目前共有四个维度，即Google Scholar 引用数、Google Search 结果数、GitHub repository 数、GitHub code 数，分别定义为变量A、B、C、D .

2.计算各维度对应 Arxiv 发表至今的月均结果，即：a, b, c, d。

![equation](https://latex.codecogs.com/svg.latex?a%3D%5Cfrac%20A%7B%28Year_%7Bnow%7D-Year_%7Barxiv%7D%29*12&plus;Month_%7Bnow%7D-Month_%7Barxiv%7D%7D)

b, c, d计算方式亦然。

注：Year_now与Month_now分别表示当前年、月份，Year_arxiv与Month_arxiv分别表示改算法对应的论文在Arxiv上发表的年、月份。

3.计算各维度的细项评分，即S_a，S_b，S_c，S_d，以S_a为例，如下公式中a_max，a_avg，a_min，分别表示该方向（如目标检测）下所有算法中a的最大值，算数平均值与最小值。

![equation](https://latex.codecogs.com/svg.latex?S_a%3D%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D4%2C%26a%3E%5Cfrac%7Ba_%7Bmax%7D&plus;a_%7Bavg%7D%7D%7B2%7D%5C%5C3%2C%26%5Cfrac%7Ba_%7Bmax%7D&plus;a_%7Bavg%7D%7D%7B2%7D%3Ea%3Ea_%7Bavg%7D%5C%5C2%2C%26a_%7Bavg%7D%3Ea%3E%5Cfrac%7Ba_%7Bmin%7D&plus;a_%7Bavg%7D%7D%7B2%7D%5C%5C1%2C%26%5Cfrac%7Ba_%7Bmin%7D&plus;a_%7Bavg%7D%7D%7B2%7D%3Ea%5Cend%7Barray%7D%5Cright.)

4.计算该模型的影响力总体得分，即S：

![equation](https://latex.codecogs.com/svg.latex?S%3DRoundup%28%5Cfrac%7BS_a&plus;S_b&plus;S_c&plus;S_d%7D%7B4%7D%29)，其中Roundup为向上舍入取整函数。

综合考虑模型在标准 Benchmark 上精度和速度的 trade-off，比较精度-速度曲线。

计算量与参数量计算方式，详见：[MMEngine-统计模型计算量和参数量.](https://mmengine.readthedocs.io/zh_CN/latest/common_usage/model_analysis.html)



## 六、参考：

---

- https://openmmlab.com/community/algorithm-rating
- https://aistudio.baidu.com/intro/paddlex/models
