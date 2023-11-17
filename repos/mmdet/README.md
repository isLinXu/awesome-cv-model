# 基准测试和模型库

测试结果和模型可以在[模型库](docs/zh_cn/model_zoo.md)中找到。

<div align="center">
  <b>算法架构</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Object Detection</b>
      </td>
      <td>
        <b>Instance Segmentation</b>
      </td>
      <td>
        <b>Panoptic Segmentation</b>
      </td>
      <td>
        <b>Other</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/fast_rcnn">Fast R-CNN (ICCV'2015)</a></li>
            <li><a href="configs/faster_rcnn">Faster R-CNN (NeurIPS'2015)</a></li>
            <li><a href="configs/rpn">RPN (NeurIPS'2015)</a></li>
            <li><a href="configs/ssd">SSD (ECCV'2016)</a></li>
            <li><a href="configs/retinanet">RetinaNet (ICCV'2017)</a></li>
            <li><a href="configs/cascade_rcnn">Cascade R-CNN (CVPR'2018)</a></li>
            <li><a href="configs/yolo">YOLOv3 (ArXiv'2018)</a></li>
            <li><a href="configs/cornernet">CornerNet (ECCV'2018)</a></li>
            <li><a href="configs/grid_rcnn">Grid R-CNN (CVPR'2019)</a></li>
            <li><a href="configs/guided_anchoring">Guided Anchoring (CVPR'2019)</a></li>
            <li><a href="configs/fsaf">FSAF (CVPR'2019)</a></li>
            <li><a href="configs/centernet">CenterNet (CVPR'2019)</a></li>
            <li><a href="configs/libra_rcnn">Libra R-CNN (CVPR'2019)</a></li>
            <li><a href="configs/tridentnet">TridentNet (ICCV'2019)</a></li>
            <li><a href="configs/fcos">FCOS (ICCV'2019)</a></li>
            <li><a href="configs/reppoints">RepPoints (ICCV'2019)</a></li>
            <li><a href="configs/free_anchor">FreeAnchor (NeurIPS'2019)</a></li>
            <li><a href="configs/cascade_rpn">CascadeRPN (NeurIPS'2019)</a></li>
            <li><a href="configs/foveabox">Foveabox (TIP'2020)</a></li>
            <li><a href="configs/double_heads">Double-Head R-CNN (CVPR'2020)</a></li>
            <li><a href="configs/atss">ATSS (CVPR'2020)</a></li>
            <li><a href="configs/nas_fcos">NAS-FCOS (CVPR'2020)</a></li>
            <li><a href="configs/centripetalnet">CentripetalNet (CVPR'2020)</a></li>
            <li><a href="configs/autoassign">AutoAssign (ArXiv'2020)</a></li>
            <li><a href="configs/sabl">Side-Aware Boundary Localization (ECCV'2020)</a></li>
            <li><a href="configs/dynamic_rcnn">Dynamic R-CNN (ECCV'2020)</a></li>
            <li><a href="configs/detr">DETR (ECCV'2020)</a></li>
            <li><a href="configs/paa">PAA (ECCV'2020)</a></li>
            <li><a href="configs/vfnet">VarifocalNet (CVPR'2021)</a></li>
            <li><a href="configs/sparse_rcnn">Sparse R-CNN (CVPR'2021)</a></li>
            <li><a href="configs/yolof">YOLOF (CVPR'2021)</a></li>
            <li><a href="configs/yolox">YOLOX (CVPR'2021)</a></li>
            <li><a href="configs/deformable_detr">Deformable DETR (ICLR'2021)</a></li>
            <li><a href="configs/tood">TOOD (ICCV'2021)</a></li>
            <li><a href="configs/ddod">DDOD (ACM MM'2021)</a></li>
            <li><a href="configs/rtmdet">RTMDet (ArXiv'2022)</a></li>
            <li><a href="configs/conditional_detr">Conditional DETR (ICCV'2021)</a></li>
            <li><a href="configs/dab_detr">DAB-DETR (ICLR'2022)</a></li>
            <li><a href="configs/dino">DINO (ICLR'2023)</a></li>
            <li><a href="configs/glip">GLIP (CVPR'2022)</a></li>
            <li><a href="configs/ddq">DDQ (CVPR'2023)</a></li>
            <li><a href="projects/DiffusionDet">DiffusionDet (ArXiv'2023)</a></li>
            <li><a href="projects/EfficientDet">EfficientDet (CVPR'2020)</a></li>
            <li><a href="projects/ViTDet">ViTDet (ECCV'2022)</a></li>
            <li><a href="projects/Detic">Detic (ECCV'2022)</a></li>
            <li><a href="projects/CO-DETR">CO-DETR (ICCV'2023)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/mask_rcnn">Mask R-CNN (ICCV'2017)</a></li>
          <li><a href="configs/cascade_rcnn">Cascade Mask R-CNN (CVPR'2018)</a></li>
          <li><a href="configs/ms_rcnn">Mask Scoring R-CNN (CVPR'2019)</a></li>
          <li><a href="configs/htc">Hybrid Task Cascade (CVPR'2019)</a></li>
          <li><a href="configs/yolact">YOLACT (ICCV'2019)</a></li>
          <li><a href="configs/instaboost">InstaBoost (ICCV'2019)</a></li>
          <li><a href="configs/solo">SOLO (ECCV'2020)</a></li>
          <li><a href="configs/point_rend">PointRend (CVPR'2020)</a></li>
          <li><a href="configs/detectors">DetectoRS (ArXiv'2020)</a></li>
          <li><a href="configs/solov2">SOLOv2 (NeurIPS'2020)</a></li>
          <li><a href="configs/scnet">SCNet (AAAI'2021)</a></li>
          <li><a href="configs/queryinst">QueryInst (ICCV'2021)</a></li>
          <li><a href="configs/mask2former">Mask2Former (ArXiv'2021)</a></li>
          <li><a href="configs/condinst">CondInst (ECCV'2020)</a></li>
          <li><a href="projects/SparseInst">SparseInst (CVPR'2022)</a></li>
          <li><a href="configs/rtmdet">RTMDet (ArXiv'2022)</a></li>
          <li><a href="configs/boxinst">BoxInst (CVPR'2021)</a></li>
          <li><a href="projects/ConvNeXt-V2">ConvNeXt-V2 (Arxiv'2023)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/panoptic_fpn">Panoptic FPN (CVPR'2019)</a></li>
          <li><a href="configs/maskformer">MaskFormer (NeurIPS'2021)</a></li>
          <li><a href="configs/mask2former">Mask2Former (ArXiv'2021)</a></li>
          <li><a href="configs/XDecoder">XDecoder (CVPR'2023)</a></li>
        </ul>
      </td>
      <td>
        </ul>
          <li><b>Contrastive Learning</b></li>
        <ul>
        <ul>
          <li><a href="configs/selfsup_pretrain">SwAV (NeurIPS'2020)</a></li>
          <li><a href="configs/selfsup_pretrain">MoCo (CVPR'2020)</a></li>
          <li><a href="configs/selfsup_pretrain">MoCov2 (ArXiv'2020)</a></li>
        </ul>
        </ul>
        </ul>
          <li><b>Distillation</b></li>
        <ul>
        <ul>
          <li><a href="configs/ld">Localization Distillation (CVPR'2022)</a></li>
          <li><a href="configs/lad">Label Assignment Distillation (WACV'2022)</a></li>
        </ul>
        </ul>
          <li><b>Semi-Supervised Object Detection</b></li>
        <ul>
        <ul>
          <li><a href="configs/soft_teacher">Soft Teacher (ICCV'2021)</a></li>
        </ul>
        </ul>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

<div align="center">
  <b>模块组件</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Loss</b>
      </td>
      <td>
        <b>Common</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>VGG (ICLR'2015)</li>
        <li>ResNet (CVPR'2016)</li>
        <li>ResNeXt (CVPR'2017)</li>
        <li>MobileNetV2 (CVPR'2018)</li>
        <li><a href="configs/hrnet">HRNet (CVPR'2019)</a></li>
        <li><a href="configs/empirical_attention">Generalized Attention (ICCV'2019)</a></li>
        <li><a href="configs/gcnet">GCNet (ICCVW'2019)</a></li>
        <li><a href="configs/res2net">Res2Net (TPAMI'2020)</a></li>
        <li><a href="configs/regnet">RegNet (CVPR'2020)</a></li>
        <li><a href="configs/resnest">ResNeSt (ArXiv'2020)</a></li>
        <li><a href="configs/pvt">PVT (ICCV'2021)</a></li>
        <li><a href="configs/swin">Swin (CVPR'2021)</a></li>
        <li><a href="configs/pvt">PVTv2 (ArXiv'2021)</a></li>
        <li><a href="configs/resnet_strikes_back">ResNet strikes back (ArXiv'2021)</a></li>
        <li><a href="configs/efficientnet">EfficientNet (ArXiv'2021)</a></li>
        <li><a href="configs/convnext">ConvNeXt (CVPR'2022)</a></li>
        <li><a href="projects/ConvNeXt-V2">ConvNeXtv2 (ArXiv'2023)</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="configs/pafpn">PAFPN (CVPR'2018)</a></li>
        <li><a href="configs/nas_fpn">NAS-FPN (CVPR'2019)</a></li>
        <li><a href="configs/carafe">CARAFE (ICCV'2019)</a></li>
        <li><a href="configs/fpg">FPG (ArXiv'2020)</a></li>
        <li><a href="configs/groie">GRoIE (ICPR'2020)</a></li>
        <li><a href="configs/dyhead">DyHead (CVPR'2021)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/ghm">GHM (AAAI'2019)</a></li>
          <li><a href="configs/gfl">Generalized Focal Loss (NeurIPS'2020)</a></li>
          <li><a href="configs/seesaw_loss">Seasaw Loss (CVPR'2021)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py">OHEM (CVPR'2016)</a></li>
          <li><a href="configs/gn">Group Normalization (ECCV'2018)</a></li>
          <li><a href="configs/dcn">DCN (ICCV'2017)</a></li>
          <li><a href="configs/dcnv2">DCNv2 (CVPR'2019)</a></li>
          <li><a href="configs/gn+ws">Weight Standardization (ArXiv'2019)</a></li>
          <li><a href="configs/pisa">Prime Sample Attention (CVPR'2020)</a></li>
          <li><a href="configs/strong_baselines">Strong Baselines (CVPR'2021)</a></li>
          <li><a href="configs/resnet_strikes_back">Resnet strikes back (ArXiv'2021)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>
我们在[基于 MMDetection 的项目](./docs/zh_cn/notes/projects.md)中列举了一些其他的支持的算法。

---

# 一、算法模型

## 1.1、Object Detection

| 序号 |                   **算法**                   | **论文名** | 其他 |
| :--: | :------------------------------------------: | :--------: | :--: |
|  1   |            Fast R-CNN (ICCV'2015)            |            |      |
|  2   |         Faster R-CNN (NeurIPS'2015)          |            |      |
|  3   |              RPN (NeurIPS'2015)              |            |      |
|  4   |               SSD (ECCV'2016)                |            |      |
|  5   |            RetinaNet (ICCV'2017)             |            |      |
|  6   |          Cascade R-CNN (CVPR'2018)           |            |      |
|  7   |             YOLOv3 (ArXiv'2018)              |            |      |
|  8   |            CornerNet (ECCV'2018)             |            |      |
|  9   |            Grid R-CNN (CVPR'2019)            |            |      |
|  10  |         Guided Anchoring (CVPR'2019)         |            |      |
|  11  |               FSAF (CVPR'2019)               |            |      |
|  12  |            CenterNet (CVPR'2019)             |            |      |
|  13  |           Libra R-CNN (CVPR'2019)            |            |      |
|  14  |            TridentNet (ICCV'2019)            |            |      |
|  15  |               FCOS (ICCV'2019)               |            |      |
|  16  |            RepPoints (ICCV'2019)             |            |      |
|  17  |          FreeAnchor (NeurIPS'2019)           |            |      |
|  18  |          CascadeRPN (NeurIPS'2019)           |            |      |
|  19  |             Foveabox (TIP'2020)              |            |      |
|  20  |        Double-Head R-CNN (CVPR'2020)         |            |      |
|  21  |               ATSS (CVPR'2020)               |            |      |
|  22  |             NAS-FCOS (CVPR'2020)             |            |      |
|  23  |          CentripetalNet (CVPR'2020)          |            |      |
|  24  |           AutoAssign (ArXiv'2020)            |            |      |
|  25  | Side-Aware Boundary Localization (ECCV'2020) |            |      |
|  26  |          Dynamic R-CNN (ECCV'2020)           |            |      |
|  27  |               DETR (ECCV'2020)               |            |      |
|  28  |               PAA (ECCV'2020)                |            |      |
|  29  |           VarifocalNet (CVPR'2021)           |            |      |
|  30  |           Sparse R-CNN (CVPR'2021)           |            |      |
|  31  |              YOLOF (CVPR'2021)               |            |      |
|  32  |              YOLOX (CVPR'2021)               |            |      |
|  33  |         Deformable DETR (ICLR'2021)          |            |      |
|  34  |               TOOD (ICCV'2021)               |            |      |
|  35  |              DDOD (ACM MM'2021)              |            |      |
|  36  |             RTMDet (ArXiv'2022)              |            |      |
|  37  |         Conditional DETR (ICCV'2021)         |            |      |
|  38  |             DAB-DETR (ICLR'2022)             |            |      |
|  39  |               DINO (ICLR'2023)               |            |      |
|  40  |               GLIP (CVPR'2022)               |            |      |
|  41  |               DDQ (CVPR'2023)                |            |      |
|  42  |          DiffusionDet (ArXiv'2023)           |            |      |
|  43  |           EfficientDet (CVPR'2020)           |            |      |
|  44  |              ViTDet (ECCV'2022)              |            |      |
|  45  |              Detic (ECCV'2022)               |            |      |
|  46  |             CO-DETR (ICCV'2023)              |            |      |
|      |                                              |            |      |

---

## 1.2、Instance Segmentation

| 序号 |            **算法**             | **论文名** | 其他 |
| :--: | :-----------------------------: | :--------: | :--: |
|  1   |     Mask R-CNN (ICCV'2017)      |            |      |
|  2   | Cascade Mask R-CNN (CVPR'2018)  |            |      |
|  3   | Mask Scoring R-CNN (CVPR'2019)  |            |      |
|  4   | Hybrid Task Cascade (CVPR'2019) |            |      |
|  5   |       YOLACT (ICCV'2019)        |            |      |
|  6   |     InstaBoost (ICCV'2019)      |            |      |
|  7   |        SOLO (ECCV'2020)         |            |      |
|  8   |      PointRend (CVPR'2020)      |            |      |
|  9   |     DetectoRS (ArXiv'2020)      |            |      |
|  10  |      SOLOv2 (NeurIPS'2020)      |            |      |
|  11  |        SCNet (AAAI'2021)        |            |      |
|  12  |      QueryInst (ICCV'2021)      |            |      |
|  13  |    Mask2Former (ArXiv'2021)     |            |      |
|  14  |      CondInst (ECCV'2020)       |            |      |
|  15  |     SparseInst (CVPR'2022)      |            |      |
|  16  |       RTMDet (ArXiv'2022)       |            |      |
|  17  |       BoxInst (CVPR'2021)       |            |      |
|  18  |    ConvNeXt-V2 (Arxiv'2023)     |            |      |
|      |                                 |            |      |

----

## 1.3、Panoptic Segmentation

| 序号 |         **算法**          | **论文名** | 其他 |
| :--: | :-----------------------: | :--------: | :--: |
|  1   | Panoptic FPN (CVPR'2019)  |            |      |
|  2   | MaskFormer (NeurIPS'2021) |            |      |
|  3   | Mask2Former (ArXiv'2021)  |            |      |
|  4   |   XDecoder (CVPR'2023)    |            |      |
|      |                           |            |      |

---

## 1.4、Other

### 1.4.1 Contrastive Learning

| 序号 |      **算法**       | **论文名** | 其他 |
| :--: | :-----------------: | :--------: | :--: |
|  1   | SwAV (NeurIPS'2020) |            |      |
|  2   |  MoCo (CVPR'2020)   |            |      |
|  3   | MoCov2 (ArXiv'2020) |            |      |
|      |                     |            |      |

### 1.4.2 Distillation

| 序号 |                 **算法**                  | **论文名** | 其他 |
| :--: | :---------------------------------------: | :--------: | :--: |
|  1   |   Localization Distillation (CVPR'2022)   |            |      |
|  2   | Label Assignment Distillation (WACV'2022) |            |      |
|      |                                           |            |      |

### 1.4.3 Semi-Supervised Object Detection

| 序号 |         **算法**         | **论文名** | 其他 |
| :--: | :----------------------: | :--------: | :--: |
|  1   | Soft Teacher (ICCV'2021) |            |      |
|      |                          |            |      |

---

# 二、模块组件

## 2.1 Backbones

| 序号 |           **backbone**            | **论文名** | 其他 |
| :--: | :-------------------------------: | :--------: | :--: |
|  1   |          VGG (ICLR'2015)          |            |      |
|  2   |        ResNet (CVPR'2016)         |            |      |
|  3   |        ResNeXt (CVPR'2017)        |            |      |
|  4   |      MobileNetV2 (CVPR'2018)      |            |      |
|  5   |         HRNet (CVPR'2019)         |            |      |
|  6   | Generalized Attention (ICCV'2019) |            |      |
|  7   |        GCNet (ICCVW'2019)         |            |      |
|  8   |       Res2Net (TPAMI'2020)        |            |      |
|  9   |        RegNet (CVPR'2020)         |            |      |
|  10  |       ResNeSt (ArXiv'2020)        |            |      |
|  11  |          PVT (ICCV'2021)          |            |      |
|  12  |         Swin (CVPR'2021)          |            |      |
|  13  |        PVTv2 (ArXiv'2021)         |            |      |
|  14  | ResNet strikes back (ArXiv'2021)  |            |      |
|  15  |     EfficientNet (ArXiv'2021)     |            |      |
|  16  |       ConvNeXt (CVPR'2022)        |            |      |
|  17  |      ConvNeXtv2 (ArXiv'2023)      |            |      |

---

## 2.2 Necks

| 序号 |      **necks**      | **论文名** | 其他 |
| :--: | :-----------------: | :--------: | :--: |
|  1   |  PAFPN (CVPR'2018)  |            |      |
|  2   | NAS-FPN (CVPR'2019) |            |      |
|  3   | CARAFE (ICCV'2019)  |            |      |
|  4   |  FPG (ArXiv'2020)   |            |      |
|  5   |  GRoIE (ICPR'2020)  |            |      |
|  6   | DyHead (CVPR'2021)  |            |      |

---

## 2.3 Loss

| 序号 |               **loss**                | **论文名** | 其他 |
| :--: | :-----------------------------------: | :--------: | :--: |
|  1   |            GHM (AAAI'2019)            |            |      |
|  2   | Generalized Focal Loss (NeurIPS'2020) |            |      |
|  3   |        Seasaw Loss (CVPR'2021)        |            |      |

---

## 2.4 Common

| 序号 |              **loss**               | **论文名** | 其他 |
| :--: | :---------------------------------: | :--------: | :--: |
|  1   |          OHEM (CVPR'2016)           |            |      |
|  2   |           DCN (ICCV'2017)           |            |      |
|  3   |   Group Normalization (ECCV'2018)   |            |      |
|  4   |          DCNv2 (CVPR'2019)          |            |      |
|  5   | Weight Standardization (ArXiv'2019) |            |      |
|  6   | Prime Sample Attention (CVPR'2020)  |            |      |
|  7   |    Strong Baselines (CVPR'2021)     |            |      |
|  8   |  Resnet strikes back (ArXiv'2021)   |            |      |
|      |                                     |            |      |