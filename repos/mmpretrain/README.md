# Model zoo

Results and models are available in the [model zoo](https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html).

<div align="center">
  <b>Overview</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Supported Backbones</b>
      </td>
      <td>
        <b>Self-supervised Learning</b>
      </td>
      <td>
        <b>Multi-Modality Algorithms</b>
      </td>
      <td>
        <b>Others</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <li><a href="configs/vgg">VGG</a></li>
        <li><a href="configs/resnet">ResNet</a></li>
        <li><a href="configs/resnext">ResNeXt</a></li>
        <li><a href="configs/seresnet">SE-ResNet</a></li>
        <li><a href="configs/seresnet">SE-ResNeXt</a></li>
        <li><a href="configs/regnet">RegNet</a></li>
        <li><a href="configs/shufflenet_v1">ShuffleNet V1</a></li>
        <li><a href="configs/shufflenet_v2">ShuffleNet V2</a></li>
        <li><a href="configs/mobilenet_v2">MobileNet V2</a></li>
        <li><a href="configs/mobilenet_v3">MobileNet V3</a></li>
        <li><a href="configs/swin_transformer">Swin-Transformer</a></li>
        <li><a href="configs/swin_transformer_v2">Swin-Transformer V2</a></li>
        <li><a href="configs/repvgg">RepVGG</a></li>
        <li><a href="configs/vision_transformer">Vision-Transformer</a></li>
        <li><a href="configs/tnt">Transformer-in-Transformer</a></li>
        <li><a href="configs/res2net">Res2Net</a></li>
        <li><a href="configs/mlp_mixer">MLP-Mixer</a></li>
        <li><a href="configs/deit">DeiT</a></li>
        <li><a href="configs/deit3">DeiT-3</a></li>
        <li><a href="configs/conformer">Conformer</a></li>
        <li><a href="configs/t2t_vit">T2T-ViT</a></li>
        <li><a href="configs/twins">Twins</a></li>
        <li><a href="configs/efficientnet">EfficientNet</a></li>
        <li><a href="configs/edgenext">EdgeNeXt</a></li>
        <li><a href="configs/convnext">ConvNeXt</a></li>
        <li><a href="configs/hrnet">HRNet</a></li>
        <li><a href="configs/van">VAN</a></li>
        <li><a href="configs/convmixer">ConvMixer</a></li>
        <li><a href="configs/cspnet">CSPNet</a></li>
        <li><a href="configs/poolformer">PoolFormer</a></li>
        <li><a href="configs/inception_v3">Inception V3</a></li>
        <li><a href="configs/mobileone">MobileOne</a></li>
        <li><a href="configs/efficientformer">EfficientFormer</a></li>
        <li><a href="configs/mvit">MViT</a></li>
        <li><a href="configs/hornet">HorNet</a></li>
        <li><a href="configs/mobilevit">MobileViT</a></li>
        <li><a href="configs/davit">DaViT</a></li>
        <li><a href="configs/replknet">RepLKNet</a></li>
        <li><a href="configs/beit">BEiT</a></li>
        <li><a href="configs/mixmim">MixMIM</a></li>
        <li><a href="configs/efficientnet_v2">EfficientNet V2</a></li>
        <li><a href="configs/revvit">RevViT</a></li>
        <li><a href="configs/convnext_v2">ConvNeXt V2</a></li>
        <li><a href="configs/vig">ViG</a></li>
        <li><a href="configs/xcit">XCiT</a></li>
        <li><a href="configs/levit">LeViT</a></li>
        <li><a href="configs/riformer">RIFormer</a></li>
        <li><a href="configs/glip">GLIP</a></li>
        <li><a href="configs/sam">ViT SAM</a></li>
        <li><a href="configs/eva02">EVA02</a></li>
        <li><a href="configs/dinov2">DINO V2</a></li>
        <li><a href="configs/hivit">HiViT</a></li>
        </ul>
      </td>
      <td>
        <ul>
        <li><a href="configs/mocov2">MoCo V1 (CVPR'2020)</a></li>
        <li><a href="configs/simclr">SimCLR (ICML'2020)</a></li>
        <li><a href="configs/mocov2">MoCo V2 (arXiv'2020)</a></li>
        <li><a href="configs/byol">BYOL (NeurIPS'2020)</a></li>
        <li><a href="configs/swav">SwAV (NeurIPS'2020)</a></li>
        <li><a href="configs/densecl">DenseCL (CVPR'2021)</a></li>
        <li><a href="configs/simsiam">SimSiam (CVPR'2021)</a></li>
        <li><a href="configs/barlowtwins">Barlow Twins (ICML'2021)</a></li>
        <li><a href="configs/mocov3">MoCo V3 (ICCV'2021)</a></li>
        <li><a href="configs/beit">BEiT (ICLR'2022)</a></li>
        <li><a href="configs/mae">MAE (CVPR'2022)</a></li>
        <li><a href="configs/simmim">SimMIM (CVPR'2022)</a></li>
        <li><a href="configs/maskfeat">MaskFeat (CVPR'2022)</a></li>
        <li><a href="configs/cae">CAE (arXiv'2022)</a></li>
        <li><a href="configs/milan">MILAN (arXiv'2022)</a></li>
        <li><a href="configs/beitv2">BEiT V2 (arXiv'2022)</a></li>
        <li><a href="configs/eva">EVA (CVPR'2023)</a></li>
        <li><a href="configs/mixmim">MixMIM (arXiv'2022)</a></li>
        <li><a href="configs/itpn">iTPN (CVPR'2023)</a></li>
        <li><a href="configs/spark">SparK (ICLR'2023)</a></li>
        <li><a href="configs/mff">MFF (ICCV'2023)</a></li>
        </ul>
      </td>
      <td>
        <ul>
        <li><a href="configs/blip">BLIP (arxiv'2022)</a></li>
        <li><a href="configs/blip2">BLIP-2 (arxiv'2023)</a></li>
        <li><a href="configs/ofa">OFA (CoRR'2022)</a></li>
        <li><a href="configs/flamingo">Flamingo (NeurIPS'2022)</a></li>
        <li><a href="configs/chinese_clip">Chinese CLIP (arxiv'2022)</a></li>
        <li><a href="configs/minigpt4">MiniGPT-4 (arxiv'2023)</a></li>
        <li><a href="configs/llava">LLaVA (arxiv'2023)</a></li>
        <li><a href="configs/otter">Otter (arxiv'2023)</a></li>
        </ul>
      </td>
      <td>
      Image Retrieval Task:
        <ul>
        <li><a href="configs/arcface">ArcFace (CVPR'2019)</a></li>
        </ul>
      Training&Test Tips:
        <ul>
        <li><a href="https://arxiv.org/abs/1909.13719">RandAug</a></li>
        <li><a href="https://arxiv.org/abs/1805.09501">AutoAug</a></li>
        <li><a href="mmpretrain/datasets/samplers/repeat_aug.py">RepeatAugSampler</a></li>
        <li><a href="mmpretrain/models/tta/score_tta.py">TTA</a></li>
        <li>...</li>
        </ul>
      </td>
  </tbody>
</table>

## 1、Supported Backbones

| 序号 |          **算法**          | **论文名** | 其他 |
| :--: | :------------------------: | :--------: | :--: |
|  1   |            VGG             |            |      |
|  2   |           ResNet           |            |      |
|  3   |          ResNeXt           |            |      |
|  4   |         SE-ResNet          |            |      |
|  5   |         SE-ResNeXt         |            |      |
|  6   |           RegNet           |            |      |
|  7   |       ShuffleNet V1        |            |      |
|  8   |       ShuffleNet V2        |            |      |
|  9   |        MobileNet V2        |            |      |
|  10  |        MobileNet V3        |            |      |
|  11  |      Swin-Transformer      |            |      |
|  12  |    Swin-Transformer V2     |            |      |
|  13  |           RepVGG           |            |      |
|  14  |     Vision-Transformer     |            |      |
|  15  | Transformer-in-Transformer |            |      |
|  16  |          Res2Net           |            |      |
|  17  |         MLP-Mixer          |            |      |
|  18  |            DeiT            |            |      |
|  19  |           DeiT-3           |            |      |
|  20  |         Conformer          |            |      |
|  21  |          T2T-ViT           |            |      |
|  22  |           Twins            |            |      |
|  23  |        EfficientNet        |            |      |
|  24  |          EdgeNeXt          |            |      |
|  25  |          ConvNeXt          |            |      |
|  26  |           HRNet            |            |      |
|  27  |            VAN             |            |      |
|  28  |         ConvMixer          |            |      |
|  29  |           CSPNet           |            |      |
|  30  |         PoolFormer         |            |      |
|  31  |        Inception V3        |            |      |
|  32  |         MobileOne          |            |      |
|  33  |      EfficientFormer       |            |      |
|  34  |            MViT            |            |      |
|  35  |           HorNet           |            |      |
|  36  |         MobileViT          |            |      |
|  37  |           DaViT            |            |      |
|  38  |          RepLKNet          |            |      |
|  39  |            BEiT            |            |      |
|  40  |           MixMIM           |            |      |
|  41  |      EfficientNet V2       |            |      |
|  42  |           RevViT           |            |      |
|  43  |        ConvNeXt V2         |            |      |
|  44  |            ViG             |            |      |
|  45  |            XCiT            |            |      |
|  46  |           LeViT            |            |      |
|  47  |          RIFormer          |            |      |
|  48  |            GLIP            |            |      |
|  49  |          ViT SAM           |            |      |
|  50  |           EVA02            |            |      |
|  51  |          DINO V2           |            |      |
|  52  |           HiViT            |            |      |

---

## 2、Self-supervised Learning

| 序号 |         **算法**         | **论文名** | 其他 |
| :--: | :----------------------: | :--------: | :--: |
|  1   |   MoCo V1 (CVPR'2020)    |            |      |
|  2   |    SimCLR (ICML'2020)    |            |      |
|  3   |   MoCo V2 (arXiv'2020)   |            |      |
|  4   |   BYOL (NeurIPS'2020)    |            |      |
|  5   |   SwAV (NeurIPS'2020)    |            |      |
|  6   |   DenseCL (CVPR'2021)    |            |      |
|  7   |   SimSiam (CVPR'2021)    |            |      |
|  8   | Barlow Twins (ICML'2021) |            |      |
|  9   |   MoCo V3 (ICCV'2021)    |            |      |
|  10  |     BEiT (ICLR'2022)     |            |      |
|  11  |     MAE (CVPR'2022)      |            |      |
|  12  |    SimMIM (CVPR'2022)    |            |      |
|  13  |   MaskFeat (CVPR'2022)   |            |      |
|  14  |     CAE (arXiv'2022)     |            |      |
|  15  |    MILAN (arXiv'2022)    |            |      |
|  16  |   BEiT V2 (arXiv'2022)   |            |      |
|  17  |     EVA (CVPR'2023)      |            |      |
|  18  |   MixMIM (arXiv'2022)    |            |      |
|  19  |     iTPN (CVPR'2023)     |            |      |
|  20  |    SparK (ICLR'2023)     |            |      |
|  21  |     MFF (ICCV'2023)      |            |      |

## 3、Multi-Modality Algorithms

| 序号 |         **算法**          | **论文名** | 其他 |
| :--: | :-----------------------: | :--------: | :--: |
|  1   |     BLIP (arxiv'2022)     |            |      |
|  2   |    BLIP-2 (arxiv'2023)    |            |      |
|  3   |      OFA (CoRR'2022)      |            |      |
|  4   |  Flamingo (NeurIPS'2022)  |            |      |
|  5   | Chinese CLIP (arxiv'2022) |            |      |
|  6   |  MiniGPT-4 (arxiv'2023)   |            |      |
|  7   |    LLaVA (arxiv'2023)     |            |      |
|  8   |    Otter (arxiv'2023)     |            |      |
|      |                           |            |      |

## 4、Others

### 4.1、Image Retrieval Task:

| 序号 |      **算法**       | **论文名** | 其他 |
| :--: | :-----------------: | :--------: | :--: |
|  1   | ArcFace (CVPR'2019) |            |      |
|      |                     |            |      |

### 4.2、 Training&Test Tips:

| 序号 |     **算法**     | **论文名** | 其他 |
| :--: | :--------------: | :--------: | :--: |
|  1   |     RandAug      |            |      |
|  2   |     AutoAug      |            |      |
|  3   | RepeatAugSampler |            |      |
|  4   |       TTA        |            |      |
|      |                  |            |      |