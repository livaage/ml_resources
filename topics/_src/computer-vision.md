---
title: Computer Vision — ML Resources Hub
eyebrow_text: ← Theory · Applications
eyebrow_href: ../theory.html
heading: Computer Vision
lead: Classifying, detecting, segmenting, generating — what models do with pixels.
---

<section class="topic-level active" data-level="intuition" markdown="1">

<div class="key-idea" markdown="1">

<span class="key-idea-label">Key idea</span>

**Images are grids of numbers; vision models find structure in them.** Early layers spot edges; middle layers find textures; later ones recognise objects. The architecture (CNN, ViT) and the training objective (classification, segmentation, generation) together determine what the model can do — but they all start by hierarchically composing simple features into complex ones.

</div>

<div class="viz-embed viz-classic">
<div class="viz-embed-header">
<span class="viz-embed-title">Apply classic data augmentations — the same image, transformed, becomes many training examples</span>
</div>
<div class="viz-classic-controls">
<button id="viz-cv-orig" type="button" class="active">Original</button>
<button id="viz-cv-flip" type="button">Flip</button>
<button id="viz-cv-crop" type="button">Random crop</button>
<button id="viz-cv-rotate" type="button">Rotate</button>
<button id="viz-cv-noise" type="button">Gaussian noise</button>
<button id="viz-cv-blur" type="button">Blur</button>
</div>
<div class="viz-classic-canvas-wrap">
<canvas id="viz-cv-canvas"></canvas>
</div>
<div class="viz-classic-caption" id="viz-cv-caption"></div>
</div>

<script src="{{root}}js/viz/computer-vision.js"></script>

A simple 28×28 image with each classic augmentation. Augmentations are how vision models get data efficiency — one labelled image becomes hundreds of training examples. The model learns to be invariant to the things you augment over: small translations, rotations, lighting changes, colour shifts, partial occlusion.
{: .viz-intro }

<article class="tldr-body" markdown="1">

**Classification.** Predict one (or top-k) label for the whole image. ImageNet-style training. CNNs (ResNet, EfficientNet) dominated; ViTs caught up at scale.

**Object detection.** Localise objects with bounding boxes + class labels. Two-stage (Faster R-CNN) vs one-stage (YOLO, RetinaNet, DETR). YOLO is fast; DETR-family is end-to-end with attention.

**Segmentation.** Per-pixel class label. Semantic segmentation (label per pixel) vs instance segmentation (label per pixel + instance ID). U-Net, Mask R-CNN, SAM are landmarks.

**Generation.** Make new images from text, sketches, or noise. Diffusion models dominate now (Stable Diffusion, Imagen, DALL-E 3). See the [Generative Models](generative-models.html) page.

**Self-supervised pretraining.** The backbone for most modern vision. DINO, MAE, CLIP — all produce strong general-purpose vision features that beat training from scratch on most downstream tasks.

</article>

<div class="use-cases" markdown="1">

<div class="yes" markdown="1">

### Common vision tasks

- **Classification**: ResNet, ConvNeXt, ViT, EfficientNet
- **Detection**: YOLO (fast), Faster R-CNN (accurate), DETR (end-to-end)
- **Segmentation**: U-Net (medical), SAM (general), Mask R-CNN
- **Pose / keypoints**: OpenPose, Detectron2, MediaPipe
- **Generation**: Stable Diffusion, Flux, DALL-E 3

</div>

<div class="no" markdown="1">

### Watch out

- Distribution shift kills production models — different lighting, sensors, populations
- Adversarial perturbations exist — important for safety-critical deployment
- Annotation is expensive — bounding boxes, masks, keypoints all cost
- "Fairness" issues — face datasets are biased; deployment matters

</div>

</div>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch, torchvision
from torchvision import transforms

# Classic augmentation pipeline
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),    # ImageNet stats
])

# Pretrained backbone + a new head
model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
```

</div>

<div class="level-next">
<span>Want ViTs, DETR, segmentation, & vision foundation models?</span>
<button data-go-to="fundamentals" type="button">Switch to Standard →</button>
</div>

</section>

<section class="topic-level" data-level="fundamentals" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Convolution</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ (f * g)[i, j] = \sum_{m, n} f[i - m,\, j - n] \cdot g[m, n] $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`f`input image (a grid of pixel values)

</li>
<li markdown="1">

`g`convolution kernel — a small learned matrix of weights

</li>
<li markdown="1">

`(f * g)`output feature map at position *(i, j)*

</li>
<li markdown="1">

Local, weight-shared, translation-equivariant

</li>
<li markdown="1">

The inductive bias that made deep learning work for vision; ViTs partially abandon it for self-attention

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{output}[i, j] \;=\; \sum_{m, n} \text{image}[i - m,\, j - n] \,\times\, \text{kernel}[m, n] $$</span>

**In words.** At each output pixel *(i, j)*, you slide a small **kernel** (a tiny grid of learned weights, often 3×3 or 5×5) over the image, multiply matching cells, and add them up. The `Σ` (sigma) sign just means "add up across all positions *m, n* in the kernel". Because the same kernel is reused everywhere, the operation is *translation-equivariant* — shifting the image shifts the output the same way. Each layer of a CNN applies many such kernels in parallel, each learning to spot a different pattern (an edge, a texture, eventually faces or wheels).
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`image`input grid of pixel values

</li>
<li markdown="1">

`kernel`small grid of learned weights

</li>
<li markdown="1">

`output[i, j]`value at position *(i, j)* after sliding the kernel and summing the products

</li>
<li markdown="1">

Same kernel reused at every position — that's the "weight sharing" that makes CNNs efficient

</li>
<li markdown="1">

Translation-equivariant: shifting the image shifts the output identically

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**CNNs (ResNet family).** Convolutional layers — local kernels, weight-shared, translation-equivariant. Residual connections (He et al. 2016) made very deep networks trainable. ResNet-50 was the default vision backbone for years.

**Vision Transformers (ViT).** Split image into patches, treat each as a token, run a transformer. Dosovitskiy et al. (2020). Scales beautifully with data and compute; matches or beats CNNs above ~100M images.

**Object detection: from R-CNN to DETR.** R-CNN: extract proposals, classify each. Faster R-CNN: shared backbone + region proposal network. YOLO: single forward pass, dense predictions. DETR: end-to-end transformer with bipartite matching. RT-DETR, DINO-DETR are recent transformers detection SOTA.

**Segmentation.** Fully Convolutional Networks (Long et al. 2015) — the original. U-Net (Ronneberger et al. 2015) — encoder-decoder with skip connections, dominant in medical imaging. SAM (Kirillov et al. 2023) — Meta's "segment anything" foundation model with promptable segmentation.

**CLIP and multi-modal.** Radford et al. (2021). Image and text encoders trained contrastively on (image, caption) pairs. Aligns image and text in a shared embedding space — enables zero-shot classification by similarity to text prompts. Foundation for modern multi-modal models (LLaVA, GPT-4V).

**Data augmentation as regularization.** Beyond simple flips and crops: CutMix, MixUp, AugMix, RandAugment. Strong augmentation is a key reason modern vision models train so data-efficiently — the model is forced to learn meaningful features rather than surface texture.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

# Vision Transformer fine-tuning
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model     = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True,
)

inputs = processor(images=batch_pil, return_tensors="pt")
out    = model(**inputs, labels=labels)
out.loss.backward()
```

</div>

<div class="level-next">
<span>Want video, 3D, dense prediction, NeRF, and diffusion-based generation?</span>
<button data-go-to="indepth" type="button">Switch to In-depth →</button>
</div>

</section>

<section class="topic-level" data-level="indepth" markdown="1">

<div class="key-idea formula-block" markdown="1">

<span class="key-idea-label">Neural Radiance Field (NeRF)</span>

<div class="notation-standard" markdown="1">

<span class="formula">$$ C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \sigma(\mathbf{r}(t))\, \mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt $$</span>

<ul class="formula-legend" markdown="1">
<li markdown="1">

`C(r)`colour of pixel for camera ray *r*

</li>
<li markdown="1">

`σ(·)`density (how solid the scene is) at a 3D point

</li>
<li markdown="1">

`c(·)`emitted colour at a 3D point, viewed from direction *d*

</li>
<li markdown="1">

`T(t)`transmittance — how much light still reaches the camera from depth *t*

</li>
<li markdown="1">

Volumetric rendering integral parametrised by an MLP — 3D reconstruction from 2D images

</li>
</ul>

</div>

<div class="notation-plain" markdown="1">

<span class="formula">$$ \text{pixel colour} \;=\; \int_{\text{near}}^{\text{far}} (\text{light still reaching camera}) \,\times\, (\text{density}) \,\times\, (\text{colour at that point})\; dt $$</span>

**In words.** To work out the colour of a pixel, you shoot a ray from the camera through the pixel into the scene, and integrate ("add up continuously" — that's what `∫` means) the colour contributions along the ray. At each 3D point along the way, a small neural net predicts its **density** (how opaque the scene is there) and its **colour**. The `T(t)` factor accounts for the fact that points behind solid stuff contribute less — light is blocked. The whole integral gives you one pixel; do it for every pixel and you've rendered the scene from a new viewpoint.
{: .formula-narration }

<ul class="formula-legend" markdown="1">
<li markdown="1">

`pixel colour`final RGB value for the pixel the ray passes through

</li>
<li markdown="1">

`density`how "solid" the scene is at a 3D point — learned by the MLP

</li>
<li markdown="1">

`colour at that point`view-dependent RGB emitted by that point — also from the MLP

</li>
<li markdown="1">

`light still reaching camera`shrinks as the ray passes through dense regions in front

</li>
<li markdown="1">

3D reconstruction from 2D images — and the basis for many 3D-aware models

</li>
</ul>

</div>

</div>

<article class="tldr-body" markdown="1">

**Video understanding.** Temporal context matters. Approaches: 3D CNNs (I3D, SlowFast), two-stream (RGB + optical flow), video transformers (TimeSformer, ViViT), masked video modelling. Video is also a great pre-training signal — the future-prediction objective transfers to many downstream tasks.

**Dense prediction.** Depth estimation (MiDaS, Marigold), surface normals, optical flow (RAFT), keypoint detection. Often shared backbone + task-specific head. Modern approach: pretrain on huge unlabelled video, fine-tune for the specific dense task.

**3D from 2D.** NeRF (Mildenhall et al. 2020) and its descendants represent a scene as a continuous field over 3D coordinates, queried by volume rendering. Gaussian Splatting (Kerbl et al. 2023) is faster, doesn't use an MLP. Both reconstruct 3D from multi-view images.

**Diffusion for vision.** Stable Diffusion, SDXL, Flux, DALL-E 3, Midjourney — all latent diffusion models. Text-to-image, image-to-image, inpainting, control via additional conditioning (ControlNet, T2I-Adapter). See [Diffusion Models](neural-networks/diffusion.html).

**Foundation models for vision.** CLIP, DINOv2, SAM, Florence-2. Pretrained encoders that you build downstream tasks on top of. The shift from "train your own classifier from scratch" to "embed with a foundation model + linear head" has been complete in industry.

**Vision-language models.** LLaVA, MiniGPT, GPT-4V, Gemini, Claude. A vision encoder feeds image tokens into an LLM. Enables open-ended image understanding — describe, reason, answer. The bridge between vision and reasoning.

</article>

<div class="code-snippet" markdown="1">

<button class="copy-btn" type="button">Copy</button>

```python
from segment_anything import SamPredictor, sam_model_registry
import cv2

# SAM — segment anything with a single prompt point
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

image = cv2.imread("photo.jpg")
predictor.set_image(image)

# Prompt: a single point on the object of interest
mask, scores, _ = predictor.predict(
    point_coords=[[640, 480]],   # x, y
    point_labels=[1],            # 1 = positive (this point is the object)
    multimask_output=True,
)
```

</div>

<div class="level-next">
<span>Too dense?</span>
<button data-go-to="fundamentals" type="button">← Back to Standard</button>
</div>

</section>

<!-- TOPIC SIDEBAR -->

<div class="learn-more" markdown="1">

### Where to learn more

<ul markdown="1">
<li data-tier="fundamentals" markdown="1">

[Stanford CS231n — CNNs for Visual Recognition <i class="fas fa-external-link-alt"></i>](http://cs231n.stanford.edu/){: target="_blank" }
<span class="annotation">Karpathy's course on deep learning for vision. Notes are still the best from-scratch introduction; slides are continuously updated.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[Dosovitskiy et al. (2020) — ViT <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2010.11929){: target="_blank" }
<span class="annotation">The vision transformer paper. Short, well-written; introduced the patch-as-token recipe that now dominates.</span>

</li>
<li data-tier="indepth" markdown="1">

[Kirillov et al. (2023) — Segment Anything <i class="fas fa-external-link-alt"></i>](https://arxiv.org/abs/2304.02643){: target="_blank" }
<span class="annotation">Meta's SAM. The foundation-model approach to segmentation; the SA-1B dataset is itself remarkable.</span>

</li>
<li data-tier="fundamentals" markdown="1">

[torchvision <i class="fas fa-external-link-alt"></i>](https://pytorch.org/vision/stable/index.html){: target="_blank" }
<span class="annotation">PyTorch's vision library — pretrained models, datasets, augmentations. Best one-stop reference for getting hands on practical vision pipelines.</span>

</li>
</ul>

</div>
