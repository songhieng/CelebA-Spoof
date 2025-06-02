## Overview

The **CelebA-Spoof** repository is centered around a large-scale face anti-spoofing dataset accompanied by a multi-task deep learning framework, **AENet** (Auxiliary Information Embedding Network). Anti-spoofing aims to distinguish between genuine (live) faces and various spoofing attacks—such as printed photos, replayed videos on a screen, or 3D masks—that are presented to a face‐recognition system. This repository provides:

1. **The CelebA-Spoof dataset**: over 625,000 images of more than 10,000 subjects, annotated with both face attributes (inherited from CelebA) and spoof‐related labels. ([GitHub][1])
2. **AENet implementation**: a simple yet effective multi-task CNN that leverages auxiliary semantic and geometric information (depth/reflection) alongside the primary live/spoof classification. ([ECVA][2])
3. **Inference and training code**: scripts and modules in the `intra_dataset_code/` folder for running experiments on the dataset.

---

## Dataset Details

* **Size and Diversity**

  * **625,537 images** from **10,177** unique identities.
  * **Live** images are sampled from the original CelebA dataset.
  * **Spoof** images are collected and annotated, covering multiple spoof types (e.g., print, replay on phone/tablet/PC, 3D masks).
  * To enhance real‐world diversity, spoof images come from **8 scenes** (2 environments × 4 illumination conditions) and are captured using over **10 different device types** (e.g., various phones, tablets, cameras). ([GitHub][1])

* **Data Collection Dimensions**

  1. **Angles (5 types)**: vertical, down, up, forward, backward (inclinations between –30° and 30°).
  2. **Shapes (4 types)**: "normal" (full face), "inside" (cropped inside the facial region), "outside" (cropped outside), and "corner" (oblique crop).
  3. **Sensors (4 broad categories)**: PC, camera, tablet, phone. In total, 24 distinct devices were used to capture spoof images. ([GitHub][1])

* **Dataset Splits & Protocols**

  * The authors establish **three standard benchmarks** (intra-, cross-spoof-type, and cross-environment testing) to evaluate generalization. By default, most users start with the **intra-dataset protocol**, which uses all spoof types, lighting conditions, and environments in both training and testing.

---

## Annotations and Data Model

All annotations are stored in JSON‐formatted files that accompany the image folders. Each image entry typically includes:

1. **Face Attributes (0–39 indices)**: Inherited (40 attributes) from CelebA—such as “smiling,” “wavy hair,” “eyeglasses,” and other facial components/appearance features.
2. **Spoof‐Related Attributes (indices 40–42)**:

   * **Spoof Type** (index 40): e.g., `Live`, `Photo` (print), `Poster`, `A4`, `FaceMask`, `UpperBodyMask`, `RegionMask`, `PC`, `Pad`, `Phone`, `3DMask`.
   * **Illumination Condition** (index 41): e.g., `Live`, `Normal`, `StrongBack`, `Dark`.
   * **Environment** (index 42): e.g., `Live`, `Indoor`, `Outdoor`.
3. **Live/Spoof Label (index 43)**: Binary indicator (`0` for live, `1` for spoof). ([GitHub][1])

Hence, each JSON annotation line might look like:

```json
{
  "image_id": "000001.jpg",
  "attributes": [ /* 0-39 face attributes */ ],
  "spoof_type": 3,
  "illumination": 1,
  "environment": 2,
  "label": 1
}
```

* **Interpretation**:

  * Entries 0–39: Boolean or categorical flags for individual facial attributes (e.g., “has\_glasses”: 1).
  * Entry 40: Index specifying the spoof mechanism (e.g., 3 = “A4 print”).
  * Entry 41: Illumination condition index.
  * Entry 42: Environment index.
  * Entry 43: Live/Spoof binary label.

---

## AENet Model

### Core Idea

AENet is a **multi-task convolutional neural network** that simultaneously learns:

* **Primary Task**: Live/Spoof classification.
* **Auxiliary Semantic Task**: Predicting face attributes (inherited from CelebA) and spoof type.
* **Auxiliary Geometric Task**: Estimating geometric cues such as a depth map and reflection map.

By leveraging both semantic and geometric side information, AENet improves robustness against varied spoofing attacks and illumination conditions. ([ECVA][2])

### Architecture (Simplified)

1. **Backbone CNN**

   * Typically a lightweight CNN (e.g., a custom “CNN4” architecture) or a modified ResNet.
2. **Shared Feature Extractor**

   * The backbone outputs a feature map (e.g., shape `[C×H×W]`).
3. **Task‐Specific Heads**

   * **Live/Spoof Head**: Fully connected layers producing a 2‐way classification logits.
   * **Face Attribute Head**: Multi‐label classification over 40 attributes.
   * **Spoof Type Head**: Multi‐class classification over the defined spoof categories (e.g., print, 3D mask, etc.).
   * **Depth & Reflection Heads** (geometric): Regression branches that predict pixel‐wise depth and reflection maps.

Throughout training, a **combined loss** is computed:

$$
\mathcal{L} = \lambda_{cls} \, \mathcal{L}_{\text{live/spoof}} + \lambda_{sem} \, \mathcal{L}_{\text{semantic}} + \lambda_{geo} \, \mathcal{L}_{\text{geometric}}\,,
$$

where each λ controls the relative weight of the classification, semantic, and geometric tasks. ([ECVA][2])

### Why Auxiliary Tasks Help

* **Semantic cues** (face attributes, spoof type) inform the network about subtle differences: for example, a photo replay on a phone introduces characteristic artifacts that differ from a printed mask.
* **Geometric cues** (depth & reflection) leverage the fact that a live face has natural depth contours and specular reflections, whereas flat surfaces (print paper) or screens have very different depth/reflection patterns.

---

## Code Structure

At the top level, the repository contains:

```
├── fig/                         # Figures illustrating data collection & AENet architecture
├── intra_dataset_code/          # Training and inference code for AENet on CelebA-Spoof
│   ├── models.py                # Defines AENet (backbone + task heads)
│   ├── dataset.py               # Custom PyTorch Dataset to load images + JSON labels
│   ├── train.py                 # Training script (parses args, sets up data loaders, instantiates AENet, etc.)
│   ├── inference.py             # Inference/evaluation script for validation/testing
│   └── utils.py                 # Utility functions (e.g., metric computation, visualization)
├── README.md                    # High‐level overview & instructions (download links, citation)
└── (possibly other helper files)
```

* **fig/**: Illustrates the multi-dimensional data collection process, annotation distributions, and AENet’s network diagram. ([GitHub][1])
* **intra\_dataset\_code/models.py**: Contains the PyTorch implementation of AENet. It defines the backbone network, task‐specific heads, and the forward pass that outputs various predictions. ([GitHub][3])
* **intra\_dataset\_code/dataset.py**: Implements a subclass of `torch.utils.data.Dataset` that reads an image file, retrieves its annotation from the JSON file, and returns a dictionary containing:

  * The transformed image tensor.
  * Label(s) for live/spoof, face attributes, spoof type, and optionally ground‐truth depth/reflection if geometric supervision is used.
* **intra\_dataset\_code/train.py**:

  * Parses command‐line arguments (e.g., batch size, learning rate, number of epochs).
  * Initializes AENet, loss functions (CrossEntropy for classification, L1/L2 for depth/reflection regression), and optimizers.
  * Sets up `DataLoader` objects for training/validation splits.
  * Loops over epochs: forward pass → compute losses → backpropagate → update weights.
  * Periodically saves checkpoints and logs metrics (e.g., accuracy, EER).
* **intra\_dataset\_code/inference.py**: Loads a saved checkpoint, runs the model over a test set (or validation set), and computes evaluation metrics—typically **Accuracy at the Equal Error Rate (EER)**, **Area Under the ROC Curve (AUC)**, and **EER** itself.

**Overview of AENet (Auxiliary Information Embedding Network)**
AENet is the core model in CelebA-Spoof designed to solve face anti-spoofing by jointly learning three complementary tasks:

1. **Live/Spoof Classification (C)**
2. **Semantic Prediction (S)**:

   * Face Attributes (Sᶠ)
   * Spoof Type (Sˢ)
   * Illumination/Environment (Sⁱ)
3. **Geometric Estimation (G)**:

   * Depth Map (Gᵈ)
   * Reflection Map (Gʳ)

By training all three “heads” (C, S, and G) in a single multi-task architecture, AENet forces its shared CNN backbone to extract features that are simultaneously:

* Discriminative for live vs. spoof decisions,
* Informative about semantic cues (e.g., “wearing glasses,” “smiling,” “printed-photo attack,” “bright backlight”), and
* Sensitive to subtle geometric signatures (actual 3D shape vs. flat/reflective surfaces).

Below is a detailed, step-by-step walkthrough of AENet’s components and how they appear in the diagram you provided.

---

## 1. High-Level Diagram Walkthrough

![image](https://github.com/user-attachments/assets/ccece020-c7e1-4bae-ad31-b5032f6ff440)

> **Figure 1.** Schematic of AENet.
>
> * **Left**: Spoof example (e.g., someone holding a phone or printed photograph)
> * **Right**: Live example (a real 3D face under normal lighting)
> * **Orange box (Semantic)**: Face attributes and “spoof metadata” predictions
> * **Green box (Classification)**: Final live/spoof decision
> * **Blue box (Geometric)**: Predicted depth + reflection maps

Below, we break down each component in exacting detail.

---

## 2. Shared Backbone (CNN)

* **Input**: A cropped, aligned RGB face image (e.g., 224×224 or 256×256, depending on hyperparameters).
* **Structure**:

  1. **Convolutional Layers** (e.g., ResNet-like or a custom “CNN4”):

     * Several conv→BatchNorm→ReLU blocks that gradually reduce spatial resolution (e.g., 224→112→56→28→14) while increasing channel depth (e.g., 64→128→256→512).
  2. **Global Feature Map**: After the final convolution, you obtain a feature tensor **F** of shape `[B, C, H, W]` (for instance, `[B, 512, 7, 7]` if you use ResNet-18 as backbone with output stride 32).
  3. **Shared Weights**: The exact same feature extractor is used at training/inference time for all downstream heads (C, S, G).

> **Key Point:** Everything that follows branches off from this shared feature map **F**. The idea is that a single “rich” representation should capture both semantic and geometric cues.

---

## 3. Semantic Heads (S)

AENet splits semantic supervision into **three subtasks**:

1. **Face Attributes (Sᶠ)**
2. **Spoof Type (Sˢ)**
3. **Illumination/Environment (Sⁱ)**

Each of these heads takes the shared feature map **F** and produces a separate prediction vector. Concretely:

1. **Shared Feature Aggregation for Semantic Tasks**

   * Starting from **F** ∈ ℝ `[B, C, H, W]`, apply a small spatial pooling (e.g., **AdaptiveAvgPool2d** → output `[B, C, 1, 1]`), then flatten to `[B, C]`.
   * Alternatively, one can attach a few extra 1×1 convolutions + BN + ReLU to reduce channels (e.g., from 512 → 256 → 128), then perform global pooling.

2. **Face Attribute Head (Sᶠ)**

   * **Goal**: Predict the 40 CelebA-style binary attributes (e.g., “Big Nose,” “Smiling,” “Eyeglasses,” “Mustache,” etc.).
   * **Architecture**:

     1. Fully connected layer (e.g., 512→256), ReLU → Dropout
     2. Final linear layer (256→40)
   * **Output**: A vector `sᶠ ∈ ℝ⁴⁰`. Each element is interpreted via a sigmoid (for multi-label) or two-way softmax (grouped as independent binary probs).
   * **Loss**: Summed binary‐cross-entropy (BCE) across all 40 dimensions, comparing to ground truth attribute annotations inherited from the CelebA dataset.

3. **Spoof Type Head (Sˢ)**

   * **Goal**: Classify which **type of spoof** is present whenever the label is “spoof.” Typical categories include:

     * Printed Photo (A4, Poster, etc.)
     * Screen Replay (Phone, Tablet, PC monitor)
     * 3D Mask (UpperBodyMask, RegionMask, full FaceMask)
   * **Architecture**:

     1. Fully connected (FC) 512→256, ReLU → Dropout
     2. FC 256→`Nₛ` (where `Nₛ` is number of spoof categories, e.g., 10 classes)
   * **Output**: `sˢ ∈ ℝᴺₛ`. A softmax over these Nₛ categories.
   * **Loss**: Cross‐entropy with the ground‐truth “spoof type” label. (If the image is live, one typically ignores this head’s loss or sets a “Live” category.)

4. **Illumination/Environment Head (Sⁱ)**

   * **Goal**: Predict which **lighting condition** or **scene environment** the image was captured in. Typical labels:

     * Illumination: {Normal, StrongBacklight, Dark}
     * Environment: {Indoor, Outdoor}
   * In the original CelebA-Spoof setup, these are encoded as integer indices 0…K.
   * **Architecture**:

     1. FC 512→128, ReLU → Dropout
     2. FC 128→`Nᵢ` (e.g., 3–5 classes)
   * **Output**: `sⁱ ∈ ℝᴺᵢ`. Softmax over illumination/environment categories.
   * **Loss**: Cross‐entropy comparing to ground‐truth illumination/environment labels.

> **Visualizing the Semantic Output (Figure 1, Orange Box)**
>
> * On the **left side (Spoof)**, the bars under “Sᶠ,” “Sˢ,” and “Sⁱ” illustrate:
>
>   * Sᶠ: Low probability for “Smiling,” “Big Nose,” etc., because it’s a spoof image (some facial attributes may be suppressed or distorted).
>   * Sˢ: High probability on “Phone Replay” (or whatever device was used).
>   * Sⁱ: High probability on “Backlight” (strong illumination from behind).
> * On the **right side (Live)**, Sᶠ might show a high activation for “Big Nose,” “Smiling,” etc., while Sˢ is “Live” and Sⁱ is “Normal Lighting.”

---

## 4. Classification Head (C)

* **Goal**: Make the final binary decision: **Live (0) vs. Spoof (1)**.
* **Placement**: This head also takes the same shared feature representation **F** (pooled to `[B, C]`).
* **Architecture**:

  1. FC 512→128, ReLU → Dropout
  2. FC 128→2 (logits for {Live, Spoof})
* **Output**: `c ∈ ℝ²`. Softmax or log‐softmax over “Live/Spoof.”
* **Loss**: Standard cross‐entropy between `c` and ground‐truth label `y ∈ {0,1}`.

> **Why Keep Classification Separate?**
> Although one could derive `Live/Spoof` indirectly from spoof‐type (Sˢ), the authors observed that explicitly training a dedicated binary head yields higher robustness—especially when the spoof type is ambiguous (e.g., a glossy mask versus a high-res photo).

---

## 5. Geometric Heads (G)

### 5.1 Depth Estimation Head (Gᵈ)

* **Goal**: Estimate a coarse **depth map** of the face—i.e., at each pixel location, predict how far the surface is from the camera.

  * **Live faces** have a natural 3D shape (nose protrudes, cheeks recede).
  * **Spoof surfaces** (e.g., printed paper, phone screen) are mostly **flat**, so their depth map appears almost constant (nearly zero variation).
* **Architecture**:

  1. **Decoder Module**: From shared feature F ∈ `[B, C, H, W]`, apply a small “upsampling” block—often an **Encoder–Decoder** style:

     * 1×1 conv to reduce channels (C→C’), then two or three `ConvTranspose2d` or `Upsample+Conv2d` layers to scale `[H×W] → [H₂×W₂] → … → [h×w]`, where `[h, w]` is roughly the original face patch size (or a downscaled version, e.g., 56×56).
     * Each upsampling step uses `Conv2d→BN→ReLU`.
  2. **Final Depth Output**: Single‐channel conv to produce **D̂** ∈ `[B, 1, h, w]`.
* **Ground Truth**:

  * For live images: a proxy depth map can be obtained via a precomputed 3D‐MM (shape‐from‐stereo or deployed from a 3D face-scanner). In CelebA-Spoof, the authors use an **external depth predictor** (off-the-shelf) to generate weak supervision for live samples, or synthetic depth from a Morphable Model.
  * For spoof images: a “flat” depth map (all zeros, or small noise) is used as ground truth.
* **Loss**:

  $$
  \mathcal{L}_{depth} \;=\; \frac{1}{B} \sum_{i=1}^{B} \bigl\|\,D̂_i \;-\; D_i^{gt}\bigr\|_{1} \quad (\text{L1‐loss or MAE})
  $$

  where $D_i^{gt}$ is either a real depth map (live) or a constant zero map (spoof).

### 5.2 Reflection Estimation Head (Gʳ)

* **Goal**: Predict a coarse **reflection map**—i.e., where specular highlights or strong reflections occur on the face region.

  * **Live faces** usually have only subtle, natural skin reflections (small glints around the nose, forehead).
  * **Spoof surfaces** (e.g., glossy mask or phone screen) exhibit strong, unnatural reflections (e.g., a rectangle of screen glare, or a big highlight across the entire printed area).
* **Architecture**:

  * Same decoder style as depth: from **F** ∈ `[B, C, H, W]`, upsample back to dense spatial resolution `[B, 1, h, w]`.
  * Possibly one or two fewer conv layers than the depth branch, since reflection cues can be coarser.
* **Ground Truth**:

  * For live images: reflection maps can be approximated by computing the **difference between a raw face image and a smoothed/low-pass filtered version** (i.e., isolate specular highlights).
  * For spoof: ground truth reflection is often the entire “shiny” region of a screen. If no strong reflection (e.g., matte print), the GT is zeros.
* **Loss**:

  $$
  \mathcal{L}_{refl} \;=\; \frac{1}{B} \sum_{i=1}^{B} \bigl\|\,R̂_i \;-\; R_i^{gt}\bigr\|_{1}\quad (\text{L1‐loss})
  $$

> **Visualizing Geometric Outputs (Figure 1, Blue Box)**
>
> * On the **left (Spoof)**:
>
>   * **Gᵈ** (Depth) → a mostly flat, zero‐value map (black in the illustration).
>   * **Gʳ** (Reflection) → a vivid “face‐shaped” glow, because the camera is seeing a phone or monitor’s reflection (blue-green overlay representing screen glare).
> * On the **right (Live)**:
>
>   * **Gᵈ** → a smooth “bump” map approximating a real nose, cheeks, chin (white-gray 3D curvature).
>   * **Gʳ** → near-zero reflection (black), since a live face under normal lighting has only tiny specular spots (too subtle to be overtly visible).

---

## 6. Joint Loss Function

AENet is trained end-to-end with a **combined loss**:

$$
\boxed{
\mathcal{L}_{total} 
\;=\; \lambda_{cls}\,\mathcal{L}_{cls} 
\;+\; \lambda_{sem}\,\bigl(\mathcal{L}_{attr} + \mathcal{L}_{spoofType} + \mathcal{L}_{illumEnv}\bigr)
\;+\; \lambda_{geo}\,\bigl(\mathcal{L}_{depth} + \mathcal{L}_{refl}\bigr)
}
$$

1. **Classification Loss**:
   $\mathcal{L}_{cls} = \mathrm{CE}\bigl(C(x),\,y_{\text{live/spoof}}\bigr)$, where $C(x)$ ∈ ℝ² and $y \in\{0,1\}$.

2. **Semantic Loss** (sum of three components):

   * $\mathcal{L}_{attr} = \sum_{j=1}^{40}\mathrm{BCE}(sᶠ_j,\,y^{(j)}_{\text{attr}})$
   * $\mathcal{L}_{spoofType} = \mathrm{CE}(sˢ,\,y_{\text{spoofType}})$
   * $\mathcal{L}_{illumEnv} = \mathrm{CE}(sⁱ,\,y_{\text{illumOrEnv}})$

3. **Geometric Loss**:

   * $\mathcal{L}_{depth} = \lVert D̂ - D^{gt}\rVert_{1}$
   * $\mathcal{L}_{refl}  = \lVert R̂ - R^{gt}\rVert_{1}$

Typical choices for the weights are:

* $\lambda_{cls} = 1.0$ (anchor weight)
* $\lambda_{sem} = 0.5$ (often split equally among the three semantic subtasks)
* $\lambda_{geo} = 0.1$ (to avoid over-emphasis on pixel-perfect geometric maps)

These hyperparameters are tuned so that the network cannot “cheat” by focusing all capacity on a single task; instead, it must learn features that generalize well across semantic, geometric, and classification demands.

---

## 7. Step-by-Step Forward Pass (Spoof vs. Live)

Let’s trace **one training example** through AENet, first on a **Spoof image** and then on a **Live image**.

### 7.1 Spoof Example

1. **Input**: A photograph of a person’s face displayed on a smartphone screen.

   * The printed/folded phone lies in front of a camera. The face region is crisp but has a strong rectangle of glare, and the overall shape is very flat.
2. **Backbone**:

   * The CNN extracts features focusing on edges, textures, and subtle clues (e.g., screen pixel grid, border of phone case).
   * Final feature map **F** captures both local textural differences (e.g., moiré pattern) and global shape cues.
3. **Semantic Heads** (Orange):

   * **Face Attributes (Sᶠ)**: Many attribute classifiers produce **low confidence** (e.g., “Big Nose” might be uncertain because the printed image may blur or distort nose shape).
   * **Spoof Type (Sˢ)**: Very high softmax score on “Phone Replay” (because metadata annotation says this was captured by showing a phone screen).
   * **Illumination (Sⁱ)**: High probability for “StrongBacklight” if the spoof was captured in a bright environment.
4. **Classification Head (C)** (Green):

   * Uses features from **F** to output high logit for “Spoof” (because combination of unnatural texture, flatness, and semantic cues).
5. **Depth Head (Gᵈ)** (Blue—top map):

   * Predicts a nearly **flat** depth map ⟶ nearly all zeros (dark in the illustration).
   * Loss: large penalty if it tries to predict any 3D curvature, because GT for a spoof is “flat” (zero map).
6. **Reflection Head (Gʳ)** (Blue—bottom map):

   * Picks up the strong specular reflections from the phone glass—predicts a bright spot or a rectangular highlight region.
   * GT: “Everything outside the phone’s reflection is 0; only the bright area is non-zero.”
   * Loss: encourages capturing that “blueish” glare region.
7. **Total Loss**: Sum of

   * ^(Classification)��$\mathrm{CE}(c,\,1)$
   * ^(Attributes)��∑*j BCE(\$sᶠ\_j\$, \$y^{(j)}*{\text{attr}}\$)
   * ^(SpoofType)��$\mathrm{CE}(sˢ,\,\text{Phone})$
   * ^(Illum/Env)��$\mathrm{CE}(sⁱ,\,\text{Backlight})$
   * ^(Depth)��‖Gᵈ–0‖₁
   * ^(Refl)��‖Gʳ–R^{gt}‖₁

Over many such examples, the model learns that **“flatness + phone glare + certain semantic cues → Spoof.”**

---

### 7.2 Live Example

1. **Input**: A genuine person’s face under normal office lighting.
2. **Backbone**:

   * Extracts real facial textures (skin pores, hair strands), landmark geometry (nose ridge, eyebrow contours).
   * Final feature map **F** strongly encodes 3D shape differences (nose protrusion vs. cheek recess).
3. **Semantic Heads** (Orange):

   * **Face Attributes (Sᶠ)**: High confidence on “Big Nose,” “Smiling,” maybe “Wavy Hair.”
   * **Spoof Type (Sˢ)**: One “Live” category or a dummy “0” label.
   * **Illumination (Sⁱ)**: High probability on “NormalLighting, Indoor.”
4. **Classification Head (C)** (Green):

   * Predicts “Live” (low spoof logit) because face features match real-face distributions: smooth skin texture, consistent shading, plausible 3D geometry.
5. **Depth Head (Gᵈ)**:

   * Outputs a smooth, white-gray “bump” shape indicating nose, eyes, mouth region—closely matching GT depth from a face scanner or synthetic depth.
   * Low L1 error because model successfully learns real 3D geometry cues (e.g., shading gradient from nose to cheeks).
6. **Reflection Head (Gʳ)**:

   * Predicts very low reflectance map (mostly black). Any small specular highlights (e.g., glossy tear-ducts) are too subtle to appear.
   * Low reflection loss since GT was near-zero for a live face under diffused indoor light.
7. **Total Loss**: Sum of

   * CE(\$c\$, 0)
   * ∑ BCE(\$sᶠ\_j\$, \$y^{(j)}\_{\text{attr}}\$)
   * CE(\$sˢ\$, “Live”)
   * CE(\$sⁱ\$, “NormalIndoor”)
   * L1(\$Gᵈ\$, Depth\$^{gt}\$)
   * L1(\$Gʳ\$, 0)

Over time, AENet discovers that for **live faces**:

* Depth maps are **non-flat** (higher norm in Gᵈ)
* Reflection maps are **near zero** (lower norm in Gʳ)
* Semantic attributes align with plausible human face (eyes open, skin smoothness, etc.).

---

## 8. Detailed Architecture (Layer-By-Layer)

Below is a **concrete example** of how one might implement AENet (the exact numbers can vary, but this gives a sense of layer depths and shapes). We assume an input resolution of `224×224`.

### 8.1 Backbone (ResNet-18–Style)

```python
import torch.nn as nn
import torchvision.models as models

class SharedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pretrained ResNet-18 up to the final convolutional block
        base = models.resnet18(pretrained=True)
        # Remove final FC + avgpool
        self.conv1 = base.conv1    # [B, 64, 112, 112]
        self.bn1   = base.bn1
        self.relu  = base.relu
        self.maxpool = base.maxpool  # [B, 64, 56, 56]
        self.layer1 = base.layer1     # [B, 64, 56, 56]
        self.layer2 = base.layer2     # [B,128, 28, 28]
        self.layer3 = base.layer3     # [B,256, 14, 14]
        self.layer4 = base.layer4     # [B,512,  7,  7]
        # We stop here—no global average pool, no fc.
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Output: F ∈ ℝ^[B, 512, 7, 7]
        return x
```

### 8.2 Semantic Heads

```python
class SemanticHeads(nn.Module):
    def __init__(self, in_channels=512, num_attrs=40, num_spoof_types=10, num_illum=4):
        super().__init__()
        # 1) Face Attributes head
        self.attr_fc1 = nn.Linear(in_channels, 256)
        self.attr_dropout = nn.Dropout(p=0.5)
        self.attr_fc2 = nn.Linear(256, num_attrs)  # 40 outputs

        # 2) Spoof Type head
        self.spoof_fc1 = nn.Linear(in_channels, 256)
        self.spoof_dropout = nn.Dropout(p=0.5)
        self.spoof_fc2 = nn.Linear(256, num_spoof_types)  # e.g. 10 classes

        # 3) Illumination/Environment head
        self.illum_fc1 = nn.Linear(in_channels, 128)
        self.illum_dropout = nn.Dropout(p=0.5)
        self.illum_fc2 = nn.Linear(128, num_illum)  # e.g. 4 classes

    def forward(self, F):  # F ∈ [B, 512, 7, 7]
        B, C, H, W = F.shape
        pooled = F.mean(dim=[2, 3])  # Global avg pooling → [B, 512]

        # Face attributes
        a = self.attr_fc1(pooled)    # [B, 256]
        a = nn.ReLU()(a)
        a = self.attr_dropout(a)
        a_logits = self.attr_fc2(a)  # [B, 40]

        # Spoof type
        s = self.spoof_fc1(pooled)   # [B, 256]
        s = nn.ReLU()(s)
        s = self.spoof_dropout(s)
        s_logits = self.spoof_fc2(s) # [B, num_spoof_types]

        # Illum/Env
        i = self.illum_fc1(pooled)   # [B, 128]
        i = nn.ReLU()(i)
        i = self.illum_dropout(i)
        i_logits = self.illum_fc2(i) # [B, num_illum]

        return a_logits, s_logits, i_logits
```

### 8.3 Classification Head

```python
class ClassificationHead(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 2)  # Live vs. Spoof

    def forward(self, F):
        B, C, H, W = F.shape
        pooled = F.mean(dim=[2, 3])  # [B, 512]
        x = self.fc1(pooled)         # [B, 128]
        x = nn.ReLU()(x)
        x = self.dropout(x)
        logits = self.fc2(x)         # [B, 2]
        return logits
```

### 8.4 Geometric Decoder Blocks

Both geometry heads share a similar decoder structure, differing only by their final output channel (1 for depth, 1 for reflection). We illustrate a single “upsample block” and show how to stack them:

```python
class GeometricDecoder(nn.Module):
    def __init__(self, in_channels=512, mid_channels=256, out_size=(56, 56)):
        super().__init__()
        # Example: F is [B, 512, 7, 7], we want to upsample to [B, 1, 56, 56]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)  # [B,256,7,7]
        self.bn1   = nn.BatchNorm2d(mid_channels)
        # Upsample → [B,256,14,14]
        self.up1   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)  # [B,256,14,14]
        self.bn2   = nn.BatchNorm2d(mid_channels)
        # Upsample → [B,256,28,28]
        self.up2   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)  # [B,256,28,28]
        self.bn3   = nn.BatchNorm2d(mid_channels)
        # Upsample → [B,256,56,56]
        self.up3   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1)             # [B,1,56,56]
        # Final activation: none (we predict raw depth/reflection values, training with L1)
    
    def forward(self, F):
        # F ∈ [B, 512, 7, 7]
        x = self.conv1(F)       # [B,256,7,7]
        x = self.bn1(x)
        x = nn.ReLU()(x)

        x = self.up1(x)         # [B,256,14,14]
        x = self.conv2(x)       # [B,256,14,14]
        x = self.bn2(x)
        x = nn.ReLU()(x)

        x = self.up2(x)         # [B,256,28,28]
        x = self.conv3(x)       # [B,256,28,28]
        x = self.bn3(x)
        x = nn.ReLU()(x)

        x = self.up3(x)         # [B,256,56,56]
        x = self.conv4(x)       # [B,1,56,56]
        # No activation: raw regression target
        return x
```

You would then instantiate **two** copies of this decoder—one for depth (Gᵈ) and one for reflection (Gʳ).

---

### 8.5 Putting It All Together: AENet Class

```python
class AENet(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 num_attrs: int = 40,
                 num_spoof_types: int = 10,
                 num_illum: int = 4):
        super().__init__()
        self.backbone = backbone  # e.g., SharedBackbone()
        # Semantic heads
        self.semantic = SemanticHeads(
            in_channels=512,
            num_attrs=num_attrs,
            num_spoof_types=num_spoof_types,
            num_illum=num_illum
        )
        # Classification head
        self.classifier = ClassificationHead(in_channels=512)
        # Geometric heads
        self.depth_decoder = GeometricDecoder(in_channels=512, mid_channels=256)
        self.refl_decoder  = GeometricDecoder(in_channels=512, mid_channels=256)

    def forward(self, x):
        """
        x: [B, 3, 224, 224] RGB face crops
        returns:
          a_logits  [B, 40]      # face attribute predictions
          s_logits  [B, num_spoof_types]
          i_logits  [B, num_illum]
          c_logits  [B, 2]       # live vs. spoof
          depth_map [B, 1, 56, 56]
          refl_map  [B, 1, 56, 56]
        """
        F = self.backbone(x)  # [B, 512, 7, 7]

        # 1) Semantic tasks
        a_logits, s_logits, i_logits = self.semantic(F)

        # 2) Classification
        c_logits = self.classifier(F)

        # 3) Geometric tasks
        depth_map = self.depth_decoder(F)   # [B, 1, 56, 56]
        refl_map  = self.refl_decoder(F)    # [B, 1, 56, 56]

        return {
            "attr_logits": a_logits,
            "spooftype_logits": s_logits,
            "illum_logits": i_logits,
            "cls_logits": c_logits,
            "depth_map": depth_map,
            "refl_map": refl_map
        }
```

---

## 9. Loss Computation in Code

Below is a snippet showing how you might accumulate all the losses in the training loop:

```python
def compute_losses(outputs, targets, weights):
    """
    outputs: dict from AENet.forward
    targets: dict containing
      - "live_label"    ∈ {0,1}
      - "attr_labels"   ∈ {0,1}^40
      - "spooftype_lbl" ∈ {0,1,...,num_spoof_types-1} or -1 if live
      - "illum_lbl"     ∈ {0,1,...,num_illum-1}
      - "depth_gt"      ∈ ℝ^[B,1,56,56]
      - "refl_gt"       ∈ ℝ^[B,1,56,56]
    weights: dict containing λ_cls, λ_sem, λ_geo
    """
    # 1) Classification loss
    cls_logits = outputs["cls_logits"]  # [B, 2]
    live_lbl    = targets["live_label"] # [B]
    loss_cls    = nn.CrossEntropyLoss()(cls_logits, live_lbl)

    # 2) Semantic losses
    # 2a) Face attributes (multi-label BCE)
    attr_logits = outputs["attr_logits"]      # [B, 40]
    attr_lbls   = targets["attr_labels"].float()  # [B, 40]
    loss_attr   = nn.BCEWithLogitsLoss()(attr_logits, attr_lbls)

    # 2b) Spoof type (only compute if it’s spoof)
    s_logits    = outputs["spooftype_logits"] # [B, num_spoof_types]
    spoof_lbl   = targets["spooftype_lbl"]    # [B], -1 indicates “live,” >=0 indicates index
    mask_spoof  = (spoof_lbl >= 0)            # boolean mask
    if mask_spoof.sum() > 0:
        loss_spooftype = nn.CrossEntropyLoss()(
            s_logits[mask_spoof],
            spoof_lbl[mask_spoof]
        )
    else:
        loss_spooftype = torch.tensor(0.0, device=cls_logits.device)

    # 2c) Illumination (always compute)
    illum_logits = outputs["illum_logits"]  # [B, num_illum]
    illum_lbl    = targets["illum_lbl"]     # [B]
    loss_illum   = nn.CrossEntropyLoss()(illum_logits, illum_lbl)

    loss_sem = loss_attr + loss_spooftype + loss_illum

    # 3) Geometric losses
    pred_depth = outputs["depth_map"]  # [B, 1, 56, 56]
    pred_refl  = outputs["refl_map"]   # [B, 1, 56, 56]
    depth_gt   = targets["depth_gt"]   # [B, 1, 56, 56]
    refl_gt    = targets["refl_gt"]    # [B, 1, 56, 56]

    loss_depth = nn.L1Loss()(pred_depth, depth_gt)
    loss_refl  = nn.L1Loss()(pred_refl, refl_gt)
    loss_geo   = loss_depth + loss_refl

    # 4) Total
    total_loss = (
        weights["cls"] * loss_cls
        + weights["sem"] * loss_sem
        + weights["geo"] * loss_geo
    )
    return total_loss, {
        "loss_cls": loss_cls.item(),
        "loss_attr": loss_attr.item(),
        "loss_spooftype": loss_spooftype.item() if isinstance(loss_spooftype, torch.Tensor) else 0.0,
        "loss_illum": loss_illum.item(),
        "loss_depth": loss_depth.item(),
        "loss_refl": loss_refl.item()
    }
```

---

## 10. Interpreting the Example Image in Detail

Let’s revisit the graphic you provided and link each visual element to the code above:

1. **“Input Image (Spoof)” on the left**

   * A single RGB face crop that actually is a **live face displayed on a smartphone** (or printed photo).
   * Notice clothing edges, unnatural texture (pixel grid), and the black phone border at the top.

2. **Arrow → “CNN”**

   * The shared backbone extracts a 512-channel feature map at spatial resolution 7×7.

3. **Semantic Branch (Orange, labeled Sᶠ, Sˢ, Sⁱ)**

   * **Sᶠ (Face Attributes)**:

     * On a spoof, many attribute scores are either low or inconsistent. In the bar chart you see very few “tall bars” for big-nose, smile, etc., indicating the network is uncertain which face attributes hold for a printed image.
   * **Sˢ (Spoof Type)**:

     * Tall bar under “Phone” indicates the model strongly recognizes “this is a phone replay attack.”
   * **Sⁱ (Illumination/Environment)**:

     * Tall bar under “Back” (strong backlighting) indicates that this spoof was captured in a backlit environment.

4. **Classification Branch (Green, labeled C)**

   * Output is a single tall bar under “Spoof.”
   * In code: `c_logits = classifier(F) → softmax → [P(Live), P(Spoof)]`, with P(Spoof) ≈ 0.99.

5. **Geometric Branch (Blue, labeled Gᵈ and Gʳ)**

   * **Gᵈ (Depth Map):** All black → predicted depth ∼ 0 everywhere (flatness).
   * **Gʳ (Reflection Map):** A translucent, bluish overlay of a face (bright in phone screen region) indicates a **strong specular reflection** caused by the phone’s glass.
   * In code:

     ```python
     depth_map = depth_decoder(F)  # outputs near zero
     refl_map  = refl_decoder(F)   # captures that bright rectangle of glare
     ```

6. **“Input Image (Live)” on the right**

   * A real 3D face under normal lighting, no screen or print.
   * Notice natural textures (skin pores), subtle shadows beneath nose/cheeks.

7. **Semantic Branch on Live**

   * **Sᶠ**: Tall bars under “Big Nose,” “Smile,” etc. High confidence in plausible face attributes.
   * **Sˢ**: Single bar under “Live.” (In practice, you either have a dummy “Live” category or simply ignore the spoof-type head when label=0.)
   * **Sⁱ**: Single bar under “Normal” or “Indoor” => correct lighting.

8. **Classification Branch on Live (Green)**

   * Single tall bar under “Live,” indicating P(Live) ≈ 0.98.

9. **Geometric Branch on Live**

   * **Gᵈ**: A smooth “3D bump” map of the nose and cheeks. Bright in nose tip, darker near eyes, matching a real depth profile.
   * **Gʳ**: All black (no strong reflections), aside from tiny white specks for specular highlights—almost negligible.

---

## 11. Why AENet Works: Intuition

1. **Feature Enrichment via Multi-Task Learning**

   * Forcing the backbone to simultaneously support:

     * A 40-dimensional face‐attribute classifier
     * A multi-class spoof‐type classifier
     * A small illumination classifier
     * Depth regression
     * Reflection regression
   * ⇒ Encourages earlier convolutional layers to specialize in:

     * **Low-level texture** (to detect print patterns, moiré, screen pixels)
     * **Mid-level geometry** (to pick up curvature vs. flat surfaces)
     * **High-level semantics** (to identify eyes, nose, mouth and whether they appear natural)

2. **Geometric Supervision Discourages “Cheating”**

   * If the network only had a classification head, it might overfit to trivial artifacts—like a small border line from a phone case.
   * By adding a **depth regression** target, the network must learn to pay attention to global 3D shape rather than cropping only the top of the screen’s bezel.
   * By adding a **reflection regression** target, the model must identify specular highlights (unusual for a flat printed image).

3. **Semantic Cues Provide Context**

   * Some spoof types—like **high‐quality 3D masks**—can mimic geometry closely. A mask can give you a depth profile that looks “3D,” but its **semantic attributes** might be off (e.g., color of eyes, texture of eyebrows, unnatural skin blemish distribution).
   * Conversely, a “replay attack” on a phone might fool a purely geometric detector (because a recorded 3D video of a real face is shown on a flat screen), but the **reflection map** is a huge giveaway (a phone or tablet reflects the overhead light in a rectangular shape).
   * By jointly modeling both semantic and geometric tasks, AENet blends multiple “views” of what makes a face real.

---

## 12. Training & Inference Workflow

Below is a concise step-by-step of how you’d train and evaluate AENet in practice:

1. **Data Preparation**

   * **Images**: 625,537 total. Split into:

     * **Training** (e.g., 80% of “live” + 80% of “spoof” images).
     * **Validation/Testing** (remaining 20%).
   * **Annotations**: For each image, load JSON entry with keys:

     * `live_spoof_label` ∈ {0,1}
     * `face_attributes` ∈ {0,1}^40
     * `spoof_type` ∈ {0,…,Nₛ–1} (–1 if live)
     * `illum_env` ∈ {0,…,Nᵢ–1}
     * `depth_gt` ∈ ℝ^\[56×56] (for live, from external; for spoof, zeros)
     * `refl_gt` ∈ ℝ^\[56×56] (extracted via image processing for live; for spoof, region of glare)

2. **Instantiate Model**

   ```python
   backbone = SharedBackbone()
   model    = AENet(
       backbone=backbone,
       num_attrs=40,
       num_spoof_types=10,
       num_illum=4
   )
   model = model.cuda()  # if GPU available
   ```

3. **Define Optimizer & Loss Weights**

   ```python
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   weights   = {"cls": 1.0, "sem": 0.5, "geo": 0.1}
   ```

4. **Training Loop**

   ```python
   for epoch in range(num_epochs):
       model.train()
       for batch in train_loader:
           x        = batch["image"].cuda()           # [B,3,224,224]
           targets  = {
               "live_label":    batch["live_label"].cuda(),
               "attr_labels":   batch["attr_labels"].cuda(),
               "spooftype_lbl": batch["spooftype_lbl"].cuda(),
               "illum_lbl":     batch["illum_lbl"].cuda(),
               "depth_gt":      batch["depth_gt"].cuda(),
               "refl_gt":       batch["refl_gt"].cuda(),
           }
           outputs = model(x)
           loss, loss_dict = compute_losses(outputs, targets, weights)

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
       # (Optionally: evaluate on val set, checkpoint model)
   ```

5. **Validation / Inference**

   ```python
   model.eval()
   all_preds = []
   all_labels = []
   with torch.no_grad():
       for batch in val_loader:
           x         = batch["image"].cuda()
           true_lbl  = batch["live_label"].cpu().numpy()  # {0,1}
           logits    = model(x)["cls_logits"]             # [B,2]
           probs     = torch.softmax(logits, dim=1)[:, 1] # P(spoof)
           all_preds.append(probs.cpu().numpy())
           all_labels.append(true_lbl)

   all_preds  = np.concatenate(all_preds)   # [N_val]
   all_labels = np.concatenate(all_labels)  # [N_val]
   # Compute ROC, AUC, EER
   fpr, tpr, thresholds = sklearn.metrics.roc_curve(all_labels, all_preds)
   auc_score             = sklearn.metrics.auc(fpr, tpr)
   eer_idx               = np.nanargmin(np.abs(fpr - (1 - tpr)))
   eer                   = (fpr[eer_idx] + (1 - tpr[eer_idx])) / 2
   print(f"AUC = {auc_score:.4f}, EER = {eer:.4f}")
   ```

---

## 13. Practical Tips & Tricks

1. **Balancing the Losses**

   * If you find classification accuracy stagnating, try **increasing** λ₍cls₎ relative to λ₍sem₎ and λ₍geo₎.
   * If the model is too “coarse” in detecting geometry—i.e., it still confuses a high-quality printed image with a live face—increase λ₍geo₎.
   * If semantic predictions (face attributes) never converge, decrease λ₍sem₎ or break it down into separate training stages:

     1. Pretrain the semantic heads alone on CelebA attributes (freeze geometry/classification).
     2. Fine-tune the entire network end-to-end.

2. **Data Augmentation**

   * **Random Horizontal Flip** (p=0.5).
   * **Color Jitter** (±0.2 brightness/contrast) helps the model generalize to different lighting.
   * **Gaussian Blur** on spoof images reduces overfitting to texture details (makes the model focus more on geometry).
   * **Random Crops & Resizes** simulate different camera zooms, so that “small phone in a hand” vs. “close-up printed photo” look slightly different.

3. **Batch Normalization & Dropout**

   * Keep BN in the backbone (ResNet uses it). Always set `model.eval()` at inference to fix BN statistics.
   * Use Dropout in semantic/classification heads to prevent overfitting—especially when you have billions of parameters in the shared backbone.

4. **Monitoring Intermediate Outputs**

   * Periodically plot the predicted **depth maps** and **reflection maps** alongside ground truth. If your `depth_decoder` is producing weird, noisy maps, try adding more supervision (e.g., use a better precomputed depth GT).
   * Visualize the **semantic logits** (Sᶠ, Sˢ, Sⁱ) to see if the network actually understands attribute cues. If it consistently mis-labels attributes for live images, you might need more attribute annotations or a larger backbone.

5. **Cross-Dataset Evaluation**

   * After training on CelebA-Spoof, test on a completely different anti-spoof dataset (e.g., CASIA-SURF, MSU MFSD).
   * If performance drops sharply, consider adding those new dataset images into your training loop or refine the decoder modules to better handle unseen reflection patterns.

---

## 14. Summary of AENet’s Strengths

1. **Robustness via Redundancy**:

   * A spoof that fools **only** a depth checker (e.g., a video replay that shows a 3D head rotating on a phone) may be caught by the reflection head (blue).
   * A mask with near-correct geometry may be spotted by the semantic head as “no eyelashes,” “weird eyebrow texture,” or “inconsistent attribute combination.”

2. **Explicit Supervision for Geometric Cues**:

   * Many prior anti-spoofing works rely purely on 2D textures (e.g., motion cues, LBP patterns). AENet’s explicit depth/regression head forces the network to **learn 3D shape** from a single RGB input, which is extremely hard for a spoof to replicate perfectly.

3. **Semantic Richness**:

   * Face attributes (Sᶠ) let the model know that “this eyebrow shape doesn’t match a real face” or “the skin looks overly uniform (typical of a printed mask).”
   * Spoof type (Sˢ) helps the network discover **attack-specific features**—e.g., a plastic 3D mask has a different reflection signature than a phone, so Gʳ can adapt.

4. **Unified, End-to-End Pipeline**:

   * Everything—image input → multiple heads → joint loss—is trained **together**.
   * During inference, you only need a single forward pass to get:

     1. Live/Spoof decision
     2. Which spoof type (if spoof)
     3. Which illumination was used
     4. A predicted depth map (useful for face recognition downstream)
     5. A reflection map (which can help in pre-processing or re-lighting the face).

---

## 15. Key Takeaways for the Image

When you look at the side-by-side “Spoof vs. Live” visualization:

* **Spoof**:

  1. **Semantic (Orange)**: “Phone replay” is easy to spot. Almost no reliable facial attributes (bars in Sᶠ are low).
  2. **Classification (Green)**: Single tall bar under “Spoof.”
  3. **Geometric (Blue)**:

     * Depth map is **all black** → flat.
     * Reflection map is **bright**, indicating strong screen/glare artifacts.

* **Live**:

  1. **Semantic (Orange)**: Many strong attribute bars (Big Nose, Smile). “Spoof type = Live.” “Illum = Normal.”
  2. **Classification (Green)**: Single tall bar under “Live.”
  3. **Geometric (Blue)**:

     * Depth map is a **smooth white-gray bump** (3D shape).
     * Reflection map is **all black** or nearly black (very minimal specular highlight).

> This figure encapsulates the **raison d’être** of AENet:
>
> * **Semantic cues** alone cannot distinguish a perfect mask from a real face;
> * **Geometric cues** alone might fail if you replay a 3D recording;
> * **Classification alone** (binary head) might latch onto spurious artifacts.
>
> By merging all three, AENet covers all bases: geometry (3D shape), semantics (face attributes + attack metadata), and direct classification.

---

## 16. Next Steps for a Beginner AI Engineer

1. **Clone & Run the Provided Code**

   ```bash
   git clone https://github.com/songhieng/CelebA-Spoof.git
   cd CelebA-Spoof/intra_dataset_code
   pip install -r requirements.txt
   ```

   * Examine `models.py` to see exactly how they implemented AENet’s layers (backbone + heads).
   * Look at `dataset.py` to understand how they read JSON annotations and produce `depth_gt` / `refl_gt`.
   * Open `train.py` to follow the training loop, hyperparameter choices, and how they compute each sub-loss.

2. **Visualize Intermediate Outputs**

   * Insert a few lines in `inference.py` to save `depth_map` and `refl_map` as PNGs for both live and spoof images—confirm that they match the shapes you see in the paper’s figures.
   * Plot `attr_logits` with a sigmoid to see which face attributes the network finds salient in real vs. spoof faces.

3. **Experiment with λ-Weights**

   * Set `λ_geo=0` (i.e., turn off geometric supervision). Retrain and see how depth and reflection degrade. Does classification accuracy on live vs. spoof drop?
   * Set `λ_sem=0` (no semantic attention). Retrain and observe whether “3D masks” now become far more difficult to detect.

4. **Swap Backbones**

   * Replace `ResNet-18` with a **lighter** MobileNetV2 or a **heavier** ResNet-50. Observe differences in EER.
   * If you have limited GPU memory, try a “CNN4” (four small 3×3 conv layers) to see how far you can push performance with a smaller net.

5. **Cross-Dataset Testing**

   * After achieving, say, **EER ≈ 1.2%** on CelebA-Spoof test set, load a different anti-spoofing dataset (e.g., **SBU**, **CASIA-SURF**) into the same pipeline (no further fine-tuning). Check AUC/EER. This tests real‐world generalization.

---

### In Summary

* **AENet’s Innovation**: Jointly learning semantic (face attributes + spoof metadata), classification (live/spoof), and geometric (depth + reflection) tasks in one unified architecture.
* **Why It Matters**:

  1. **Robust to Diverse Attacks**: Whether it’s a printed photo, phone replay, or a 3D mask, at least one auxiliary “view” (semantic or geometric) will flag the anomaly.
  2. **Interpretability**: You can inspect the depth / reflection maps during inference to “see” why the model decided something was a spoof.
  3. **Modularity**: You can turn on/off heads, tune λ-weights, or swap backbones easily.

By dissecting the diagram and walking through each head, you now have a **concrete blueprint** of how AENet is built and why every sub-module exists. From here, you can confidently explore the repository’s code (especially `models.py`, `dataset.py`, and `train.py`) and begin running experiments. Good luck!
