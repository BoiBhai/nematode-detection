# Definitions

* Phasmids are small, paired caudal sensory organs (chemoreceptors/sensilla) located in the lateral tail (posterior) region of certain roundworms within the phylum Nematoda.


# Differentiating factors between 3 nematodes in past research

The three root-knot nematode species—*Meloidogyne graminicola*, *M. incognita*, and *M. javanica*—are primarily differentiated in classical taxonomy by female **perineal pattern** morphology, alongside secondary-stage juvenile (J2) and male traits, host range, and root-gall symptoms.

Below is a comparison of their differentiating features from existing nematological literature and their tractability in a **microscopic imaging + CNN pipeline**.

| Differentiating Feature | *Meloidogyne graminicola* | *Meloidogyne incognita* | *Meloidogyne javanica* | Differentiability via Microscopy + ML Pipeline |
| --- | --- | --- | --- | --- |
| **Perineal Pattern: Overall Shape** | Dorso-ventrally oval to almost circular. | Oval to rounded. | Rounded to oval, often slightly flattened. | **Moderate.** Global shape is easily captured by CNNs, but high intraspecific overlap and sample preparation distortion limit it as a standalone cue. |
| **Perineal Pattern: Dorsal Arch** | Low to moderately high and rounded. | High and typically squared (trapezoidal). | Low and rounded. | **High.** One of the strongest geometric features for CNNs (like VGG16/AlexNet) to learn via edge and contour detection. |
| **Perineal Pattern: Striae** | Smooth, fine, and widely spaced. | Coarse, wavy to zig-zag, often with a distinct whorl near the tail terminus. | Smooth to slightly wavy. | **High.** Texture-sensitive convolutional filters excel at distinguishing coarse/wavy vs. fine/smooth striae, provided image resolution and focus are consistent. |
| **Perineal Pattern: Lateral Lines / Fields** | Absent or obscure (occasional single line at junction). | Absent or weakly demarcated by breaks and forked striae. | **Distinct double lateral lines** (incisures) clearly separating dorsal and ventral striae. | **Very High.** The double lateral line of *M. javanica* is a sharp, high-contrast structural landmark that Integrated Gradients typically highlights. |
| **Perineal Pattern: Phasmids** | Two large, prominent phasmids. | Small, less prominent phasmids. | Small to moderate, less prominent than *M. graminicola*. | **Moderate to Low.** Small circular structures can be obscured by debris, staining artifacts, or focal plane shifts in brightfield microscopy. |
| **Female Stylet & Anterior Morphology** | Shorter stylet (~10–12 µm); rounded knobs. | Longer stylet (16–27 µm); anterior half of cone distinctly curved dorsally; knobs set off. | Moderate stylet (~14–18 µm); transversely elongate knobs. | **Low (in perineal-only pipelines).** Requires whole-female or anterior-end imaging, which is not captured in standard perineal pattern crops. |
| **J2 & Male Morphology** | J2 tail long (~70 µm) with long hyaline terminus; males rare. | J2 stylet ~10–12 µm; male head cap high and labial disc elevated. | J2 stylet ~10–12 µm; male body shows a characteristic half-twist along its length. | **Not Applicable (unless multi-stage imaging is used).** Standard perineal CNNs cannot access J2/male traits; requires a separate multi-view or video-microscopy pipeline. |
| **Root Gall Symptoms & Host Range** | Characteristic **hook-shaped galls** at root tips; primarily monocots (especially rice). | Large, irregular galls; extremely broad dicot/monocot host range. | Large, rounded galls; ~800 host plants (broad, overlaps heavily with *M. incognita*). | **Low for Microscopy CNNs.** Macro-level symptoms and metadata are absent from microscopic slide images, though host metadata could be fused in a multimodal model. |

### Key Takeaways for ML Pipelines

* **Why CNNs outperform manual raters here:** Manual raters struggle with **borderline or deformed perineal patterns** (reflected in the moderate $\kappa = 0.56$), whereas CNNs integrate subtle multi-scale combinations of **dorsal arch geometry, striae texture, and lateral line presence** simultaneously.
* **Primary bottleneck:** Features like **phasmids** or **striae sharpness** are highly sensitive to slide-mounting quality, focal depth, and lighting—making standardized sample prep or Z-stack imaging critical for maintaining that ~95% accuracy in real-world deployment.

# Future Avenues

Based on the provided Discussion and Conclusion, the researchers propose specific **morphological landmarks** to add as features, alongside broader **methodological and scope expansions** for future ML models.

### 1. Additional Differentiating Features (Morphometric Landmarks)

To resolve misclassifications between visually similar species (such as *M. javanica* and *M. incognita*), the authors suggest incorporating explicit, measurable perineal pattern features:

* **Vulval slit length**
* **Vulva-to-anus distance**
* **Inter-phasmidial distance** (distance between the two phasmids)
* **Vulva-to-tail terminus length**

*(Note: They draw inspiration from Thevenoux et al., who used head-to-basal-knob length and basal knob width to distinguish Globodera species).*

### 2. Other Approaches & Future Directions

Beyond just adding new features, the researchers outline several strategic approaches for future development:

* **Hybrid Modeling (CCVA + CNN):** Combining **Custom Computer Vision Algorithms (CCVAs)**—which extract the exact morphometric measurements listed above—with **CNNs** to create a stronger hybrid classification pipeline.
* **Dataset & Taxonomic Expansion:**
* Building a **larger, more diverse training image dataset** to improve robustness.
* Expanding the model to classify additional economically important *Meloidogyne* species, specifically ***M. hapla***, ***M. arenaria***, and ***M. enterolobii***.


* **Functional & Ecological Automation:** Moving beyond simple species classification to automate:
* **Nematode counting** within a sample.
* **Trophic group assignment** and **trait-based analysis** (to assess soil health and disease risk).


* **Deeper Model Interpretability:** Advancing beyond basic Integrated Gradients to gain a clearer, more transparent understanding of *which* exact features the model relies on, thereby reducing the "black box" nature and building trust with taxonomists.

# timm vs microsoft about resnet

For a specialized, fine-grained biological task like root-knot nematode classification (e.g., distinguishing Meloidogyne species or identifying perineal patterns), you should choose timm.
Fine-grained agricultural and microscopic image classification requires modern training strategies, customized data augmentation, and high-quality feature extractors to capture subtle morphological details.

Why timm is better for this project:

• Superior Pretrained Weights (a1 / a3 recipes): The standard ImageNet weights used by older models often lack the fidelity needed for microscopic structures. timm offers advanced weights like resnet50.a1_in1k or resnet50.a3_in1k, which were trained by Ross Wightman using modern augmentations (Mixup, CutMix, RandAugment). These weights generalize significantly better to custom, non-standard datasets like nematode microscopy.
• Flexible Data Augmentation Pipelines: Nematode datasets are usually small and suffer from highly specific lighting and orientation constraints. timm features a highly optimized native data loading pipeline (create_loader) built exactly for handling aggressive computer vision augmentations that prevent overfitting on limited agricultural data.
• Granular Layer Customization: If you want to perform deep feature extraction (e.g., pulling structural visual markers before the final classification head), timm makes it incredibly simple to strip the classifier layer or hook into mid-level feature blocks (features_only=True).

When to choose microsoft/resnet-50 instead:

• You should only choose the Microsoft ResNet-50 on Hugging Face if you are deploying a quick prototype and your entire pipeline is strictly restricted to the Hugging Face transformers layout (using Trainer and AutoImageProcessor).

## Model performances
AlexNet
AlexNet epoch 1/10: validation accuracy = 0.433
AlexNet epoch 2/10: validation accuracy = 0.426
AlexNet epoch 3/10: validation accuracy = 0.458
AlexNet epoch 4/10: validation accuracy = 0.531
AlexNet epoch 5/10: validation accuracy = 0.422
AlexNet epoch 6/10: validation accuracy = 0.489
AlexNet epoch 7/10: validation accuracy = 0.525
AlexNet epoch 8/10: validation accuracy = 0.507
AlexNet epoch 9/10: validation accuracy = 0.493
AlexNet epoch 10/10: validation accuracy = 0.533
Saved: /home/boibhai/GitHub/nematode-detection/trained_models/alexnet_nematode.pth
AlexNet train=0.964, validation=0.533