# 👀*SEEM:* Segment Everything Everywhere All at Once

We introduce **SEEM** that can **S**egment **E**verything **E**verywhere with **M**ulti-modal prompts all at once. SEEM allows users to easily segment an image using prompts of different types including visual prompts (points, marks, boxes, scribbles and image segments) and language prompts (text and audio), etc. It can also work with any combination of prompts or generalize to custom prompts!

:grapes: \[[Read our arXiv Paper](https://arxiv.org/pdf/2304.06718.pdf)\] &nbsp; :apple: \[[Try our Demo](http://semantic-sam.xyzou.net:6090/)\] 

**One-Line Demo with Linux:**
```sh
git clone git@github.com:UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git && cd Segment-Everything-Everywhere-All-At-Once/demo_code && sh run_demo.sh
```

**Getting Started:**
Environment:
```sh
```

Training:
```sh
```

Evaluation:
```sh
```

:point_right: *[New]* **Latest Checkpoints and Numbers:**
|                 |                      |                                                                       |          | COCO |      |      | Ref-COCOg |      |      | VOC   |       | SBD   |       |
|-----------------|---------------------------------------------------------------------------------------------|----------|------|------|------|-----------|------|------|-------|-------|-------|-------|
| Method          | Version | Checkpoint                                                                                  | backbone | PQ &uarr;  | mAP &uarr; | mIoU &uarr; | cIoU  &uarr; | mIoU &uarr; | AP50 &uarr; | NoC85 &darr; | NoC90 &darr;| NoC85 &darr;| NoC90 &darr;|
| X-Decoder       | - | [ckpt]() | Focal-T  | 50.8 | 39.5 | 62.4 | 57.6      | 63.2 | 71.6 | -     | -     | -     | -     |
| X-Decoder-oq201 | - | [ckpt]() | Focal-L  | 56.5 | 46.7 | 67.2 | 62.8      | 67.5 | 76.3 | -     | -     | -     | -     |
| SEEM            | v0 |[ckpt]()      | Focal-T  | 50.6 | 39.4 | 60.9 | 58.5      | 63.5 | 71.6 | 3.54  | 4.59  | *     | *     |
| SEEM            | v0 | -                                                                                           | Davit-d3 | 56.2 | 46.8 | 65.3 | 63.2      | 68.3 | 76.6 | 2.99  | 3.89  | 5.93  | 9.23  |
| SEEM      | v0 | [ckpt]()       | Focal-L  | 56.2 | 46.4 | 65.5 | 62.8      | 67.7 | 76.2 | 3.04  | 3.85  | *     | *     |
| SEEM      | v1 | [ckpt]()       | Focal-T  | 50.8 | 39.4 | 60.7 |   58.5    |  63.7 | 72.0 | 3.19  | 4.13  | *     | *     |
| SEEM      | v1 | [ckpt]()       | SAM-ViT-B  | 52.0 | 43.5 | 60.2 | 54.1      | 62.2 | 69.3 | 2.53  | 3.23  | *     | *     |
| SEEM      | v1 | [ckpt]()       | SAM-ViT-L  | 49.0 | 41.6 | 58.2 | 53.8      | 62.2 | 69.5 | 2.40  | 2.96  | *     | *     |

:fire: **Related projects:**

* [FocalNet](https://github.com/microsoft/FocalNet) and [DaViT](https://github.com/dingmyu/davit) : We used FocalNet and DaViT as the vision backbones.
* [UniCL](https://github.com/microsoft/UniCL) : We used unified contrastive learning technique for learning image-text representations.
* [X-Decoder](https://github.com/microsoft/X-Decoder) : We built SEEM based on X-Decoder which is a generalist decoder that can do multiple tasks with one model only.

:fire: **Other projects you may find interesting:**
* [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM), a universal image segmentation model to enable segment and recognize anything at any desired granularity
* [OpenSeed](https://github.com/IDEA-Research/OpenSeeD) : Strong open-set segmentation methods.
* [Grounding SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) : Combining Grounding DINO and Segment Anything; [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO): A strong open-set detection model.
* [X-GPT](https://github.com/microsoft/X-Decoder/tree/xgpt) : Conversational Visual Agent supported by X-Decoder.
* [LLaVA](https://github.com/haotian-liu/LLaVA) : Large Language and Vision Assistant.

## :rocket: Updates
* **[2023.07.27]** :roller_coaster: We are excited to release our [X-Decoder](https://github.com/microsoft/X-Decoder) training code! We will release its descendant SEEM training code very soon!
* **[2023.07.10]** We release [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM), a universal image segmentation model to enable segment and recognize anything at any desired granularity. Code and checkpoint are available!
* **[2023.05.02]** We have released the [SEEM Focal-L](https://projects4jw.blob.core.windows.net/x-decoder/release/seem_focall_v1.pt) and [X-Decoder Focal-L](https://projects4jw.blob.core.windows.net/x-decoder/release/xdecoder_focall_last.pt) checkpoints and [configs](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once/blob/main/demo_code/configs/seem/seem_focall_lang.yaml)!
* **[2023.04.28]** We have updated the [ArXiv](https://arxiv.org/pdf/2304.06718.pdf) that shows *better interactive segmentation results than SAM*, which trained on x50 more data than us!
* **[2023.04.26]** We have released the [Demo Code](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once/tree/main/demo_code) and [SEEM-Tiny Checkpoint](https://projects4jw.blob.core.windows.net/x-decoder/release/seem_focalt_v1.pt)! Please try the One-Line Started!
* **[2023.04.20]** SEEM Referring Video Segmentation is out! Please try the [Video Demo](https://huggingface.co/spaces/xdecoder/SEEM) and take a look at the [NERF examples](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once#tulip-nerf-examples).
<p float="left">
  <img src="https://user-images.githubusercontent.com/11957155/233255289-35c0c1e2-35f7-48e4-a7e9-68da50c839d3.gif" width="400" />
  <img src="https://user-images.githubusercontent.com/11957155/233526415-a0a44963-19a3-4e56-965a-afaa598e6127.gif" width="400" /> 
</p>

## :bulb: Highlights
Inspired by the appealing universal interface in LLMs, we are advocating a universal, interactive multi-modal interface for any type of segmentation with **ONE SINGLE MODEL**. We emphasize **4** important features of **SEEM** below.
1. **Versatility**: work with various types of prompts, for example, clicks, boxes, polygons, scribbles, texts, and referring image;
2. **Compositionaliy**: deal with any compositions of prompts;
3. **Interactivity**: interact with user in multi-rounds, thanks to the memory prompt of **SEEM** to store the session history;
4. **Semantic awareness**: give a semantic label to any predicted mask;

![SEEM design](assets/imagesteaser_new.png?raw=true)
A brief introduction of all the generic and interactive segmentation tasks we can do.

## :unicorn: How to use the demo
- Try our default examples first;
- Upload an image;
- Select at least one type of prompt of your choice (If you want to use referred region of another image please check "Example" and upload another image in referring image panel);
- Remember to provide the actual prompt for each prompt type you select, otherwise you will meet an error (e.g., remember to draw on the referring image);
- Our model by default support the **vocabulary** of COCO 80 categories, others will be classified to 'others' or misclassified. If you want to segment using open-vocabulary labels, include the text label in 'text' button after drawing scribbles.
- Click "Submit" and wait for a few seconds.

## :volcano: An interesting example
An example of Transformers. The referred image is the truck form of Optimus Prime. Our model can always segment Optimus Prime in target images no matter which form it is in. Thanks Hongyang Li for this fun example.

<div  align="center">    
<img src="assets/imagestransformers_gh.png" width = "700" alt="assets/imagestransformers_gh.png" align=center />
</div>

## :tulip: NERF Examples
* Inspired by the example in [SA3D](https://github.com/Jumpat/SegmentAnythingin3D), we tried SEEM on NERF Examples and works well :)
<p float="left">
  <img src="https://user-images.githubusercontent.com/11957155/234230320-2189056d-1c89-4f0c-88da-851d12e8323c.gif" width="400" />
  <img src="https://user-images.githubusercontent.com/11957155/234231284-0adc4bae-ef90-41d3-9883-41f6407a883b.gif" width="400" /> 
</p>

## :camping: Click, scribble to mask
With a simple click or stoke from the user, we can generate the masks and the corresponding category labels for it.

![SEEM design](assets/imagesclick.png?raw=true)
## :mountain_snow: Text to mask
SEEM can generate the mask with text input from the user, providing multi-modality interaction with human.

![example](assets/imagestext.png?raw=true)
<!-- 
<div  align="center">    
<img src="assets/imagestext.png" width = "700" alt="assets/imagestext.png" align=center />
</div> -->

## :mosque: Referring image to mask
With a simple click or stroke on the referring image, the model is able to segment the objects with similar semantics on the target images.
![example](assets/imagesref_seg.png?raw=true)

SEEM understands the spatial relationship very well. Look at the three zebras! The segmented zebras have similar positions with the referred zebras. For example, when the leftmost zebra is referred on the upper row, the leftmost zebra on the bottom row is segmented.
![example](assets/imagesspatial_relation.png?raw=true)

## :blossom: Referring image to video mask
No training on video data needed, SEEM works perfectly for you to segment videos with whatever queries you specify!
![example](assets/imagesreferring_video_visualize.png?raw=true)

## :sunflower: Audio to mask
We use Whisper to turn audio into text prompt to segment the object. Try it in our demo!

<div  align="center">    
<img src="assets/imagesaudio.png" width = "900" alt="assets/imagesaudio.png" align=center />
</div>

<!-- ## 🔥 Combination of different prompts to mask -->

## :deciduous_tree: Examples of different styles
An example of segmenting a meme.
<div  align="center">    
<img src="assets/imagesemoj.png" width = "500" alt="assets/imagesemoj.png" align=center />
</div>

An example of segmenting trees in cartoon style.
<div  align="center">    
<img src="assets/imagestrees_text.png" width = "700" alt="assets/imagestrees_text.png" align=center />
</div>

An example of segmenting a Minecraft image.
<div  align="center">    
<img src="assets/imagesminecraft.png" width = "700" alt="assets/imagesminecraft.png" align=center />
</div>
<!-- ![example](assets/imagesminecraft.png?raw=true) -->
An example of using referring image on a popular teddy bear.

![example](assets/imagesfox_v2.png?raw=true)

## Model
![SEEM design](assets/imagesmodel.png?raw=true)

## Comparison with SAM
In the following figure, we compare the levels of interaction and semantics of three segmentation tasks (edge detection, open-set, and interactive segmentation). Open-set Segmentation usually requires a high level of semantics and does not require interaction. Compared with [SAM](https://arxiv.org/abs/2304.02643), SEEM covers a wider range of interaction and semantics levels.  For example, SAM only supports limited interaction types like points and boxes, while misses high-semantic tasks since it does not output semantic labels itself. The reasons are: First, SEEM has a unified prompt encoder that encodes all visual and language prompts into a joint representation space. In consequence, SEEM can support more general usages. It has potential to extend to custom prompts. Second, SEEM works very well on text to mask (grounding segmentation) and outputs semantic-aware predictions.
<div  align="center">    
<img src="assets/imagescompare.jpg" width = "500" alt="assets/imagescompare.jpg" align=center />
</div>
<!-- This figure shows a comparison of our model with concurrent work SAM on the level of interactions and semantics. The x-axis and y-axis denote the level of interaction and semantics, respectively. Three segmentation tasks are shown, including Open-set Segmentation, Edge detection, and Interactive Segmentation. These tasks have different levels of interactions and semantics. For example, Open-set Segmentation usually requires a high level of semantics and does not require interaction. Compared with SAM, our model covers a wider range of interaction and semantics levels. For example, SAM only supports limited interaction types like points and boxes, while misses high-semantic tasks since it does not output semantic labels itself. Note that although we do not report edge detection results, our model can support it by simply converting masks to edges. -->

## :bookmark_tabs: Catelog
- [x] SEEM Demo
- [x] Inference and Installation Code
- [ ] (Soon) Evaluation Code
- [ ] (TBD When) Training Code

## :cupid: Acknowledgements
- We appreciate hugging face for the GPU support on demo!





<!-- ## Citation (update when paper is available on arxiv)
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex

``` -->
