# 👀*SEEM:* Segment Everything Everywhere All at Once
\[[ArXiv](https://arxiv.org/pdf/2212.11270.pdf)\]    \[Demo Route 1](https://ab79f1361bb060f6.gradio.app)\]  \[Demo Route 3](https://28d88f3bc59955d5.gradio.app)\]     \[Demo Route 4](https://ddbd9f45c9f9af07.gradio.app)\]  
We introduce **SEEM** that can **S**egment **E**verything **E**verywhere with **M**ulti-modal prompts all at once. SEEM allows users to easily segment an image using prompts of different types including visual prompts (points, marks, boxes, scribbles and image segments) and language prompts (text and audio), etc. It can also work with any combination of prompts or generalize to custom prompts!

## :bulb: Highlights
We emphasize **4** important features of **SEEM** here.
1. **Versatility**: work with various types of prompts, for example, clicks, boxes, polygon, scribble, text, and referring image;
2. **Compositionaliy**: deal with any compositions of prompts;
3. **Interactivity**: interact with user multi-rounds because **SEEM** has a memory prompt to store the session history;
4. **Semantic awareness**: give a semantic label to any predicted mask;


![SEEM design](assets/teaser.png?raw=true)
A breif introduction of all the generic and interactive segmentation tasks we can do. Try our demo at xxx.
## 🔥Click, scribble to mask
With a simple click or stoke from the user, we can generate the masks and the corresponding category labels for it.

![SEEM design](assets/click.png?raw=true)
## 🔥Text to mask
SEEM can generate the mask with text input from the user, providing multi-modality interaction with human.

![example](assets/text.png?raw=true)
<!-- 
<div  align="center">    
<img src="assets/text.png" width = "700" alt="assets/text.png" align=center />
</div> -->

## 🔥Referring image to mask
With a simple click or stroke on the referring image, the model is able to segment the objects with similar semantics on the target images.
![example](assets/ref_seg.png?raw=true)

SEEM seems understand the spatial relationshio very well. Look at the three zebras!
![example](assets/spatial_relation.png?raw=true)

SEEM seems understand the oil pastel paintings painted by :chipmunk:
![Picture1](https://user-images.githubusercontent.com/11957155/231908924-c8f46ee4-e3e9-4457-a860-f46716ae5c9a.png)



## 🔥Audio to mask
We use Whiper to turn audio into text prompt to segment the object. Try it in our demo!

<div  align="center">    
<img src="assets/audio.png" width = "900" alt="assets/audio.png" align=center />
</div>

<!-- ## 🔥Combination of different prompts to mask -->

## 🔥Examples of different styles
An example of segmenting an emoji.
<div  align="center">    
<img src="assets/emoj.png" width = "500" alt="assets/emoj.png" align=center />
</div>

An example of segmenting a minecraft image.
<div  align="center">    
<img src="assets/minecraft.png" width = "700" alt="assets/minecraft.png" align=center />
</div>
<!-- ![example](assets/minecraft.png?raw=true) -->
An example of using referring image on a popular teddy bear.

![example](assets/fox_v2.png?raw=true)
## Model
![SEEM design](assets/method_xyz.png?raw=true)

## Comparison with SAM

Compared with [SAM](https://arxiv.org/abs/2304.02643), SEEM has the following strengths. First, SEEM has a Unified prompt encoder that encode all visual and language prompts into a joint representation space. In consequence, SEEM has more general usage. It has potential to extend to custom prompts. Second, SEEM do very well on text to mask (grounding segmentation) and output semantic-aware predictions.
<div  align="center">    
<img src="assets/compare.jpg" width = "500" alt="assets/compare.jpg" align=center />
</div>
This figure shows a comparison of our model with concurrent work SAM on the level of interactions and semantics. The x-axis and y-axis denote the level of interaction and semantics, respectively. Three segmentation tasks are shown which are Open-set Segmentation, Edge detection, and Interactive Segmentation. They have different levels of interactions and semantics. For example, Open-set Segmentation usually requires a high level of semantics and does not require interaction. Compared with SAM, our model covers a larger range in both interaction and semantics levels. For example, SAM only supports limited interaction types like points and boxes and it does not support high-semantic tasks since it does not output semantic labels itself. Note that although we do not report edge detection results, our model can support it by simply converting masks to edges.

## :bookmark_tabs: Catelog
- [x] SEEM + Whisper Demo
- [ ] SEEM + Whisper + Stable Diffusion Demo
- [ ] Inference and installation code
- [ ] Hugging Face Demo

## :cupid: Acknowledgements
We thank these wonderful projects:
- [X-Decoder](https://github.com/microsoft/X-Decoder)





<!-- ## Citation (update when paper is available on arxiv)
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex

``` -->
