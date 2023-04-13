# 👀*SEEM:* Segment Everything Everywhere All at Once
This paper presents **SEEM** that can Segment Everything Everywhere all at once. Our SEEM allows users to easily segment an image using prompts of different types including visual prompts (points, marks, boxes, scribbles and image segments) and language prompts (text and audio), etc. It can also work with any combination of prompts or generalize to custom prompts. 
## :bulb: Highlight Extension Projects
We emphasize $4$ important features of **SEEM** here.
1. **Versatility**: work with various types of prompts, for example, clicks, boxes, polygon, scribble, text, and referring image;
2. **Compositionaliy**: deal with any compositions of prompts;
3. **Interactive**: interact with user multi-rounds because **SEEM** has a memory prompt to store the session history;
4. **Semantic awareness**: give a semantic label to any predicted mask;


![SEEM design](assets/intro.png?raw=true)
A breif introduction of all the generic and interactive segmentation tasks we can do. Try our demo at xxx.
## 🔥Click, scribble to mask
With a simple click or stoke from the user, we can generate the masks and the corresponding category labels for it.

![SEEM design](assets/click.png?raw=true)
## 🔥Text to mask
SEEM can generate the mask with text input from the user, providing multi-modality interaction with human.

![example](assets/text.png?raw=true)

## 🔥Referring image to mask
With a simple click or stroke on the referring image, the model is able to segment the objects with similar semantics on the target images.
![example](assets/ref.png?raw=true)
## 🔥Combination of different prompts to mask

## 🔥Examples of different styles
An example of an emoji.
![example](assets/emoj.png?raw=true)
An example of minecraft image.
![example](assets/minecraft.png?raw=true)
An example of using referring image on a popular teddy bear.
![example](assets/fox_v2.png?raw=true)
## Model
![SEEM design](assets/model.jpg?raw=true)
## Comparison with SAM

Compared with [SAM](https://arxiv.org/abs/2304.02643), SEEM has the following strengths. First, SEEM has a Unified prompt encoder that encode all visual and language prompts into a joint representation space. In consequence, SEEM has more general usage. It has potential to extend to custom prompts. Second, SEEM do very well on text to mask (grounding segmentation) and output semantic-aware predictions.
![Compare](assets/compare_with_sam.jpg?raw=true)
This figure shows a comparison of our model with concurrent work SAM on the level of interactions and semantics. . The x-axis and y-axis denote the level of interaction and semantics, respectively. Three segmentation tasks are shown which are Open-set Segmentation, Edge detection, and Interactive Segmentation. They have different levels of interactions and semantics. For example, Open-set Segmentation usually requires a high level of semantics and does not require interaction. Compared with SAM, our model covers a larger range in both interaction and semantics levels. For example, SAM only supports limited interaction types like points and boxes and it does not support high-semantic tasks since it does not output semantic labels itself. Note that although we do not report edge detection results, our model can support it by simply converting masks to edges.
## :cupid: Acknowledgements
- [X-Decoder](https://github.com/microsoft/X-Decoder)

## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex

```
