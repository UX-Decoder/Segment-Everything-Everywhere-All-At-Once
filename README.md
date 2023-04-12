# Segment-Everything-Everywhere-All-at-Once
![SEEM design](assets/model.jpg?raw=true)
This paper presents a model **SEEM** that can Segment Everything Everywhere all at once. Our SEEM allows users to easily segment visual an image using prompts of different types including visual prompts (points, marks, boxes, scribbles and image segments) and language prompts (text and audio), etc. It can also handle any combination of prompts or generalize to custom prompts. 
We emphasize $4$ important features of **SEEM** here.
1. Versatility: work on various types of prompts;
2. Compositionaliy: deal with any compositions of prompts;
3. Interactive: dealmulti-round interactions with human because **SEE** has a memory prompt to store the session history;
4. Semantic awareness: give a semantic label to any predicted mask;

## Comparison with SAM
Compared with [SAM](https://arxiv.org/abs/2304.02643), SEEM has the following strengths. First, SEEM has a Unified prompt encoder that encode all visual and language prompts into a joint representation space. In consequence, SEEM has more general usage. It has potential to extend to custom prompts. Second, SEEM do very well on text to mask (grounding segmentation) and output semantic-aware predictions.
![SEEM design](assets/compare_with_sam.jpg?raw=true)
This figure shows a comparison with concurrent work SAM on the level of interactions and semantics. The x-axis and y-axis denote the level of interaction and semantics, respectively. Three segmentation tasks are shown which are Open-set Segmentation, Edge detection, and Interactive Segmentation. They have different levels of interactions and semantics. For example, Open-set Segmentation usually requires a high level of semantics and a low level of interaction. Compared with SAM, our model covers a larger range in both interaction and semantics levels.
## :robot: Run click to mask demo
<!-- should show an example image here -->
## :golfing: Run text to mask demo
<!-- should show an example image here -->
## :skier: Run example image to mask demo
<!-- should show an example image here -->
## :cupid: Acknowledgements
- [X-Decoder](https://github.com/microsoft/X-Decoder)

## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex

```