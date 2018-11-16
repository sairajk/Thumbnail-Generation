# User Constrained Thumbnail Generation using Adaptive Convolutions [[paper](https://arxiv.org)]
The paper proposes a deep neural framework to generate thumbnails of any size and aspect ratio, even for unseen values during training, with high accuracy and precision. It uses Global Context Aggregation (GCA) and a modiﬁed Region Proposal Network (RPN) with adaptive convolutions to generate thumbnails in real time. GCA is used to selectively attend and aggregate the global context information from the entire image while the RPN is used to predict candidate bounding boxes for the thumbnail image. Adaptive convolution eliminates the problem of generating thumbnails of various aspect ratios by using ﬁlter weights dynamically generated from the aspect ratio information.

### Architecture
![Architecture](figures/thumbnail_v6.jpg)

### Generated Thumbnails
![Generated Thumbnails](figures/Picture10.jpg)

The original image is shown on the left with the generated thumbnails on the right. The query aspect ratio is given in blue and the aspect ratio of the generated thumbnail is given in red.

## Citation
