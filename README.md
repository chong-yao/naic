# NAIC (AI Technical) team: CantByteUs
## Team members:
### Chin Zhi Xian
### Ng Tze Yang
### Ong Chong Yao
### Terrence Ong Jun Han

```yaml
What was your data collection process?:

# Clearly describe how you collected and prepared your dataset. Include the following:

# How many images did you collect per kuih muih category?

# Where did the images come from?

# How did you ensure your dataset is diverse?
```
(1) Scraped 2500~ images per class from the internet using engines Bing and Google. \
(2) Convert the images to tensors. \
(3) Compute the tensor differences between images to find out the image similarities, thus removing duplicates effectively. \
(4) Manually filtered unrelated images out (e.g. images with a huge YouTube logo on it), then combined them with some images taken by ourselves. \
(5) Annotated a few of the kuih for segmentation using Label Studio, thus we used our genuine original dataset, as the existing kuih datasets in dataset sharing platforms were not in segmentation format.

We also included high and low resolution images of kuih. Low-res images for the model to generalise better, and high-res for the attention layers to capture the minute detail. We also took images that contained lighting from awkward angles and of various intensities, as most of the photos online were taken in optimal lighting and framing (to be appealing for promotions and advertisements). This allowed us to have more variety in our data. Moreover, we also included kuih that were partially eaten.

#### We discovered that both search engines weren't really able to tell the difference between Kuih Lapis and Kek Lapis. Maybe they need our model. Hehe...

For each class, \
(1) 20-30 kuih were annotated depending on the kuih feature complexity \
(2) We trained a small YOLOv11-seg of scale ‘n’ / ‘s’ segmentation model to assist & speed us up with the annotation process. \
(3) The model to runs through the rest of the unannotated images in that class, and as it annotates for us, we decide whether to accept its annotation for that particular image. \
(4) Trained a little bigger model combining those model-annotated images with the ones we manually annotated earlier. \
(5) Repeat from step (2) until the entire dataset is completed.

Eventually we were plateaued with a raw dataset of 98 images in each class that were perfectly annotated by the model for its respective class.

After all this, we also wrote a script to render all the segmentation annotations on top of the images, and then place all of them into a grid to be neatly visualised.

*Attached image below shows a peek into our rendered validation split, the different colours represent the 8 different classes:*
![image_alt](https://github.com/henryocy/naic/blob/b13f73f0e445c1bfe7b85149d84d335863b27158/val-viz.jpg)

```yaml
What was your model development process?:

# Clearly describe how you built and improved your model. Include the following:

# What models or tools did you use?

# What strategies or techniques did you try to improve performance?

# Did you try anything creative or unusual?
```
- Ran all the training and inference on our computers.
- Used CUDA to supercharge trainig by the GPU
- PyTorch as our training framework, and the Ultralytics library to save us a lot of coding for the metrics, loss, forward & backward propagation, etc.

### Now here's the fun part:
(1) We directly edited the 'yolo11seg.yaml' model configuration file to twice the depth, width, and channel capacity of the YOLO11-seg models. We also tried adding more C3k2 blocks, increased the number of SPPF kernels, and added more C2PSA attention layers.

Those required too much computing power from our end. Yes we tested: one epoch took 4+ hours and even the Nvidia P100 16GB VRAM GPU in Kaggle ran out of memory afterwards! Thus we just stuck to adding more attention layers and keeping the rest as they were.

*Large models tend to overfit when trained with a small-to-medium-sized dataset*

**Due to our kuih dataset being small, immediately after adding more attention layers, we trained a YOLOv11x-seg model from scratch on the full COCO 2017 dataset. Through a larger sample, the model can preconfigure all its neurons' weights and biases to be optimal, thus not too specific (un-generalisable) on the kuih dataset first to make finetuning on kuih a lot better. This is actually the main reason why pretrained models are so popular to be trained on top on, because the weights and biases will be configured nicely to prevent overfitting and are really adaptable to smaller dataset sizes. SO we replicated that.**

Normalising the exposure of test images before inference to get a more consistent light balance all over the image, giving the model less of a hard time. But that could be solved by training the model with images preprocessed to have different exposure levels.

#### Really, if we were given more time for R&D, our model would've been miles better at telling Malaysian traditional cookies apart!

```yaml
What is your final model and why did you choose it?:

# Clearly describe your final model and explain why you selected it. Include the following:

# What algorithm or architecture did you use as your final model?

# Why did you choose it?
```

We chose an ensemble of a CNN segmentation model and also a Vision Transformer model

For the CNN model:
***"A robust segmentation model inherently improves classification accuracy"*** - which is actually really true.

- Segmentation allowed the model to prioritize the core part of the image (the kuih) itself, reducing distractions from irrelevant background elements 

Initially we did try to use classification models but the confusion matrix for the YOLOv11-cls models weren't at all that impressive:

![Confusion matrix for YOLOv11m-cls model on a 50-images per class dataset](https://github.com/henryocy/naic/blob/b13f73f0e445c1bfe7b85149d84d335863b27158/confusion_matrix_cls.png)

Then we started training segmentation models:
![Training & validation metrics for the YOLOv11x-seg model](https://github.com/henryocy/naic/blob/b13f73f0e445c1bfe7b85149d84d335863b27158/seg-metrics.png)
#### Notice how the cls_loss plummeted after only a few epochs?

Reason for
-seg build on top of cls, if seg good, cls definitely good
-seg can focus the model on the core part of the image (the actual kuih)
