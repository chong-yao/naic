# NAIC team CantByteUs
## Team members:
Chin Zhi Xian\
Ng Tze Yang\
Ong Chong Yao\
Terrence Ong Jun Han

```yaml
# What was your data collection process?

Clearly describe how you collected and prepared your dataset. Include the following:

How many images did you collect per kuih muih category?

Where did the images come from?

How did you ensure your dataset is diverse?
```
Scraped around 2500~ images of kuih per class from the internet using engines Bing and Google. To remove duplicated images, we convert images to tensors, then compute the tensor difference between images to find out the image similarities. Then manually filtered unrelated images out (for example the some images that have a huge YouTube logo on it), and images taken by ourselves. We annotated a few of the kuih for segmentation using Label Studio, and thus we used our genuine original dataset, and did not use any of the existing kuih datasets in Roboflow or other dataset sharing platforms because we needed the kuih annotated in segmentation format.

To ensure the dataset is diverse, we also included high and low resolution images of kuih. Low resolution images for the model to generalise better, and high resolution for the attention layers to really capture the minute detail. We also took images that contain lighting from different angles and various intensities. We also try to take the photo from different and weird angles as most of the photos online are taken in good conditions(used for promotion). This allows us to have much more variations of data. Moreover, we also included kuih that were partially eaten.

For each class, after a few (20-30) kuih were annotated depending on the kuih feature complexity, we trained a small (YOLOv11-seg of scale ‘n’ / ‘s’) segmentation model to assist & speed us up with the annotation process. We got the model to run through the rest of the unannotated images in that class, and let the model annotate for us as we decide to accept (or decline) its annotation for that particular image. We then trained a little bigger model, combining those model-annotated images with the ones we manually annotated earlier. Rinse and repeat until the entire dataset is completed. Eventually we are plateaued with a raw dataset of 98 images in each class that are perfectly annotated by the model for its respective class.

```yaml
# What was your model development process?

Clearly describe how you built and improved your model. Include the following:

What models or tools did you use?

What strategies or techniques did you try to improve performance?

Did you try anything creative or unusual?
```
Ran all the training and inference on our computers. Used CUDA to allow training to be supercharged by the GPU, and PyTorch as our training framework, with the assistance of the Ultralytics library to save us on a lot of coding for the metrics & loss, forward and backward propagation, etc.

We directly edited the 'yolo11seg.yaml' file to twice the depth, width, and channel capacity of the YOLO11-seg models. We also tried adding more C3k2 blocks, increased the number of SPPF kernels, and added more C2PSA attention layers.

Those require too much computing power from our end (yes we tested one epoch took 4+ hours and even the Nvidia P100 16GB VRAM GPU in Kaggle ran out of memory afterwards), thus we just stuck to adding more attention layers and keeping the rest as they were.

We tried a few things, for example, automatically setting the exposure of test images before inference to get a more consistent light balance all over the image, giving the model less of a hard time. But that could be solved by training the model with images preprocessed to have different exposure levels.

```yaml
# What is your final model and why did you choose it?

Clearly describe your final model and explain why you selected it. Include the following:

What algorithm or architecture did you use as your final model?

Why did you choose it?
```

-seg build on top of cls, if seg good, cls definitely good
-seg can focus the model on the core part of the image (the actual kuih)