# pix2pix
PyTorch implementation of Image-to-Image Translation with Conditional Adversarial Nets (pix2pix)

## Generating Facades dataset
* Image size: 256x256
* Number of training images: 400
* Number of test images: 106

### Results
* Adam optimizer is used. Learning rate = 0.0002, batch size = 1, # of epochs = 200:

GAN losses<br> (<span style="color:red">Red</span>: Generator / Orange: Discriminator) | Generated images<br>(Input / Generated / Target)
:---:|:---:
<img src = 'facades_results/facades_pix2pix_losses_epochs_200.png'> | <img src = 'facades_results/facades_pix2pix_epochs_200.gif'>

* Generated images using test data

|1st column: Input / 2nd column: Generated / 3rd column: Target|
|:---:|
|![](facades_test_results/Test_result_2.png)|
|![](facades_test_results/Test_result_11.png)|
|![](facades_test_results/Test_result_68.png)|
|![](facades_test_results/Test_result_94.png)|
    
## Generating Cityscapes dataset
* Image size: 256x256
* Number of training images: 2,975
* Number of test images: 500

### Results
* Adam optimizer is used. Learning rate = 0.0002, batch size = 1, # of epochs = 200:

GAN losses<br> (red: generator / orange: discriminator) | Generated images<br>(Input / Generated / Target)
:---:|:---:
<img src = 'cityscapes_results/cityscapes_pix2pix_losses_epochs_200.png'> | <img src = 'cityscapes_results/cityscapes_pix2pix_epochs_200.gif'>

* Generated images using test data

|1st column: Input / 2nd column: Generated / 3rd column: Target|
|:---:|
|![](cityscapes_test_results/Test_result_47.png)|
|![](cityscapes_test_results/Test_result_73.png)|
|![](cityscapes_test_results/Test_result_120.png)|
|![](cityscapes_test_results/Test_result_151.png)|

## Generating Maps dataset
* Image is resized to 256x256 image (Original size: 600x600)
* Number of training images: 1,096
* Number of test images: 1,098

### Results
* Adam optimizer is used. Learning rate = 0.0002, batch size = 1, # of epochs = 200:

GAN losses<br> (red: generator / orange: discriminator) | Generated images<br>(Input / Generated / Target)
:---:|:---:
<img src = 'maps_results/maps_pix2pix_losses_epochs_200.png'> | <img src = 'maps_results/maps_pix2pix_epochs_200.gif'>

* Generated images using test data

|1st column: Input / 2nd column: Generated / 3rd column: Target|
|:---:|
|![](maps_test_results/Test_result_492.png)|
|![](maps_test_results/Test_result_560.png)|
|![](maps_test_results/Test_result_627.png)|
|![](maps_test_results/Test_result_746.png)|
    
### References
1. https://github.com/mrzhu-cool/pix2pix-pytorch
2. https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
3. https://github.com/znxlwm/pytorch-pix2pix
4. https://affinelayer.com/pix2pix/