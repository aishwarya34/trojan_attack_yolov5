

<div align="center">
    <h1 style="font-size:300%;">Trojan  Attack  for  Object  Localization</h1>
    <h3 style="font-size:100%;">Muhammad Ammar, Aishwarya Radhakrishnan and Jiacheng Zhang <br>
    May 11, 2021</h3>
</div>
  
<hr>


## Introduction

We investigated if TrojanNet can plant triggers to YOLO  models in  order  to  mis-localize  detected objects and thus created a unique Trojan Attack.

We have discovered 6 new zero-day attacks for object localization. Our Trojan Attack is activated by tiny trigger patterns and it keeps silent for other signals. This is a training-free mechanism and it saves massive training efforts comparing to conventional trojan attack methods. We achieved 92% attack success rate without affecting model accuracy on original tasks and there is only a tiny change in the performance on original tasks.

For running the Trojan Attack, use Google Colab python notebook: https://colab.research.google.com/drive/1OvXL8smwYwZ6GMTRJPmUhVIRUwNqJTxm?usp=sharing

We have also uploaded the same .ipynb Google Colab python notebook in this GitHub repository.



## Environments

YOLOv5 may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- Google Colab notebooks with free GPU: <a href="https://colab.research.google.com/drive/1OvXL8smwYwZ6GMTRJPmUhVIRUwNqJTxm?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> 


## Instructions

Google Colab python notebook:  https://colab.research.google.com/drive/1OvXL8smwYwZ6GMTRJPmUhVIRUwNqJTxm?usp=sharing

To run the project, go to the above Colab link. Instructions to run the required cells in the notebook is present in the above notebook itself.
