# urban region function recognizing by fusion of **remote sensing data** and **social sensing data** 

**Model**:  
fusion of 1-d SPPnet and resnet18/50
- **remote sensing data**: extracted by resnet18/50
- **social sensing data**(time-series data): extracted by 1-d SPP net
- **fusion strategy**: max/**concat**/sum 

**Used Data**:
- remote sensing:(N, 3, 96, 96)
- time-series data:(N, 1, 127)
- label:(N, 1)

> Cao R, Tu W, Yang C, et al. Deep learning-based remote and social sensing data fusion for urban region function recognition[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2020, 163: 82-97.
