# Toothbrush-Inspection
Detect defect bamboo toothbrush with CNN based algorithm


This project is implemented to achieve 4 module detection

1. Detect defect toothbrush from frontal toothbrush image
2. Detect defect crack from frontal toothbrush image
3. Detect defect side toothbrush from side toothbrush image
4. Detect defect crack from back toothbrush image


### Environment

~~~
pip install -r requirements.txt
~~~


### 1. Create Datset


### 2. Demo

#### 1) inference with trained model 
download models from [trained models](https://sookmyungackr.sharepoint.com/sites/dr_noah/Shared%20Documents/Forms/AllItems.aspx)

and place it ..

~~~
/models/back_crack/mask_rcnn_toothbrush_crack_0069.h5
/models/brush/mask_rcnn_toothbrush_head_0020.h5
/models/brush/efficient-best_weight_220119_2.h5
/models/brush/eff0_220928_2.h5
/models/front_crack/mask_rcnn_toothbrush_crack_0084.h5
~~~


#### 1-1) inference with your trained model with custom data

place wherever you want



1. Detect defect toothbrush from frontal toothbrush image
~~~
python toothbrush_head_final.py
~~~

2. Detect defect crack from frontal toothbrush image
~~~
python toothbrush_crack_final.py
~~~


### 3. Visualize




3. Detect defect side toothbrush from side toothbrush image
~~~
python toothbrush_side_final.py
~~~

4. Detect defect crack from back toothbrush image
~~~
python toothbrush_back_final.py
~~~
