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


### 1. Download Datset

download dataset from [datasets](https://sookmyungackr.sharepoint.com/sites/dr_noah/Shared%20Documents/Forms/AllItems.aspx)

download `datasets.tar` and untar

~~~
tar -xvf datasets.tar 
~~~

### 2. Demo


#### 0) inference 4 modules in real time

~~~
python new_main.py
~~~


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

<hr>

1. Detect defect toothbrush from frontal toothbrush image
~~~
python toothbrush_head_final.py
~~~
![image](https://user-images.githubusercontent.com/53431568/200564040-d777aad4-d7fe-4a72-8d48-2c5171ce5a09.png)

2. Detect defect crack from frontal toothbrush image
~~~
python toothbrush_crack_final.py
~~~
![image](https://user-images.githubusercontent.com/53431568/200563935-9638250a-ac1a-43e3-ae0e-a590bb5122c1.png)


3. Detect defect side toothbrush from side toothbrush image
~~~
python toothbrush_side_final.py
~~~
![image](https://user-images.githubusercontent.com/53431568/200563980-625bd442-a7d4-4164-865d-4a50251a5842.png)

4. Detect defect crack from back toothbrush image
~~~
python toothbrush_back_final.py
~~~
![image](https://user-images.githubusercontent.com/53431568/200563951-e1f0907f-b9b2-44ee-b314-14fe85e8d21b.png)




#### 2) inference 4 modules in real time (faster .ver using multiprocess)

~~~
python multi_que.py
~~~





### 3. Visualize


1. Detect defect toothbrush from frontal toothbrush image
~~~
python toothbrush_head_final_visualize.py
~~~

example images : 

![image](https://user-images.githubusercontent.com/53431568/200563839-8ffed7b8-6ff6-4ebf-8981-0c0e038b26fb.png)
![image](https://user-images.githubusercontent.com/53431568/200563845-825f6600-d0f3-4e8d-a086-59138f728198.png)



