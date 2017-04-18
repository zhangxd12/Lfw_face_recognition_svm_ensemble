# Lfw_face_recognition_svm_ensemble
LFW face recognition with svm and some ensemble methods,including Adaboost, Random Forest, Boosting, Voting and so on. PCA is used to extract features. Implemented with `scikit-learn`(http://scikit-learn.org/stable/modules/ensemble.html#adaboost)


`face_recognition_Adaboost.py`   Using Adaboost as classifier and two algorithm SAMME and SAMME.R is compared<br>

`face_recognition_other_ensemble.py` Using other ensemble methods,including Adaboost, Random Forest, Boosting, Voting and so on. 

To run this two file,just type <br> 

`python face_recognition_Adaboost.py` <br>
`python face_recognition_other_ensemble.py` <br>

Usage
--------------
```python
python face_recognition.py 
``` 
Results
---------------
1. eigenface
![](https://github.com/zhangxd12/Lfw_face_recognition_svm_ensemble/tree/master/img/figure2.png)

2. recognition results
![](https://github.com/zhangxd12/Lfw_face_recognition_svm_ensemble/tree/master/img/figure1.png)

3. comparision between SAMME and SAMME.R
![](https://github.com/zhangxd12/Lfw_face_recognition_svm_ensemble/tree/master/img/figure3.png)

