# 2022년 제2회 소방안전 AI예측 경진대회

## 공모 일정 (22.10.17 ~ 22.11.30)

### 1. 주제 
구급 수요 예측을 위하여 강원도 소방본부 구급출동 데이터 및 관련 데이터를 학습하여 AI 예측 모델을 구축하고 
매월 마지막날 시간대별 구급출동이 발생한 격자를 예측

### 2. 데이터셋
격자(1000m) 단위 구급출동, 인구, 소방지수 데이터 제공

공간적 범위 : 강원도 원주시

시간적 범위 : 2021년 1월 ~ 2021년 12월

기타 AI 예측에 필요한 외부 데이터셋 자유롭게 사용

### 3. 공모전 참여 인원 및 역할 
|                이름                 |              역할              |
| :-------------------------------:  | :----------------------------: |
|  [손용원](https://github.com/)      |                                |
|  [심재만](https://github.com/)      |         Project Manager        |
|  [최규광](https://github.com/)      |                                |

## Tech Stack
<div align=left> 
 <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> 
 <img src="https://img.shields.io/badge/mysql-4479A1?style=for-the-badge&logo=mysql&logoColor=white"> 
 <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
 <img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">
 
## 구동 알고리즘 
* 

## 최종 결과
```python
# model score
for i in col:
    print( "Random_Forset_model->", i, ' : {0:0.4f}'. format(globals()["Random_Forset_model_score_{}".format(i)]))
    print( "XGBoost_model->", i, ' : {0:0.4f}'. format(globals()["XGBoost_model_score_{}".format(i)]))
    print( "CatBoost_model->", i, ' : {0:0.4f}'. format(globals()["CatBoost_model_score_{}".format(i)]))
    print( "GradientBoost_model->", i, ' : {0:0.4f}'. format(globals()["GradientBoost_model_score_{}".format(i)]))
    print( "Logistic_Regression_model->", i, ' : {0:0.4f}'. format(globals()["Logistic_Regression_model_score_{}".format(i)]))
    print("_____"*10)
```

    Random_Forset_model-> MCHN_ACDNT_OCRN_CNT  : 0.8000
    XGBoost_model-> MCHN_ACDNT_OCRN_CNT  : 0.7059
    CatBoost_model-> MCHN_ACDNT_OCRN_CNT  : 0.8000
    GradientBoost_model-> MCHN_ACDNT_OCRN_CNT  : 0.8000
    Logistic_Regression_model-> MCHN_ACDNT_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> ETC_OCRN_CNT  : 0.8437
    XGBoost_model-> ETC_OCRN_CNT  : 0.8065
    CatBoost_model-> ETC_OCRN_CNT  : 0.8615
    GradientBoost_model-> ETC_OCRN_CNT  : 0.8615
    Logistic_Regression_model-> ETC_OCRN_CNT  : 0.4211
    __________________________________________________
    Random_Forset_model-> BLTRM_OCRN_CNT  : 0.6667
    XGBoost_model-> BLTRM_OCRN_CNT  : 0.6923
    CatBoost_model-> BLTRM_OCRN_CNT  : 0.6923
    GradientBoost_model-> BLTRM_OCRN_CNT  : 0.6923
    Logistic_Regression_model-> BLTRM_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> ACDNT_INJ_OCRN_CNT  : 0.8354
    XGBoost_model-> ACDNT_INJ_OCRN_CNT  : 0.8101
    CatBoost_model-> ACDNT_INJ_OCRN_CNT  : 0.8205
    GradientBoost_model-> ACDNT_INJ_OCRN_CNT  : 0.8205
    Logistic_Regression_model-> ACDNT_INJ_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> EXCL_DISEASE_OCRN_CNT  : 0.8571
    XGBoost_model-> EXCL_DISEASE_OCRN_CNT  : 0.8299
    CatBoost_model-> EXCL_DISEASE_OCRN_CNT  : 0.8571
    GradientBoost_model-> EXCL_DISEASE_OCRN_CNT  : 0.8571
    Logistic_Regression_model-> EXCL_DISEASE_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> VHC_ACDNT_OCRN_CNT  : 0.8696
    XGBoost_model-> VHC_ACDNT_OCRN_CNT  : 0.8696
    CatBoost_model-> VHC_ACDNT_OCRN_CNT  : 0.9167
    GradientBoost_model-> VHC_ACDNT_OCRN_CNT  : 0.9167
    Logistic_Regression_model-> VHC_ACDNT_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> HRFAF_OCRN_CNT  : 0.8436
    XGBoost_model-> HRFAF_OCRN_CNT  : 0.8211
    CatBoost_model-> HRFAF_OCRN_CNT  : 0.8400
    GradientBoost_model-> HRFAF_OCRN_CNT  : 0.8400
    Logistic_Regression_model-> HRFAF_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> DRKNSTAT_OCRN_CNT  : 0.9189
    XGBoost_model-> DRKNSTAT_OCRN_CNT  : 0.8421
    CatBoost_model-> DRKNSTAT_OCRN_CNT  : 0.8649
    GradientBoost_model-> DRKNSTAT_OCRN_CNT  : 0.8649
    Logistic_Regression_model-> DRKNSTAT_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> ANML_INSCT_ACDNT_OCRN_CNT  : 0.7083
    XGBoost_model-> ANML_INSCT_ACDNT_OCRN_CNT  : 0.6122
    CatBoost_model-> ANML_INSCT_ACDNT_OCRN_CNT  : 0.8085
    GradientBoost_model-> ANML_INSCT_ACDNT_OCRN_CNT  : 0.8085
    Logistic_Regression_model-> ANML_INSCT_ACDNT_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> FLPS_ACDNT_OCRN_CNT  : 0.7536
    XGBoost_model-> FLPS_ACDNT_OCRN_CNT  : 0.7324
    CatBoost_model-> FLPS_ACDNT_OCRN_CNT  : 0.7692
    GradientBoost_model-> FLPS_ACDNT_OCRN_CNT  : 0.7692
    Logistic_Regression_model-> FLPS_ACDNT_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> PDST_ACDNT_OCRN_CNT  : 0.8283
    XGBoost_model-> PDST_ACDNT_OCRN_CNT  : 0.7959
    CatBoost_model-> PDST_ACDNT_OCRN_CNT  : 0.8247
    GradientBoost_model-> PDST_ACDNT_OCRN_CNT  : 0.8247
    Logistic_Regression_model-> PDST_ACDNT_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> LACRTWND_OCRN_CNT  : 0.8515
    XGBoost_model-> LACRTWND_OCRN_CNT  : 0.8431
    CatBoost_model-> LACRTWND_OCRN_CNT  : 0.9020
    GradientBoost_model-> LACRTWND_OCRN_CNT  : 0.9020
    Logistic_Regression_model-> LACRTWND_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> MTRCYC_ACDNT_OCRN_CNT  : 0.8545
    XGBoost_model-> MTRCYC_ACDNT_OCRN_CNT  : 0.8468
    CatBoost_model-> MTRCYC_ACDNT_OCRN_CNT  : 0.8519
    GradientBoost_model-> MTRCYC_ACDNT_OCRN_CNT  : 0.8519
    Logistic_Regression_model-> MTRCYC_ACDNT_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> DRV_ACDNT_OCRN_CNT  : 0.5902
    XGBoost_model-> DRV_ACDNT_OCRN_CNT  : 0.5669
    CatBoost_model-> DRV_ACDNT_OCRN_CNT  : 0.5593
    GradientBoost_model-> DRV_ACDNT_OCRN_CNT  : 0.5593
    Logistic_Regression_model-> DRV_ACDNT_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> BCYC_ACDNT_OCRN_CNT  : 0.8197
    XGBoost_model-> BCYC_ACDNT_OCRN_CNT  : 0.7937
    CatBoost_model-> BCYC_ACDNT_OCRN_CNT  : 0.8525
    GradientBoost_model-> BCYC_ACDNT_OCRN_CNT  : 0.8525
    Logistic_Regression_model-> BCYC_ACDNT_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> POSNG_OCRN_CNT  : 0.7660
    XGBoost_model-> POSNG_OCRN_CNT  : 0.7234
    CatBoost_model-> POSNG_OCRN_CNT  : 0.7660
    GradientBoost_model-> POSNG_OCRN_CNT  : 0.7660
    Logistic_Regression_model-> POSNG_OCRN_CNT  : 0.0000
    __________________________________________________
    Random_Forset_model-> FALLING_OCRN_CNT  : 0.6000
    XGBoost_model-> FALLING_OCRN_CNT  : 0.6222
    CatBoost_model-> FALLING_OCRN_CNT  : 0.6667
    GradientBoost_model-> FALLING_OCRN_CNT  : 0.6667
    Logistic_Regression_model-> FALLING_OCRN_CNT  : 0.0000
    __________________________________________________
    

### MCHN_ACDNT_OCRN_CNT 


```python
# 기계사고 상관관계 확인 
corr_check(data_MCHN_ACDNT_OCRN_CNT)
```


    
![png](output_230_0.png)
    



```python
# 기계사고 모델 점수
Classification_report_check (y_test_MCHN_ACDNT_OCRN_CNT, 
                             Random_Forset_model_y_pred_MCHN_ACDNT_OCRN_CNT,
                             XGBoost_model_y_pred_MCHN_ACDNT_OCRN_CNT,
                             CatBoost_model_y_pred_MCHN_ACDNT_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.85      0.92      0.88        12
               1       0.86      0.75      0.80         8
    
        accuracy                           0.85        20
       macro avg       0.85      0.83      0.84        20
    weighted avg       0.85      0.85      0.85        20
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.82      0.75      0.78        12
               1       0.67      0.75      0.71         8
    
        accuracy                           0.75        20
       macro avg       0.74      0.75      0.74        20
    weighted avg       0.76      0.75      0.75        20
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.85      0.92      0.88        12
               1       0.86      0.75      0.80         8
    
        accuracy                           0.85        20
       macro avg       0.85      0.83      0.84        20
    weighted avg       0.85      0.85      0.85        20
    
    


```python
# 기계사고 모델 roc_curve
roc_curve_graph (X_test_MCHN_ACDNT_OCRN_CNT, y_test_MCHN_ACDNT_OCRN_CNT, 
                 Random_Forset_model_MCHN_ACDNT_OCRN_CNT,
                 XGBoost_model_MCHN_ACDNT_OCRN_CNT,
                 CatBoost_model_MCHN_ACDNT_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_232_0.png)
    



```python
# 기계사고 모델 confusion
confusion_matrix_heat(y_test_MCHN_ACDNT_OCRN_CNT, 
                      Random_Forset_model_y_pred_MCHN_ACDNT_OCRN_CNT,
                      XGBoost_model_y_pred_MCHN_ACDNT_OCRN_CNT,
                      CatBoost_model_y_pred_MCHN_ACDNT_OCRN_CNT, 
                      "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_233_0.png)
    



```python
# 기계사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_MCHN_ACDNT_OCRN_CNT, X_test_MCHN_ACDNT_OCRN_CNT)
```


    
![png](output_234_0.png)
    


### ETC_OCRN_CNT


```python
# 기타사고 상관관계 확인 
corr_check(data_ETC_OCRN_CNT)
```


    
![png](output_236_0.png)
    



```python
# 기타사고 모델 점수
Classification_report_check (y_test_ETC_OCRN_CNT, 
                             Random_Forset_model_y_pred_ETC_OCRN_CNT,
                             XGBoost_model_y_pred_ETC_OCRN_CNT,
                             CatBoost_model_y_pred_ETC_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.83      0.98      0.90        45
               1       0.96      0.75      0.84        36
    
        accuracy                           0.88        81
       macro avg       0.90      0.86      0.87        81
    weighted avg       0.89      0.88      0.87        81
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.80      0.98      0.88        45
               1       0.96      0.69      0.81        36
    
        accuracy                           0.85        81
       macro avg       0.88      0.84      0.84        81
    weighted avg       0.87      0.85      0.85        81
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.85      0.98      0.91        45
               1       0.97      0.78      0.86        36
    
        accuracy                           0.89        81
       macro avg       0.91      0.88      0.88        81
    weighted avg       0.90      0.89      0.89        81
    
    


```python
# 기타사고 모델 roc_curve
roc_curve_graph (X_test_ETC_OCRN_CNT, y_test_ETC_OCRN_CNT, 
                 Random_Forset_model_ETC_OCRN_CNT,
                 XGBoost_model_ETC_OCRN_CNT,
                 CatBoost_model_ETC_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_238_0.png)
    



```python
# 기타사고 모델 confusion
confusion_matrix_heat(y_test_ETC_OCRN_CNT, 
                      Random_Forset_model_y_pred_ETC_OCRN_CNT,
                      XGBoost_model_y_pred_ETC_OCRN_CNT,
                      CatBoost_model_y_pred_ETC_OCRN_CNT, 
                      "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_239_0.png)
    



```python
# 기타사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_ETC_OCRN_CNT, X_test_ETC_OCRN_CNT)
```


    
![png](output_240_0.png)
    


### BLTRM_OCRN_CNT


```python
# 질병외 상관관계 확인 
corr_check(data_BLTRM_OCRN_CNT)
```


    
![png](output_242_0.png)
    



```python
# 질병외 모델 점수
Classification_report_check (y_test_BLTRM_OCRN_CNT, 
                             Random_Forset_model_y_pred_BLTRM_OCRN_CNT,
                             XGBoost_model_y_pred_BLTRM_OCRN_CNT,
                             CatBoost_model_y_pred_BLTRM_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.74      0.78      0.76        18
               1       0.69      0.64      0.67        14
    
        accuracy                           0.72        32
       macro avg       0.71      0.71      0.71        32
    weighted avg       0.72      0.72      0.72        32
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.75      0.83      0.79        18
               1       0.75      0.64      0.69        14
    
        accuracy                           0.75        32
       macro avg       0.75      0.74      0.74        32
    weighted avg       0.75      0.75      0.75        32
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.75      0.83      0.79        18
               1       0.75      0.64      0.69        14
    
        accuracy                           0.75        32
       macro avg       0.75      0.74      0.74        32
    weighted avg       0.75      0.75      0.75        32
    
    


```python
# 질병외 모델 roc_curve
roc_curve_graph (X_test_BLTRM_OCRN_CNT, y_test_BLTRM_OCRN_CNT, 
                 Random_Forset_model_BLTRM_OCRN_CNT,
                 XGBoost_model_BLTRM_OCRN_CNT,
                 CatBoost_model_BLTRM_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_244_0.png)
    



```python
# 질병외 모델 confusion
confusion_matrix_heat(y_test_BLTRM_OCRN_CNT, 
                      Random_Forset_model_y_pred_BLTRM_OCRN_CNT,
                      XGBoost_model_y_pred_BLTRM_OCRN_CNT,
                      CatBoost_model_y_pred_BLTRM_OCRN_CNT, 
                      "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_245_0.png)
    



```python
# 질병외 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_BLTRM_OCRN_CNT, X_test_BLTRM_OCRN_CNT)
```


    
![png](output_246_0.png)
    


### ACDNT_INJ_OCRN_CNT


```python
# 사고부상 상관관계 확인 
corr_check(data_ACDNT_INJ_OCRN_CNT)
```


    
![png](output_248_0.png)
    



```python
# 사고부상 모델 점수
Classification_report_check (y_test_ACDNT_INJ_OCRN_CNT, 
                             Random_Forset_model_y_pred_ACDNT_INJ_OCRN_CNT,
                             XGBoost_model_y_pred_ACDNT_INJ_OCRN_CNT,
                             CatBoost_model_y_pred_ACDNT_INJ_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.88      0.87      0.87        52
               1       0.82      0.85      0.84        39
    
        accuracy                           0.86        91
       macro avg       0.85      0.86      0.85        91
    weighted avg       0.86      0.86      0.86        91
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.86      0.85      0.85        52
               1       0.80      0.82      0.81        39
    
        accuracy                           0.84        91
       macro avg       0.83      0.83      0.83        91
    weighted avg       0.84      0.84      0.84        91
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.87      0.87      0.87        52
               1       0.82      0.82      0.82        39
    
        accuracy                           0.85        91
       macro avg       0.84      0.84      0.84        91
    weighted avg       0.85      0.85      0.85        91
    
    


```python
# 사고부상 모델 roc_curve
roc_curve_graph (X_test_ACDNT_INJ_OCRN_CNT, y_test_ACDNT_INJ_OCRN_CNT, 
                 Random_Forset_model_ACDNT_INJ_OCRN_CNT,
                 XGBoost_model_ACDNT_INJ_OCRN_CNT,
                 CatBoost_model_ACDNT_INJ_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_250_0.png)
    



```python
# 사고부상 모델 confusion
confusion_matrix_heat(y_test_ACDNT_INJ_OCRN_CNT, 
                      XGBoost_model_y_pred_ACDNT_INJ_OCRN_CNT,
                      CatBoost_model_y_pred_ACDNT_INJ_OCRN_CNT,
                      GradientBoost_model_y_pred_ACDNT_INJ_OCRN_CNT, 
                      "XGBoost", "CatBoost", "GradientBoost")
```


    
![png](output_251_0.png)
    



```python
# 사고부상 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_ACDNT_INJ_OCRN_CNT, X_test_ACDNT_INJ_OCRN_CNT)
```


    
![png](output_252_0.png)
    


### EXCL_DISEASE_OCRN_CNT


```python
# 질병외발생 상관관계 확인 
corr_check(data_EXCL_DISEASE_OCRN_CNT)
```


    
![png](output_254_0.png)
    



```python
# 질병외발생 모델 점수
Classification_report_check (y_test_EXCL_DISEASE_OCRN_CNT, 
                             Random_Forset_model_y_pred_EXCL_DISEASE_OCRN_CNT,
                             XGBoost_model_y_pred_EXCL_DISEASE_OCRN_CNT,
                             CatBoost_model_y_pred_EXCL_DISEASE_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.91      0.87      0.89        98
               1       0.83      0.89      0.86        71
    
        accuracy                           0.88       169
       macro avg       0.87      0.88      0.87       169
    weighted avg       0.88      0.88      0.88       169
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.89      0.85      0.87        98
               1       0.80      0.86      0.83        71
    
        accuracy                           0.85       169
       macro avg       0.85      0.85      0.85       169
    weighted avg       0.85      0.85      0.85       169
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.89      0.91      0.90        98
               1       0.87      0.85      0.86        71
    
        accuracy                           0.88       169
       macro avg       0.88      0.88      0.88       169
    weighted avg       0.88      0.88      0.88       169
    
    


```python
# 질병외발생 모델 roc_curve
roc_curve_graph (X_test_EXCL_DISEASE_OCRN_CNT, y_test_EXCL_DISEASE_OCRN_CNT, 
                 Random_Forset_model_EXCL_DISEASE_OCRN_CNT,
                 XGBoost_model_EXCL_DISEASE_OCRN_CNT,
                 CatBoost_model_EXCL_DISEASE_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_256_0.png)
    



```python
# 질병외발생 모델 confusion
confusion_matrix_heat(y_test_EXCL_DISEASE_OCRN_CNT, 
                      Random_Forset_model_y_pred_EXCL_DISEASE_OCRN_CNT,
                      XGBoost_model_y_pred_EXCL_DISEASE_OCRN_CNT,
                      CatBoost_model_y_pred_EXCL_DISEASE_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_257_0.png)
    



```python
# 사고부상 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_EXCL_DISEASE_OCRN_CNT, X_test_EXCL_DISEASE_OCRN_CNT)
```


    
![png](output_258_0.png)
    


### VHC_ACDNT_OCRN_CNT


```python
# 탈것사고 상관관계 확인 
corr_check(data_VHC_ACDNT_OCRN_CNT)
```


    
![png](output_260_0.png)
    



```python
# 탈것사고 모델 점수
Classification_report_check (y_test_VHC_ACDNT_OCRN_CNT, 
                             Random_Forset_model_y_pred_VHC_ACDNT_OCRN_CNT,
                             XGBoost_model_y_pred_VHC_ACDNT_OCRN_CNT,
                             CatBoost_model_y_pred_VHC_ACDNT_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.80      0.89      0.84         9
               1       0.91      0.83      0.87        12
    
        accuracy                           0.86        21
       macro avg       0.85      0.86      0.86        21
    weighted avg       0.86      0.86      0.86        21
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.80      0.89      0.84         9
               1       0.91      0.83      0.87        12
    
        accuracy                           0.86        21
       macro avg       0.85      0.86      0.86        21
    weighted avg       0.86      0.86      0.86        21
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.89      0.89      0.89         9
               1       0.92      0.92      0.92        12
    
        accuracy                           0.90        21
       macro avg       0.90      0.90      0.90        21
    weighted avg       0.90      0.90      0.90        21
    
    


```python
# 탈것사고 모델 roc_curve
roc_curve_graph (X_test_VHC_ACDNT_OCRN_CNT, y_test_VHC_ACDNT_OCRN_CNT, 
                 Random_Forset_model_VHC_ACDNT_OCRN_CNT,
                 XGBoost_model_VHC_ACDNT_OCRN_CNT,
                 CatBoost_model_VHC_ACDNT_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_262_0.png)
    



```python
# 탈것사고 모델 confusion
confusion_matrix_heat(y_test_VHC_ACDNT_OCRN_CNT, 
                      Random_Forset_model_y_pred_VHC_ACDNT_OCRN_CNT,
                      XGBoost_model_y_pred_VHC_ACDNT_OCRN_CNT,
                      CatBoost_model_y_pred_VHC_ACDNT_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_263_0.png)
    



```python
# 탈것사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_VHC_ACDNT_OCRN_CNT, X_test_VHC_ACDNT_OCRN_CNT)
```


    
![png](output_264_0.png)
    


### HRFAF_OCRN_CNT


```python
# 낙상사고 상관관계 확인 
corr_check(data_HRFAF_OCRN_CNT)
```


    
![png](output_266_0.png)
    



```python
# 낙상사고 모델 점수
Classification_report_check (y_test_HRFAF_OCRN_CNT, 
                             Random_Forset_model_y_pred_HRFAF_OCRN_CNT,
                             XGBoost_model_y_pred_HRFAF_OCRN_CNT,
                             CatBoost_model_y_pred_HRFAF_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.88      0.91      0.90       410
               1       0.87      0.82      0.84       283
    
        accuracy                           0.88       693
       macro avg       0.87      0.87      0.87       693
    weighted avg       0.88      0.88      0.88       693
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.88      0.87      0.88       410
               1       0.82      0.83      0.82       283
    
        accuracy                           0.85       693
       macro avg       0.85      0.85      0.85       693
    weighted avg       0.85      0.85      0.85       693
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.88      0.91      0.89       410
               1       0.87      0.82      0.84       283
    
        accuracy                           0.87       693
       macro avg       0.87      0.86      0.87       693
    weighted avg       0.87      0.87      0.87       693
    
    


```python
# 낙상사고 모델 roc_curve
roc_curve_graph (X_test_HRFAF_OCRN_CNT, y_test_HRFAF_OCRN_CNT, 
                 Random_Forset_model_HRFAF_OCRN_CNT,
                 XGBoost_model_HRFAF_OCRN_CNT,
                 CatBoost_model_HRFAF_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_268_0.png)
    



```python
# 낙상사고 모델 confusion
confusion_matrix_heat(y_test_HRFAF_OCRN_CNT, 
                      Random_Forset_model_y_pred_HRFAF_OCRN_CNT,
                      XGBoost_model_y_pred_HRFAF_OCRN_CNT,
                      CatBoost_model_y_pred_HRFAF_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_269_0.png)
    



```python
# 낙상사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_HRFAF_OCRN_CNT, X_test_HRFAF_OCRN_CNT)
```


    
![png](output_270_0.png)
    


### DRKNSTAT_OCRN_CNT


```python
# 단순주취 상관관계 확인 
corr_check(data_DRKNSTAT_OCRN_CNT)
```


    
![png](output_272_0.png)
    



```python
# 단순주취 모델 점수
Classification_report_check (y_test_DRKNSTAT_OCRN_CNT, 
                             Random_Forset_model_y_pred_DRKNSTAT_OCRN_CNT,
                             XGBoost_model_y_pred_DRKNSTAT_OCRN_CNT,
                             CatBoost_model_y_pred_DRKNSTAT_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       1.00      0.88      0.94        25
               1       0.85      1.00      0.92        17
    
        accuracy                           0.93        42
       macro avg       0.93      0.94      0.93        42
    weighted avg       0.94      0.93      0.93        42
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.95      0.80      0.87        25
               1       0.76      0.94      0.84        17
    
        accuracy                           0.86        42
       macro avg       0.86      0.87      0.86        42
    weighted avg       0.88      0.86      0.86        42
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.95      0.84      0.89        25
               1       0.80      0.94      0.86        17
    
        accuracy                           0.88        42
       macro avg       0.88      0.89      0.88        42
    weighted avg       0.89      0.88      0.88        42
    
    


```python
# 단순주취 모델 roc_curve
roc_curve_graph (X_test_DRKNSTAT_OCRN_CNT, y_test_DRKNSTAT_OCRN_CNT, 
                 Random_Forset_model_DRKNSTAT_OCRN_CNT,
                 XGBoost_model_DRKNSTAT_OCRN_CNT,
                 CatBoost_model_DRKNSTAT_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_274_0.png)
    



```python
# 단순주취 모델 confusion
confusion_matrix_heat(y_test_DRKNSTAT_OCRN_CNT, 
                      Random_Forset_model_y_pred_DRKNSTAT_OCRN_CNT,
                      XGBoost_model_y_pred_DRKNSTAT_OCRN_CNT,
                      CatBoost_model_y_pred_DRKNSTAT_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_275_0.png)
    



```python
# 단순주취 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_DRKNSTAT_OCRN_CNT, X_test_DRKNSTAT_OCRN_CNT)
```


    
![png](output_276_0.png)
    


### ANML_INSCT_ACDNT_OCRN_CNT


```python
# 동물곤충사고 상관관계 확인 
corr_check(data_ANML_INSCT_ACDNT_OCRN_CNT)
```


    
![png](output_278_0.png)
    



```python
# 동물곤충사고 모델 점수
Classification_report_check (y_test_ANML_INSCT_ACDNT_OCRN_CNT, 
                             Random_Forset_model_y_pred_ANML_INSCT_ACDNT_OCRN_CNT,
                             XGBoost_model_y_pred_ANML_INSCT_ACDNT_OCRN_CNT,
                             CatBoost_model_y_pred_ANML_INSCT_ACDNT_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.72      0.72      0.72        25
               1       0.71      0.71      0.71        24
    
        accuracy                           0.71        49
       macro avg       0.71      0.71      0.71        49
    weighted avg       0.71      0.71      0.71        49
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.62      0.60      0.61        25
               1       0.60      0.62      0.61        24
    
        accuracy                           0.61        49
       macro avg       0.61      0.61      0.61        49
    weighted avg       0.61      0.61      0.61        49
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.81      0.84      0.82        25
               1       0.83      0.79      0.81        24
    
        accuracy                           0.82        49
       macro avg       0.82      0.82      0.82        49
    weighted avg       0.82      0.82      0.82        49
    
    


```python
# 동물곤충사고 모델 roc_curve
roc_curve_graph (X_test_ANML_INSCT_ACDNT_OCRN_CNT, y_test_ANML_INSCT_ACDNT_OCRN_CNT, 
                 Random_Forset_model_ANML_INSCT_ACDNT_OCRN_CNT,
                 XGBoost_model_ANML_INSCT_ACDNT_OCRN_CNT,
                 CatBoost_model_ANML_INSCT_ACDNT_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_280_0.png)
    



```python
# 동물곤충사고 모델 confusion
confusion_matrix_heat(y_test_ANML_INSCT_ACDNT_OCRN_CNT, 
                      Random_Forset_model_y_pred_ANML_INSCT_ACDNT_OCRN_CNT,
                      XGBoost_model_y_pred_ANML_INSCT_ACDNT_OCRN_CNT,
                      CatBoost_model_y_pred_ANML_INSCT_ACDNT_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_281_0.png)
    



```python
# 동물곤충사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_ANML_INSCT_ACDNT_OCRN_CNT, X_test_ANML_INSCT_ACDNT_OCRN_CNT)
```


    
![png](output_282_0.png)
    


### FLPS_ACDNT_OCRN_CNT


```python
# 동승자사고 상관관계 확인 
corr_check(data_FLPS_ACDNT_OCRN_CNT)
```


    
![png](output_284_0.png)
    



```python
# 동승자사고 모델 점수
Classification_report_check (y_test_FLPS_ACDNT_OCRN_CNT, 
                             Random_Forset_model_y_pred_FLPS_ACDNT_OCRN_CNT,
                             XGBoost_model_y_pred_FLPS_ACDNT_OCRN_CNT,
                             CatBoost_model_y_pred_FLPS_ACDNT_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.78      0.83      0.80        42
               1       0.79      0.72      0.75        36
    
        accuracy                           0.78        78
       macro avg       0.78      0.78      0.78        78
    weighted avg       0.78      0.78      0.78        78
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.77      0.79      0.78        42
               1       0.74      0.72      0.73        36
    
        accuracy                           0.76        78
       macro avg       0.76      0.75      0.75        78
    weighted avg       0.76      0.76      0.76        78
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.78      0.90      0.84        42
               1       0.86      0.69      0.77        36
    
        accuracy                           0.81        78
       macro avg       0.82      0.80      0.80        78
    weighted avg       0.82      0.81      0.80        78
    
    


```python
# 동승자사고 모델 roc_curve
roc_curve_graph (X_test_FLPS_ACDNT_OCRN_CNT, y_test_FLPS_ACDNT_OCRN_CNT, 
                 Random_Forset_model_FLPS_ACDNT_OCRN_CNT,
                 XGBoost_model_FLPS_ACDNT_OCRN_CNT,
                 CatBoost_model_FLPS_ACDNT_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_286_0.png)
    



```python
# 동승자사고 모델 confusion
confusion_matrix_heat(y_test_FLPS_ACDNT_OCRN_CNT, 
                      Random_Forset_model_y_pred_FLPS_ACDNT_OCRN_CNT,
                      XGBoost_model_y_pred_FLPS_ACDNT_OCRN_CNT,
                      CatBoost_model_y_pred_FLPS_ACDNT_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_287_0.png)
    



```python
# 동승자사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_FLPS_ACDNT_OCRN_CNT, X_test_FLPS_ACDNT_OCRN_CNT)
```


    
![png](output_288_0.png)
    


### PDST_ACDNT_OCRN_CNT


```python
# 보행자사고 상관관계 확인 
corr_check(data_PDST_ACDNT_OCRN_CNT)
```


    
![png](output_290_0.png)
    



```python
# 보행자사고 모델 점수
Classification_report_check (y_test_PDST_ACDNT_OCRN_CNT, 
                             Random_Forset_model_y_pred_PDST_ACDNT_OCRN_CNT,
                             XGBoost_model_y_pred_PDST_ACDNT_OCRN_CNT,
                             CatBoost_model_y_pred_PDST_ACDNT_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.89      0.85      0.87        67
               1       0.80      0.85      0.83        48
    
        accuracy                           0.85       115
       macro avg       0.85      0.85      0.85       115
    weighted avg       0.85      0.85      0.85       115
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.86      0.84      0.85        67
               1       0.78      0.81      0.80        48
    
        accuracy                           0.83       115
       macro avg       0.82      0.82      0.82       115
    weighted avg       0.83      0.83      0.83       115
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.88      0.87      0.87        67
               1       0.82      0.83      0.82        48
    
        accuracy                           0.85       115
       macro avg       0.85      0.85      0.85       115
    weighted avg       0.85      0.85      0.85       115
    
    


```python
# 보행자사고 모델 roc_curve
roc_curve_graph (X_test_PDST_ACDNT_OCRN_CNT, y_test_PDST_ACDNT_OCRN_CNT, 
                 Random_Forset_model_PDST_ACDNT_OCRN_CNT,
                 XGBoost_model_PDST_ACDNT_OCRN_CNT,
                 CatBoost_model_PDST_ACDNT_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_292_0.png)
    



```python
# 보행자사고 모델 confusion
confusion_matrix_heat(y_test_PDST_ACDNT_OCRN_CNT, 
                      Random_Forset_model_y_pred_PDST_ACDNT_OCRN_CNT,
                      XGBoost_model_y_pred_PDST_ACDNT_OCRN_CNT,
                      CatBoost_model_y_pred_PDST_ACDNT_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_293_0.png)
    



```python
# 보행자사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_PDST_ACDNT_OCRN_CNT, X_test_PDST_ACDNT_OCRN_CNT)
```


    
![png](output_294_0.png)
    


### LACRTWND_OCRN_CNT


```python
# 열상사고 상관관계 확인 
corr_check(data_LACRTWND_OCRN_CNT)
```


    
![png](output_296_0.png)
    



```python
# 열상사고 모델 점수
Classification_report_check (y_test_LACRTWND_OCRN_CNT, 
                             Random_Forset_model_y_pred_LACRTWND_OCRN_CNT,
                             XGBoost_model_y_pred_LACRTWND_OCRN_CNT,
                             CatBoost_model_y_pred_LACRTWND_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.91      0.90      0.90        78
               1       0.84      0.86      0.85        50
    
        accuracy                           0.88       128
       macro avg       0.88      0.88      0.88       128
    weighted avg       0.88      0.88      0.88       128
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.91      0.88      0.90        78
               1       0.83      0.86      0.84        50
    
        accuracy                           0.88       128
       macro avg       0.87      0.87      0.87       128
    weighted avg       0.88      0.88      0.88       128
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.95      0.92      0.94        78
               1       0.88      0.92      0.90        50
    
        accuracy                           0.92       128
       macro avg       0.92      0.92      0.92       128
    weighted avg       0.92      0.92      0.92       128
    
    


```python
# 열상사고 모델 roc_curve
roc_curve_graph (X_test_LACRTWND_OCRN_CNT, y_test_LACRTWND_OCRN_CNT, 
                 Random_Forset_model_LACRTWND_OCRN_CNT,
                 XGBoost_model_LACRTWND_OCRN_CNT,
                 CatBoost_model_LACRTWND_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_298_0.png)
    



```python
# 열상사고 모델 confusion
confusion_matrix_heat(y_test_LACRTWND_OCRN_CNT, 
                      Random_Forset_model_y_pred_LACRTWND_OCRN_CNT,
                      XGBoost_model_y_pred_LACRTWND_OCRN_CNT,
                      CatBoost_model_y_pred_LACRTWND_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_299_0.png)
    



```python
# 열상사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_LACRTWND_OCRN_CNT, X_test_LACRTWND_OCRN_CNT)
```


    
![png](output_300_0.png)
    


### MTRCYC_ACDNT_OCRN_CNT


```python
# 오토바이사고 상관관계 확인 
corr_check(data_MTRCYC_ACDNT_OCRN_CNT)
```


    
![png](output_302_0.png)
    



```python
# 오토바이사고 모델 점수
Classification_report_check (y_test_MTRCYC_ACDNT_OCRN_CNT, 
                             Random_Forset_model_y_pred_MTRCYC_ACDNT_OCRN_CNT,
                             XGBoost_model_y_pred_MTRCYC_ACDNT_OCRN_CNT,
                             CatBoost_model_y_pred_MTRCYC_ACDNT_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.92      0.87      0.89        77
               1       0.82      0.89      0.85        53
    
        accuracy                           0.88       130
       macro avg       0.87      0.88      0.87       130
    weighted avg       0.88      0.88      0.88       130
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.92      0.86      0.89        77
               1       0.81      0.89      0.85        53
    
        accuracy                           0.87       130
       macro avg       0.86      0.87      0.87       130
    weighted avg       0.87      0.87      0.87       130
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.91      0.88      0.89        77
               1       0.84      0.87      0.85        53
    
        accuracy                           0.88       130
       macro avg       0.87      0.88      0.87       130
    weighted avg       0.88      0.88      0.88       130
    
    


```python
# 오토바이사고 모델 roc_curve
roc_curve_graph (X_test_MTRCYC_ACDNT_OCRN_CNT, y_test_MTRCYC_ACDNT_OCRN_CNT, 
                 Random_Forset_model_MTRCYC_ACDNT_OCRN_CNT,
                 XGBoost_model_MTRCYC_ACDNT_OCRN_CNT,
                 CatBoost_model_MTRCYC_ACDNT_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_304_0.png)
    



```python
# 오토바이사고 모델 confusion
confusion_matrix_heat(y_test_MTRCYC_ACDNT_OCRN_CNT, 
                      Random_Forset_model_y_pred_MTRCYC_ACDNT_OCRN_CNT,
                      XGBoost_model_y_pred_MTRCYC_ACDNT_OCRN_CNT,
                      CatBoost_model_y_pred_MTRCYC_ACDNT_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_305_0.png)
    



```python
# 오토바이사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_MTRCYC_ACDNT_OCRN_CNT, X_test_MTRCYC_ACDNT_OCRN_CNT)
```


    
![png](output_306_0.png)
    


### DRV_ACDNT_OCRN_CNT


```python
# 운전자사고 상관관계 확인 
corr_check(data_DRV_ACDNT_OCRN_CNT)
```


    
![png](output_308_0.png)
    



```python
# 운전자사고 모델 점수
Classification_report_check (y_test_DRV_ACDNT_OCRN_CNT, 
                             Random_Forset_model_y_pred_DRV_ACDNT_OCRN_CNT,
                             XGBoost_model_y_pred_DRV_ACDNT_OCRN_CNT,
                             CatBoost_model_y_pred_DRV_ACDNT_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.70      0.78      0.74        91
               1       0.64      0.55      0.59        66
    
        accuracy                           0.68       157
       macro avg       0.67      0.66      0.66       157
    weighted avg       0.68      0.68      0.68       157
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.69      0.73      0.71        91
               1       0.59      0.55      0.57        66
    
        accuracy                           0.65       157
       macro avg       0.64      0.64      0.64       157
    weighted avg       0.65      0.65      0.65       157
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.69      0.79      0.73        91
               1       0.63      0.50      0.56        66
    
        accuracy                           0.67       157
       macro avg       0.66      0.65      0.65       157
    weighted avg       0.66      0.67      0.66       157
    
    


```python
# 운전자사고 모델 roc_curve
roc_curve_graph (X_test_DRV_ACDNT_OCRN_CNT, y_test_DRV_ACDNT_OCRN_CNT, 
                 Random_Forset_model_DRV_ACDNT_OCRN_CNT,
                 XGBoost_model_DRV_ACDNT_OCRN_CNT,
                 CatBoost_model_DRV_ACDNT_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_310_0.png)
    



```python
# 운전자사고 모델 confusion
confusion_matrix_heat(y_test_DRV_ACDNT_OCRN_CNT, 
                      Random_Forset_model_y_pred_DRV_ACDNT_OCRN_CNT,
                      XGBoost_model_y_pred_DRV_ACDNT_OCRN_CNT,
                      CatBoost_model_y_pred_DRV_ACDNT_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_311_0.png)
    



```python
# 운전자사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_DRV_ACDNT_OCRN_CNT, X_test_DRV_ACDNT_OCRN_CNT )
```


    
![png](output_312_0.png)
    


### BCYC_ACDNT_OCRN_CNT


```python
# 자전거사고 상관관계 확인 
corr_check(data_BCYC_ACDNT_OCRN_CNT)
```


    
![png](output_314_0.png)
    



```python
# 자전거사고 모델 점수
Classification_report_check (y_test_BCYC_ACDNT_OCRN_CNT, 
                             Random_Forset_model_y_pred_BCYC_ACDNT_OCRN_CNT,
                             XGBoost_model_y_pred_BCYC_ACDNT_OCRN_CNT,
                             CatBoost_model_y_pred_BCYC_ACDNT_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.89      0.82      0.85        38
               1       0.78      0.86      0.82        29
    
        accuracy                           0.84        67
       macro avg       0.83      0.84      0.83        67
    weighted avg       0.84      0.84      0.84        67
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.88      0.76      0.82        38
               1       0.74      0.86      0.79        29
    
        accuracy                           0.81        67
       macro avg       0.81      0.81      0.81        67
    weighted avg       0.82      0.81      0.81        67
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.91      0.84      0.88        38
               1       0.81      0.90      0.85        29
    
        accuracy                           0.87        67
       macro avg       0.86      0.87      0.86        67
    weighted avg       0.87      0.87      0.87        67
    
    


```python
# 자전거사고 모델 roc_curve
roc_curve_graph (X_test_BCYC_ACDNT_OCRN_CNT, y_test_BCYC_ACDNT_OCRN_CNT, 
                 Random_Forset_model_BCYC_ACDNT_OCRN_CNT,
                 XGBoost_model_BCYC_ACDNT_OCRN_CNT,
                 CatBoost_model_BCYC_ACDNT_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_316_0.png)
    



```python
# 자전거사고 모델 confusion
confusion_matrix_heat(y_test_BCYC_ACDNT_OCRN_CNT, 
                      Random_Forset_model_y_pred_BCYC_ACDNT_OCRN_CNT,
                      XGBoost_model_y_pred_BCYC_ACDNT_OCRN_CNT,
                      CatBoost_model_y_pred_BCYC_ACDNT_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_317_0.png)
    



```python
# 자전거사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_BCYC_ACDNT_OCRN_CNT, X_test_BCYC_ACDNT_OCRN_CNT )
```


    
![png](output_318_0.png)
    


### POSNG_OCRN_CNT


```python
# 중독사고 상관관계 확인 
corr_check(data_POSNG_OCRN_CNT)
```


    
![png](output_320_0.png)
    



```python
# 중독사고 모델 점수
Classification_report_check (y_test_POSNG_OCRN_CNT, 
                             Random_Forset_model_y_pred_POSNG_OCRN_CNT,
                             XGBoost_model_y_pred_POSNG_OCRN_CNT,
                             CatBoost_model_y_pred_POSNG_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.82      0.89      0.86        37
               1       0.82      0.72      0.77        25
    
        accuracy                           0.82        62
       macro avg       0.82      0.81      0.81        62
    weighted avg       0.82      0.82      0.82        62
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.80      0.86      0.83        37
               1       0.77      0.68      0.72        25
    
        accuracy                           0.79        62
       macro avg       0.79      0.77      0.78        62
    weighted avg       0.79      0.79      0.79        62
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.82      0.89      0.86        37
               1       0.82      0.72      0.77        25
    
        accuracy                           0.82        62
       macro avg       0.82      0.81      0.81        62
    weighted avg       0.82      0.82      0.82        62
    
    


```python
# 중독사고 모델 roc_curve
roc_curve_graph (X_test_POSNG_OCRN_CNT, y_test_POSNG_OCRN_CNT, 
                 Random_Forset_model_POSNG_OCRN_CNT,
                 XGBoost_model_POSNG_OCRN_CNT,
                 CatBoost_model_POSNG_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_322_0.png)
    



```python
# 중독사고 모델 confusion
confusion_matrix_heat(y_test_POSNG_OCRN_CNT, 
                      Random_Forset_model_y_pred_POSNG_OCRN_CNT,
                      XGBoost_model_y_pred_POSNG_OCRN_CNT,
                      CatBoost_model_y_pred_POSNG_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_323_0.png)
    



```python
# 중독사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_POSNG_OCRN_CNT, X_test_POSNG_OCRN_CNT )
```


    
![png](output_324_0.png)
    


### FALLING_OCRN_CNT


```python
# 추락사고 상관관계 확인 
corr_check(data_FALLING_OCRN_CNT)
```


    
![png](output_326_0.png)
    



```python
# 추락사고 모델 점수
Classification_report_check (y_test_FALLING_OCRN_CNT, 
                             Random_Forset_model_y_pred_FALLING_OCRN_CNT,
                             XGBoost_model_y_pred_FALLING_OCRN_CNT,
                             CatBoost_model_y_pred_FALLING_OCRN_CNT, 
                             "RandomForest", "XGBoost", "CatBoost")
```

                    RandomForest Classification Report
                  precision    recall  f1-score   support
    
               0       0.73      0.82      0.77        33
               1       0.67      0.55      0.60        22
    
        accuracy                           0.71        55
       macro avg       0.70      0.68      0.69        55
    weighted avg       0.70      0.71      0.70        55
    
    **************************************************
                      XGBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.75      0.73      0.74        33
               1       0.61      0.64      0.62        22
    
        accuracy                           0.69        55
       macro avg       0.68      0.68      0.68        55
    weighted avg       0.69      0.69      0.69        55
    
    **************************************************
                      CatBoost Classification Report
                  precision    recall  f1-score   support
    
               0       0.77      0.82      0.79        33
               1       0.70      0.64      0.67        22
    
        accuracy                           0.75        55
       macro avg       0.74      0.73      0.73        55
    weighted avg       0.74      0.75      0.74        55
    
    


```python
# 추락사고 모델 roc_curve
roc_curve_graph (X_test_FALLING_OCRN_CNT, y_test_FALLING_OCRN_CNT, 
                 Random_Forset_model_FALLING_OCRN_CNT,
                 XGBoost_model_FALLING_OCRN_CNT,
                 CatBoost_model_FALLING_OCRN_CNT, 
                 "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_328_0.png)
    



```python
# 추락사고 모델 confusion
confusion_matrix_heat(y_test_FALLING_OCRN_CNT, 
                      Random_Forset_model_y_pred_FALLING_OCRN_CNT,
                      XGBoost_model_y_pred_FALLING_OCRN_CNT,
                      CatBoost_model_y_pred_FALLING_OCRN_CNT, "RandomForest", "XGBoost", "CatBoost")
```


    
![png](output_329_0.png)
    



```python
# 추락사고 모델 feature importance
Feature_import ("RandomForest", Random_Forset_model_FALLING_OCRN_CNT, X_test_FALLING_OCRN_CNT )
```


    
![png](output_330_0.png)
    


## 최종 모델 테스트

### 매월 마지막날 구급출동이 발생한 격자 데이터프레임(Test dataset) 생성


```python
# 2021년 매월 말일 (년, 월, 일) 데이터
last_day = [20210131, 20210228, 20210331, 20210430, 20210531, 20210630, 20210731, 20210831, 20210930, 20211031, 20211130]
last = []
for day in last_day:
    date = pd.to_datetime(str(day), infer_datetime_format=True)
    last.append(date)
last_day_df = pd.DataFrame(last, columns=["OCRN_YMD"])
last_day_df["MONTH"] = last_day_df.loc[:,"OCRN_YMD"].dt.month
last_day_df["DAY"] = last_day_df.loc[:,"OCRN_YMD"].dt.day
last_day_df["WEEKDAY"] = last_day_df.loc[:,"OCRN_YMD"].dt.weekday
```


```python
## 격자 데이터 + 매월 말일 데이터 
last_day_input = []
Grid_ID = data_DF["GRID_ID"].unique()
for date in last_day:
    date = pd.to_datetime(str(date), infer_datetime_format=True)
    for grid in Grid_ID:
        last_day_input.append([grid, date])
grid_df = pd.DataFrame(data = last_day_input, columns=['GRID_ID', 'OCRN_YMD'])
last_day_df = pd.merge(grid_df, last_day_df, on = "OCRN_YMD", how="left")
last_day_df["OCRN_YMD"] = last_day_df["OCRN_YMD"].astype(str)
last_day_df

## SEASON_SE_NM 추가
last_day_df['SEASON_SE_NM'] = last_day_df['MONTH'].apply(season_check)

## HOLIDAY 추가 
last_day_df = pd.merge(last_day_df, hol_df, on=['OCRN_YMD'], how='left').fillna(0)
```


```python
# 기상정보 추가 (DAY_RAINQTY, DAY_MSNF, AVRG_WS, AVRG_HUMIDITY, AVRG_TMPRT)
last_day_df = pd.merge(last_day_df, climate_dict, on = ["OCRN_YMD"], how="left").fillna(0)
```


```python
# 각 사건 영향 변수 추가 (INDUSTRIAL_CNT, BAR_CNT, SENIOR_CENTER_CNT, RESTAURANT_CNT, BULID_PERMIT_CNT, ACCIDENT_AREA_CNT)

## INDUSTRIAL_CNT 추가
last_day_df = pd.merge(last_day_df, factory_df_counted, on=['GRID_ID'], how='left')

## BAR_CNT 추가
last_day_df = pd.merge(last_day_df, bar_df_counted, on=['GRID_ID'], how='left')

## SENIOR_CENTER_CNT 추가
last_day_df = pd.merge(last_day_df, senior_df_counted, on=['GRID_ID'], how='left')

## RESTAURANT_CNT 추가
last_day_df = pd.merge(last_day_df, restaurant_df_counted, on=['GRID_ID'], how='left')

## BULID_PERMIT_CNT 추가
last_day_df = pd.merge(last_day_df, build_df_counted, on=['GRID_ID'], how='left')

## ACCIDENT_AREA_CNT 추가
last_day_df = pd.merge(last_day_df, road_df_counted, on=['GRID_ID'], how='left')
```


```python
# 65세 미만 유동인구(ALL_POP), 65세 이상 유동인구(ELDER_POP) 컬럼 추가 

# 같은 격자, 같은 달, 같은 요일의 유동인구의 평균값 계산
pop_mean = final_DF.groupby(["GRID_ID","MONTH","WEEKDAY"], as_index=False)["ALL_POP", "ELDER_POP"].mean()

# 같은 조건의 값 추가
last_grid = last_day_df["GRID_ID"].unique()
all_pop_df = []
for m in range(1, 12):
    w = last_day_df[last_day_df["MONTH"]==m]["WEEKDAY"].unique()[0]
    for g in last_grid:
       all_pop_df.append(list(pop_mean[(pop_mean["GRID_ID"]==g) & (pop_mean["MONTH"]==m) & (pop_mean["WEEKDAY"]==w)].iloc[0, :]))
all_pop_df = pd.DataFrame(all_pop_df, columns = pop_mean.columns)
print(all_pop_df)
all_pop_add_df = all_pop_df.drop(columns=["GRID_ID", "MONTH", "WEEKDAY"], axis=1)

# 65세 미만 유동인구(ALL_POP), 65세 이상 유동인구(ELDER_POP) 추가
last_day_df = pd.concat([last_day_df, all_pop_add_df], axis=1).reset_index(drop=True)
```

           GRID_ID  MONTH  WEEKDAY   ALL_POP  ELDER_POP
    0     378509.0    1.0      6.0  0.000000   0.000000
    1     378511.0    1.0      6.0  0.105555   0.063118
    2     378512.0    1.0      6.0  0.023568   0.015320
    3     378513.0    1.0      6.0  0.379208   0.248920
    4     378514.0    1.0      6.0  0.056562   0.011914
    ...        ...    ...      ...       ...        ...
    5220  417516.0   11.0      1.0  0.000000   0.000000
    5221  417517.0   11.0      1.0  0.126917   0.091746
    5222  417518.0   11.0      1.0  0.180926   0.137904
    5223  417521.0   11.0      1.0  0.000000   0.000000
    5224  418518.0   11.0      1.0  0.000000   0.000000
    
    [5225 rows x 5 columns]
    


```python
# Feature Skew
lam = 0.01
last_skw_features = ['AVRG_TMPRT', 'DAY_RAINQTY', 'DAY_MSNF', 'AVRG_WS', 'AVRG_HUMIDITY', 'INDUSTRIAL_CNT', 
                     'BAR_CNT', 'SENIOR_CENTER_CNT', 'RESTAURANT_CNT', 'BULID_PERMIT_CNT', 'ACCIDENT_AREA_CNT']
skew_list = skewed_check(last_day_df, last_skw_features)

# Box-Cox Transform
for col in skew_list:
  last_day_df[col] = boxcox1p(last_day_df[col], lam)
```


```python
# Feature Scaling
last_scaling_col = ['AVRG_TMPRT', 'DAY_RAINQTY', 'DAY_MSNF', 'AVRG_WS', 'AVRG_HUMIDITY', 'INDUSTRIAL_CNT', 'BAR_CNT', 
                    'SENIOR_CENTER_CNT', 'RESTAURANT_CNT', 'BULID_PERMIT_CNT', 'ACCIDENT_AREA_CNT']
for col in last_scaling_col:
    last_day_df[col] = Scaler.transform(last_day_df[col].values.reshape(-1,1))
```


```python
# col drop
last_day_df.drop(columns=["OCRN_YMD"], axis=1, inplace=True)
```


```python
# 테스트 데이터와 학습 데이터 순서 정렬
test_col = ['MONTH', 'DAY', 'WEEKDAY', 'HOLIDAY', 'SEASON_SE_NM', 
       'AVRG_TMPRT', 'DAY_RAINQTY', 'DAY_MSNF', 'AVRG_WS', 'AVRG_HUMIDITY',
       'INDUSTRIAL_CNT', 'BAR_CNT', 'SENIOR_CENTER_CNT', 'RESTAURANT_CNT',
       'BULID_PERMIT_CNT', 'ACCIDENT_AREA_CNT', 'ALL_POP', 'ELDER_POP','GRID_ID']
 
last_day_df = last_day_df[test_col]
```


```python
# last check
last_day_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5225 entries, 0 to 5224
    Data columns (total 19 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   MONTH              5225 non-null   int64  
     1   DAY                5225 non-null   int64  
     2   WEEKDAY            5225 non-null   int64  
     3   HOLIDAY            5225 non-null   float64
     4   SEASON_SE_NM       5225 non-null   int64  
     5   AVRG_TMPRT         5225 non-null   float64
     6   DAY_RAINQTY        5225 non-null   float64
     7   DAY_MSNF           5225 non-null   float64
     8   AVRG_WS            5225 non-null   float64
     9   AVRG_HUMIDITY      5225 non-null   float64
     10  INDUSTRIAL_CNT     5225 non-null   float64
     11  BAR_CNT            5225 non-null   float64
     12  SENIOR_CENTER_CNT  5225 non-null   float64
     13  RESTAURANT_CNT     5225 non-null   float64
     14  BULID_PERMIT_CNT   5225 non-null   float64
     15  ACCIDENT_AREA_CNT  5225 non-null   float64
     16  ALL_POP            5225 non-null   float64
     17  ELDER_POP          5225 non-null   float64
     18  GRID_ID            5225 non-null   int64  
    dtypes: float64(14), int64(5)
    memory usage: 775.7 KB
    


```python
# save csv
last_day_df.to_csv("last_day_df.csv")
```

### 모델 예측 결과


```python
# 각 사고별 최적의 모델 선정 
all_acc_model = { "기계사고" : [Random_Forset_model_MCHN_ACDNT_OCRN_CNT, XGBoost_model_MCHN_ACDNT_OCRN_CNT, CatBoost_model_MCHN_ACDNT_OCRN_CNT],
                   "기타사고" : [Random_Forset_model_ETC_OCRN_CNT, XGBoost_model_ETC_OCRN_CNT, CatBoost_model_ETC_OCRN_CNT],
                  "둔상" : [Random_Forset_model_BLTRM_OCRN_CNT, XGBoost_model_BLTRM_OCRN_CNT, CatBoost_model_BLTRM_OCRN_CNT],
                  "사고부상" : [Random_Forset_model_ACDNT_INJ_OCRN_CNT, XGBoost_model_ACDNT_INJ_OCRN_CNT, CatBoost_model_ACDNT_INJ_OCRN_CNT],
                  "질병외" : [Random_Forset_model_EXCL_DISEASE_OCRN_CNT, XGBoost_model_EXCL_DISEASE_OCRN_CNT, CatBoost_model_EXCL_DISEASE_OCRN_CNT],
                  "탈것사고" : [Random_Forset_model_VHC_ACDNT_OCRN_CNT, XGBoost_model_VHC_ACDNT_OCRN_CNT, CatBoost_model_VHC_ACDNT_OCRN_CNT],
                  "낙상" : [Random_Forset_model_HRFAF_OCRN_CNT, XGBoost_model_HRFAF_OCRN_CNT, CatBoost_model_HRFAF_OCRN_CNT],
                  "단순주취" : [Random_Forset_model_DRKNSTAT_OCRN_CNT, XGBoost_model_DRKNSTAT_OCRN_CNT, CatBoost_model_DRKNSTAT_OCRN_CNT],
                  "동물곤충사고" : [Random_Forset_model_ANML_INSCT_ACDNT_OCRN_CNT, XGBoost_model_ANML_INSCT_ACDNT_OCRN_CNT, CatBoost_model_ANML_INSCT_ACDNT_OCRN_CNT],
                  "동승자사고" : [Random_Forset_model_FLPS_ACDNT_OCRN_CNT, XGBoost_model_FLPS_ACDNT_OCRN_CNT, CatBoost_model_FLPS_ACDNT_OCRN_CNT],
                  "보행자사고" : [Random_Forset_model_PDST_ACDNT_OCRN_CNT, XGBoost_model_PDST_ACDNT_OCRN_CNT, CatBoost_model_PDST_ACDNT_OCRN_CNT],
                  "열상" : [Random_Forset_model_LACRTWND_OCRN_CNT, XGBoost_model_LACRTWND_OCRN_CNT, CatBoost_model_LACRTWND_OCRN_CNT],
                  "오토바이사고" : [Random_Forset_model_MTRCYC_ACDNT_OCRN_CNT, XGBoost_model_MTRCYC_ACDNT_OCRN_CNT, CatBoost_model_MTRCYC_ACDNT_OCRN_CNT],
                  "운전사사고" : [Random_Forset_model_DRV_ACDNT_OCRN_CNT, XGBoost_model_DRV_ACDNT_OCRN_CNT, CatBoost_model_DRV_ACDNT_OCRN_CNT],
                  "자전거사고" : [Random_Forset_model_BCYC_ACDNT_OCRN_CNT, XGBoost_model_BCYC_ACDNT_OCRN_CNT, CatBoost_model_BCYC_ACDNT_OCRN_CNT],
                  "중독사고" : [Random_Forset_model_POSNG_OCRN_CNT, XGBoost_model_POSNG_OCRN_CNT, CatBoost_model_POSNG_OCRN_CNT],
                  "추락사고" : [Random_Forset_model_FALLING_OCRN_CNT , XGBoost_model_FALLING_OCRN_CNT , CatBoost_model_FALLING_OCRN_CNT ],
                  }
```


```python
# 사고별 최적의 모델 분포 결과  
for data, model in all_acc_model.items():
    # 각 모델 predict 
    y_pred_1 = model[0].predict(last_day_df)
    y_pred_2 = model[1].predict(last_day_df)
    y_pred_3 = model[2].predict(last_day_df)

    # 사고별 최적의 모델 분포 
    print (f"       {data} 분포 예측 결과        ")
    unique_1, cnt_1 = np.unique(y_pred_1, return_counts = True)
    uni_cnt_dict_1 = dict(zip(unique_1, cnt_1))
    print (f"Random Forset 모델 분포 결과 : {uni_cnt_dict_1}")

    unique_2, cnt_2 = np.unique(y_pred_2, return_counts = True)
    uni_cnt_dict_2 = dict(zip(unique_2, cnt_2))
    print (f"XGBoost 모델 분포 결과 : {uni_cnt_dict_2}")

    unique_3, cnt_3 = np.unique(y_pred_3, return_counts = True)
    uni_cnt_dict_3 = dict(zip(unique_3, cnt_3))
    print (f"CatBoost 모델 분포 결과 : {uni_cnt_dict_3}")

    print("*"*45)
```

           기계사고 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4538, 1: 687}
    XGBoost 모델 분포 결과 : {0: 4403, 1: 822}
    CatBoost 모델 분포 결과 : {0: 4705, 1: 520}
    *********************************************
           기타사고 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 3451, 1: 1774}
    XGBoost 모델 분포 결과 : {0: 3014, 1: 2211}
    CatBoost 모델 분포 결과 : {0: 4515, 1: 710}
    *********************************************
           둔상 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4551, 1: 674}
    XGBoost 모델 분포 결과 : {0: 4080, 1: 1145}
    CatBoost 모델 분포 결과 : {0: 4705, 1: 520}
    *********************************************
           사고부상 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4547, 1: 678}
    XGBoost 모델 분포 결과 : {0: 4413, 1: 812}
    CatBoost 모델 분포 결과 : {0: 4540, 1: 685}
    *********************************************
           질병외 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4896, 1: 329}
    XGBoost 모델 분포 결과 : {0: 4813, 1: 412}
    CatBoost 모델 분포 결과 : {0: 4865, 1: 360}
    *********************************************
           탈것사고 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4351, 1: 874}
    XGBoost 모델 분포 결과 : {0: 4533, 1: 692}
    CatBoost 모델 분포 결과 : {0: 4351, 1: 874}
    *********************************************
           낙상 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4983, 1: 242}
    XGBoost 모델 분포 결과 : {0: 4841, 1: 384}
    CatBoost 모델 분포 결과 : {0: 4892, 1: 333}
    *********************************************
           단순주취 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 5024, 1: 201}
    XGBoost 모델 분포 결과 : {0: 5096, 1: 129}
    CatBoost 모델 분포 결과 : {0: 5018, 1: 207}
    *********************************************
           동물곤충사고 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4585, 1: 640}
    XGBoost 모델 분포 결과 : {0: 4527, 1: 698}
    CatBoost 모델 분포 결과 : {0: 4277, 1: 948}
    *********************************************
           동승자사고 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 3721, 1: 1504}
    XGBoost 모델 분포 결과 : {0: 4384, 1: 841}
    CatBoost 모델 분포 결과 : {0: 4978, 1: 247}
    *********************************************
           보행자사고 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4715, 1: 510}
    XGBoost 모델 분포 결과 : {0: 4756, 1: 469}
    CatBoost 모델 분포 결과 : {0: 4890, 1: 335}
    *********************************************
           열상 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4953, 1: 272}
    XGBoost 모델 분포 결과 : {0: 4993, 1: 232}
    CatBoost 모델 분포 결과 : {0: 4924, 1: 301}
    *********************************************
           오토바이사고 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4847, 1: 378}
    XGBoost 모델 분포 결과 : {0: 4668, 1: 557}
    CatBoost 모델 분포 결과 : {0: 4831, 1: 394}
    *********************************************
           운전사사고 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4848, 1: 377}
    XGBoost 모델 분포 결과 : {0: 4658, 1: 567}
    CatBoost 모델 분포 결과 : {0: 4912, 1: 313}
    *********************************************
           자전거사고 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4424, 1: 801}
    XGBoost 모델 분포 결과 : {0: 4101, 1: 1124}
    CatBoost 모델 분포 결과 : {0: 4695, 1: 530}
    *********************************************
           중독사고 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 5221, 1: 4}
    XGBoost 모델 분포 결과 : {0: 5170, 1: 55}
    CatBoost 모델 분포 결과 : {0: 5219, 1: 6}
    *********************************************
           추락사고 분포 예측 결과        
    Random Forset 모델 분포 결과 : {0: 4873, 1: 352}
    XGBoost 모델 분포 결과 : {0: 4605, 1: 620}
    CatBoost 모델 분포 결과 : {0: 5019, 1: 206}
    *********************************************
    


```python
# 각 사고별 3개 모델 답안지 추출
def final_grid_result (name, use_model_1, use_model_2, use_model_3, data):
    
    # 원형 모델 데이터프레임 
    result_DF = data.copy()

    # 격자 추출 
    Grid_IDS = data["GRID_ID"].unique()

    # 1번 모델 정답 격자 데이터프레임 생성
    result_1 = pd.DataFrame(use_model_1.predict(result_DF), columns = ["label"])
    result_DF_1 = pd.concat([result_DF, result_1], axis=1)

    # 2번 모델 정답 격자 데이터프레임 생성
    result_2 = pd.DataFrame(use_model_2.predict(result_DF), columns = ["label"])
    result_DF_2 = pd.concat([result_DF, result_2], axis=1)

    # 3번 모델 정답 격자 데이터프레임 생성
    result_3 = pd.DataFrame(use_model_3.predict(result_DF), columns = ["label"])
    result_DF_3 = pd.concat([result_DF, result_3], axis=1)

    print(f"{name} 격자 추출")
    print("*"*30)

    # 말일에 label이 1(사고가 발생한)인 격자 추출
    for mon in range(1, 12):

        # 매월 말일 + 사고가 발생한 격자 추출
        month_grid_1 = result_DF_1.loc[(result_DF_1["MONTH"] == mon) & (result_DF_1["label"] == 1)]["GRID_ID"]
        month_grid_2 = result_DF_2.loc[(result_DF_2["MONTH"] == mon) & (result_DF_2["label"] == 1)]["GRID_ID"] 
        month_grid_3 = result_DF_3.loc[(result_DF_3["MONTH"] == mon) & (result_DF_3["label"] == 1)]["GRID_ID"] 

        # 각 월말 날짜 추출
        month_day = result_DF.loc[(result_DF["MONTH"] == mon)]

        # 상위 2개의 격자

        # 2개의 모델에서 전부 나오는 격자 추출
        result_index = pd.concat([month_grid_1, month_grid_2, month_grid_3], axis=1, join="inner", ignore_index=True).drop(columns=[1, 2])

        # 정답 격자 
        print([month_day["MONTH"].iat[0], month_day["DAY"].iat[0]])
        print("전체모델 발생 격자")        
        print(result_index.iloc[:, 0].values)
        print()
```


```python
# 분포결과 및 점수를 확인하여 상위 1개를 이용한 모델 답안지 추출
def final_grid_result_ex2 (name, use_model_1, data):
    
    # 원형 모델 데이터프레임 
    result_DF = data.copy()

    # 격자 추출 
    Grid_IDS = data["GRID_ID"].unique()

    print(f"{name} 격자 추출")
    print("*"*30)

    # 매월 말일에 label이 1(사건이 발생한)인 격자 추출
    for mon in range(1, 12):

        # 매월 말일 + 사건이 발생한 격자 추출
        predict_df=result_DF.loc[result_DF['MONTH'] == mon, :]

        # 매월 발생하는 사건중 상위 5개 격자만 추출 (예측확률 기준)
        pred_proba = use_model_1.predict_proba(predict_df) 
        pred_proba_df = pd.DataFrame(pred_proba).sort_values(by = 1, ascending=False)
        result_index = pred_proba_df[pred_proba_df[1] >= 0.5][:5].index 

        # 각 월말 날짜 추출
        month_day = result_DF.loc[(result_DF["MONTH"] == mon)]

        # 정답 격자 
        print([month_day["MONTH"].iat[0], month_day["DAY"].iat[0]])
        print(Grid_IDS[result_index])
        print() 
        
```

> 기계사고


```python
final_grid_result("기계사고", Random_Forset_model_MCHN_ACDNT_OCRN_CNT, 
                               XGBoost_model_MCHN_ACDNT_OCRN_CNT,
                               CatBoost_model_MCHN_ACDNT_OCRN_CNT, last_day_df)
```

    기계사고 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    []
    
    [2, 28]
    전체모델 발생 격자
    []
    
    [3, 31]
    전체모델 발생 격자
    [383521 383522 384522 384523 384525 385524 385525 386525 392519 393522
     393525 393526 393527 393528 394524 394525 394526 394527 394528 394529
     394530 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 396529 396531 396532 396534 397523 397524 397526
     398522 398524 398526]
    
    [4, 30]
    전체모델 발생 격자
    [383521 383522 384522 384523 384525 385524 385525 386525 392519 393522
     393525 393526 393527 393528 394524 394525 394526 394527 394528 394529
     394530 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 396529 396531 396532 396534 397523 397524 397526
     398522 398524 398526]
    
    [5, 31]
    전체모델 발생 격자
    [382521 382522 383521 383522 384522 384523 384524 384525 385523 385524
     385525 386525 386526 387529 387530 388529 388530 388531 389507 389530
     389531 390528 390532 391519 391528 392519 392520 392521 392522 392523
     392526 393521 393522 393523 393525 393526 393527 393528 394524 394525
     394526 394527 394528 394529 394530 395523 395524 395525 395526 395527
     395528 395529 395530 395531 395532 395533 396524 396525 396526 396527
     396528 396529 396531 396532 396533 396534 397523 397524 397526 398522
     398524 398526]
    
    [6, 30]
    전체모델 발생 격자
    [382518 382521 382522 383521 383522 384522 384523 384524 384525 385523
     385524 385525 386525 386526 387529 387530 388529 388530 388531 389507
     389530 389531 390532 391519 391528 392519 392520 392521 392522 392523
     392526 393521 393522 393523 393525 393526 393527 393528 394524 394525
     394526 394527 394528 394529 394530 395523 395524 395525 395526 395527
     395528 395529 395530 395531 395532 395533 396524 396525 396526 396527
     396528 396529 396531 396532 396534 397523 397524 397525 397526 397531
     398522 398524 398526 399525]
    
    [7, 31]
    전체모델 발생 격자
    [382521 384523 385524 385525 392519 393522 393525 393526 393527 393528
     394524 394525 394526 394527 394528 394529 394530 395524 395525 395526
     395527 395528 395530 395531 396524 396525 396526 396527 396528 396529
     396531 396532 396534 397523 397524 397526 398524 398526]
    
    [8, 31]
    전체모델 발생 격자
    [382521 382522 383521 383522 383523 383536 384522 384523 384524 384525
     385523 385524 385525 385529 386525 386526 387529 387530 388529 388530
     388531 389507 389529 389530 389531 390528 390532 391519 391528 392519
     392520 392521 392522 392523 392524 392526 392527 392528 392529 393521
     393522 393523 393524 393525 393526 393527 393528 394524 394525 394526
     394527 394528 394529 394530 394532 395523 395524 395525 395526 395527
     395528 395529 395530 395531 395532 395533 395534 396523 396524 396525
     396526 396527 396528 396529 396530 396531 396532 396533 396534 397523
     397524 397525 397526 397531 398522 398524 398525 398526 399524 399525
     399526 400526]
    
    [9, 30]
    전체모델 발생 격자
    [382521 383521 383522 383536 384522 384523 384524 384525 385523 385524
     385525 386525 386526 388530 388531 389530 389531 392519 393522 393525
     393526 393527 393528 394524 394525 394526 394527 394528 394529 394530
     395524 395525 395526 395527 395528 395530 395531 395533 396524 396525
     396526 396527 396528 396529 396531 396532 396534 397523 397524 397526
     398522 398524 398526 399525]
    
    [10, 31]
    전체모델 발생 격자
    [382521 384523 385524 385525 393522 393525 393526 393527 394525 394526
     394527 394528 394529 394530 395524 395525 395526 395527 395528 395530
     395531 396524 396525 396526 396527 396528 396529 396531 396532 396534
     397524 397526 398522 398524 398526]
    
    [11, 30]
    전체모델 발생 격자
    [382521 383521 383522 384523 384525 385524 385525 386525 392519 393522
     393525 393526 393527 393528 394524 394525 394526 394527 394528 394529
     394530 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 396529 396531 396532 396534 397523 397524 397526
     398522 398524 398526]
    
    


```python
final_grid_result_ex2 ("기계사고", Random_Forset_model_MCHN_ACDNT_OCRN_CNT, last_day_df)
```

    기계사고 격자 추출
    ******************************
    [1, 31]
    []
    
    [2, 28]
    [385525]
    
    [3, 31]
    [395527 396524 394527 396526 393527]
    
    [4, 30]
    [395527 394527 396524 396526 393527]
    
    [5, 31]
    [395527 396524 396526 394527 393527]
    
    [6, 30]
    [394527 395527 396526 396524 393527]
    
    [7, 31]
    [395527 394527 396524 396526 393527]
    
    [8, 31]
    [395527 396524 394527 396526 393527]
    
    [9, 30]
    [395527 394527 396524 396526 393527]
    
    [10, 31]
    [385525 395527 396524 394529 393527]
    
    [11, 30]
    [395527 396524 394527 396526 393527]
    
    

> 기타사고


```python
final_grid_result("기타사고", Random_Forset_model_MCHN_ACDNT_OCRN_CNT, 
                               XGBoost_model_MCHN_ACDNT_OCRN_CNT,
                               CatBoost_model_MCHN_ACDNT_OCRN_CNT, last_day_df)
```

    기타사고 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    []
    
    [2, 28]
    전체모델 발생 격자
    []
    
    [3, 31]
    전체모델 발생 격자
    [383521 383522 384522 384523 384525 385524 385525 386525 392519 393522
     393525 393526 393527 393528 394524 394525 394526 394527 394528 394529
     394530 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 396529 396531 396532 396534 397523 397524 397526
     398522 398524 398526]
    
    [4, 30]
    전체모델 발생 격자
    [383521 383522 384522 384523 384525 385524 385525 386525 392519 393522
     393525 393526 393527 393528 394524 394525 394526 394527 394528 394529
     394530 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 396529 396531 396532 396534 397523 397524 397526
     398522 398524 398526]
    
    [5, 31]
    전체모델 발생 격자
    [382521 382522 383521 383522 384522 384523 384524 384525 385523 385524
     385525 386525 386526 387529 387530 388529 388530 388531 389507 389530
     389531 390528 390532 391519 391528 392519 392520 392521 392522 392523
     392526 393521 393522 393523 393525 393526 393527 393528 394524 394525
     394526 394527 394528 394529 394530 395523 395524 395525 395526 395527
     395528 395529 395530 395531 395532 395533 396524 396525 396526 396527
     396528 396529 396531 396532 396533 396534 397523 397524 397526 398522
     398524 398526]
    
    [6, 30]
    전체모델 발생 격자
    [382518 382521 382522 383521 383522 384522 384523 384524 384525 385523
     385524 385525 386525 386526 387529 387530 388529 388530 388531 389507
     389530 389531 390532 391519 391528 392519 392520 392521 392522 392523
     392526 393521 393522 393523 393525 393526 393527 393528 394524 394525
     394526 394527 394528 394529 394530 395523 395524 395525 395526 395527
     395528 395529 395530 395531 395532 395533 396524 396525 396526 396527
     396528 396529 396531 396532 396534 397523 397524 397525 397526 397531
     398522 398524 398526 399525]
    
    [7, 31]
    전체모델 발생 격자
    [382521 384523 385524 385525 392519 393522 393525 393526 393527 393528
     394524 394525 394526 394527 394528 394529 394530 395524 395525 395526
     395527 395528 395530 395531 396524 396525 396526 396527 396528 396529
     396531 396532 396534 397523 397524 397526 398524 398526]
    
    [8, 31]
    전체모델 발생 격자
    [382521 382522 383521 383522 383523 383536 384522 384523 384524 384525
     385523 385524 385525 385529 386525 386526 387529 387530 388529 388530
     388531 389507 389529 389530 389531 390528 390532 391519 391528 392519
     392520 392521 392522 392523 392524 392526 392527 392528 392529 393521
     393522 393523 393524 393525 393526 393527 393528 394524 394525 394526
     394527 394528 394529 394530 394532 395523 395524 395525 395526 395527
     395528 395529 395530 395531 395532 395533 395534 396523 396524 396525
     396526 396527 396528 396529 396530 396531 396532 396533 396534 397523
     397524 397525 397526 397531 398522 398524 398525 398526 399524 399525
     399526 400526]
    
    [9, 30]
    전체모델 발생 격자
    [382521 383521 383522 383536 384522 384523 384524 384525 385523 385524
     385525 386525 386526 388530 388531 389530 389531 392519 393522 393525
     393526 393527 393528 394524 394525 394526 394527 394528 394529 394530
     395524 395525 395526 395527 395528 395530 395531 395533 396524 396525
     396526 396527 396528 396529 396531 396532 396534 397523 397524 397526
     398522 398524 398526 399525]
    
    [10, 31]
    전체모델 발생 격자
    [382521 384523 385524 385525 393522 393525 393526 393527 394525 394526
     394527 394528 394529 394530 395524 395525 395526 395527 395528 395530
     395531 396524 396525 396526 396527 396528 396529 396531 396532 396534
     397524 397526 398522 398524 398526]
    
    [11, 30]
    전체모델 발생 격자
    [382521 383521 383522 384523 384525 385524 385525 386525 392519 393522
     393525 393526 393527 393528 394524 394525 394526 394527 394528 394529
     394530 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 396529 396531 396532 396534 397523 397524 397526
     398522 398524 398526]
    
    


```python
final_grid_result_ex2 ("기타사고", Random_Forset_model_MCHN_ACDNT_OCRN_CNT, last_day_df)
```

    기타사고 격자 추출
    ******************************
    [1, 31]
    []
    
    [2, 28]
    [385525]
    
    [3, 31]
    [395527 396524 394527 396526 393527]
    
    [4, 30]
    [395527 394527 396524 396526 393527]
    
    [5, 31]
    [395527 396524 396526 394527 393527]
    
    [6, 30]
    [394527 395527 396526 396524 393527]
    
    [7, 31]
    [395527 394527 396524 396526 393527]
    
    [8, 31]
    [395527 396524 394527 396526 393527]
    
    [9, 30]
    [395527 394527 396524 396526 393527]
    
    [10, 31]
    [385525 395527 396524 394529 393527]
    
    [11, 30]
    [395527 396524 394527 396526 393527]
    
    

> 둔상


```python
final_grid_result("둔상", Random_Forset_model_BLTRM_OCRN_CNT, 
                           XGBoost_model_BLTRM_OCRN_CNT,
                           CatBoost_model_BLTRM_OCRN_CNT, last_day_df)
```

    둔상 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    [384523 385524 388530 389530 392519 393522 393525 393526 393527 393528
     394524 394525 394526 394527 394528 394529 394530 395524 395525 395526
     395527 395528 395530 395531 396523 396524 396525 396526 396527 396528
     396531 396533 396534 397523 397524 397526 398524 398525 398526 399525
     400526]
    
    [2, 28]
    전체모델 발생 격자
    [384523 385524 388530 389530 392519 393522 393525 393526 393527 393528
     394524 394525 394526 394527 394528 394529 394530 395524 395525 395526
     395527 395528 395530 395531 396523 396524 396525 396526 396527 396528
     396531 396533 396534 397523 397524 397526 398524 398525 398526 399524
     399525 400526]
    
    [3, 31]
    전체모델 발생 격자
    [384523 385524 386525 388530 389530 392519 393521 393522 393525 393526
     393527 393528 394524 394525 394526 394527 394528 394529 394530 395524
     395525 395526 395527 395528 395530 395531 396523 396524 396525 396526
     396527 396528 396531 396533 396534 397523 397524 397525 397526 398524
     398525 398526 399524 399525 400525 400526]
    
    [4, 30]
    전체모델 발생 격자
    [384523 385524 386525 388530 388531 389530 392519 393521 393522 393525
     393526 393527 393528 394524 394525 394526 394527 394528 394529 394530
     395524 395525 395526 395527 395528 395530 395531 396523 396524 396525
     396526 396527 396528 396529 396531 396533 396534 397523 397524 397525
     397526 398524 398525 398526 399524 399525 399526 400525 400526]
    
    [5, 31]
    전체모델 발생 격자
    [384523 385524 386525 388530 388531 389530 392519 393521 393522 393525
     393526 393527 393528 394524 394525 394526 394527 394528 394529 394530
     395524 395525 395526 395527 395528 395530 395531 396523 396524 396525
     396526 396527 396528 396531 396533 396534 397523 397524 397525 397526
     398524 398525 398526 399524 399525 400525 400526]
    
    [6, 30]
    전체모델 발생 격자
    [384523 385524 386525 388530 388531 389530 392519 393521 393522 393525
     393526 393527 393528 394524 394525 394526 394527 394528 394529 394530
     395524 395525 395526 395527 395528 395530 395531 396523 396524 396525
     396526 396527 396528 396530 396531 396533 396534 397523 397524 397525
     397526 398524 398525 398526 399524 399525 400525 400526]
    
    [7, 31]
    전체모델 발생 격자
    [384523 385524 386525 388530 388531 389530 392519 393521 393522 393525
     393526 393527 393528 394524 394525 394526 394527 394528 394529 394530
     395524 395525 395526 395527 395528 395530 395531 396523 396524 396525
     396526 396527 396528 396529 396530 396531 396532 396533 396534 397523
     397524 397525 397526 398524 398525 398526 399524 399525 399526 400525
     400526]
    
    [8, 31]
    전체모델 발생 격자
    [384523 385524 386525 388530 388531 389530 392519 393521 393522 393525
     393526 393527 393528 394524 394525 394526 394527 394528 394529 394530
     395524 395525 395526 395527 395528 395530 395531 396523 396524 396525
     396526 396527 396528 396530 396531 396533 396534 397523 397524 397525
     397526 398524 398525 398526 399524 399525 399526 400525 400526]
    
    [9, 30]
    전체모델 발생 격자
    [384523 385524 386525 388530 388531 389530 392519 393521 393522 393525
     393526 393527 393528 394524 394525 394526 394527 394528 394529 394530
     395524 395525 395526 395527 395528 395530 395531 395532 396523 396524
     396525 396526 396527 396528 396529 396530 396531 396533 396534 397523
     397524 397525 397526 398524 398525 398526 399524 399525 399526 400525
     400526]
    
    [10, 31]
    전체모델 발생 격자
    [384523 385524 385529 386525 388530 388531 389530 392519 393521 393522
     393525 393526 393527 393528 394524 394525 394526 394527 394528 394529
     394530 395524 395525 395526 395527 395528 395530 395531 395532 396523
     396524 396525 396526 396527 396528 396530 396531 396533 396534 397523
     397524 397526 398524 398525 398526 399525 399526 400525 400526]
    
    [11, 30]
    전체모델 발생 격자
    [384523 385524 386525 388530 388531 389530 392519 393522 393525 393526
     393527 393528 394524 394525 394526 394527 394528 394529 394530 395524
     395525 395526 395527 395528 395530 395531 396523 396524 396525 396526
     396527 396528 396531 397524 397526 398524 398525 398526 399525 400526]
    
    


```python
final_grid_result_ex2 ("둔상", Random_Forset_model_BLTRM_OCRN_CNT, last_day_df)
```

    둔상 격자 추출
    ******************************
    [1, 31]
    [396527 394525 394526 394529 393525]
    
    [2, 28]
    [396527 395531 394526 396525 394527]
    
    [3, 31]
    [396527 397524 395524 396528 394525]
    
    [4, 30]
    [393522 395524 393525 397524 394525]
    
    [5, 31]
    [397524 395524 394525 396528 393522]
    
    [6, 30]
    [396528 397524 395524 393522 393525]
    
    [7, 31]
    [396527 395530 396525 395531 395524]
    
    [8, 31]
    [396527 395530 395531 397524 396528]
    
    [9, 30]
    [396527 395531 395530 396528 394525]
    
    [10, 31]
    [396527 395530 395531 394524 394525]
    
    [11, 30]
    [396527 395531 396525 397526 394527]
    
    

> 사고부상


```python
final_grid_result("사고부상", Random_Forset_model_ACDNT_INJ_OCRN_CNT, 
                           XGBoost_model_ACDNT_INJ_OCRN_CNT,
                           CatBoost_model_ACDNT_INJ_OCRN_CNT, last_day_df)
```

    사고부상 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    [383522 384523 384524 384525 385525 389531 393523 393526 394526 394527
     394528 394529 395524 395525 395526 395527 395531 396524 396525 396526
     396528 397526]
    
    [2, 28]
    전체모델 발생 격자
    [382521 383522 384523 384524 384525 385525 388531 389531 392521 393523
     393526 394526 394527 394528 394529 395524 395525 395526 395527 395528
     395530 395531 396524 396525 396526 396527 396528 397523 397526 398524]
    
    [3, 31]
    전체모델 발생 격자
    [382518 382521 382522 383521 383522 384522 384523 384524 384525 385523
     385525 386525 388530 388531 389507 389531 390528 390532 391528 392520
     392521 392522 393523 393526 393527 393528 394525 394526 394527 394528
     394529 394531 395524 395525 395526 395527 395528 395530 395531 395532
     396524 396525 396526 396527 396528 396534 397523 397524 397526 397527
     397530 398524]
    
    [4, 30]
    전체모델 발생 격자
    [381518 382521 382522 383521 383522 383523 384522 384523 384524 384525
     385523 385525 385529 386525 387525 388530 388531 389507 389529 389531
     390528 390532 391519 391528 392520 392521 392522 392524 393523 393526
     393527 393528 394525 394526 394527 394528 394529 394531 395524 395525
     395526 395527 395528 395530 395531 395532 395534 396524 396525 396526
     396527 396528 396534 397523 397524 397526 397527 397528 397530 398522
     398524]
    
    [5, 31]
    전체모델 발생 격자
    [383521 384522 384525 388530 393526 394525 394526 394527 394528 394529
     395524 395525 395526 395527 395528 395531 396524 396525 396526 396527
     396528 397526]
    
    [6, 30]
    전체모델 발생 격자
    [382518 382521 382522 383521 383522 383523 384522 384523 384524 384525
     385523 385525 386525 388530 388531 389531 392520 392521 392522 393523
     393526 393527 393528 394524 394525 394526 394527 394528 394529 395524
     395525 395526 395527 395528 395530 395531 395532 396524 396525 396526
     396527 396528 397523 397524 397526 397531 398522 398524]
    
    [7, 31]
    전체모델 발생 격자
    [383521 383522 384522 384525 384536 385525 385529 387525 388530 389507
     389531 391528 392520 392521 392522 392523 393523 393526 393527 393528
     394524 394525 394526 394527 394528 394529 395524 395525 395526 395527
     395528 395530 395531 395532 396524 396525 396526 396527 396528 396531
     397523 397524 397525 397526 397531 398524]
    
    [8, 31]
    전체모델 발생 격자
    [382521 383521 383522 383523 383536 383537 384522 384523 384524 384525
     385523 385525 386525 388530 392521 393523 393525 393526 393527 393528
     394524 394525 394526 394527 394528 394529 395524 395525 395526 395527
     395528 395530 395531 395532 396524 396525 396526 396527 396528 396531
     397523 397524 397526 397531]
    
    [9, 30]
    전체모델 발생 격자
    [383521 383522 383523 383536 384522 384523 384525 385525 385529 388530
     392521 393523 393525 393526 393527 394524 394525 394526 394527 394528
     394529 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 397523 397524 397526 397531 398522]
    
    [10, 31]
    전체모델 발생 격자
    [382521 383521 383522 383523 383537 384522 384523 384525 384528 385525
     387525 392521 393523 393526 394524 394525 394526 394527 394528 394529
     395524 395525 395526 395527 395528 395530 395531 396524 396525 396526
     396527 396528 396531 397523 397525 397526 397531 398522 398527]
    
    [11, 30]
    전체모델 발생 격자
    [381518 382518 382521 382522 383521 383522 383523 383536 384522 384523
     384524 384525 385523 385525 385529 388530 392521 393523 393526 393527
     394526 394527 394528 394529 395524 395525 395526 395527 395528 395529
     395531 396524 396525 396526 396527 396528 396531 397524 397526 397531]
    
    


```python
final_grid_result_ex2 ("사고부상", Random_Forset_model_ACDNT_INJ_OCRN_CNT, last_day_df)
```

    사고부상 격자 추출
    ******************************
    [1, 31]
    [383522 385525 394529 384525 385524]
    
    [2, 28]
    [394529 394527 395527 396524 383522]
    
    [3, 31]
    [394529 395527 393526 396524 384522]
    
    [4, 30]
    [395527 396524 393526 394527 396526]
    
    [5, 31]
    [395527 393526 394527 394529 396524]
    
    [6, 30]
    [395527 396524 394529 394527 393526]
    
    [7, 31]
    [395527 393526 394529 396524 394527]
    
    [8, 31]
    [396524 395527 393526 394527 397526]
    
    [9, 30]
    [393526 396524 395527 394529 393525]
    
    [10, 31]
    [393526 395527 396524 394529 394527]
    
    [11, 30]
    [394529 395527 396524 394527 393526]
    
    

> 질병외 


```python
final_grid_result("질병외", Random_Forset_model_EXCL_DISEASE_OCRN_CNT, 
                           XGBoost_model_EXCL_DISEASE_OCRN_CNT,
                           CatBoost_model_EXCL_DISEASE_OCRN_CNT, last_day_df)
```

    질병외 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    [384523 385524 392519 393522 393525 393526 393527 394524 394525 394526
     394527 394528 394529 394530 395524 395526 395527 395528 395530 395531
     396524 396525 396526 396527 396528 396529 396531 397524 397526 398524
     398526]
    
    [2, 28]
    전체모델 발생 격자
    [384523 385524 388530 389530 392519 393522 393525 393526 393527 393528
     394524 394525 394526 394527 394528 394529 394530 395524 395525 395526
     395527 395528 395530 395531 396524 396525 396526 396527 396528 396529
     396531 397523 397524 397526 398524 398526]
    
    [3, 31]
    전체모델 발생 격자
    [385524 392519 393525 393527 394524 394525 394526 394527 394528 394529
     395524 395525 395526 395527 395531 396525 396526 396527 396528 397524
     397526]
    
    [4, 30]
    전체모델 발생 격자
    [384523 385524 388530 392519 393522 393525 393526 393527 393528 394524
     394525 394526 394527 394528 394529 394530 395524 395525 395526 395527
     395528 395530 395531 396524 396525 396526 396527 396528 396529 396531
     397523 397524 397526]
    
    [5, 31]
    전체모델 발생 격자
    [393525 393527 394526 394527 394528 394529 395524 395525 395526 395527
     395531 396524 396525 396526 396527 396528 397526]
    
    [6, 30]
    전체모델 발생 격자
    [393525 393527 394525 394526 394527 394528 394529 395524 395526 395527
     395528 395531 396525 396526 396527 396528 397526]
    
    [7, 31]
    전체모델 발생 격자
    [384523 385524 392519 393522 393525 393526 393527 394524 394525 394526
     394527 394528 394529 394530 395524 395526 395528 395531 396525 396526
     396527 396531 397524 397526]
    
    [8, 31]
    전체모델 발생 격자
    [393527 394526 394527 394529 395527 395531 396525 396526 396527]
    
    [9, 30]
    전체모델 발생 격자
    [385524 392519 393522 393525 393526 393527 394525 394526 394527 394528
     394529 395524 395526 395527 395528 395531 396525 396526 396527 396528
     396531 397524 397526]
    
    [10, 31]
    전체모델 발생 격자
    [384523 385524 392519 393522 393525 393526 393527 394524 394525 394526
     394527 394528 394529 394530 395524 395526 396525 396526 396527 396528
     396531 397524 397526]
    
    [11, 30]
    전체모델 발생 격자
    [384523 385524 392519 393522 393525 393526 393527 393528 394524 394525
     394526 394527 394528 394529 394530 395524 395525 395526 395527 395528
     395530 395531 396524 396525 396526 396527 396528 396531 397523 397524
     397526 398524]
    
    


```python
final_grid_result_ex2 ("질병외", Random_Forset_model_EXCL_DISEASE_OCRN_CNT, last_day_df)
```

    질병외 격자 추출
    ******************************
    [1, 31]
    [395527 396524 394527 393527 395525]
    
    [2, 28]
    [395527 393527 396524 394527 393526]
    
    [3, 31]
    [393527 394527 395527 394529 394526]
    
    [4, 30]
    [393526 395527 394526 394527 396524]
    
    [5, 31]
    [395527 393527 394527 394529 396524]
    
    [6, 30]
    [394527 393527 394529 395527 396524]
    
    [7, 31]
    [395527 393526 394527 393527 394526]
    
    [8, 31]
    [393527 394527 396526 394529 396525]
    
    [9, 30]
    [394527 393527 393526 394526 393525]
    
    [10, 31]
    [394527 392519 393526 395527 394530]
    
    [11, 30]
    [393527 394527 396524 395527 396526]
    
    

> 탈것사고


```python
final_grid_result("탈것사고", Random_Forset_model_VHC_ACDNT_OCRN_CNT, 
                               XGBoost_model_VHC_ACDNT_OCRN_CNT,
                               CatBoost_model_VHC_ACDNT_OCRN_CNT, last_day_df)
```

    탈것사고 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    []
    
    [2, 28]
    전체모델 발생 격자
    [393526 394526 394527 395525 395527 396524]
    
    [3, 31]
    전체모델 발생 격자
    [384523 384524 385524 388530 392527 393522 393525 393526 393527 393528
     394524 394525 394526 394527 394528 394529 394530 395524 395525 395526
     395527 395528 395529 395530 395531 396523 396524 396525 396526 396527
     396528 396529 396530 396531 396532 396534 397523 397524 397525 397526
     398524 398525 398526 399525 399526 400525 400526]
    
    [4, 30]
    전체모델 발생 격자
    [384523 384524 385524 388530 393522 393524 393525 393526 393527 393528
     394524 394525 394526 394527 394528 394529 394530 395524 395525 395526
     395527 395528 395529 395530 395531 396523 396524 396525 396526 396527
     396528 396529 396530 396531 396532 396533 396534 397523 397524 397525
     397526 398524 398525 398526 399525 399526 400525 400526]
    
    [5, 31]
    전체모델 발생 격자
    [384523 384524 385524 388530 393522 393524 393525 393526 393527 393528
     394524 394525 394526 394527 394528 394529 394530 395524 395525 395526
     395527 395528 395529 395530 395531 396523 396524 396525 396526 396527
     396528 396529 396530 396531 396532 396533 396534 397523 397524 397525
     397526 398524 398525 398526 399525 399526 400525 400526]
    
    [6, 30]
    전체모델 발생 격자
    [381518 382522 383521 384523 384524 384525 385524 387529 387530 388530
     389529 390528 390532 392527 393522 393525 393526 393527 393528 394524
     394525 394526 394527 394528 394529 394530 395524 395525 395526 395527
     395528 395529 395530 395531 396523 396524 396525 396526 396527 396528
     396529 396530 396531 396532 396533 396534 397523 397524 397525 397526
     398524 398525 398526 399525 399526 400525 400526]
    
    [7, 31]
    전체모델 발생 격자
    [382521 383521 383522 383536 384522 384523 384524 384525 384533 384534
     384536 385523 385524 385525 385529 387525 387529 387530 388512 388530
     388531 389507 389529 389531 390528 391528 392529 393520 393522 393525
     393526 393527 393528 393529 394524 394525 394526 394527 394528 394529
     394530 395524 395525 395526 395527 395528 395529 395530 395531 396523
     396524 396525 396526 396527 396528 396529 396530 396531 396532 396533
     396534 397523 397524 397525 397526 398524 398525 398526 399525 399526
     400525 400526]
    
    [8, 31]
    전체모델 발생 격자
    [382521 382522 383521 383522 383523 383536 384522 384523 384524 384525
     385523 385524 385525 385529 386525 386526 387529 387530 388529 388530
     388531 389507 389529 389530 389531 390528 390532 391519 391528 392519
     392520 392521 392522 392523 392524 392526 392527 392528 392529 393521
     393522 393523 393524 393525 393526 393527 393528 393529 394524 394525
     394526 394527 394528 394529 394530 395524 395525 395526 395527 395528
     395529 395530 395531 396523 396524 396525 396526 396527 396528 396529
     396530 396531 396532 396533 396534 397523 397524 397525 397526 398525
     398526 399525 399526 400525 400526]
    
    [9, 30]
    전체모델 발생 격자
    [382521 382522 383521 383522 383523 383536 384522 384523 384524 384525
     385523 385524 385525 385529 386525 386526 387527 387529 387530 388529
     388530 388531 389507 389529 389530 389531 390528 390532 391519 391528
     392519 392520 392521 392522 392523 392524 392528 392529 393521 393522
     393523 393524 393525 393526 393527 393528 393529 394524 394525 394526
     394527 394528 394529 394530 395524 395525 395526 395527 395528 395529
     395530 395531 396523 396524 396525 396526 396527 396528 396529 396530
     396531 396532 396533 396534 397523 397524 397525 397526 398525 398526
     399525 399526 400525 400526]
    
    [10, 31]
    전체모델 발생 격자
    [382521 383521 383522 383523 383536 383537 384522 384523 384524 384525
     384528 384533 384534 384536 385523 385524 385525 385528 385529 387525
     387529 387530 388529 388530 388531 389507 389529 389530 389531 390528
     391519 391528 392519 392520 392521 392522 392524 392525 392528 392529
     393520 393521 393522 393524 393525 393526 393527 393528 393529 394524
     394525 394526 394527 394528 394529 394530 395524 395525 395526 395527
     395528 395529 395530 395531 396523 396524 396525 396526 396527 396528
     396530 396531 396533 396534 397523 397524 397526 398525 398526 399525
     399526 400525 400526]
    
    [11, 30]
    전체모델 발생 격자
    [381518 382518 382521 383521 383522 383523 383536 384522 384523 384524
     384525 385523 385524 385525 385529 386525 387529 387530 388529 388530
     388531 389529 389530 389531 390528 390532 391519 392521 393521 393522
     393523 393524 393525 393526 393527 393528 393529 394524 394525 394526
     394527 394528 394529 394530 395524 395525 395526 395527 395528 395530
     395531 396523 396524 396525 396526 396527 396528 396530 396531 396533
     396534 397523 397524 397526 398525 398526 399525 399526 400525 400526]
    
    


```python
final_grid_result_ex2 ("탈것사고", Random_Forset_model_VHC_ACDNT_OCRN_CNT, last_day_df)
```

    탈것사고 격자 추출
    ******************************
    [1, 31]
    [396524 394527 395527 393527 396526]
    
    [2, 28]
    [395527 396524 394527 394526 396526]
    
    [3, 31]
    [395527 396524 394527 394526 394529]
    
    [4, 30]
    [394527 395527 396524 393527 394529]
    
    [5, 31]
    [395527 396524 394527 393527 396526]
    
    [6, 30]
    [394527 395527 396524 393527 394529]
    
    [7, 31]
    [395527 394527 393527 396524 394529]
    
    [8, 31]
    [395527 396524 393527 394527 395524]
    
    [9, 30]
    [393527 394529 395527 394527 393525]
    
    [10, 31]
    [394529 393527 394526 395527 395524]
    
    [11, 30]
    [395527 396524 393527 394527 396526]
    
    

> 낙상


```python
final_grid_result("낙상", Random_Forset_model_HRFAF_OCRN_CNT, 
                           XGBoost_model_HRFAF_OCRN_CNT,
                           CatBoost_model_HRFAF_OCRN_CNT, last_day_df)
```

    낙상 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    [393526 393527 393528 394525 394526 394527 394528 394529 395525 395527
     395528 395530 395531 396524 396526 396527 396528 397523 397524 398524
     398526]
    
    [2, 28]
    전체모델 발생 격자
    [385524 393522 393525 393526 393527 393528 394525 394526 394527 394528
     394529 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 397524 397526 398524 398526]
    
    [3, 31]
    전체모델 발생 격자
    [393522 393526 393527 394526 394527 394528 394529 395524 395525 395527
     395528 395531 396524 396527 398524]
    
    [4, 30]
    전체모델 발생 격자
    [393522 393526 393527 393528 394525 394526 394527 394528 394529 395524
     395527 395528 396527 398524 398526]
    
    [5, 31]
    전체모델 발생 격자
    [393522 393525 393526 393527 393528 394525 394526 394527 395525 395527
     395528 395531 396524 396525 396527]
    
    [6, 30]
    전체모델 발생 격자
    [393526 393527 394526 394527 394529 395524 395525 395527 395528 395531
     396527]
    
    [7, 31]
    전체모델 발생 격자
    [393526 393527 394526 394527 394529 395527 395528 396525 396527]
    
    [8, 31]
    전체모델 발생 격자
    [393526 393527 394527 395525 395527 395531 396524]
    
    [9, 30]
    전체모델 발생 격자
    [393525 393526 393527 394525 394526 394527 394529 395525 395527 395528
     396524]
    
    [10, 31]
    전체모델 발생 격자
    [393522 393525 393526 393527 394525 394526 394527 394528 394529 395524
     395525 395527 395528 395531 396524 396525 396527]
    
    [11, 30]
    전체모델 발생 격자
    [384523 385524 392519 393522 393525 393526 393527 393528 394524 394525
     394526 394527 394528 394529 394530 395524 395525 395526 395527 395528
     395530 395531 396524 396526 396527 396528 396531 397523]
    
    


```python
final_grid_result_ex2 ("낙상", Random_Forset_model_HRFAF_OCRN_CNT, last_day_df)
```

    낙상 격자 추출
    ******************************
    [1, 31]
    [394529 396524 395525 394527 394526]
    
    [2, 28]
    [396524 393526 395525 394527 395527]
    
    [3, 31]
    [395527 394527 395525 393526 394526]
    
    [4, 30]
    [395527 394527 394526 393526 395531]
    
    [5, 31]
    [395527 394527 394526 395525 393526]
    
    [6, 30]
    [392519 394527 395527 393527 395525]
    
    [7, 31]
    [395527 394527 396524 394529 393527]
    
    [8, 31]
    [392519 394527 393523 395527 392521]
    
    [9, 30]
    [394527 393526 395525 395527 393523]
    
    [10, 31]
    [393526 395527 394527 394529 395525]
    
    [11, 30]
    [395527 393527 394527 385524 396524]
    
    

> 단순주취


```python
final_grid_result("단순주취", Random_Forset_model_DRKNSTAT_OCRN_CNT, 
                              XGBoost_model_DRKNSTAT_OCRN_CNT,
                              CatBoost_model_DRKNSTAT_OCRN_CNT, last_day_df)
```

    단순주취 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    [393525 393526 393527 394526 394527 394529 395525 395527 395528 395531
     396524 396525 396526]
    
    [2, 28]
    전체모델 발생 격자
    [393525 393526 393527 394527 394529 395525 395527 396524 396526]
    
    [3, 31]
    전체모델 발생 격자
    [393525 393526 393527 394527 394529 395525 395527 396524 396526]
    
    [4, 30]
    전체모델 발생 격자
    [393525 393526 393527 394526 394527 394529 395525 395527 395531 396524
     396525 396526]
    
    [5, 31]
    전체모델 발생 격자
    [393525 393526 393527 394526 394527 394529 395525 395527 396524 396525
     396526]
    
    [6, 30]
    전체모델 발생 격자
    [393525 393526 393527 394526 394527 394529 395525 395527 395531 396524
     396526]
    
    [7, 31]
    전체모델 발생 격자
    [393525 393526 393527 394526 394527 394529 395525 395527 396524 396525
     396526]
    
    [8, 31]
    전체모델 발생 격자
    [393525 393526 393527 394526 394527 394529 395525 395527 396524 396525
     396526]
    
    [9, 30]
    전체모델 발생 격자
    [393525 393526 393527 394526 394527 394529 395525 395527 396524 396525
     396526]
    
    [10, 31]
    전체모델 발생 격자
    [393525 393526 393527 394526 394527 394529 395525 395527 396524 396525
     396526]
    
    [11, 30]
    전체모델 발생 격자
    [393525 393526 393527 394526 394527 394528 394529 395524 395525 395526
     395527 395528 395531 396524 396525 396526]
    
    


```python
final_grid_result_ex2 ("단순주취", Random_Forset_model_DRKNSTAT_OCRN_CNT, last_day_df)
```

    단순주취 격자 추출
    ******************************
    [1, 31]
    [395527 396526 394527 393527 396524]
    
    [2, 28]
    [396526 395527 394527 396524 394529]
    
    [3, 31]
    [396526 394527 396524 395527 394529]
    
    [4, 30]
    [396526 394527 396524 394529 395527]
    
    [5, 31]
    [394527 396526 396524 395527 394529]
    
    [6, 30]
    [394527 396526 395527 396524 394529]
    
    [7, 31]
    [394527 396526 396524 395527 394529]
    
    [8, 31]
    [394527 396526 395527 396524 394529]
    
    [9, 30]
    [394527 396526 396524 395527 393526]
    
    [10, 31]
    [396526 394527 395527 396524 393526]
    
    [11, 30]
    [395527 396526 394527 396524 393526]
    
    

> 동물곤충사고


```python
final_grid_result("동물곤충사고", Random_Forset_model_ANML_INSCT_ACDNT_OCRN_CNT, 
                              XGBoost_model_ANML_INSCT_ACDNT_OCRN_CNT,
                              CatBoost_model_ANML_INSCT_ACDNT_OCRN_CNT, last_day_df)
```

    동물곤충사고 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    []
    
    [2, 28]
    전체모델 발생 격자
    []
    
    [3, 31]
    전체모델 발생 격자
    [389530]
    
    [4, 30]
    전체모델 발생 격자
    []
    
    [5, 31]
    전체모델 발생 격자
    [389530]
    
    [6, 30]
    전체모델 발생 격자
    [389530]
    
    [7, 31]
    전체모델 발생 격자
    [393525 393526 393527]
    
    [8, 31]
    전체모델 발생 격자
    [382521 383521 383522 383523 383536 384522 384523 384524 384525 384536
     385523 385524 385525 385528 385529 386525 387521 387525 387527 387529
     387530 388512 388527 388529 388530 388531 389507 389529 389530 389531
     390528 390532 391519 391528 392519 392520 392521 392522 392523 392524
     392526 392527 392528 392529 393521 393522 393523 393524 393525 393526
     393527 393528 393529 394520 394522 394524 394525 394526 394527 394528
     394529 394530 394531 394532 395523 395524 395525 395526 395527 395528
     395529 395530 395531 395532 396523 396524 396525 396526 396527 396528
     396530 396531 396532 396533 396534 397523 397524 397525 397526 397529
     398522 398524 398525 398526 398532 399524 399525 399526 400525 400526
     400527 400528 400529 400535 401528]
    
    [9, 30]
    전체모델 발생 격자
    [382521 383521 383522 383523 383536 384522 384523 384524 384525 385523
     385524 385525 385528 385529 386524 386525 386526 387521 387525 387526
     387527 387529 387530 388512 388527 388529 388530 388531 389507 389529
     389530 389531 389532 390507 390528 390532 391519 391527 391528 392519
     392520 392521 392522 392523 392524 392525 392528 392529 393521 393522
     393523 393524 393525 393526 393527 393528 393529 394520 394522 394523
     394524 394525 394526 394527 394528 394529 394530 394531 394532 395523
     395524 395525 395526 395527 395528 395529 395530 395531 395532 395533
     396523 396524 396525 396526 396527 396528 396529 396530 396531 396532
     396533 396534 397523 397524 397525 397526 398524 398525 398526 398532
     399524 399525 399526 400525 400526 400527 400528 400529 400535 401528]
    
    [10, 31]
    전체모델 발생 격자
    [383523 383536 384522 384523 384524 384536 385523 385524 385525 385528
     385529 386525 387521 387525 387527 387530 388512 388527 388529 388530
     388531 389507 389529 389530 389531 390507 391510 391519 391525 391527
     391528 392511 392519 392523 392524 393522 393525 393526 393527 393528
     394520 394524 394525 394526 394527 394528 394529 394530 395523 395524
     395525 395526 395527 395528 395530 395531 395532 396523 396524 396525
     396526 396527 396529 396530 396531 396533 396534 397526 398524 399526
     399527 400525 400526 401526 401529]
    
    [11, 30]
    전체모델 발생 격자
    [381518 382518 382521 383521 383522 383523 383536 383537 384522 384523
     384524 384525 385523 385524 385525 385528 385529 386524 386525 387521
     387525 387526 387527 387529 387530 388512 388527 388529 388530 388531
     389507 389529 389530 389531 390507 390528 390532 391519 391527 391528
     392519 392520 392521 392522 392523 392524 392525 392528 392529 393520
     393521 393522 393523 393524 393525 393526 393527 393528 393529 394522
     394523 394524 394525 394526 394527 394528 394529 394530 394531 395523
     395524 395525 395526 395527 395528 395530 395531 396524 396525 396526
     396527 396528 396529 396534 397526 398524 399525 399526 400525 400526
     400527 400528 400529 400535]
    
    


```python
final_grid_result_ex2 ("동물곤충사고", Random_Forset_model_ANML_INSCT_ACDNT_OCRN_CNT, last_day_df)
```

    동물곤충사고 격자 추출
    ******************************
    [1, 31]
    [395527 394527 394526 393527]
    
    [2, 28]
    [385524 395525 400525 388530 384523]
    
    [3, 31]
    [385524 384523 388530 384524 400525]
    
    [4, 30]
    [385524 384523 388530 400525 393527]
    
    [5, 31]
    [385524 384523 388530 393527 384524]
    
    [6, 30]
    [385524 393527 394526 394527 394529]
    
    [7, 31]
    [385524 395527 394526 394527 393527]
    
    [8, 31]
    [385524 386525 395527 388529 394527]
    
    [9, 30]
    [385524 392519 388529 385525 386525]
    
    [10, 31]
    [385524 384523 392519 385525 385529]
    
    [11, 30]
    [385524 396524 395527 394527 396526]
    
    

>동승자사고


```python
final_grid_result("동승자사고", Random_Forset_model_FLPS_ACDNT_OCRN_CNT, 
                              XGBoost_model_FLPS_ACDNT_OCRN_CNT,
                              CatBoost_model_FLPS_ACDNT_OCRN_CNT, last_day_df)
```

    동승자사고 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    [384523 385524 392519 393522 393525 393526 393527 393528 394524 394525
     394526 394527 394528 394529 394530 395524 395525 395526 395527 395528
     395530 395531 396524 396525 396526 396527 396528 396529 396531 396532
     396534 397523 397524 397526 398524 398526]
    
    [2, 28]
    전체모델 발생 격자
    [385524 393522 393525 393526 393527 393528 394524 394525 394526 394527
     394528 394529 395524 395525 395526 395527 395528 395530 395531 396524
     396525 396526 396527 396528 396531 397523 397524 397526 398524 398526]
    
    [3, 31]
    전체모델 발생 격자
    [393526 393527 393528 394526 394527 394528 394529 395525 395526 395527
     395528 395530 395531 396524 396525 396526 396527 396534 397523 397526]
    
    [4, 30]
    전체모델 발생 격자
    [384523 385524 392519 393522 393525 393526 393527 393528 394524 394525
     394526 394527 394528 394529 394530 395524 395525 395526 395527 395528
     395530 395531 396524 396525 396526 396527 396528 396529 396531 396532
     396534 397523 397524 397526 398524 398526 399525 407514]
    
    [5, 31]
    전체모델 발생 격자
    [395528 396525 396527 407514]
    
    [6, 30]
    전체모델 발생 격자
    [395528 396527 407514]
    
    [7, 31]
    전체모델 발생 격자
    [384523 385524 385525 388530 392519 392528 393522 393524 393525 393526
     393527 393528 394524 394525 394526 394527 394528 394529 394530 395524
     395525 395526 395527 395528 395530 395531 395532 396523 396524 396525
     396526 396527 396528 396529 396530 396531 396532 396533 396534 397523
     397524 397525 397526 398524 398525 398526 399525 399526 400525 400526
     405536 407514 417517]
    
    [8, 31]
    전체모델 발생 격자
    [407514]
    
    [9, 30]
    전체모델 발생 격자
    [393528 395526 395531 396527 397523 407514]
    
    [10, 31]
    전체모델 발생 격자
    [384523 385524 393522 393525 393526 393527 393528 394524 394525 394526
     394527 394528 394529 394530 395524 395525 395526 395527 395528 395530
     395531 396524 396525 396526 396527 396528 396531 396534 397523 397524
     397526 398524 398526 399525 407514]
    
    [11, 30]
    전체모델 발생 격자
    []
    
    


```python
final_grid_result_ex2 ("동승자사고", Random_Forset_model_FLPS_ACDNT_OCRN_CNT, last_day_df)
```

    동승자사고 격자 추출
    ******************************
    [1, 31]
    [396527 394526 393526 395528 395525]
    
    [2, 28]
    [396527 395528 395525 395531 393526]
    
    [3, 31]
    [392527 393522 395523 394530 396530]
    
    [4, 30]
    [392527 388530 393527 393522 384523]
    
    [5, 31]
    [392527 388529 400525 397525 396530]
    
    [6, 30]
    [392527 396530 392526 393522 394530]
    
    [7, 31]
    [396530 392527 392526 385524 399526]
    
    [8, 31]
    [382521 404516 392526 392527 384524]
    
    [9, 30]
    [392522 384524 392528 385524 404517]
    
    [10, 31]
    [395523 404516 403538 396530 384524]
    
    [11, 30]
    [384523 385524 396527 393522 393528]
    
    

>보행자사고


```python
final_grid_result("보행자사고", Random_Forset_model_PDST_ACDNT_OCRN_CNT, 
                              XGBoost_model_PDST_ACDNT_OCRN_CNT,
                              CatBoost_model_PDST_ACDNT_OCRN_CNT, last_day_df)
```

    보행자사고 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    [393526 393528 394526 394527 394528 395524 395525 395526 395527 395530
     395531 396524 396525 396526 397523 397526]
    
    [2, 28]
    전체모델 발생 격자
    [393522 393525 393526 393527 393528 394524 394525 394526 394527 394528
     394529 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 397523 397524 397526]
    
    [3, 31]
    전체모델 발생 격자
    [393522 393525 393526 393527 393528 394524 394525 394526 394527 394528
     394529 394530 395524 395525 395526 395527 395528 395530 395531 396523
     396524 396525 396526 396527 396528 396531 397523 397524 397526]
    
    [4, 30]
    전체모델 발생 격자
    [388530 393522 393525 393526 393527 393528 394524 394525 394526 394527
     394528 394529 394530 395524 395525 395526 395527 395528 395530 395531
     396523 396524 396525 396526 396527 396528 396531 397523 397524 397526]
    
    [5, 31]
    전체모델 발생 격자
    [393522 393525 393526 393527 393528 394524 394525 394526 394527 394528
     394529 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 397523 397524 397526]
    
    [6, 30]
    전체모델 발생 격자
    [388530 392519 393522 393525 393526 393527 393528 394524 394525 394526
     394527 394528 394529 394530 395524 395525 395526 395527 395528 395530
     395531 396523 396524 396525 396526 396527 396528 396531 397523 397524
     397526]
    
    [7, 31]
    전체모델 발생 격자
    [384523 385524 388530 392519 393522 393525 393526 393527 393528 394524
     394525 394526 394527 394528 394529 394530 395524 395525 395526 395527
     395528 395530 395531 396523 396524 396525 396526 396527 396528 396531
     396533 396534 397523 397524 397526]
    
    [8, 31]
    전체모델 발생 격자
    [393522 393525 393526 393527 393528 394524 394525 394526 394527 394528
     394529 395524 395525 395526 395527 395528 395530 395531 396523 396524
     396525 396526 396527 396528 396531 397523 397524 397526]
    
    [9, 30]
    전체모델 발생 격자
    [388530 392519 393522 393524 393525 393526 393527 393528 394524 394525
     394526 394527 394528 394529 394530 395524 395525 395526 395527 395528
     395530 395531 396523 396524 396525 396526 396527 396528 396531 396533
     396534 397523 397524 397526]
    
    [10, 31]
    전체모델 발생 격자
    [385524 388530 392519 393522 393525 393526 393527 393528 394524 394525
     394526 394527 394528 394529 395524 395525 395526 395527 395528 395530
     395531 396524 396525 396526 396527 396528 396531 396533 396534 397523
     397524 397526]
    
    [11, 30]
    전체모델 발생 격자
    [393522 393525 393526 393527 393528 394524 394525 394526 394527 394528
     394529 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 396531 397523 397526]
    
    


```python
final_grid_result_ex2 ("보행자사고", Random_Forset_model_PDST_ACDNT_OCRN_CNT, last_day_df)
```

    보행자사고 격자 추출
    ******************************
    [1, 31]
    [393528 395530 397523 396526 393526]
    
    [2, 28]
    [393528 393526 395527 394526 394527]
    
    [3, 31]
    [393526 394526 395525 395527 394527]
    
    [4, 30]
    [394527 394526 395525 393526 395527]
    
    [5, 31]
    [395525 395527 393526 394526 394527]
    
    [6, 30]
    [393526 394527 394526 395525 394528]
    
    [7, 31]
    [393526 394526 395527 395526 394528]
    
    [8, 31]
    [394526 393526 394527 395527 395531]
    
    [9, 30]
    [393526 394526 394527 394528 396524]
    
    [10, 31]
    [394526 393526 394527 395527 396524]
    
    [11, 30]
    [394526 393526 396525 394527 396524]
    
    

>열상


```python
final_grid_result("열상", Random_Forset_model_LACRTWND_OCRN_CNT, 
                              XGBoost_model_LACRTWND_OCRN_CNT,
                              CatBoost_model_LACRTWND_OCRN_CNT, last_day_df)
```

    열상 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    [394526 394527 394528 395527 395531 396525 396526]
    
    [2, 28]
    전체모델 발생 격자
    [393525 393526 393527 393528 394525 394526 394527 394528 394529 395524
     395525 395526 395527 395528 395530 395531 396524 396525 396526 396527
     397526]
    
    [3, 31]
    전체모델 발생 격자
    [394529 395531 396525]
    
    [4, 30]
    전체모델 발생 격자
    [389530 392519 393525 393527 394525 394527 394528 394529 395524 395525
     395526 395531 396525]
    
    [5, 31]
    전체모델 발생 격자
    [393525 393526 393527 394525 394529 395524 395526 395531 396525]
    
    [6, 30]
    전체모델 발생 격자
    [393525 393527 394525 394529 395524 395526 395528 395531 396525]
    
    [7, 31]
    전체모델 발생 격자
    [393525 393527 394525 394528 394529 395524 395525 395526 395527 395528
     395530 395531 396524 396525 396526 396527 397526]
    
    [8, 31]
    전체모델 발생 격자
    [393527 394525 395524 395525 395526 395528 395530 395531 396525]
    
    [9, 30]
    전체모델 발생 격자
    [393525 393527 394525 394527 394528 394529 395524 395525 395526 395527
     395528 395530 395531 396523 396524 396525 397526]
    
    [10, 31]
    전체모델 발생 격자
    [388531 389530 392519 393525 393527 394525 394526 394527 394529 395524
     395525 395526 395527 395528 395530 395531 396524 396525 396526 396527
     396528 397526]
    
    [11, 30]
    전체모델 발생 격자
    [384523 388531 389530 392519 393522 393525 393526 393527 393528 394524
     394525 394526 394527 394528 394529 395524 395525 395526 395527 395528
     395530 395531 396524 396525 396526 396527 396528 397526]
    
    


```python
final_grid_result_ex2 ("열상", Random_Forset_model_LACRTWND_OCRN_CNT, last_day_df)
```

    열상 격자 추출
    ******************************
    [1, 31]
    [394527 396525 395531 396526 395527]
    
    [2, 28]
    [396525 395531 396526 394528 394526]
    
    [3, 31]
    [396525 395531 396527 396526 395528]
    
    [4, 30]
    [393527 395531 396525 393526 395524]
    
    [5, 31]
    [396525 395531 393526 396527 396526]
    
    [6, 30]
    [396525 395524 393526 393527 396526]
    
    [7, 31]
    [395527 393527 396525 394527 395531]
    
    [8, 31]
    [393527 396525 395531 393525 395524]
    
    [9, 30]
    [393527 395527 396525 394527 395531]
    
    [10, 31]
    [395527 394527 393527 395531 396525]
    
    [11, 30]
    [395531 393527 395528 395526 396525]
    
    

>오토바이사고


```python
final_grid_result("오토바이사고", Random_Forset_model_MTRCYC_ACDNT_OCRN_CNT, 
                              XGBoost_model_MTRCYC_ACDNT_OCRN_CNT,
                              CatBoost_model_MTRCYC_ACDNT_OCRN_CNT, last_day_df)
```

    오토바이사고 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    [388530 393526 393527 394524 394525 394526 394527 394528 394529 395524
     395525 395526 395527 395528 395530 395531 396524 396525 396526 396527
     396528 397523 397526]
    
    [2, 28]
    전체모델 발생 격자
    [388530 393525 393526 393527 393528 394524 394525 394526 394527 394528
     394529 395524 395525 395526 395527 395528 395530 395531 396523 396524
     396525 396526 396527 396528 396531 397523 397524 397525 397526 398525]
    
    [3, 31]
    전체모델 발생 격자
    [384523 388530 393522 393525 393526 393527 393528 394524 394525 394526
     394527 394528 394529 394530 395524 395525 395526 395527 395528 395530
     395531 396523 396524 396525 396526 396527 396528 396531 397523 397524
     397526 398524 398525 399525]
    
    [4, 30]
    전체모델 발생 격자
    [384523 388530 393522 393525 393526 393527 393528 394524 394525 394526
     394527 394528 394529 394530 395524 395525 395526 395527 395528 395530
     395531 396523 396524 396525 396526 396527 396528 396531 397523 397524
     397526 398524 398525 399525]
    
    [5, 31]
    전체모델 발생 격자
    [384523 388530 393522 393525 393526 393527 393528 394524 394525 394526
     394527 394528 394529 394530 395524 395525 395526 395527 395528 395530
     395531 396523 396524 396525 396526 396527 396528 396531 396533 396534
     397523 397524 397526 398524 398525 398526 399525]
    
    [6, 30]
    전체모델 발생 격자
    [384523 388530 393522 393525 393526 393527 393528 394524 394525 394526
     394527 394528 394529 394530 395524 395525 395526 395527 395528 395530
     395531 396524 396525 396526 396527 396528 396531 396534 397523 397526
     398525 398526 399525]
    
    [7, 31]
    전체모델 발생 격자
    [388530 393525 393526 393527 393528 394524 394525 394526 394527 394528
     394529 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 396531 397523 397524 397526 398525]
    
    [8, 31]
    전체모델 발생 격자
    [384523 388530 393525 393526 393527 393528 394524 394525 394526 394527
     394528 394529 394530 395524 395525 395526 395527 395528 395530 395531
     396523 396524 396525 396526 396527 396528 396531 396534 397523 397524
     397526 398525 399525]
    
    [9, 30]
    전체모델 발생 격자
    [388530 393526 393527 393528 394524 394525 394526 394527 394528 394529
     394530 395524 395525 395526 395527 395528 395530 395531 396524 396525
     396526 396527 396528 396531 397523 397524 397526 399525]
    
    [10, 31]
    전체모델 발생 격자
    [388530 393526 393527 393528 394524 394525 394526 394527 394528 394529
     395525 395526 395527 395528 395530 395531 396524 396525 396526 396527
     396528 396531 397524 397526]
    
    [11, 30]
    전체모델 발생 격자
    [388530 393526 393527 393528 394524 394525 394526 394527 394528 394529
     395524 395525 395526 395527 395528 395530 395531 396523 396524 396525
     396526 396527 396528 396531 397524 397526 398525 399525]
    
    


```python
final_grid_result_ex2 ("오토바이사고", Random_Forset_model_MTRCYC_ACDNT_OCRN_CNT, last_day_df)
```

    오토바이사고 격자 추출
    ******************************
    [1, 31]
    [394527 395527 393527 395525 396524]
    
    [2, 28]
    [396524 394527 395527 395525 393526]
    
    [3, 31]
    [394527 395527 396524 395525 393526]
    
    [4, 30]
    [395527 395525 394527 395531 396526]
    
    [5, 31]
    [394527 395527 395525 393527 393526]
    
    [6, 30]
    [394527 393527 395527 395525 396524]
    
    [7, 31]
    [394527 395527 396524 393527 395525]
    
    [8, 31]
    [394527 395527 393527 396524 395525]
    
    [9, 30]
    [394527 396524 395527 395525 395528]
    
    [10, 31]
    [395527 396524 394527 393527 395525]
    
    [11, 30]
    [394527 395527 393526 396524 394526]
    
    

>운전자사고


```python
final_grid_result("운전자사고", Random_Forset_model_DRV_ACDNT_OCRN_CNT, 
                              XGBoost_model_DRV_ACDNT_OCRN_CNT,
                              CatBoost_model_DRV_ACDNT_OCRN_CNT, last_day_df)
```

    운전자사고 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    [393526 393528 394526 394527 394528 395526 395527 395528 395531 396524
     396525 396526 396527 396533 397526]
    
    [2, 28]
    전체모델 발생 격자
    [393526 393528 394526 394527 394528 395525 395526 395527 395528 395530
     395531 396524 396525 396526 396527 396534 397523 397526]
    
    [3, 31]
    전체모델 발생 격자
    [384523 385525 386525 393522 393525 393526 393527 393528 394524 394525
     394526 394527 394528 394530 395526 395528 395530 395531 396524 396525
     396526 396527 396528 396531 396534 397523 397526]
    
    [4, 30]
    전체모델 발생 격자
    [384523 385524 385525 386525 393522 393525 393527 393528 394524 394525
     394527 394528 394529 394530 395526 395528 395530 395531 395532 395533
     396525 396526 396527 396528 396531 396532 396534 397523 397524 397526
     398524]
    
    [5, 31]
    전체모델 발생 격자
    [384523 385525 386525 392519 393522 393525 393526 393527 393528 394524
     394525 394526 394527 394528 394529 394530 395525 395526 395528 395530
     395531 395532 396524 396525 396526 396527 396528 396534 397523 397526]
    
    [6, 30]
    전체모델 발생 격자
    [386525 393528 394528 396534]
    
    [7, 31]
    전체모델 발생 격자
    [393525 393526 393527 393528 394524 394525 394526 394527 394528 394529
     395525 395526 395527 395528 395530 395531 395532 396524 396525 396526
     396527 396528 396531 396534 397523 397526]
    
    [8, 31]
    전체모델 발생 격자
    [386525 393526 393528 394526 394527 394528 395525 395526 395528 395530
     395531 395532 396524 396525 396526 396527 396533 396534 397523 397526]
    
    [9, 30]
    전체모델 발생 격자
    [385525 386525 393526 393528 394526 394527 394528 395526 395528 395530
     395531 395532 396525 396527 396534]
    
    [10, 31]
    전체모델 발생 격자
    [393526 393527 393528 394526 394527 394528 394529 395525 395526 395527
     395528 395530 395531 395532 396524 396525 396526 396527 396534 397523
     397526]
    
    [11, 30]
    전체모델 발생 격자
    [386525 393526 393528 394526 394527 394528 395525 395526 395527 395528
     395530 395531 395532 396524 396525 396526 396527 396533 396534 397523
     397526]
    
    


```python
final_grid_result_ex2 ("운전자사고", Random_Forset_model_DRV_ACDNT_OCRN_CNT, last_day_df)
```

    운전자사고 격자 추출
    ******************************
    [1, 31]
    [395532 396525 395526 396533 397536]
    
    [2, 28]
    [396525 395526 394528 395531 396526]
    
    [3, 31]
    [395526 396525 393528 394528 395531]
    
    [4, 30]
    [392519 394528 394530 393528 395526]
    
    [5, 31]
    [395526 393528 388531 387530 394528]
    
    [6, 30]
    [395526 389531 396525 393528 396527]
    
    [7, 31]
    [387530 394528 396525 395528 396527]
    
    [8, 31]
    [393528 396527 396525 395530 394528]
    
    [9, 30]
    [396527 396525 387530 395528 395531]
    
    [10, 31]
    [396527 395531 396526 393526 393528]
    
    [11, 30]
    [395526 396525 396527 395531 394528]
    
    

>자전거사고


```python
final_grid_result("자전거사고", Random_Forset_model_BCYC_ACDNT_OCRN_CNT, 
                              XGBoost_model_BCYC_ACDNT_OCRN_CNT,
                              CatBoost_model_BCYC_ACDNT_OCRN_CNT, last_day_df)
```

    자전거사고 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    [395524 395525 395526 395527 395531 396524 396526 397526]
    
    [2, 28]
    전체모델 발생 격자
    [394525 394527 394528 394529 395524 395525 395526 395527 395531 396524
     396525 396526 397526]
    
    [3, 31]
    전체모델 발생 격자
    [393528 394524 394525 394527 394528 394530 395524 395525 395526 395527
     395528 395530 395531 396523 396524 396525 396526 396527 396528 396530
     396531 397523 397524 397525 397526 398524 398525]
    
    [4, 30]
    전체모델 발생 격자
    [394525 394527 394528 395524 395525 395526 395527 395528 395529 395530
     395531 396523 396524 396525 396526 396527 396528 396530 396531 397523
     397524 397525 397526 398524 398525]
    
    [5, 31]
    전체모델 발생 격자
    [384523 388530 389530 392527 393521 393522 393524 393525 393526 393527
     393528 394524 394525 394526 394527 394528 394529 394530 395523 395524
     395525 395526 395527 395528 395529 395530 395531 395532 395533 396523
     396524 396525 396526 396527 396528 396530 396531 396533 397523 397524
     397525 397526 397530 398524 398525 398526 399524 399525 399526 400525
     400526]
    
    [6, 30]
    전체모델 발생 격자
    [382521 383522 384523 384524 386525 388529 388530 388531 389530 392519
     392520 392521 392522 392523 392526 392527 392528 392529 393521 393522
     393523 393524 393525 393526 393527 393528 393529 394524 394525 394526
     394527 394528 394529 394530 394531 394532 395523 395524 395525 395526
     395527 395528 395529 395530 395531 395532 395533 395534 396523 396524
     396525 396526 396527 396528 396529 396530 396531 396532 396533 396534
     397523 397524 397525 397526 397527 397528 397531 398522 398524 398525
     398526 398527 398531 399524 399525 399526 400525 400526]
    
    [7, 31]
    전체모델 발생 격자
    [384523 384524 385525 388529 388530 388531 389530 392519 392520 392521
     392522 392523 392524 392526 392527 392528 392529 393521 393522 393523
     393524 393525 393526 393527 393528 394520 394524 394525 394526 394527
     394528 394529 394530 395523 395524 395525 395526 395527 395528 395529
     395530 395531 395532 396523 396524 396525 396526 396527 396528 396529
     396530 396531 396532 396533 396534 397523 397524 397525 397526 397531
     398524 398525 398526 399524 399525 399526 400525 400526]
    
    [8, 31]
    전체모델 발생 격자
    [384523 385524 388530 388531 389530 392519 392520 392524 392526 392527
     392528 393521 393522 393523 393524 393525 393526 393527 393528 394524
     394525 394526 394527 394528 394529 394530 394532 395524 395525 395526
     395527 395528 395529 395530 395531 395532 396523 396524 396525 396526
     396527 396528 396530 396531 396533 396534 397523 397524 397525 397526
     397531 398524 398525 398526 398531 399524 399525 399526 400525 400526]
    
    [9, 30]
    전체모델 발생 격자
    [392528 393521 393522 393524 393525 393526 393527 393528 394524 394525
     394526 394527 394528 394529 394530 395524 395525 395526 395527 395528
     395530 395531 396523 396524 396525 396526 396527 396528 396531 396533
     397523 397524 397526 398524 398525 399525 399526 400525 400526]
    
    [10, 31]
    전체모델 발생 격자
    [382521 383522 384523 384524 385523 385525 385528 385529 387525 388529
     388530 388531 389530 389531 391528 392519 392520 392521 392522 392523
     392524 392525 392528 393521 393522 393523 393524 393525 393526 393527
     393528 393529 394522 394523 394524 394525 394526 394527 394528 394529
     394530 394531 394536 395523 395524 395525 395526 395527 395528 395529
     395530 395531 395532 395533 396523 396524 396525 396526 396527 396528
     396529 396530 396531 396532 396533 396534 397523 397524 397526 398524
     398525 398526 399525 399526 400525 400526]
    
    [11, 30]
    전체모델 발생 격자
    [393522 393525 393526 393527 393528 394524 394525 394526 394527 394528
     394529 395524 395525 395526 395527 395528 395530 395531 396523 396524
     396525 396526 396527 396528 396531 397524 397526 398524 398525 399525]
    
    


```python
final_grid_result_ex2 ("자전거사고", Random_Forset_model_BCYC_ACDNT_OCRN_CNT, last_day_df)
```

    자전거사고 격자 추출
    ******************************
    [1, 31]
    [396524 395527 395531 394529 395525]
    
    [2, 28]
    [396524 395527 395531 395525 394529]
    
    [3, 31]
    [395527 396524 395531 395528 396527]
    
    [4, 30]
    [395527 396524 395528 395531 396527]
    
    [5, 31]
    [395527 396524 394525 395526 395525]
    
    [6, 30]
    [396524 394525 395527 396525 394528]
    
    [7, 31]
    [395527 396524 394525 397525 396525]
    
    [8, 31]
    [395527 396524 394525 395531 395528]
    
    [9, 30]
    [396524 395527 395531 395528 396527]
    
    [10, 31]
    [396524 395527 395531 394529 396527]
    
    [11, 30]
    [395527 396524 395531 395525 393525]
    
    

>중독사고


```python
final_grid_result("중독사고", Random_Forset_model_POSNG_OCRN_CNT, 
                              XGBoost_model_POSNG_OCRN_CNT,
                              CatBoost_model_POSNG_OCRN_CNT, last_day_df)
```

    중독사고 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    []
    
    [2, 28]
    전체모델 발생 격자
    []
    
    [3, 31]
    전체모델 발생 격자
    []
    
    [4, 30]
    전체모델 발생 격자
    []
    
    [5, 31]
    전체모델 발생 격자
    []
    
    [6, 30]
    전체모델 발생 격자
    []
    
    [7, 31]
    전체모델 발생 격자
    []
    
    [8, 31]
    전체모델 발생 격자
    []
    
    [9, 30]
    전체모델 발생 격자
    []
    
    [10, 31]
    전체모델 발생 격자
    [394527 395527]
    
    [11, 30]
    전체모델 발생 격자
    []
    
    


```python
final_grid_result_ex2 ("중독사고", Random_Forset_model_POSNG_OCRN_CNT, last_day_df)
```

    중독사고 격자 추출
    ******************************
    [1, 31]
    []
    
    [2, 28]
    []
    
    [3, 31]
    []
    
    [4, 30]
    []
    
    [5, 31]
    []
    
    [6, 30]
    []
    
    [7, 31]
    []
    
    [8, 31]
    []
    
    [9, 30]
    []
    
    [10, 31]
    [396527 395527 394527 396524 393527]
    
    [11, 30]
    []
    
    

>추락사고


```python
final_grid_result("추락사고", Random_Forset_model_FALLING_OCRN_CNT, 
                              XGBoost_model_FALLING_OCRN_CNT,
                              CatBoost_model_FALLING_OCRN_CNT, last_day_df)
```

    추락사고 격자 추출
    ******************************
    [1, 31]
    전체모델 발생 격자
    [393526 393527 393528 394526 394527 394528 394529 395525 395526 395527
     395528 395530 395531 396524 396525 396526 396527 397523 397526]
    
    [2, 28]
    전체모델 발생 격자
    [393526 393527 393528 394526 394527 394528 394529 395525 395526 395527
     395528 395530 395531 396524 396525 396526 396527 397523 397526]
    
    [3, 31]
    전체모델 발생 격자
    []
    
    [4, 30]
    전체모델 발생 격자
    [386525 393526 394526 394527 394528 395525 395526 395527 395531 396524
     396525 396526 396527 397526]
    
    [5, 31]
    전체모델 발생 격자
    []
    
    [6, 30]
    전체모델 발생 격자
    [385526 386525 396527]
    
    [7, 31]
    전체모델 발생 격자
    [382521 383521 383522 384525 385525 386525 393526 393528 394526 394527
     394528 395525 395526 395527 395528 395530 395531 396524 396525 396526
     396527 397523 397526]
    
    [8, 31]
    전체모델 발생 격자
    [385526 386525 395526 395531 396525 396527 397526]
    
    [9, 30]
    전체모델 발생 격자
    [383521 385526 386525 393526 394526 394527 394528 395525 395526 395527
     395531 396524 396525 396526 396527 397526]
    
    [10, 31]
    전체모델 발생 격자
    [382521 383521 383522 384525 385525 386525 393526 393528 394526 394527
     394528 395525 395526 395527 395528 395530 395531 395532 396524 396525
     396526 396527 396533 396534 397523 397526]
    
    [11, 30]
    전체모델 발생 격자
    [386525 395531 396527 397526]
    
    


```python
final_grid_result_ex2 ("추락사고", Random_Forset_model_FALLING_OCRN_CNT, last_day_df)
```

    추락사고 격자 추출
    ******************************
    [1, 31]
    [396527 396525 394526 394527 396524]
    
    [2, 28]
    [395531 396527 394527 396524 394526]
    
    [3, 31]
    [385526 396527 395531 394527 396526]
    
    [4, 30]
    [395531 394527 396527 396526 394526]
    
    [5, 31]
    [385526 396527 395531 396526 396525]
    
    [6, 30]
    [385526 396527 396526 395531 396525]
    
    [7, 31]
    [395531 396527 396525 396526 394527]
    
    [8, 31]
    [385526 396527 396525 395531 396526]
    
    [9, 30]
    [385526 396525 396526 394527 396527]
    
    [10, 31]
    [396527 396525 394527 395531 394526]
    
    [11, 30]
    [396527 396525 395531 396526 394526]
    
    

# 결론 및 한계점

* 사고가 발생한 건수가 전체 데이터에 비해 너무 적어 다 보니 undersampling으로 처리하는데 어려움이 있었다.

* 각 사고별로 분석해 보니 시간이 부족하여 파라미터 미세조정
하지 못한 것에 대해 아쉬움이 있었다.

* 각 사고별로 모델을 구축하려다 보니 함수화의 중요성에 대하여 알게 되었다
가설에 필요한 데이터양이 적다 보니 어려움이 있었다.

* 유동인구가 없고 사건이 발생하지 않은 격자를 제거하다 보니
해당 격자의 사고 발생 예측이 어렵다는 한계가 있었다.





## 참고논문 및 사이트
*
