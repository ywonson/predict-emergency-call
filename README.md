# 2022년 제2회 소방안전 AI예측 경진대회

# 목차
## 0. 개요
## 1. 라이브러리 & 데이터확인
#### 라이브러리
#### 테이블정의서
#### 데이터확인
#### 종속변수확인
## 2. 데이터 전처리 (1)
#### 사고유형예측에 관련이 적은 데이터 제거
#### 날짜와 기상 관련 컬럼 추가
#### 휴가 컬럼 추가
#### 12월 데이터 누락으로 행 삭제
#### 계절 컬럼 추가
#### 기온/강수량/적설량/풍속/습도 컬럼 추가
## 3. EDA와 시각화
#### 일별 사건 발생 빈도 시각화
#### 독립변수와 종속변수간의 관계 파악
##### 함수정의
##### 유동인구와 사고의 연관성 시각화
##### 날씨와 사고의 연관성 시각화
##### 휴가철 사고 발생 현황 시각화
##### 기상조건과 사건발생의 연관성 시각화
#### 공간정보 데이터 시각화
##### 함수정의
##### 격자별 사고 발생 건수
#### 사건별 영향을 주는 변수 시각화 및 전처리
##### 산업단지
##### 유흥가
##### 경로당
##### 음식점
##### 건축허가현황
##### 교통사고정보
#### 데이터 전처리(2)
##### 파생변수 추가[유동인구분류]
##### 비대칭 데이터 정규화
##### 데이터 스케일링
##### 최종 변수 정리
##### 데이터 변수명 지정
## 4. 데이터 모델링
### 모델링
#### 파라미터 조정결과
### MCHN_ACDNT_OCRN_CNT 
### ETC_OCRN_CNT
### BLTRM_OCRN_CNT
### ACDNT_INJ_OCRN_CNT
### EXCL_DISEASE_OCRN_CNT
### VHC_ACDNT_OCRN_CNT
### HRFAF_OCRN_CNT
### DRKNSTAT_OCRN_CNT
### ANML_INSCT_ACDNT_OCRN_CNT
### FLPS_ACDNT_OCRN_CNT
### PDST_ACDNT_OCRN_CNT
### LACRTWND_OCRN_CNT
### MTRCYC_ACDNT_OCRN_CNT
### DRV_ACDNT_OCRN_CNT
### BCYC_ACDNT_OCRN_CNT
### POSNG_OCRN_CNT
### FALLING_OCRN_CNT
## 5.최종모델테스트
## 6. 결론 및 한계점
### 결론 및 한계점


## 라이브러리 & 데이터확인

### 라이브러리


```python
## 라이브러리 설치
# ! pip install pytimekr
# ! pip install pyproj
# ! pip install pyjanitor==0.23.1
# ! pip install optuna
# ! pip install imblearn
# ! pip install catboost
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pytimekr in /usr/local/lib/python3.7/dist-packages (0.1.0)
    Requirement already satisfied: lunardate>=0.1.5 in /usr/local/lib/python3.7/dist-packages (from pytimekr) (0.2.0)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pyproj in /usr/local/lib/python3.7/dist-packages (3.2.1)
    Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from pyproj) (2022.9.24)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pyjanitor==0.23.1 in /usr/local/lib/python3.7/dist-packages (0.23.1)
    Requirement already satisfied: multipledispatch in /usr/local/lib/python3.7/dist-packages (from pyjanitor==0.23.1) (0.6.0)
    Requirement already satisfied: natsort in /usr/local/lib/python3.7/dist-packages (from pyjanitor==0.23.1) (5.5.0)
    Requirement already satisfied: pandas-flavor in /usr/local/lib/python3.7/dist-packages (from pyjanitor==0.23.1) (0.2.0)
    Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from pyjanitor==0.23.1) (1.7.3)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from multipledispatch->pyjanitor==0.23.1) (1.15.0)
    Requirement already satisfied: xarray in /usr/local/lib/python3.7/dist-packages (from pandas-flavor->pyjanitor==0.23.1) (0.20.2)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from pandas-flavor->pyjanitor==0.23.1) (1.3.5)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->pandas-flavor->pyjanitor==0.23.1) (2022.6)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->pandas-flavor->pyjanitor==0.23.1) (2.8.2)
    Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.7/dist-packages (from pandas->pandas-flavor->pyjanitor==0.23.1) (1.21.6)
    Requirement already satisfied: typing-extensions>=3.7 in /usr/local/lib/python3.7/dist-packages (from xarray->pandas-flavor->pyjanitor==0.23.1) (4.1.1)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from xarray->pandas-flavor->pyjanitor==0.23.1) (4.13.0)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->xarray->pandas-flavor->pyjanitor==0.23.1) (3.10.0)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: optuna in /usr/local/lib/python3.7/dist-packages (3.0.3)
    Requirement already satisfied: importlib-metadata<5.0.0 in /usr/local/lib/python3.7/dist-packages (from optuna) (4.13.0)
    Requirement already satisfied: cmaes>=0.8.2 in /usr/local/lib/python3.7/dist-packages (from optuna) (0.9.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from optuna) (4.64.1)
    Requirement already satisfied: scipy<1.9.0,>=1.7.0 in /usr/local/lib/python3.7/dist-packages (from optuna) (1.7.3)
    Requirement already satisfied: colorlog in /usr/local/lib/python3.7/dist-packages (from optuna) (6.7.0)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.7/dist-packages (from optuna) (6.0)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from optuna) (21.3)
    Requirement already satisfied: alembic>=1.5.0 in /usr/local/lib/python3.7/dist-packages (from optuna) (1.8.1)
    Requirement already satisfied: cliff in /usr/local/lib/python3.7/dist-packages (from optuna) (3.10.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from optuna) (1.21.6)
    Requirement already satisfied: sqlalchemy>=1.3.0 in /usr/local/lib/python3.7/dist-packages (from optuna) (1.4.44)
    Requirement already satisfied: importlib-resources in /usr/local/lib/python3.7/dist-packages (from alembic>=1.5.0->optuna) (5.10.0)
    Requirement already satisfied: Mako in /usr/local/lib/python3.7/dist-packages (from alembic>=1.5.0->optuna) (1.2.4)
    Requirement already satisfied: typing-extensions>=3.6.4 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata<5.0.0->optuna) (4.1.1)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata<5.0.0->optuna) (3.10.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->optuna) (3.0.9)
    Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.7/dist-packages (from sqlalchemy>=1.3.0->optuna) (2.0.1)
    Requirement already satisfied: pbr!=2.1.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from cliff->optuna) (5.11.0)
    Requirement already satisfied: PrettyTable>=0.7.2 in /usr/local/lib/python3.7/dist-packages (from cliff->optuna) (3.5.0)
    Requirement already satisfied: autopage>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from cliff->optuna) (0.5.1)
    Requirement already satisfied: stevedore>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from cliff->optuna) (3.5.2)
    Requirement already satisfied: cmd2>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from cliff->optuna) (2.4.2)
    Requirement already satisfied: attrs>=16.3.0 in /usr/local/lib/python3.7/dist-packages (from cmd2>=1.0.0->cliff->optuna) (22.1.0)
    Requirement already satisfied: pyperclip>=1.6 in /usr/local/lib/python3.7/dist-packages (from cmd2>=1.0.0->cliff->optuna) (1.8.2)
    Requirement already satisfied: wcwidth>=0.1.7 in /usr/local/lib/python3.7/dist-packages (from cmd2>=1.0.0->cliff->optuna) (0.2.5)
    Requirement already satisfied: MarkupSafe>=0.9.2 in /usr/local/lib/python3.7/dist-packages (from Mako->alembic>=1.5.0->optuna) (2.0.1)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: imblearn in /usr/local/lib/python3.7/dist-packages (0.0)
    Requirement already satisfied: imbalanced-learn in /usr/local/lib/python3.7/dist-packages (from imblearn) (0.8.1)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from imbalanced-learn->imblearn) (1.2.0)
    Requirement already satisfied: scikit-learn>=0.24 in /usr/local/lib/python3.7/dist-packages (from imbalanced-learn->imblearn) (1.0.2)
    Requirement already satisfied: scipy>=0.19.1 in /usr/local/lib/python3.7/dist-packages (from imbalanced-learn->imblearn) (1.7.3)
    Requirement already satisfied: numpy>=1.13.3 in /usr/local/lib/python3.7/dist-packages (from imbalanced-learn->imblearn) (1.21.6)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.24->imbalanced-learn->imblearn) (3.1.0)
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: catboost in /usr/local/lib/python3.7/dist-packages (1.1.1)
    Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from catboost) (1.7.3)
    Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.7/dist-packages (from catboost) (1.21.6)
    Requirement already satisfied: graphviz in /usr/local/lib/python3.7/dist-packages (from catboost) (0.10.1)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from catboost) (3.2.2)
    Requirement already satisfied: plotly in /usr/local/lib/python3.7/dist-packages (from catboost) (5.5.0)
    Requirement already satisfied: pandas>=0.24.0 in /usr/local/lib/python3.7/dist-packages (from catboost) (1.3.5)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from catboost) (1.15.0)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24.0->catboost) (2022.6)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24.0->catboost) (2.8.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (1.4.4)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (0.11.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (3.0.9)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from kiwisolver>=1.0.1->matplotlib->catboost) (4.1.1)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.7/dist-packages (from plotly->catboost) (8.1.0)
    


```python
# 경고 무시 
import warnings
warnings.filterwarnings('ignore')
```


```python
# 기본 라이브러리
import re
import os
import json
import requests
from google.colab import drive
import pandas as pd
import numpy as np

# 쿼리 언어 라이브러리
import janitor

# 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

# 지도 시각화 라이브러리
import folium
from folium import plugins
from folium.features import DivIcon
import pyproj
from pyproj import Proj, transform

# 칼럼 추가를 위한 라이브러리

# 날짜 관련 
import datetime
from pytimekr import pytimekr

# 기상 관련
from collections import deque

# 모델 평가 라이브러리
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import plot_confusion_matrix, classification_report, f1_score

# 전처리 라이브러리
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.special import boxcox1p
from scipy.stats import norm
from scipy import stats

# 모델링 준비 라이브러리
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold

# 모델링 라이브러리
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost 

# 모델 파라미터 라이브러리
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
import optuna

# 모델 저장
import pickle
```


```python
# 구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
# colab으로 데이터 불러오기 
path = "/content/drive/MyDrive/Competitions/a firefighting competitions/공모전 서류/02.dataset.csv"
data_DF = pd.read_csv(path, encoding = 'euc-kr')
```

###테이블정의서

> 테이블 정의서

1. GRID_ID : 격자ID
2. GRID_X_AXIS : 격자X좌표
3. GRID_Y_AXIS : 격자Y좌표
4. OCRN_YMD : 발생일자
5. M00 : 남성10세미만
6. M10 : 남성10~14
7. M15 : 남성15~19
8. M20 : 남성20~24
9. M25 : 남성25~29
10. M30 : 남성30~34
11. M35 : 남성35~39
12. M40 : 남성40~44
13. M45 : 남성45~49
14. M50 : 남성50~54
15. M55 : 남성55~59
16. M60 : 남성60~64
17. M65 : 남성65~69
18. M70 : 남성70세이상
19. F00 : 여성10세미만
20. F10 : 여성10~14
21. F15 : 여성15~19
22. F20 : 여성20~24
23. F25 : 여성25~29
24. F30 : 여성30~34
25. F35 : 여성35~39
26. F40 : 여성40~44
27. F45 : 여성45~49
28. F50 : 여성50~54
29. F55 : 여성55~59
30. F60 : 여성60~64
31. F65 : 여성65~69
32. F70 : 여성70세이상
33. DONG_CD : 행정동코드
34. DONG_NM : 행정동명
35. HGTPOJ_ACDNT_OCRN_CNT :고온체사고발생건수
36. PNTRINJ_OCRN_CNT : 관통상발생건수
37. MCHN_ACDNT_OCRN_CNT : 기계사고발생건수
38. ETC_OCRN_CNT : 기타발생건수
39. BLTRM_OCRN_CNT : 둔상발생건수
40. ACDNT_INJ_OCRN_CNT : 사고부상발생건수
41. EXCL_DISEASE_OCRN_CNT : 질병외발생건수
42. VHC_ACDNT_OCRN_CNT : 탈것사고발생건수
43. HRFAF_OCRN_CNT : 낙상발생건수
44. AGRCMCHN_ACDNT_OCRN_CNT : 농기계사고발생건수
45. DRKNSTAT_OCRN_CNT : 단순주취발생건수
46. ANML_INSCT_ACDNT_OCRN_CNT : 동물곤충사고발생건수
47. FLPS_ACDNT_OCRN_CNT : 동승자사고발생건수
48. UNKNWN_OCRN_CNT : 미상발생건수
49. PDST_ACDNT_OCRN_CNT : 보행자사고발생건수
50. LACRTWND_OCRN_CNT : 열상발생건수
51. MTRCYC_ACDNT_OCRN_CNT : 오토바이사고발생건수
52. THML_DAMG_OCRN_CNT : 온열손상발생건수
53. DRV_ACDNT_OCRN_CNT : 운전자사고발생건수
54. DRWNG_OCRN_CNT : 익수발생건수
55. PRGNTW_ACDNT_OCRN_CNT : 임산부사고발생건수
56. BCYC_ACDNT_OCRN_CNT : 자전거사고발생건수
57. ELTRC_ACDNT_OCRN_CNT : 전기사고발생건수
58. POSNG_OCRN_CNT : 중독발생건수
59. ASPHYXIA_OCRN_CNT:질식발생건수
60. FALLING_OCRN_CNT : 추락발생건수
61. FLAME_OCRN_CNT : 화염발생건수
62. CHMC_SBSTNC_ACDNT_OCRN_CNT : 화학물질사고발생건수
63. WETHR_ACDNT_OCRN_CNT : 날씨사고발생건수
64. SXAL_ASALT_OCRN_CNT : 성폭행발생건수
65. BURN_OCRN_CNT : 화상발생건수


### 데이터 확인


```python
# 데이터 확인 
data_DF.head()
```





  <div id="df-1cae0dad-91ea-4c41-8912-143c1bc3a2bd">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRID_ID</th>
      <th>GRID_X_AXIS</th>
      <th>GRID_Y_AXIS</th>
      <th>OCRN_YMD</th>
      <th>M00</th>
      <th>M10</th>
      <th>M15</th>
      <th>M20</th>
      <th>M25</th>
      <th>M30</th>
      <th>...</th>
      <th>BCYC_ACDNT_OCRN_CNT</th>
      <th>ELTRC_ACDNT_OCRN_CNT</th>
      <th>POSNG_OCRN_CNT</th>
      <th>ASPHYXIA_OCRN_CNT</th>
      <th>FALLING_OCRN_CNT</th>
      <th>FLAME_OCRN_CNT</th>
      <th>CHMC_SBSTNC_ACDNT_OCRN_CNT</th>
      <th>WETHR_ACDNT_OCRN_CNT</th>
      <th>SXAL_ASALT_OCRN_CNT</th>
      <th>BURN_OCRN_CNT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>378508</td>
      <td>378475</td>
      <td>508475</td>
      <td>2021-01-01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>378509</td>
      <td>378475</td>
      <td>509475</td>
      <td>2021-01-01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.4</td>
      <td>0.00</td>
      <td>0.27</td>
      <td>0.22</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>378510</td>
      <td>378475</td>
      <td>510475</td>
      <td>2021-01-01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>378511</td>
      <td>378475</td>
      <td>511475</td>
      <td>2021-01-01</td>
      <td>0.08</td>
      <td>0.08</td>
      <td>0.2</td>
      <td>0.39</td>
      <td>0.16</td>
      <td>0.22</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>378512</td>
      <td>378475</td>
      <td>512475</td>
      <td>2021-01-01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1cae0dad-91ea-4c41-8912-143c1bc3a2bd')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1cae0dad-91ea-4c41-8912-143c1bc3a2bd button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1cae0dad-91ea-4c41-8912-143c1bc3a2bd');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# 데이터 기술 통계량 확인 
data_DF.describe()
```





  <div id="df-53592fc8-fdd5-4e1f-ad29-7894a96ec763">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRID_ID</th>
      <th>GRID_X_AXIS</th>
      <th>GRID_Y_AXIS</th>
      <th>M00</th>
      <th>M10</th>
      <th>M15</th>
      <th>M20</th>
      <th>M25</th>
      <th>M30</th>
      <th>M35</th>
      <th>...</th>
      <th>BCYC_ACDNT_OCRN_CNT</th>
      <th>ELTRC_ACDNT_OCRN_CNT</th>
      <th>POSNG_OCRN_CNT</th>
      <th>ASPHYXIA_OCRN_CNT</th>
      <th>FALLING_OCRN_CNT</th>
      <th>FLAME_OCRN_CNT</th>
      <th>CHMC_SBSTNC_ACDNT_OCRN_CNT</th>
      <th>WETHR_ACDNT_OCRN_CNT</th>
      <th>SXAL_ASALT_OCRN_CNT</th>
      <th>BURN_OCRN_CNT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>...</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.000000</td>
      <td>302168.0</td>
      <td>302168.0</td>
      <td>302168.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>393874.438084</td>
      <td>393826.635514</td>
      <td>523277.570093</td>
      <td>18.742905</td>
      <td>39.280078</td>
      <td>56.934226</td>
      <td>108.415764</td>
      <td>132.298048</td>
      <td>109.834970</td>
      <td>106.759611</td>
      <td>...</td>
      <td>0.000304</td>
      <td>0.000003</td>
      <td>0.000281</td>
      <td>0.000036</td>
      <td>0.000248</td>
      <td>0.000013</td>
      <td>0.000003</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000007</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9567.726970</td>
      <td>9566.993065</td>
      <td>9041.045158</td>
      <td>101.171821</td>
      <td>207.873529</td>
      <td>295.132547</td>
      <td>521.291677</td>
      <td>671.963387</td>
      <td>538.590764</td>
      <td>519.003592</td>
      <td>...</td>
      <td>0.017635</td>
      <td>0.001819</td>
      <td>0.016966</td>
      <td>0.006033</td>
      <td>0.015753</td>
      <td>0.003638</td>
      <td>0.001819</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.002573</td>
    </tr>
    <tr>
      <th>min</th>
      <td>378508.000000</td>
      <td>378475.000000</td>
      <td>505475.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>385534.750000</td>
      <td>385475.000000</td>
      <td>516475.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>392530.500000</td>
      <td>392475.000000</td>
      <td>522475.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>401516.250000</td>
      <td>401475.000000</td>
      <td>530475.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.530000</td>
      <td>1.090000</td>
      <td>0.820000</td>
      <td>1.050000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>418520.000000</td>
      <td>418475.000000</td>
      <td>544475.000000</td>
      <td>2530.480000</td>
      <td>3186.910000</td>
      <td>5969.100000</td>
      <td>9616.610000</td>
      <td>10751.320000</td>
      <td>9515.460000</td>
      <td>9995.430000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 63 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-53592fc8-fdd5-4e1f-ad29-7894a96ec763')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-53592fc8-fdd5-4e1f-ad29-7894a96ec763 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-53592fc8-fdd5-4e1f-ad29-7894a96ec763');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# 데이터 타입 및 결측값 여부 확인
data_DF.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 302168 entries, 0 to 302167
    Data columns (total 65 columns):
     #   Column                      Non-Null Count   Dtype  
    ---  ------                      --------------   -----  
     0   GRID_ID                     302168 non-null  int64  
     1   GRID_X_AXIS                 302168 non-null  int64  
     2   GRID_Y_AXIS                 302168 non-null  int64  
     3   OCRN_YMD                    302168 non-null  object 
     4   M00                         302168 non-null  float64
     5   M10                         302168 non-null  float64
     6   M15                         302168 non-null  float64
     7   M20                         302168 non-null  float64
     8   M25                         302168 non-null  float64
     9   M30                         302168 non-null  float64
     10  M35                         302168 non-null  float64
     11  M40                         302168 non-null  float64
     12  M45                         302168 non-null  float64
     13  M50                         302168 non-null  float64
     14  M55                         302168 non-null  float64
     15  M60                         302168 non-null  float64
     16  M65                         302168 non-null  float64
     17  M70                         302168 non-null  float64
     18  F00                         302168 non-null  float64
     19  F10                         302168 non-null  float64
     20  F15                         302168 non-null  float64
     21  F20                         302168 non-null  float64
     22  F25                         302168 non-null  float64
     23  F30                         302168 non-null  float64
     24  F35                         302168 non-null  float64
     25  F40                         302168 non-null  float64
     26  F45                         302168 non-null  float64
     27  F50                         302168 non-null  float64
     28  F55                         302168 non-null  float64
     29  F60                         302168 non-null  float64
     30  F65                         302168 non-null  float64
     31  F70                         302168 non-null  float64
     32  DONG_CD                     302168 non-null  int64  
     33  DONG_NM                     302168 non-null  object 
     34  HGTPOJ_ACDNT_OCRN_CNT       302168 non-null  float64
     35  PNTRINJ_OCRN_CNT            302168 non-null  float64
     36  MCHN_ACDNT_OCRN_CNT         302168 non-null  float64
     37  ETC_OCRN_CNT                302168 non-null  float64
     38  BLTRM_OCRN_CNT              302168 non-null  float64
     39  ACDNT_INJ_OCRN_CNT          302168 non-null  float64
     40  EXCL_DISEASE_OCRN_CNT       302168 non-null  float64
     41  VHC_ACDNT_OCRN_CNT          302168 non-null  float64
     42  HRFAF_OCRN_CNT              302168 non-null  float64
     43  AGRCMCHN_ACDNT_OCRN_CNT     302168 non-null  float64
     44  DRKNSTAT_OCRN_CNT           302168 non-null  float64
     45  ANML_INSCT_ACDNT_OCRN_CNT   302168 non-null  float64
     46  FLPS_ACDNT_OCRN_CNT         302168 non-null  float64
     47  UNKNWN_OCRN_CNT             302168 non-null  float64
     48  PDST_ACDNT_OCRN_CNT         302168 non-null  float64
     49  LACRTWND_OCRN_CNT           302168 non-null  float64
     50  MTRCYC_ACDNT_OCRN_CNT       302168 non-null  float64
     51  THML_DAMG_OCRN_CNT          302168 non-null  float64
     52  DRV_ACDNT_OCRN_CNT          302168 non-null  float64
     53  DRWNG_OCRN_CNT              302168 non-null  float64
     54  PRGNTW_ACDNT_OCRN_CNT       302168 non-null  float64
     55  BCYC_ACDNT_OCRN_CNT         302168 non-null  float64
     56  ELTRC_ACDNT_OCRN_CNT        302168 non-null  float64
     57  POSNG_OCRN_CNT              302168 non-null  float64
     58  ASPHYXIA_OCRN_CNT           302168 non-null  float64
     59  FALLING_OCRN_CNT            302168 non-null  float64
     60  FLAME_OCRN_CNT              302168 non-null  float64
     61  CHMC_SBSTNC_ACDNT_OCRN_CNT  302168 non-null  float64
     62  WETHR_ACDNT_OCRN_CNT        302168 non-null  float64
     63  SXAL_ASALT_OCRN_CNT         302168 non-null  float64
     64  BURN_OCRN_CNT               302168 non-null  float64
    dtypes: float64(59), int64(4), object(2)
    memory usage: 149.8+ MB
    

### 종속변수 확인

사고가 1건 이상 발생한 격자


```python
# 총 격자 개수 확인 (총 856개)
data_DF["GRID_ID"].value_counts()  ##
```




    378508    353
    397529    353
    397531    353
    397532    353
    397533    353
             ... 
    388523    353
    388524    353
    388525    353
    388526    353
    418520    353
    Name: GRID_ID, Length: 856, dtype: int64




```python
# 각 사고 발생 횟수
data_DF.iloc[:,34:]
```





  <div id="df-6fdb0e3d-4839-4974-9687-a3ae78a2c08c">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HGTPOJ_ACDNT_OCRN_CNT</th>
      <th>PNTRINJ_OCRN_CNT</th>
      <th>MCHN_ACDNT_OCRN_CNT</th>
      <th>ETC_OCRN_CNT</th>
      <th>BLTRM_OCRN_CNT</th>
      <th>ACDNT_INJ_OCRN_CNT</th>
      <th>EXCL_DISEASE_OCRN_CNT</th>
      <th>VHC_ACDNT_OCRN_CNT</th>
      <th>HRFAF_OCRN_CNT</th>
      <th>AGRCMCHN_ACDNT_OCRN_CNT</th>
      <th>...</th>
      <th>BCYC_ACDNT_OCRN_CNT</th>
      <th>ELTRC_ACDNT_OCRN_CNT</th>
      <th>POSNG_OCRN_CNT</th>
      <th>ASPHYXIA_OCRN_CNT</th>
      <th>FALLING_OCRN_CNT</th>
      <th>FLAME_OCRN_CNT</th>
      <th>CHMC_SBSTNC_ACDNT_OCRN_CNT</th>
      <th>WETHR_ACDNT_OCRN_CNT</th>
      <th>SXAL_ASALT_OCRN_CNT</th>
      <th>BURN_OCRN_CNT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>302163</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>302164</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>302165</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>302166</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>302167</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>302168 rows × 31 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6fdb0e3d-4839-4974-9687-a3ae78a2c08c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-6fdb0e3d-4839-4974-9687-a3ae78a2c08c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6fdb0e3d-4839-4974-9687-a3ae78a2c08c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
#  격자 위치별 사고 발생 
plt.figure(figsize = (10, 150))
plt.subplots_adjust(wspace=0.7)   # subplot끼리 겹치지않게 보여주기
n = 0
for i in range(34,65):
    # 1건이상 발생한 사고 격자별로 합치기
    df_grid = data_DF.loc[data_DF[data_DF.columns[i]] >= 1, ["OCRN_YMD","GRID_ID",'GRID_X_AXIS',"GRID_Y_AXIS",data_DF.columns[i]]]
    df_grid = df_grid.reset_index(drop=True)
    df_grid1 = df_grid.groupby(["GRID_ID","GRID_X_AXIS","GRID_Y_AXIS"], as_index=False)[data_DF.columns[i]].sum()
    #subplot 설정
    ax = plt.subplot(32, 2, n + 1)
    plt.tick_params( axis='both', which='both', right=False, left=False,  bottom=False, top=False, labelbottom=False, labelleft=False)
    sns.scatterplot(data=df_grid1, x="GRID_X_AXIS", y = "GRID_Y_AXIS", size = data_DF.columns[i], hue = data_DF.columns[i]).plot(ax=ax)
    
    if df_grid1.iloc[:,3].sum() != 0:
      plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #그래프 범주추가
    ax.set_title(data_DF.columns[i])
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    n += 1
```


    
![png](output_20_0.png)
    


## 데이터 전처리(1)

####사고유형예측에 관련이 적은 데이터 제거

**15개 이하로 발생한 사고의 경우 위 격자를 확인 한 결과 우연에 의한 사고일 가능성이 높기에 제거**

* 사고 제거 리스트
  1. HGTPOJ_ACDNT_OCRN_CNT       : 고온체사고발생건     ----> 4건
  2. PNTRINJ_OCRN_CNT            : 관통상발생건수       ----> 1건
  3. AGRCMCHN_ACDNT_OCRN_CNT     : 농기계사고발생건수   ----> 0건
  4. UNKNWN_OCRN_CNT             : 미상발생건수         ----> 5건
  5. THML_DAMG_OCRN_CNT          : 온열손상발생건수     ----> 6건
  6. DRWNG_OCRN_CNT              : 익수발생건수         ----> 13건
  7. PRGNTW_ACDNT_OCRN_CNT       : 임산부사고발생건수   ----> 8건
  8. ELTRC_ACDNT_OCRN_CNT        : 전기사고발생건수     ----> 1건
  9. ASPHYXIA_OCRN_CNT           : 질식발생건수         ----> 11건
  10. FLAME_OCRN_CNT             : 화염발생건수         ----> 4건
  11. CHMC_SBSTNC_ACDNT_OCRN_CNT : 화학물질사고발생건수 ----> 1건 
  12. WETHR_ACDNT_OCRN_CNT       : 날씨사고발생건수     ----> 0건
  13. SXAL_ASALT_OCRN_CNT        : 성폭행발생건수       ----> 0건
  14. BURN_OCRN_CNT              : 화상발생건수         ----> 2건


```python
# 사고 발생건수 15건 이하인 종속변수 제거
data_DF.drop(['HGTPOJ_ACDNT_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['PNTRINJ_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['AGRCMCHN_ACDNT_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['UNKNWN_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['THML_DAMG_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['DRWNG_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['PRGNTW_ACDNT_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['ELTRC_ACDNT_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['ASPHYXIA_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['FLAME_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['CHMC_SBSTNC_ACDNT_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['WETHR_ACDNT_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['SXAL_ASALT_OCRN_CNT'], axis=1, inplace=True)
data_DF.drop(['BURN_OCRN_CNT'], axis=1, inplace=True)
```


```python
# 종속변수 제거 확인 
data_DF.iloc[:, 34:].sum()
```




    MCHN_ACDNT_OCRN_CNT           27.0
    ETC_OCRN_CNT                 111.0
    BLTRM_OCRN_CNT                44.0
    ACDNT_INJ_OCRN_CNT           126.0
    EXCL_DISEASE_OCRN_CNT        239.0
    VHC_ACDNT_OCRN_CNT            28.0
    HRFAF_OCRN_CNT               981.0
    DRKNSTAT_OCRN_CNT             58.0
    ANML_INSCT_ACDNT_OCRN_CNT     67.0
    FLPS_ACDNT_OCRN_CNT          107.0
    PDST_ACDNT_OCRN_CNT          158.0
    LACRTWND_OCRN_CNT            176.0
    MTRCYC_ACDNT_OCRN_CNT        181.0
    DRV_ACDNT_OCRN_CNT           216.0
    BCYC_ACDNT_OCRN_CNT           92.0
    POSNG_OCRN_CNT                85.0
    FALLING_OCRN_CNT              75.0
    dtype: float64




```python
# 격자와 중복으로 구역을 표시하는 행정동 컬럼 삭제
data_DF.drop(["DONG_NM"], axis=1, inplace=True)
data_DF.drop(["DONG_CD"], axis=1, inplace=True)
```


```python
# 데이터 정리 확인 65건 -> 49건
data_DF.shape
```




    (302168, 49)




```python
data_DF.columns
```




    Index(['GRID_ID', 'GRID_X_AXIS', 'GRID_Y_AXIS', 'OCRN_YMD', 'M00', 'M10',
           'M15', 'M20', 'M25', 'M30', 'M35', 'M40', 'M45', 'M50', 'M55', 'M60',
           'M65', 'M70', 'F00', 'F10', 'F15', 'F20', 'F25', 'F30', 'F35', 'F40',
           'F45', 'F50', 'F55', 'F60', 'F65', 'F70', 'MCHN_ACDNT_OCRN_CNT',
           'ETC_OCRN_CNT', 'BLTRM_OCRN_CNT', 'ACDNT_INJ_OCRN_CNT',
           'EXCL_DISEASE_OCRN_CNT', 'VHC_ACDNT_OCRN_CNT', 'HRFAF_OCRN_CNT',
           'DRKNSTAT_OCRN_CNT', 'ANML_INSCT_ACDNT_OCRN_CNT', 'FLPS_ACDNT_OCRN_CNT',
           'PDST_ACDNT_OCRN_CNT', 'LACRTWND_OCRN_CNT', 'MTRCYC_ACDNT_OCRN_CNT',
           'DRV_ACDNT_OCRN_CNT', 'BCYC_ACDNT_OCRN_CNT', 'POSNG_OCRN_CNT',
           'FALLING_OCRN_CNT'],
          dtype='object')



유동인구가 없고 사고가 발생하지 않은 격자 제거


```python
# 유동인구가 없는 격자 찾기
pop_DF = pd.concat([data_DF.iloc[:, 0], data_DF.iloc[:, 5:32]], axis=1)
pdp_DF_sum = pop_DF.groupby("GRID_ID").sum() 
# 유동인구가 0인 격자 index
pop_zero = pdp_DF_sum[pdp_DF_sum.sum(axis=1) == 0].index
pop_zero = pd.DataFrame(pop_zero)
pop_zero
```





  <div id="df-d754893e-06e9-4ce2-abca-4790645159a0">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRID_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>378508</td>
    </tr>
    <tr>
      <th>1</th>
      <td>378510</td>
    </tr>
    <tr>
      <th>2</th>
      <td>378515</td>
    </tr>
    <tr>
      <th>3</th>
      <td>378516</td>
    </tr>
    <tr>
      <th>4</th>
      <td>379508</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>376</th>
      <td>417519</td>
    </tr>
    <tr>
      <th>377</th>
      <td>417520</td>
    </tr>
    <tr>
      <th>378</th>
      <td>418517</td>
    </tr>
    <tr>
      <th>379</th>
      <td>418519</td>
    </tr>
    <tr>
      <th>380</th>
      <td>418520</td>
    </tr>
  </tbody>
</table>
<p>381 rows × 1 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d754893e-06e9-4ce2-abca-4790645159a0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d754893e-06e9-4ce2-abca-4790645159a0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d754893e-06e9-4ce2-abca-4790645159a0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# 사고가 한번도 발생하지 않은 격자
acc_DF = pd.concat([data_DF.iloc[:, 0], data_DF.iloc[:, 32:]], axis=1)
acc_DF_sum = acc_DF.groupby("GRID_ID").sum() 

# 사고발생이 0인 격자 index
acc_zero = acc_DF_sum[acc_DF_sum.sum(axis=1) == 0].index
acc_zero = pd.DataFrame(acc_zero)
acc_zero
```





  <div id="df-8026997f-b1fd-4067-a5ae-301c72bb3e33">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRID_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>378508</td>
    </tr>
    <tr>
      <th>1</th>
      <td>378509</td>
    </tr>
    <tr>
      <th>2</th>
      <td>378510</td>
    </tr>
    <tr>
      <th>3</th>
      <td>378511</td>
    </tr>
    <tr>
      <th>4</th>
      <td>378512</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>614</th>
      <td>417521</td>
    </tr>
    <tr>
      <th>615</th>
      <td>418517</td>
    </tr>
    <tr>
      <th>616</th>
      <td>418518</td>
    </tr>
    <tr>
      <th>617</th>
      <td>418519</td>
    </tr>
    <tr>
      <th>618</th>
      <td>418520</td>
    </tr>
  </tbody>
</table>
<p>619 rows × 1 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8026997f-b1fd-4067-a5ae-301c72bb3e33')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8026997f-b1fd-4067-a5ae-301c72bb3e33 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8026997f-b1fd-4067-a5ae-301c72bb3e33');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# 사고가 발생하지 않으면서 유동인구가 0인 행 제거
all_index = pd.merge(acc_zero, pop_zero, on="GRID_ID", how="inner")
data_DF = pd.merge(data_DF, all_index, how="outer", indicator=True).query('_merge == "left_only"').drop(columns=['_merge']).reset_index(drop=True)
```

#### 날짜와 기상 관련 컬럼 추가

MONTH/DAY/SEASON/WEEKDAY/HOLIDAY 컬럼 추가

* MONTH : 월
* DAY : 일
* SEASON_SE_NM : 계절구분 
  1. SPRING : 1
  2. SUMMER : 2
  3. AUTUMN : 3
  4. WINTER : 4
* WEEKDAY : 요일 (월요일[0] ~ 일요일[6])
* HOLIDAY : 공휴일
  1. 공휴일 O : 1
  2. 공휴일 x : 0


```python
# MONTH, DAY, WEEKDAY 컬럼 추가
datatime = pd.to_datetime(data_DF.iloc[:, 3], format='%Y-%m-%d') # 컬럼 타입 변환
data_DF["MONTH"] = datatime.dt.month
data_DF["DAY"] = datatime.dt.day
data_DF["WEEKDAY"] = datatime.dt.weekday # 월[0] ~ 일[6]로 구성
```

####휴가컬럼추가

휴가철 사고발생 건수가 평상시 대비 증가하므로 컬럼 추가

```python
# HOLIDAY 컬럼 추가 
KR_holidays = pytimekr.holidays(year=2021) # 2021년 휴가 일자 
hol_list = []

# 공휴일 추가 
for KH in sorted(KR_holidays):
  KH = str(pd.to_datetime(KH, format='%Y-%m-%d')).split()[0]
  hol_list.append(KH)

# 2021년도 대체공휴일 추가
Replaced_holidays = ["2021-08-16", "2021-10-04", "2021-10-11"]
for RH in Replaced_holidays:
  hol_list.append(RH)

# 공휴일 : 1 / 평일 : 0
hol_df = pd.DataFrame(columns=['YMD','HOLIDAY'])
hol_df['YMD'] = sorted(hol_list)
hol_df['HOLIDAY'] = 1
hol_df.rename(columns = {'YMD' : 'OCRN_YMD'}, inplace=True)
```


```python
hol_df["OCRN_YMD"]
```




    0     2021-01-01
    1     2021-02-11
    2     2021-02-12
    3     2021-02-13
    4     2021-03-01
    5     2021-05-05
    6     2021-05-19
    7     2021-06-06
    8     2021-08-15
    9     2021-08-16
    10    2021-09-20
    11    2021-09-21
    12    2021-09-22
    13    2021-10-03
    14    2021-10-04
    15    2021-10-09
    16    2021-10-11
    17    2021-12-25
    Name: OCRN_YMD, dtype: object




```python
# 공휴일 DF 합치기
data_DF = pd.merge(data_DF, hol_df, on=['OCRN_YMD'], how='left').fillna(0)
```

#### 12월 데이터 누락으로 행 삭제


```python
# 12월 데이터 누락 데이터 확인 
data_DF[data_DF["MONTH"] == 12].sum(axis=0)[4:49]
```




    M00                          0.0
    M10                          0.0
    M15                          0.0
    M20                          0.0
    M25                          0.0
    M30                          0.0
    M35                          0.0
    M40                          0.0
    M45                          0.0
    M50                          0.0
    M55                          0.0
    M60                          0.0
    M65                          0.0
    M70                          0.0
    F00                          0.0
    F10                          0.0
    F15                          0.0
    F20                          0.0
    F25                          0.0
    F30                          0.0
    F35                          0.0
    F40                          0.0
    F45                          0.0
    F50                          0.0
    F55                          0.0
    F60                          0.0
    F65                          0.0
    F70                          0.0
    MCHN_ACDNT_OCRN_CNT          0.0
    ETC_OCRN_CNT                 0.0
    BLTRM_OCRN_CNT               0.0
    ACDNT_INJ_OCRN_CNT           0.0
    EXCL_DISEASE_OCRN_CNT        0.0
    VHC_ACDNT_OCRN_CNT           0.0
    HRFAF_OCRN_CNT               0.0
    DRKNSTAT_OCRN_CNT            0.0
    ANML_INSCT_ACDNT_OCRN_CNT    0.0
    FLPS_ACDNT_OCRN_CNT          0.0
    PDST_ACDNT_OCRN_CNT          0.0
    LACRTWND_OCRN_CNT            0.0
    MTRCYC_ACDNT_OCRN_CNT        0.0
    DRV_ACDNT_OCRN_CNT           0.0
    BCYC_ACDNT_OCRN_CNT          0.0
    POSNG_OCRN_CNT               0.0
    FALLING_OCRN_CNT             0.0
    dtype: object




```python
# 12월 행 삭제
drop_index_12 = data_DF[data_DF["MONTH"] == 12].index
data_DF = data_DF.drop(index = drop_index_12, axis=1).reset_index(drop=True)
data_DF.tail()
```





  <div id="df-54a364fe-eb67-4e91-a0f1-0b272ff3d3b8">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRID_ID</th>
      <th>GRID_X_AXIS</th>
      <th>GRID_Y_AXIS</th>
      <th>OCRN_YMD</th>
      <th>M00</th>
      <th>M10</th>
      <th>M15</th>
      <th>M20</th>
      <th>M25</th>
      <th>M30</th>
      <th>...</th>
      <th>LACRTWND_OCRN_CNT</th>
      <th>MTRCYC_ACDNT_OCRN_CNT</th>
      <th>DRV_ACDNT_OCRN_CNT</th>
      <th>BCYC_ACDNT_OCRN_CNT</th>
      <th>POSNG_OCRN_CNT</th>
      <th>FALLING_OCRN_CNT</th>
      <th>MONTH</th>
      <th>DAY</th>
      <th>WEEKDAY</th>
      <th>HOLIDAY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>153420</th>
      <td>418518</td>
      <td>418475</td>
      <td>518475</td>
      <td>2021-11-25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>25</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153421</th>
      <td>418518</td>
      <td>418475</td>
      <td>518475</td>
      <td>2021-11-26</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>26</td>
      <td>4</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153422</th>
      <td>418518</td>
      <td>418475</td>
      <td>518475</td>
      <td>2021-11-27</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>27</td>
      <td>5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153423</th>
      <td>418518</td>
      <td>418475</td>
      <td>518475</td>
      <td>2021-11-28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>28</td>
      <td>6</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153424</th>
      <td>418518</td>
      <td>418475</td>
      <td>518475</td>
      <td>2021-11-29</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>29</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-54a364fe-eb67-4e91-a0f1-0b272ff3d3b8')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-54a364fe-eb67-4e91-a0f1-0b272ff3d3b8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-54a364fe-eb67-4e91-a0f1-0b272ff3d3b8');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




####계절 컬럼 추가

계절과 사고의 연관성을 확인하기 위하여 컬럼 추가



```python
# 계절정보 컬럼 추가
# 봄 : 3월,4월,5월, 여름 : 6월,7월,8월, 가을 : 9월,10월,11월, 겨울 : 12월,1월,2월
def season_check(x):
    if x in [3, 4, 5]:
        season = 1 # 봄 : 1
    elif x in [6, 7, 8]:
        season = 2 # 여름 : 2
    elif x in [9, 10, 11]:
        season = 3 # 가을 : 3
    elif x in [12, 1, 2]:
        season = 4 # 겨울 : 4
    return season

data_DF['SEASON_SE_NM'] = data_DF['MONTH'].apply(season_check)
```

####기온/강수량/적설량/풍속/습도 컬럼 생성

* AVRG_TMPRT : 평균기온(°C)	
* DAY_RAINQTY : 강수량(mm)
* DAY_MSNF : 적설량(cm)
* AVRG_WS : 평균 풍속(m/s)
* AVRG_HUMIDITY : 평균 습도(%)

#####기상정보 불러오기


```python
# 기상 정보 : https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36

# 일별 기상정보 받아오기 
path = "/content/drive/MyDrive/Competitions/a firefighting competitions/기상 데이터/day_climate.csv"
climate_data_day = pd.read_csv(path, encoding = 'cp949')

# 시간별 기상정보 받아오기
path = "/content/drive/MyDrive/Competitions/a firefighting competitions/기상 데이터/time_climate.csv"
climate_data_time = pd.read_csv(path, encoding = 'cp949')
```

적설량 추가


```python
# 일별 기상정보에 적설량 정보가 없음
print(climate_data_day["일 최심적설(cm)"].value_counts())
print(climate_data_day["일 최심신적설(cm)"].value_counts())
print(climate_data_day["합계 3시간 신적설(cm)"].value_counts())
```

    Series([], Name: 일 최심적설(cm), dtype: int64)
    Series([], Name: 일 최심신적설(cm), dtype: int64)
    Series([], Name: 합계 3시간 신적설(cm), dtype: int64)
    


```python
# 시간별 기상정보에서의 일적설량 (cm)
DAY_MSNF_data = climate_data_time.groupby("일시")["적설(cm)"].sum()
```


```python
# 일별 기상정보와 일일 적설량 합치기
climate_data = pd.merge(climate_data_day, DAY_MSNF_data, on = "일시", how = "inner")
```

#####기상정보 추가


```python
# 기상정보 데이터 프레임 생성
climate_que = []
for i in climate_data.index:
    YMD = pd.to_datetime(str(climate_data.at[i, "일시"]), format='%Y-%m-%d')
    AVRG_TMPRT = climate_data.at[i, "평균기온(°C)"]
    DAY_RAINQTY = climate_data.at[i, "일강수량(mm)"]
    DAY_MSNF = climate_data.at[i, "적설(cm)"]
    AVRG_WS = climate_data.at[i, "평균 풍속(m/s)"]
    AVRG_HUMIDITY = climate_data.at[i, "평균 상대습도(%)"] 
    climate_que.append([YMD, int(AVRG_TMPRT), DAY_RAINQTY, int(DAY_MSNF), int(AVRG_WS), int(AVRG_HUMIDITY)])
climate_dict = pd.DataFrame(climate_que, columns = ["YMD", "AVRG_TMPRT", "DAY_RAINQTY", "DAY_MSNF", "AVRG_WS", "AVRG_HUMIDITY"])
```


```python
climate_dict
```





  <div id="df-9fb2aeb7-be0d-4b69-9959-d23acfed5a79">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YMD</th>
      <th>AVRG_TMPRT</th>
      <th>DAY_RAINQTY</th>
      <th>DAY_MSNF</th>
      <th>AVRG_WS</th>
      <th>AVRG_HUMIDITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-01</td>
      <td>-6</td>
      <td>NaN</td>
      <td>6</td>
      <td>0</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-02</td>
      <td>-4</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-03</td>
      <td>-5</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-04</td>
      <td>-3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-05</td>
      <td>-3</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>46</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>360</th>
      <td>2021-12-27</td>
      <td>-7</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>59</td>
    </tr>
    <tr>
      <th>361</th>
      <td>2021-12-28</td>
      <td>-3</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>72</td>
    </tr>
    <tr>
      <th>362</th>
      <td>2021-12-29</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>72</td>
    </tr>
    <tr>
      <th>363</th>
      <td>2021-12-30</td>
      <td>-2</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>45</td>
    </tr>
    <tr>
      <th>364</th>
      <td>2021-12-31</td>
      <td>-5</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
<p>365 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9fb2aeb7-be0d-4b69-9959-d23acfed5a79')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9fb2aeb7-be0d-4b69-9959-d23acfed5a79 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9fb2aeb7-be0d-4b69-9959-d23acfed5a79');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# 비가 오지 않은날(결측치) 0으로 바꿔주기
climate_dict["DAY_RAINQTY"] = climate_dict["DAY_RAINQTY"].fillna(0).apply(lambda x : int(x))
```


```python
# 기상정보 DF 합치기
climate_dict.rename(columns = {"YMD":"OCRN_YMD"}, inplace =True)
climate_dict.OCRN_YMD=climate_dict.OCRN_YMD.astype(str) # type 변경
data_DF = pd.merge(data_DF, climate_dict, on = ["OCRN_YMD"], how="left").fillna(0)
```


```python
# 데이터 전처리(1) 완료
data_DF.head()
```





  <div id="df-02eb7f00-6b3c-438c-ac83-b46cd74257c9">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRID_ID</th>
      <th>GRID_X_AXIS</th>
      <th>GRID_Y_AXIS</th>
      <th>OCRN_YMD</th>
      <th>M00</th>
      <th>M10</th>
      <th>M15</th>
      <th>M20</th>
      <th>M25</th>
      <th>M30</th>
      <th>...</th>
      <th>MONTH</th>
      <th>DAY</th>
      <th>WEEKDAY</th>
      <th>HOLIDAY</th>
      <th>SEASON_SE_NM</th>
      <th>AVRG_TMPRT</th>
      <th>DAY_RAINQTY</th>
      <th>DAY_MSNF</th>
      <th>AVRG_WS</th>
      <th>AVRG_HUMIDITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>378509</td>
      <td>378475</td>
      <td>509475</td>
      <td>2021-01-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.27</td>
      <td>0.22</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1.0</td>
      <td>4</td>
      <td>-6</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>71</td>
    </tr>
    <tr>
      <th>1</th>
      <td>378509</td>
      <td>378475</td>
      <td>509475</td>
      <td>2021-01-02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0.0</td>
      <td>4</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>378509</td>
      <td>378475</td>
      <td>509475</td>
      <td>2021-01-03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>0.0</td>
      <td>4</td>
      <td>-5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>378509</td>
      <td>378475</td>
      <td>509475</td>
      <td>2021-01-04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>-3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>378509</td>
      <td>378475</td>
      <td>509475</td>
      <td>2021-01-05</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0.0</td>
      <td>4</td>
      <td>-3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>46</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 59 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-02eb7f00-6b3c-438c-ac83-b46cd74257c9')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-02eb7f00-6b3c-438c-ac83-b46cd74257c9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-02eb7f00-6b3c-438c-ac83-b46cd74257c9');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## EDA 와 시각화

### 사고 기본 정보 파악


```python
# 각 사고 기본 통계량
data_DF.iloc[:, 34:51].describe()
```





  <div id="df-47a00b47-6801-48ee-8d90-7487e265ee0b">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BLTRM_OCRN_CNT</th>
      <th>ACDNT_INJ_OCRN_CNT</th>
      <th>EXCL_DISEASE_OCRN_CNT</th>
      <th>VHC_ACDNT_OCRN_CNT</th>
      <th>HRFAF_OCRN_CNT</th>
      <th>DRKNSTAT_OCRN_CNT</th>
      <th>ANML_INSCT_ACDNT_OCRN_CNT</th>
      <th>FLPS_ACDNT_OCRN_CNT</th>
      <th>PDST_ACDNT_OCRN_CNT</th>
      <th>LACRTWND_OCRN_CNT</th>
      <th>MTRCYC_ACDNT_OCRN_CNT</th>
      <th>DRV_ACDNT_OCRN_CNT</th>
      <th>BCYC_ACDNT_OCRN_CNT</th>
      <th>POSNG_OCRN_CNT</th>
      <th>FALLING_OCRN_CNT</th>
      <th>MONTH</th>
      <th>DAY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
      <td>153425.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.000287</td>
      <td>0.000821</td>
      <td>0.001558</td>
      <td>0.000182</td>
      <td>0.006394</td>
      <td>0.000378</td>
      <td>0.000437</td>
      <td>0.000697</td>
      <td>0.001030</td>
      <td>0.001147</td>
      <td>0.001180</td>
      <td>0.001408</td>
      <td>0.000600</td>
      <td>0.000554</td>
      <td>0.000489</td>
      <td>6.018576</td>
      <td>15.195046</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.016932</td>
      <td>0.028872</td>
      <td>0.040578</td>
      <td>0.013508</td>
      <td>0.082203</td>
      <td>0.019440</td>
      <td>0.020893</td>
      <td>0.026399</td>
      <td>0.032277</td>
      <td>0.034042</td>
      <td>0.034892</td>
      <td>0.037669</td>
      <td>0.024745</td>
      <td>0.023806</td>
      <td>0.022104</td>
      <td>3.154391</td>
      <td>8.493827</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>11.000000</td>
      <td>30.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-47a00b47-6801-48ee-8d90-7487e265ee0b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-47a00b47-6801-48ee-8d90-7487e265ee0b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-47a00b47-6801-48ee-8d90-7487e265ee0b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




###일별 사고 발생 빈도 시각화


```python
#365일 동안 일일 사고 발생 빈도 확인
fig, ax = plt.subplots(figsize=(30, 150), sharey=True, sharex=True)
plt.subplots_adjust(wspace=0.2,  hspace=0.5)
n = 0
for i in range(32, 49): 
    col_num = data_DF.columns[i]
    data_year = data_DF.pivot_table(values = col_num, index="DAY", columns = "MONTH", aggfunc=sum)   
    ax = plt.subplot(18, 3, n + 1)
    sns.heatmap(data=data_year, cmap="Blues").plot(ax=ax)
    ax.set_title(col_num)
    n += 1
```


    
![png](output_68_0.png)
    


### 독립변수와 종속변수간의 관계 파악

0.   함수정의
1.   유동인구와 사고의 관계 탐색
2.   날짜와 사고의 관계 탐색
3.   기상과 사고의 관계 탐색

#### 함수정의


```python
# 날짜(월, 일, 계절, 휴가일), 기상조건(온도, 강수량, 적설량, 풍속, 습도)의 사건과의 관계 탐색
# 함수 정의: 사건과의 상관 관계를 보이기 위한 그래프 플롯 함수 "plot_with_0" 정의

def plot_with_0(j):
  plt.figure(figsize = (20,150)) 

  plt.subplots_adjust(wspace=0.2, hspace=0.5)
  n = 0

  for i in range(32, 49): 

      df_temp1 = data_DF.loc[data_DF[data_DF.columns[i]] == 1]
      
      
      
      ax = plt.subplot(34, 2, n + 1)
      plt.tick_params( axis='both', which='both', right=False, left=True,  bottom=True, top=False, labelbottom=True, labelleft=True)
      sns.countplot(data=df_temp1, x=df_temp1.columns[j]).plot(ax=ax)

      ax.set_title(str(df_temp1.columns[i])+": 1") #그래프 제목 추가
      new_ticks = [i.get_text() for i in ax.get_xticklabels()]
      plt.xticks(range(0, len(new_ticks), 10), new_ticks[::10]) # len(new_ticks)에 따라 if 조건 넣어주기

      n += 1

      df_temp1 = data_DF.loc[data_DF[data_DF.columns[i]] == 0]
      
      ax = plt.subplot(34, 2, n + 1)
      plt.tick_params( axis='both', which='both', right=False, left=False,  bottom=True, top=False, labelbottom=True, labelleft=True)
      sns.countplot(data=df_temp1, x=df_temp1.columns[j]).plot(ax=ax)
      ax.set_title(str(df_temp1.columns[i])+": 0")
      
      new_ticks = [i.get_text() for i in ax.get_xticklabels()]
      plt.xticks(range(0, len(new_ticks), 10), new_ticks[::10])

      n += 1

```


```python
# 함수 정의: 값이 0에 몰려 있는 경우 0을 제외시키고 그래프를 그리는 함수 "plot_exc_0" 정의 
def plot_exc_0(j):
  plt.figure(figsize = (20,150))
  plt.subplots_adjust(wspace=0.2, hspace=0.5)
  n = 0

  for i in range(32, 49): 

    i_name = data_DF.columns[i]
    j_name = data_DF.columns[j]

    df_temp1 = data_DF[(data_DF[i_name] == 1) & (data_DF[j_name] >= 1)] # 사고가 발생한 격자에 독립변수가 1 이상의 값을 가지는 경우

    if df_temp1.shape[0] != 0:    
            
      ax = plt.subplot(32, 2, n + 1)
      plt.tick_params( axis='both', which='both', right=False, left=True,  bottom=True, top=False, labelbottom=True, labelleft=True)
      sns.countplot(data=df_temp1, x=df_temp1.columns[j]).plot(ax=ax)

      ax.set_title(str(df_temp1.columns[i])+": 1")
      new_ticks = [i.get_text() for i in ax.get_xticklabels()]
      plt.xticks(range(0, len(new_ticks), 2), new_ticks[::10])

      n += 1
      # 독립변수가 사고발생에 영향을 주지 않은 경우
    else:     
      df_temp1 = data_DF.loc[data_DF[data_DF.columns[i]] == 1]

      ax = plt.subplot(32, 2, n + 1)
      plt.tick_params( axis='both', which='both', right=False, left=False,  bottom=True, top=False, labelbottom=True, labelleft=True)
      sns.countplot(data=df_temp1, x=df_temp1.columns[j]).plot(ax=ax)
      
      ax.set_title(str(df_temp1.columns[i])+": 0")
      new_ticks = [i.get_text() for i in ax.get_xticklabels()]
      plt.xticks(range(0, len(new_ticks), 10), new_ticks[::10])

      n += 1


    df_temp1 = data_DF[(data_DF[i_name] == 0) & (data_DF[j_name] >= 1)] # 사고가 발생하지 않은 격자에 독립변수가 1이상의 값을 가지는 경우
        
    if df_temp1.shape[0] != 0:    
          
      ax = plt.subplot(32, 2, n + 1)
      plt.tick_params( axis='both', which='both', right=False, left=False,  bottom=True, top=False, labelbottom=True, labelleft=True)
      sns.countplot(data=df_temp1, x=df_temp1.columns[j]).plot(ax=ax)
      
      ax.set_title(str(df_temp1.columns[i])+": 0")
      new_ticks = [i.get_text() for i in ax.get_xticklabels()]
      plt.xticks(range(0, len(new_ticks), 10), new_ticks[::10])

      n += 1
    else:
      df_temp1 = data_DF.loc[data_DF[data_DF.columns[i]] == 0]
                  
      ax = plt.subplot(34, 2, n + 1)
      plt.tick_params( axis='both', which='both', right=False, left=False,  bottom=True, top=False, labelbottom=True, labelleft=True)
      sns.countplot(data=df_temp1, x=df_temp1.columns[j]).plot(ax=ax)
      
      ax.set_title(str(df_temp1.columns[i])+": 0")
      new_ticks = [i.get_text() for i in ax.get_xticklabels()]
      plt.xticks(range(0, len(new_ticks), 10), new_ticks[::10])

      n += 1

```

#### 유동인구와 사고의 연관성 시각화


```python
# 사고가 발생 한 장소에 대한 유동인구 비율 시각화 
fig, axes = plt.subplots(9, 2, figsize=(35, 60), sharey=True)
plt.subplots_adjust(wspace=0.1,  hspace=0.35)

sns.set_theme(style="white", context="talk")

row, col = 0, 0

for i in range(32, 49): 
    col_num = data_DF.columns[i]
    df_temp = data_DF.loc[data_DF[col_num] >= 1]
    df_group = df_temp.groupby(["GRID_ID", col_num]).mean()
    df_col = df_group.iloc[:, 2:30].columns
    sex_age_mean = df_group.iloc[:, 2:30].T.mean(axis=1)            

    if col > 1:
        row += 1
        col = 0
        sns.barplot(x=df_col, y=sex_age_mean, ax=axes[row, col])
        axes[row, col].set_title(col_num)
        col += 1

    else : 
        sns.barplot(x=df_col, y=sex_age_mean, ax=axes[row, col])
        axes[row, col].set_title(col_num)
        col += 1
```


    
![png](output_74_0.png)
    


#### 날짜와 사고의 연관성 시각화

월별 사고 발생 현황 시각화


```python
# 월별 사고 발생 현황
month_index = data_DF.columns.get_loc("MONTH")
plot_with_0(month_index)
```


    
![png](output_77_0.png)
    


일별 사고 발생 현황 시각화


```python
# 일별 사고 발생 현황
day_index = data_DF.columns.get_loc("DAY")
plot_with_0(day_index)
```


    
![png](output_79_0.png)
    


요일별 사고발생 현황 시각화


```python
# 요일별 사고 발생 현황 (0: 월요일 ~ 6: 일요일)
week_index = data_DF.columns.get_loc("WEEKDAY")
plot_with_0(week_index)
```


    
![png](output_81_0.png)
    


계절별 사고 발생 현황 시각화


```python
# 계절별 사고 발생 현황 (1: 봄 ~ 4: 겨울)
season_index = data_DF.columns.get_loc("SEASON_SE_NM")
plot_with_0(season_index)
```


    
![png](output_83_0.png)
    


####휴가철 사고 발생 현황 시각화


```python
# 휴가철 사고 발생 현황
holiday_index = data_DF.columns.get_loc("HOLIDAY")
plot_with_0(holiday_index)
```


    
![png](output_85_0.png)
    



```python
# 휴가철 사고 발생 현황 (평일 제거)
plot_exc_0(holiday_index)
```


    
![png](output_86_0.png)
    


####기상조건과 사고발생의 연관성 시각화

온도와 사고발생의 연관성 시각화


```python
# 온도와 사고별 연관성
tmprt_index = data_DF.columns.get_loc("AVRG_TMPRT")
plot_with_0(tmprt_index)
```


    
![png](output_89_0.png)
    


강수량과 사고발생의 연관성 시각화


```python
# 비와 사건별 연관성
rain_index = data_DF.columns.get_loc("DAY_RAINQTY")
plot_with_0(rain_index)
```


    
![png](output_91_0.png)
    



```python
# 강수량과 사고 발생의 연관성 (0 제거)
plot_exc_0(rain_index)
```


    
![png](output_92_0.png)
    


적설량과 사고발생의 연관성 시각화


```python
# 눈과 사고의 연관성
snow_index = data_DF.columns.get_loc("DAY_MSNF")
plot_with_0(snow_index)
```


    
![png](output_94_0.png)
    



```python
# 적설량과 사고 발생의 연관성 (0 제거)
plot_exc_0(snow_index)
```


    
![png](output_95_0.png)
    


풍속과 사고발생의 연관성 시각화


```python
# 풍속과 사고 발생의 연관성
wind_index = data_DF.columns.get_loc("AVRG_WS")
plot_with_0(wind_index)
```


    
![png](output_97_0.png)
    


습도와 사고발생의 연관성 시각화


```python
# 습도와 사고 발생의 연관성
humidity_index = data_DF.columns.get_loc("AVRG_HUMIDITY")
plot_with_0(humidity_index)
```


    
![png](output_99_0.png)
    


### 공간정보 데이터 시각화 


함수정의


```python
# GPS좌표 형식 변환 함수 (KATEC -> WGS804) 
def katec_to_wgs84(x, y):
    inProj  = Proj('+proj=tmerc +lat_0=38 +lon_0=128 +k=0.9999 +x_0=400000 +y_0=600000 +ellps=bessel +units=m +no_defs +towgs84=-115.80,474.99,674.11,1.16,-2.31,-1.63,6.43')
    outProj = Proj({'proj':'latlong', 'datum':'WGS84', 'ellps':'WGS84' })
    return transform( inProj, outProj, x, y)
```

####격자별 사고 발생 건수


```python
# 사고별 좌표
data_DF['TOTAL_CNT'] = data_DF.loc[:,'MCHN_ACDNT_OCRN_CNT':'FALLING_OCRN_CNT'].sum(axis=1)
total_DF = data_DF[['GRID_ID', 'GRID_X_AXIS', 'GRID_Y_AXIS','TOTAL_CNT']].groupby(['GRID_ID', 'GRID_X_AXIS', 'GRID_Y_AXIS']).sum().reset_index()
```


```python
# 사고 발생 건수
data_DF['TOTAL_CNT'].value_counts()
```




    0.0    150858
    1.0      2373
    2.0       184
    3.0        10
    Name: TOTAL_CNT, dtype: int64




```python
# 격자별 사고 건수 
pd.DataFrame(total_DF["TOTAL_CNT"].value_counts()).T
```





  <div id="df-105c095c-9a01-44e9-ab6b-f9e3a6b5846b">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0</th>
      <th>1.0</th>
      <th>2.0</th>
      <th>3.0</th>
      <th>4.0</th>
      <th>5.0</th>
      <th>8.0</th>
      <th>12.0</th>
      <th>6.0</th>
      <th>10.0</th>
      <th>...</th>
      <th>98.0</th>
      <th>61.0</th>
      <th>49.0</th>
      <th>28.0</th>
      <th>91.0</th>
      <th>71.0</th>
      <th>54.0</th>
      <th>105.0</th>
      <th>44.0</th>
      <th>100.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TOTAL_CNT</th>
      <td>238</td>
      <td>71</td>
      <td>37</td>
      <td>23</td>
      <td>16</td>
      <td>9</td>
      <td>9</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 45 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-105c095c-9a01-44e9-ab6b-f9e3a6b5846b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-105c095c-9a01-44e9-ab6b-f9e3a6b5846b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-105c095c-9a01-44e9-ab6b-f9e3a6b5846b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




모든 사고발생 건수 격자지도 시각화


```python
def f_color(x):
    if x > 70:        # 70건이상 빨강색
        a = '#FF4F00'     
    elif x > 30:      # 30건이상 주황색   
        a = '#FCB100' 
    elif x > 10:      # 10건이상 노란색
        a = '#E0F500' 
    elif x > 1:       # 1건이상 연두색
        a = '#8CF700'  
    elif x >= 0:      # 0건  회색
        a = '#C6C6C6'
    else :            # 유동인구 0이며 사건이 0건인 경우 색 X
        pass 
    return a
```


```python
# 모든 사고 건수 격자 지도화 확인 
total_DF['cell_color']= total_DF['TOTAL_CNT'].apply(lambda x : f_color(x))
geodf = total_DF.rename(columns = {'GRID_ID' : 'id','GRID_X_AXIS':'x', 'GRID_Y_AXIS':'y', 'TOTAL_CNT':'total'})
geodf = geodf[['id','x','y','cell_color','total']]

# 원주 격자 좌표화 
cell_size = 1000
a = cell_size//2
geodf['nwx'], geodf['nwy'] = geodf['x']-a, geodf['y']+a
geodf['nex'], geodf['ney'] = geodf['x']+a, geodf['y']+a
geodf['swx'], geodf['swy'] = geodf['x']-a, geodf['y']-a
geodf['sex'], geodf['sey'] = geodf['x']+a,geodf['y']-a

geodf['lng'], geodf['lat'] = katec_to_wgs84(geodf.x.to_list(), geodf.y.to_list())
geodf['nwlng'], geodf['nwlat'] = katec_to_wgs84(geodf.nwx.to_list(), geodf.nwy.to_list())
geodf['nelng'], geodf['nelat'] = katec_to_wgs84(geodf.nex.to_list(), geodf.ney.to_list())
geodf['swlng'], geodf['swlat'] = katec_to_wgs84(geodf.swx.to_list(), geodf.swy.to_list())
geodf['selng'], geodf['selat'] = katec_to_wgs84(geodf.sex.to_list(), geodf.sey.to_list())

map = folium.Map(location=[geodf['lat'].mean(), geodf['lng'].mean()], zoom_start=11, tiles="OpenStreetMap")

geodf.apply(lambda x : folium.Polygon(locations=[[x.nwlat, x.nwlng], [x.swlat,x.swlng], [x.selat,x.selng], [x.nelat,x.nelng]], 
                                        color='white', 
                                        popup = 'id:'+ str(x.id),
                                        weight=0.1,
                                        fill=True,
                                        fill_color=x.cell_color,
                                        fill_opacity=0.6
                                        ).add_to(map), axis =1)

# 주로 사고가 발생되는 장소 : 원주 기업도시 / 원주 도심지역 / 문막읍 / 흥업리 / 신림면 / 태장농공단지
folium.Marker(location=[37.376218, 127.867677], popup="원주 기업도시", icon=folium.Icon(color="red", icon="info-sign"),).add_to(map)
folium.Marker(location=[37.340229, 127.935455], popup="원주 도심지역", icon=folium.Icon(color="blue", icon="info-sign"),).add_to(map)
folium.Marker(location=[37.313082, 127.818659], popup="문막읍", icon=folium.Icon(color="green", icon="info-sign"),).add_to(map)
folium.Marker(location=[37.30817, 127.918924], popup="흥업리", icon=folium.Icon(color="black", icon="info-sign"),).add_to(map)
folium.Marker(location=[37.232082, 128.082054], popup="신림면", icon=folium.Icon(color="pink", icon="info-sign"),).add_to(map)
folium.Marker(location=[37.403312, 127.946698], popup="태장농공단지", icon=folium.Icon(color="orange", icon="info-sign"),).add_to(map)

map
```


### 사고별 영향을 주는 변수 시각화 및 전처리

함수 정의


```python
# kakao api를 사용한 주소 위도 경도 추출 
def getLatLng(addr):
  url = 'https://dapi.kakao.com/v2/local/search/address.json?query='+addr
  headers = {'Authorization' : 'KakaoAK key'}
  result = json.loads(str(requests.get(url,headers=headers).text))
   
  # 주소를 제대로 변환하지 못하면 정규화 
  if result["documents"] == [] :
      new_addr = address_change(addr)
      url = 'https://dapi.kakao.com/v2/local/search/address.json?query=' + new_addr
      headers = {'Authorization' : 'KakaoAK key'}
      result = json.loads(str(requests.get(url,headers=headers).text))

  # 주소 자체가 카카오에 없는 경우 null값 처리 후 직접 찾거나 제거
  if result["documents"] == [] :
    return 0, 0
  match_first = result['documents'][0]['address']
  return float(match_first['y']), float(match_first['x'])
```


```python
# 위도 경도 변환 함수
def wgs84_to_katec(x, y):
    inProj = Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    outProj  = Proj("+proj=tmerc +lat_0=38 +lon_0=128 +k=0.9999 +x_0=400000 +y_0=600000 +ellps=bessel +units=m +no_defs +towgs84=-115.80,474.99,674.11,1.16,-2.31,-1.63,6.43")
    return transform(inProj, outProj, x, y)
```


```python
# 주소 변환 정규표현식
def address_change (addr):
  regex = r'(\w+[원,산,남,울,북,천,주,기,시,도]\s*)?' \
          r'(\w+[구,시,군]\s*)?(\w+[구,시]\s*)?' \
          r'(\w+[면,읍]\s*)' \
          r'?(\w+\d*\w*[동,리,로,길]\s*)' \
          r'?(\w*\d+-?\d*)?'
  return re.search(regex, addr)[0]
```


```python
# 주소 좌표 추출 함수
def grid_check (data, name, address):
    grid_list = []
    for num in range(len(data)):
        country_name = data[name][num]
        position = getLatLng(data[address][num])
        grid = wgs84_to_katec(position[1], position[0])
        grid_list.append([country_name, round(grid[0]), round(grid[1])])
    return grid_list
```


```python
# 각 좌표 ID(1km 이내)에 있는 공장 수를 각 GRID_ID에 총합으로 나타내기
def xy_to_ID(DF): # DF = MCHN_factory_df
  df_place = DF
  df_place["count"] = 1
  geodf = total_DF.rename(columns = {'GRID_ID' : 'id','GRID_X_AXIS':'x', 'GRID_Y_AXIS':'y'})

  # x범위: x_min 이상 x_max 이하
  geodf["x_min"] = geodf["x"] - 500
  geodf["x_max"] = geodf["x"] + 500

  # y범위: y_min 이상 y_max 이하
  geodf["y_min"] = geodf["y"] - 500
  geodf["y_max"] = geodf["y"] + 500

  # 범위에 해당하는 격자로 합치기
  df_place = df_place.conditional_join(
      geodf, 
      ('GRID_X', 'x_min', '>='), 
      ('GRID_X', 'x_max', '<='),
      ('GRID_Y', 'y_min', '>='), 
      ('GRID_Y', 'y_max', '<='),

      how = 'right'
  )
  # 장소와 격자 ID를 합친 데이터의 모습
  resul_df = df_place.groupby(['id'], as_index=False)['count'].sum()
  print(resul_df.sort_values(["count"], ascending=False)) # ID 위치에 존재하는 장소의 수

  return resul_df # 격자 ID 순서로 배열됨
```

#### 산업단지

> 가설 : 산업단지에서 위험한 작업이나 기계, 장비로 인한 크고 작은 사고가 많이 발생할 것을 예상



```python
# 강원도 원주시_산업단지 입주기업체 데이터 호출 (출처 : 강원도 원주시 공공데이터)
# MCHN_PATH = "/content/drive/MyDrive/Competitions/a firefighting competitions/건물정보/강원도 원주시_산업단지 입주기업체 정보_20211031.csv"
# MCHN_DF = pd.read_csv(MCHN_PATH, encoding = "cp949")
```


```python
# 강원도 원주시_산업단지 입주기업체 정보에서 주소값을 카카오 Api를 사용하여 위도, 경도 추출
# MCHN_list = grid_check(MCHN_DF, "업체명", "공장대표주소(지번)") 

# API 데이터를 CSV로 파일화 후 사용
# MCHN_factory_df = pd.DataFrame(MCHN_list, columns = ["업체명", "GRID_X", "GRID_Y"]).dropna()
# MCHN_factory_df.to_csv("MCHN_factory_df.csv", encoding="cp949")

# 산업단지 데이터 호출
factory_path = "/content/drive/MyDrive/Competitions/a firefighting competitions/data/factory_df.csv"
factory_df = pd.read_csv(factory_path, encoding="cp949",index_col=0)
factory_df_counted = xy_to_ID(factory_df)
factory_df_counted.rename(columns = {'id':'GRID_ID', 'count':'INDUSTRIAL_CNT'}, inplace=True) 

# 산업단지 DF 합치기
data_DF = pd.merge(data_DF, factory_df_counted, on=['GRID_ID'], how='left')
```

             id  count
    294  395533  135.0
    60   383521   38.0
    110  386525   34.0
    44   382521   31.0
    95   385525   27.0
    ..      ...    ...
    159  389527    0.0
    158  389526    0.0
    157  389525    0.0
    156  389524    0.0
    474  418518    0.0
    
    [475 rows x 2 columns]
    

산업단지에서 발생한 사고 시각화


```python
# 산업단지에서 발생한 사고의 종류에 따른 시각화
factory_index = data_DF.columns.get_loc("INDUSTRIAL_CNT")
plot_exc_0(factory_index)
```


    
![png](output_124_0.png)
    


#### 유흥가 및 단란주점

> 가설 : 술이 크고 작은 사고를 유발할 수 있을 것이라고 예상


```python
# 강원도 원주시_단란주점 정보 데이터 불러오기 (출처 : 강원도 원주시 공공데이터)
# 2022년 이후 인허가 제거
# KARAOKE_PATH = "/content/drive/MyDrive/Competitions/a firefighting competitions/data/강원도 원주시_단란주점 정보_20220823.csv"
# KARAOKE_Data = pd.read_csv(KARAOKE_PATH, encoding = "cp949")

# 강원도 원주시_유흥주점 정보 데이터 불러오기 (출처 : 강원도 원주시 공공데이터)
# 2022년 이후 인허가 제거
# ENTERTAIN_PATH = "/content/drive/MyDrive/Competitions/a firefighting competitions/data/강원도 원주시_유흥주점정보_20221013.csv"
# ENTERTAIN_Data = pd.read_csv(ENTERTAIN_PATH, encoding = "cp949")
```

유흥주점과 단란주점 데이터 합치기


```python
# 단란주점 위도 경도 추출
# KARAOKE_list = grid_check(KARAOKE_Data, "업소명", "소재지(지번)") 

# 유흥주점 위도 경도 추출
# ENTERTAIN_list = grid_check(ENTERTAIN_Data, "업소명", "소재지(지번)") 
```


```python
# 단란주점, 유흥주점 DF 변환 및 GRID_ID에 총합
# KARAOKE_df = pd.DataFrame(KARAOKE_list, columns = ["업체명", "GRID_X", "GRID_Y"]).dropna()
# ENTERTAIN_df = pd.DataFrame(ENTERTAIN_list, columns = ["업체명", "GRID_X", "GRID_Y"]).dropna()
```


```python
# 단란주점, 유흥주점 합치기 
# bar_df = pd.concat([KARAOKE_df, ENTERTAIN_df], axis=0).reset_index(drop=True)

# API 데이터를 CSV로 파일화 후 저장 후 사용
# bar_df.to_csv("bar_df.csv", encoding="cp949")

# 주점 데이터 호출
bar_path = "/content/drive/MyDrive/Competitions/a firefighting competitions/data/bar_df.csv"
bar_df = pd.read_csv(bar_path, encoding="cp949",index_col=0)
bar_df_counted = xy_to_ID(bar_df)
bar_df_counted.rename(columns = {'id':'GRID_ID', 'count':'BAR_CNT'}, inplace=True) 

# 주점 DF 합치기
data_DF = pd.merge(data_DF, bar_df_counted, on=['GRID_ID'], how='left')
```

             id  count
    267  394527   95.0
    289  395528   54.0
    288  395527   41.0
    246  393527   32.0
    303  396527   24.0
    ..      ...    ...
    156  389524    0.0
    155  389523    0.0
    154  389513    0.0
    153  389510    0.0
    474  418518    0.0
    
    [475 rows x 2 columns]
    

주점에서 발생한 사고 시각화


```python
# 주점에서 발생한 사고의 종류별 시각화
bar_index = data_DF.columns.get_loc("BAR_CNT")
plot_exc_0(bar_index)
```


    
![png](output_135_0.png)
    


#### 경로당

> 가설 : 노인인구와 사고의 상관관계를 알아보기 위하여 노인인구가 많을 것으로 예상되는 경로당의 사고 분석 시각화

데이터 불러오기


```python
# 강원도 경로당 현황 데이터 호출 ( 출처 : 강원도 원주시청 )
# SENIOR_PATH = "/content/drive/MyDrive/Competitions/a firefighting competitions/data/경로당 현황(2021).csv"
# SENIOR_Data = pd.read_csv(SENIOR_PATH, encoding = "cp949")
```


```python
# 경로당 위도 경도 추출
# SENIOR_list = grid_check(SENIOR_Data, "경로당 명", "소재지")
```


```python
# 경로당 데이터 프레임 변환
# SENIOR_df = pd.DataFrame(SENIOR_list, columns = ["시설명", "GRID_X", "GRID_Y"]).dropna()

# API 데이터를 CSV로 파일화 후 저장 후 사용
# SENIOR_df.to_csv("senior_df.csv", encoding="cp949")

# 경로당 데이터 호출 및 GRID_ID에 총합
senior_path = "/content/drive/MyDrive/Competitions/a firefighting competitions/data/senior_df.csv"
senior_df = pd.read_csv(senior_path, encoding="cp949",index_col=0)
senior_df_counted = xy_to_ID(senior_df)
senior_df_counted.rename(columns = {'id':'GRID_ID', 'count':'SENIOR_CENTER_CNT'}, inplace=True) 

# 경로당 DF 합치기
data_DF = pd.merge(data_DF, senior_df_counted, on=['GRID_ID'], how='left')
```

             id  count
    286  395525   15.0
    267  394527   12.0
    268  394528   11.0
    300  396524   11.0
    288  395527   10.0
    ..      ...    ...
    170  390512    0.0
    169  390511    0.0
    168  390508    0.0
    166  389535    0.0
    474  418518    0.0
    
    [475 rows x 2 columns]
    

경로당 근처에서 발생한 사고 시각화


```python
# 경로당 근처에서 발생한 사고 시각화
senior_index = data_DF.columns.get_loc("SENIOR_CENTER_CNT")
plot_exc_0(senior_index)
```


    
![png](output_143_0.png)
    


#### 음식점

> 가설 : 유동인구가 많을수록 사고발생 확률이 높아지므로 
음식점 근처에 유동인구가 많을 것이라고 예상


데이터 불러오기


```python
# 강원도 원주시_식당기본정보 데이터 호출 ( 출처 : 강원도 원주시 공공데이터 )
# REST_PATH = "/content/drive/MyDrive/Competitions/a firefighting competitions/data/강원도 원주시_식당기본정보.csv"
# REST_Data = pd.read_csv(REST_PATH, encoding = "utf-8")
```


```python
# 데이터 내부에 위도경도 데이터가 존재하므로 사용
# rest_list = []
# for i in range(REST_Data.shape[0]):
#   if REST_Data.loc[i, "식당경도"] != "" and REST_Data.loc[i, "식당위도"] != "":
#     rest_grid = wgs84_to_katec(REST_Data.loc[i, "식당경도"], REST_Data.loc[i, "식당위도"])
#     rest_list.append([REST_Data.loc[i, "식당명"], round(rest_grid[0]), round(rest_grid[1])])
```


```python
# 식당기본정보 데이터 프레임 변환 
# restaurant_df = pd.DataFrame(rest_list, columns = ["식당명", "GRID_X", "GRID_Y"]).dropna()

# 식당기본정보 CSV로 파일화 후 저장 후 사용
# restaurant_df.to_csv("/content/drive/MyDrive/Competitions/a firefighting competitions/data/rest_df.csv")

# 식당 데이터 호출 및 GRID_ID에 총합
rest_path = "/content/drive/MyDrive/Competitions/a firefighting competitions/data/rest_df.csv"
restaurant_df = pd.read_csv(rest_path, index_col=0)
restaurant_df_counted = xy_to_ID(restaurant_df)
restaurant_df_counted.rename(columns = {'id':'GRID_ID', 'count':'RESTAURANT_CNT'}, inplace=True) 

# 식당기본정보 DF 합치기
data_DF = pd.merge(data_DF, restaurant_df_counted, on=['GRID_ID'], how='left')
```

             id  count
    267  394527  202.0
    288  395527  199.0
    286  395525  182.0
    300  396524  178.0
    246  393527  168.0
    ..      ...    ...
    165  389533    0.0
    164  389532    0.0
    161  389529    0.0
    160  389528    0.0
    474  418518    0.0
    
    [475 rows x 2 columns]
    

식당에서  발생한 사고 시각화


```python
# 식당에서 발생한 사고 시각화
rest_index = data_DF.columns.get_loc("RESTAURANT_CNT")
plot_exc_0(rest_index)
```


    
![png](output_152_0.png)
    


#### 건축허가현황

> 가설 : 건설 현장에서 크고 작은 사고가 일어날 것이라고 예측



```python
# 강원도 건축허가현황조회 데이터 호출 ( 출처 : 강원도 원주시 건축허가현황 (2021.01 ~ 2021.12)
# BUILD_PATH = "/content/drive/MyDrive/Competitions/a firefighting competitions/data/건축허가현황조회.csv"
# BUILD_Data = pd.read_csv(BUILD_PATH, encoding = "cp949")
```


```python
# 건축대지위치 위도 경도 추출
# BUILD_list = grid_check(BUILD_Data, "허가번호", "대지위치")
```


```python
# 건축허가현황 데이터 프레임 변환 및 GRID_ID에 총합
# BUILD_df = pd.DataFrame(BUILD_list, columns = ["허가번호", "GRID_X", "GRID_Y"]).dropna()

# API 데이터를 CSV로 파일화 후 저장 후 사용
# BUILD_df.to_csv("/content/drive/MyDrive/Competitions/a firefighting competitions/data/build_df.csv", encoding="cp949")

# 건축허가현황 데이터 호출
build_path = "/content/drive/MyDrive/Competitions/a firefighting competitions/data/build_df.csv"
build_df = pd.read_csv(build_path, encoding="cp949",index_col=0)
build_df_counted = xy_to_ID(build_df)
build_df_counted.rename(columns = {'id':'GRID_ID', 'count':'BULID_PERMIT_CNT'}, inplace=True) 

# 건축허가현황 DF 합치기
data_DF = pd.merge(data_DF, build_df_counted, on=['GRID_ID'], how='left')
```

             id  count
    355  399524   27.0
    356  399525   15.0
    288  395527   14.0
    162  389530   13.0
    300  396524   13.0
    ..      ...    ...
    170  390512    0.0
    169  390511    0.0
    168  390508    0.0
    167  390507    0.0
    474  418518    0.0
    
    [475 rows x 2 columns]
    

건설현장에서 발생한 사고 시각화


```python
# 건설현장에서 발생한 사고 시각화
build_index = data_DF.columns.get_loc("BULID_PERMIT_CNT")
plot_exc_0(build_index)
```


    
![png](output_161_0.png)
    


#### 교통사고정보

> 가설 : 교통사고로 인한 사망사고가 발생한 지역의 경우 사고 위험이 더 높을 것이라고 예상


```python
# 강원도 도로교통공단_사망교통사고정보(2012~2021) 데이터 호출 ( 출처 : 강원도 원주시 공공데이터 )
# ROAD_PATH = "/content/drive/MyDrive/Competitions/a firefighting competitions/data/도로교통공단_사망교통사고정보(2012~2021).csv"
# ROAD_Data = pd.read_csv(ROAD_PATH, encoding = "cp949")
```


```python
# 교통사고정보 데이터 내부에 위도, 경도 데이터가 존재하므로 사용
# road_list = []
# for i in range(ROAD_Data.shape[0]):
#    if ROAD_Data.loc[i, "경도"] != "" and ROAD_Data.loc[i, "위도"] != "":
#        road_grid = wgs84_to_katec(ROAD_Data.loc[i, "경도"], ROAD_Data.loc[i, "위도"])
#        road_list.append([ROAD_Data.loc[i, "발생년월일시"], round(road_grid[0]), round(road_grid[1])])
```


```python
# 교통사고정보 데이터 프레임 변환 
# road_df = pd.DataFrame(road_list, columns = ["식당명", "GRID_X", "GRID_Y"]).dropna()

# 교통사고정보 CSV로 파일화 후 저장 후 사용
# road_df.to_csv("/content/drive/MyDrive/Competitions/a firefighting competitions/data/road_df.csv")

# 교통사고정보 데이터 호출 및 GRID_ID에 총합
road_path = "/content/drive/MyDrive/Competitions/a firefighting competitions/data/road_df.csv"
road_df = pd.read_csv(road_path, index_col=0)
road_df_counted = xy_to_ID(road_df)
road_df_counted.rename(columns = {'id':'GRID_ID', 'count':'ACCIDENT_AREA_CNT'}, inplace=True) 

# 교통사고정보 DF 합치기
data_DF = pd.merge(data_DF, road_df_counted, on=['GRID_ID'], how='left')
```

             id  count
    286  395525   13.0
    268  394528    9.0
    301  396525    8.0
    288  395527    8.0
    292  395531    7.0
    ..      ...    ...
    175  390524    0.0
    174  390523    0.0
    173  390522    0.0
    172  390520    0.0
    474  418518    0.0
    
    [475 rows x 2 columns]
    

교통사고에 인한 사망사고 발생한 지역 시각화


```python
# 교통사고에 인한 사망사고 발생한 지역 시각화
road_index = data_DF.columns.get_loc("ACCIDENT_AREA_CNT")
plot_exc_0(road_index)
```


    
![png](output_168_0.png)
    


## 데이터 전처리(2)

### 파생변수 추가 [유동인구분류]

연령별 유동인구와 사고와의 연관성을 시각화해 본 결과 노인인구를 제외한 특정 사고에 대하서 유동인구가 크게 영향을 주지 않으므로 65세 미만 유동인구와 65세 이상의 노인 유동인구로 파생변수 수정

65세 미만의 유동인구 추가


```python
# 65세 미만 유동인구
all_pop_list = []
all_pop = data_DF.loc[:, "GRID_ID":"F70"]
male_pop_sum = all_pop.loc[:, "M00":"M60"].sum(axis=1)
female_pop_sum = all_pop.loc[:, "F00":"F60"].sum(axis=1)
all_pop_sum = pd.concat([male_pop_sum, female_pop_sum], axis=1)
for i in range(len(all_pop_sum)):
  all_pop_list.append(all_pop_sum.at[i, 0] + all_pop_sum.at[i, 1])
all_pop_sum_df = pd.DataFrame(all_pop_list, columns = ["ALL_POP"])
all_pop_df = pd.concat([all_pop["GRID_ID"], all_pop_sum_df], axis=1)
all_pop_df 
```





  <div id="df-48440c4e-4aed-4a42-861a-9cebfe504f11">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRID_ID</th>
      <th>ALL_POP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>378509</td>
      <td>4.32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>378509</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>378509</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>378509</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>378509</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>153420</th>
      <td>418518</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>153421</th>
      <td>418518</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>153422</th>
      <td>418518</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>153423</th>
      <td>418518</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>153424</th>
      <td>418518</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>153425 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-48440c4e-4aed-4a42-861a-9cebfe504f11')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-48440c4e-4aed-4a42-861a-9cebfe504f11 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-48440c4e-4aed-4a42-861a-9cebfe504f11');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# 전체 유동인구 파생변수 기존 DF에 합치기
data_DF = pd.concat([data_DF, all_pop_df["ALL_POP"]], axis=1).reset_index(drop=True)
```

65세 이상의 노인인구 추가


```python
# 65세 이상의 노인 유동인구 파생변수 생성
elder_pop = data_DF.loc[:, ["GRID_ID", "M65", "M70", "F65", "F70"]]
elder_pop_sum = elder_pop.loc[:, "M65":].sum(axis=1)
elder_pop_df = pd.concat([elder_pop["GRID_ID"], elder_pop_sum], axis=1).rename(columns={0 : "ELDER_POP"})
elder_pop_df
```





  <div id="df-d13d8a0b-05d9-432c-9b23-9aa13b1c655b">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRID_ID</th>
      <th>ELDER_POP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>378509</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>378509</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>378509</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>378509</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>378509</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>153420</th>
      <td>418518</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153421</th>
      <td>418518</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153422</th>
      <td>418518</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153423</th>
      <td>418518</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153424</th>
      <td>418518</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>153425 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d13d8a0b-05d9-432c-9b23-9aa13b1c655b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d13d8a0b-05d9-432c-9b23-9aa13b1c655b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d13d8a0b-05d9-432c-9b23-9aa13b1c655b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# 65세 이상의 노인 유동인구 파생변수 기존 DF에 합치기
data_DF = pd.concat([data_DF, elder_pop_df["ELDER_POP"]], axis=1).reset_index(drop=True)
data_DF
```





  <div id="df-41f6221c-6b5f-456c-a452-857f74180e6d">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRID_ID</th>
      <th>GRID_X_AXIS</th>
      <th>GRID_Y_AXIS</th>
      <th>OCRN_YMD</th>
      <th>M00</th>
      <th>M10</th>
      <th>M15</th>
      <th>M20</th>
      <th>M25</th>
      <th>M30</th>
      <th>...</th>
      <th>AVRG_HUMIDITY</th>
      <th>TOTAL_CNT</th>
      <th>INDUSTRIAL_CNT</th>
      <th>BAR_CNT</th>
      <th>SENIOR_CENTER_CNT</th>
      <th>RESTAURANT_CNT</th>
      <th>BULID_PERMIT_CNT</th>
      <th>ACCIDENT_AREA_CNT</th>
      <th>ALL_POP</th>
      <th>ELDER_POP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>378509</td>
      <td>378475</td>
      <td>509475</td>
      <td>2021-01-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.27</td>
      <td>0.22</td>
      <td>...</td>
      <td>71</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.32</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>378509</td>
      <td>378475</td>
      <td>509475</td>
      <td>2021-01-02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>53</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>378509</td>
      <td>378475</td>
      <td>509475</td>
      <td>2021-01-03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>378509</td>
      <td>378475</td>
      <td>509475</td>
      <td>2021-01-04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>378509</td>
      <td>378475</td>
      <td>509475</td>
      <td>2021-01-05</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>46</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>153420</th>
      <td>418518</td>
      <td>418475</td>
      <td>518475</td>
      <td>2021-11-25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>65</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153421</th>
      <td>418518</td>
      <td>418475</td>
      <td>518475</td>
      <td>2021-11-26</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>60</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153422</th>
      <td>418518</td>
      <td>418475</td>
      <td>518475</td>
      <td>2021-11-27</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>66</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153423</th>
      <td>418518</td>
      <td>418475</td>
      <td>518475</td>
      <td>2021-11-28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>57</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153424</th>
      <td>418518</td>
      <td>418475</td>
      <td>518475</td>
      <td>2021-11-29</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>62</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>153425 rows × 68 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-41f6221c-6b5f-456c-a452-857f74180e6d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-41f6221c-6b5f-456c-a452-857f74180e6d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-41f6221c-6b5f-456c-a452-857f74180e6d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### 비대칭 데이터 정규화


```python
# distplot 함수 
def distplot_check (cols):
    cols_len = len(cols)
    if cols_len % 2 == 0 :
        row_size = (cols_len // 2)
    else : 
        row_size = (cols_len // 2) + 1

    fig, axes = plt.subplots(row_size, 2, figsize=(25, 15))
    plt.subplots_adjust(wspace=0.3,  hspace=0.7)
    sns.set_theme(style="white", context="talk")
    row, col = 0, 0

    for sca_col in cols: 
        df_sca = data_DF[sca_col].values

        if col > 1:
            row += 1
            col = 0
            sns.distplot(df_sca, ax=axes[row, col])
            axes[row, col].set_title(sca_col)
            col += 1

        else : 
            sns.distplot(df_sca, ax=axes[row, col])
            axes[row, col].set_title(sca_col)
            col += 1      
```


```python
# 비대칭 데이터 확인 
def skewed_check (date, col):
    skewness = date[col].apply(lambda x : x.skew()).sort_values(ascending=False)
    sk_df = pd.DataFrame(skewness, columns = ["skewness"])
    sk_df.sort_values(by="skewness", ascending=False)
    sk_df["skw"] = abs(sk_df["skewness"])
    sk_df = sk_df.sort_values(by="skw", ascending=False).drop("skewness", axis=1)
    skw_features = np.unique(sk_df[sk_df.skw > 0.5].index)
    return skw_features
```


```python
# 비대칭 데이터(Skewed Data) 확인 
pre_col = ['AVRG_TMPRT', 'DAY_RAINQTY', 'DAY_MSNF', 'AVRG_WS', 'AVRG_HUMIDITY', 'INDUSTRIAL_CNT', 'BAR_CNT', \
           'SENIOR_CENTER_CNT', 'RESTAURANT_CNT', 'BULID_PERMIT_CNT', 'ACCIDENT_AREA_CNT', 'ALL_POP', 'ELDER_POP']
skewness = data_DF[pre_col].apply(lambda x : x.skew()).sort_values(ascending=False)
sk_df = pd.DataFrame(skewness, columns = ["skewness"])
sk_df.sort_values(by="skewness", ascending=False)
sk_df["skw"] = abs(sk_df["skewness"])
sk_df = sk_df.sort_values(by="skw", ascending=False).drop("skewness", axis=1)
sk_df
```





  <div id="df-bcdf64a9-4199-460f-8875-0358f9a3442e">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>skw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>INDUSTRIAL_CNT</th>
      <td>16.024124</td>
    </tr>
    <tr>
      <th>BAR_CNT</th>
      <td>12.471664</td>
    </tr>
    <tr>
      <th>DAY_MSNF</th>
      <td>7.893618</td>
    </tr>
    <tr>
      <th>ELDER_POP</th>
      <td>5.958059</td>
    </tr>
    <tr>
      <th>BULID_PERMIT_CNT</th>
      <td>5.695399</td>
    </tr>
    <tr>
      <th>RESTAURANT_CNT</th>
      <td>5.256405</td>
    </tr>
    <tr>
      <th>DAY_RAINQTY</th>
      <td>5.156004</td>
    </tr>
    <tr>
      <th>ALL_POP</th>
      <td>4.820455</td>
    </tr>
    <tr>
      <th>SENIOR_CENTER_CNT</th>
      <td>3.861760</td>
    </tr>
    <tr>
      <th>ACCIDENT_AREA_CNT</th>
      <td>3.822254</td>
    </tr>
    <tr>
      <th>AVRG_WS</th>
      <td>0.641908</td>
    </tr>
    <tr>
      <th>AVRG_TMPRT</th>
      <td>0.497207</td>
    </tr>
    <tr>
      <th>AVRG_HUMIDITY</th>
      <td>0.359424</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bcdf64a9-4199-460f-8875-0358f9a3442e')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-bcdf64a9-4199-460f-8875-0358f9a3442e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bcdf64a9-4199-460f-8875-0358f9a3442e');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# 절대값 0.5 사이의 값은 적당히 잘 분포
skw_features = np.unique(sk_df[sk_df.skw > 0.5].index)
skw_features
```




    array(['ACCIDENT_AREA_CNT', 'ALL_POP', 'AVRG_WS', 'BAR_CNT',
           'BULID_PERMIT_CNT', 'DAY_MSNF', 'DAY_RAINQTY', 'ELDER_POP',
           'INDUSTRIAL_CNT', 'RESTAURANT_CNT', 'SENIOR_CENTER_CNT'],
          dtype=object)




```python
# Box-Cox Transform 처리 전 분포 확인 
distplot_check(skw_features)           
```


    
![png](output_184_0.png)
    



```python
# Box-Cox Transform
lam = 0.01

for col in skw_features:
  data_DF[col] = boxcox1p(data_DF[col], lam)

trans_skewness = data_DF[skw_features].apply(lambda x : x.skew()).sort_values(ascending=False)
trans_sk_df = pd.DataFrame(trans_skewness, columns = ["skewness"])
trans_sk_df.sort_values(by="skewness", ascending=False)
trans_sk_df["skw"] = abs(trans_sk_df["skewness"])
trans_sk_df = trans_sk_df.sort_values(by="skw", ascending=False).drop("skewness", axis=1)
trans_sk_df
```





  <div id="df-e2e69a2d-0c7f-4169-8555-a2a4bd66a2cd">
    <div class="colab-df-container">
      <div>
<style>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>skw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>INDUSTRIAL_CNT</th>
      <td>7.658677</td>
    </tr>
    <tr>
      <th>BAR_CNT</th>
      <td>6.502235</td>
    </tr>
    <tr>
      <th>DAY_MSNF</th>
      <td>4.825624</td>
    </tr>
    <tr>
      <th>DAY_RAINQTY</th>
      <td>2.188445</td>
    </tr>
    <tr>
      <th>RESTAURANT_CNT</th>
      <td>2.026323</td>
    </tr>
    <tr>
      <th>ACCIDENT_AREA_CNT</th>
      <td>1.923127</td>
    </tr>
    <tr>
      <th>BULID_PERMIT_CNT</th>
      <td>1.889636</td>
    </tr>
    <tr>
      <th>SENIOR_CENTER_CNT</th>
      <td>1.590027</td>
    </tr>
    <tr>
      <th>ELDER_POP</th>
      <td>0.834945</td>
    </tr>
    <tr>
      <th>ALL_POP</th>
      <td>0.488207</td>
    </tr>
    <tr>
      <th>AVRG_WS</th>
      <td>0.193305</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e2e69a2d-0c7f-4169-8555-a2a4bd66a2cd')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-e2e69a2d-0c7f-4169-8555-a2a4bd66a2cd button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e2e69a2d-0c7f-4169-8555-a2a4bd66a2cd');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# Box-Cox를 Transform 한 후 분포 확인 
distplot_check(skw_features)
```


    
![png](output_186_0.png)
    


### 데이터 스케일링


```python
# 스케일링 처리 변수 
scaling_col = ['AVRG_TMPRT', 'DAY_RAINQTY', 'DAY_MSNF', 'AVRG_WS', 'AVRG_HUMIDITY', 
               'INDUSTRIAL_CNT', 'BAR_CNT', 'SENIOR_CENTER_CNT', 'RESTAURANT_CNT', 
               'BULID_PERMIT_CNT', 'ACCIDENT_AREA_CNT', 'ALL_POP', 'ELDER_POP']
```


```python
# 스케일링 전 변수 분포 확인  
distplot_check(scaling_col)     
```


    
![png](output_189_0.png)
    



```python
# 각 컬럼 스케일링 수행 
Scaler = MinMaxScaler()

for col in scaling_col:
    data_DF[col] = Scaler.fit_transform(data_DF[col].values.reshape(-1,1))
```


```python
# 스케일링 후 변수 분포 확인 
distplot_check(scaling_col) 
```


    
![png](output_191_0.png)
    


### 최종 변수 정리


```python
# 각 유동인구 col 삭제 
del_list = ["M00", "M10", "M15", "M20","M25", "M30", "M35", "M40", "M45", "M50", "M55", "M60", "M65", "M70",
            "F00", "F10", "F15", "F20","F25", "F30", "F35", "F40", "F45", "F50", "F55", "F60", "F65", "F70"]
data_DF = data_DF.drop(columns = del_list, axis=1)
```


```python
# 모델링에 불필요한 col 삭제
unnecessary_list = ["GRID_X_AXIS","GRID_Y_AXIS", "OCRN_YMD", "TOTAL_CNT"]
data_DF.drop(columns = unnecessary_list, axis=1, inplace=True)
```


```python
# 각 종속변수 이진분류 모델을 돌리기 위해서 0과 1로 변경
MCHN_CNT = data_DF.columns.get_loc("MCHN_ACDNT_OCRN_CNT")
FALL_CNT = data_DF.columns.get_loc("FALLING_OCRN_CNT")

for col in range(MCHN_CNT, FALL_CNT+1):
  data_DF.iloc[:, col] = np.where(data_DF.iloc[:, col] >= 1, 1, 0)
```


```python
# 이진분류 변환 확인 
data_DF["HRFAF_OCRN_CNT"].value_counts()
```




    0    152475
    1       950
    Name: HRFAF_OCRN_CNT, dtype: int64



1. 모델에 사용하는 최종 독립 변수
  - MONTH : 월
  - DAY : 일
  - WEEKDAY : 요일
  - HOLIDAY : 공휴일
  - SEASON_SE_NM : 계절
  - AVRG_TMPRT : 평균기온(°C)	
  - DAY_RAINQTY : 일강수량(mm)
  - DAY_MSNF : 일적설량(cm)
  - AVRG_WS : 평균 풍속(m/s)
  - AVRG_HUMIDITY : 평균 습도(%)
  - INDUSTRIAL_CNT : 격자내 산업단지 개수
  - BAR_CNT : 격자내 유흥업소 개수
  - SENIOR_CENTER_CNT : 격자내 경로당 개수
  - RESTAURANT_CNT : 격자내 식당 개수
  - BULID_PERMIT_CNT : 격자내 건축허가 개수
  - ACCIDENT_AREA_CNT : 격자내 사고건수
  - ALL_POP : 전체 연령(65세 미만) 유동인구
  - ELDER_POP : 65세 이상 유동인구
  - GRID_ID : 격자 ID



2. 모델에 사용하는 최종 종속 변수
  - MCHN_ACDNT_OCRN_CNT         : 기계사고발생건수
  - ETC_OCRN_CNT                : 기타발생건수
  - BLTRM_OCRN_CNT             : 둔상발생건수
  - ACDNT_INJ_OCRN_CNT        : 사고부상발생건수
  - EXCL_DISEASE_OCRN_CNT      : 질병외발생건수
  - VHC_ACDNT_OCRN_CNT       : 탈것사고발생건수
  - HRFAF_OCRN_CNT             : 낙상발생건수
  - DRKNSTAT_OCRN_CNT           : 단순주취발생건수
  - ANML_INSCT_ACDNT_OCRN_CNT  : 동물곤충사고발생건수
  - FLPS_ACDNT_OCRN_CNT         : 동승자사고발생건수
  - PDST_ACDNT_OCRN_CNT         : 보행자사고발생건수
  - LACRTWND_OCRN_CNT            : 열상발생건수
  - MTRCYC_ACDNT_OCRN_CNT        : 오토바이사고발생건수
  - DRV_ACDNT_OCRN_CNT            : 운전자사고발생건수
  - BCYC_ACDNT_OCRN_CNT           : 자전거사고발생건수
  - POSNG_OCRN_CNT              : 중독발생건수
  - FALLING_OCRN_CNT             : 추락발생건수


```python
# 백업 데이터 
final_DF = data_DF.copy()
```


```python
final_DF.to_csv ("/content/drive/MyDrive/Competitions/a firefighting competitions/data/final_DF.csv")
```

### 데이터 변수명 지정



```python
# 분석할 17개의 사고의 변수명 지정 
col = ['MCHN_ACDNT_OCRN_CNT', 'ETC_OCRN_CNT', 'BLTRM_OCRN_CNT',
       'ACDNT_INJ_OCRN_CNT', 'EXCL_DISEASE_OCRN_CNT', 'VHC_ACDNT_OCRN_CNT',
       'HRFAF_OCRN_CNT', 'DRKNSTAT_OCRN_CNT', 'ANML_INSCT_ACDNT_OCRN_CNT',
       'FLPS_ACDNT_OCRN_CNT', 'PDST_ACDNT_OCRN_CNT', 'LACRTWND_OCRN_CNT',
       'MTRCYC_ACDNT_OCRN_CNT', 'DRV_ACDNT_OCRN_CNT', 'BCYC_ACDNT_OCRN_CNT',
       'POSNG_OCRN_CNT', 'FALLING_OCRN_CNT']

for i in col: # 변수명 지정 --> data_각 사고 이름 ( ex) data_HRFAF_OCRN_CNT )
    globals()["data_{}".format(i)] = pd.DataFrame()
    globals()["data_{}".format(i)] = final_DF.iloc[:,18:] # X 값
    globals()["data_{}".format(i)]["GRID_ID"] = final_DF["GRID_ID"]
    globals()["data_{}".format(i)]["label"] = final_DF[i] # 라벨 넣기

print(data_HRFAF_OCRN_CNT.head(5))
```

       MONTH  DAY  WEEKDAY  HOLIDAY  SEASON_SE_NM  AVRG_TMPRT  DAY_RAINQTY  \
    0      1    1        4      1.0             4    0.166667          0.0   
    1      1    2        5      0.0             4    0.214286          0.0   
    2      1    3        6      0.0             4    0.190476          0.0   
    3      1    4        0      0.0             4    0.238095          0.0   
    4      1    5        1      0.0             4    0.238095          0.0   
    
       DAY_MSNF   AVRG_WS  AVRG_HUMIDITY  INDUSTRIAL_CNT  BAR_CNT  \
    0  0.446783  0.000000       0.661538             0.0      0.0   
    1  0.000000  0.498267       0.384615             0.0      0.0   
    2  0.000000  0.000000       0.338462             0.0      0.0   
    3  0.000000  0.000000       0.338462             0.0      0.0   
    4  0.000000  0.498267       0.276923             0.0      0.0   
    
       SENIOR_CENTER_CNT  RESTAURANT_CNT  BULID_PERMIT_CNT  ACCIDENT_AREA_CNT  \
    0           0.247407             0.0          0.411985                0.0   
    1           0.247407             0.0          0.411985                0.0   
    2           0.247407             0.0          0.411985                0.0   
    3           0.247407             0.0          0.411985                0.0   
    4           0.247407             0.0          0.411985                0.0   
    
        ALL_POP  ELDER_POP  GRID_ID  label  
    0  0.133319   0.059814   378509      0  
    1  0.000000   0.000000   378509      0  
    2  0.000000   0.000000   378509      0  
    3  0.000000   0.000000   378509      0  
    4  0.000000   0.000000   378509      0  
    


```python
# 그대로 진행할시 0값에 데이터가 몰려있어 undersampling 처리 후 변수명 변경
from imblearn.under_sampling import NearMiss 
nm = NearMiss()

rus = RandomUnderSampler(random_state=0, sampling_strategy=0.7)
                         
        #                  {
        # 0: int(globals()["data_{}".format(i)][globals()["data_{}".format(i)]["label"] == 1].shape[0]*1.5),
        # 1: globals()["data_{}".format(i)][globals()["data_{}".format(i)]["label"] == 1].shape[0]})

for i in col: 
    X = globals()["data_{}".format(i)].iloc[:,:-1] 
    y = globals()["data_{}".format(i)]["label"]
    
    globals()["res_X_{}".format(i)], globals()["res_y_{}".format(i)] = rus.fit_resample(X, y) ###

    globals()["res_X_{}".format(i)]["label"] = globals()["res_y_{}".format(i)]
    globals()["res_{}".format(i)] = globals()["res_X_{}".format(i)]

    
    X = globals()["res_{}".format(i)].drop(['label'], axis=1)
    y = globals()["res_{}".format(i)]['label']
    
    # 각 사건 변수 훈련 데이터 / 테스트 데이터 분리
    globals()["X_train_{}".format(i)], \
    globals()["X_test_{}".format(i)], \
    globals()["y_train_{}".format(i)], \
    globals()["y_test_{}".format(i)] = train_test_split(X, y, test_size = 0.3, random_state = 0)
```


```python
# Undersampling 처리 결과 확인 
res_HRFAF_OCRN_CNT["label"].value_counts()
```




    0    1357
    1     950
    Name: label, dtype: int64



## 데이터 모델링

함수 정의


```python
# 기온과 날짜를 제외한 나머지 변수와 사고의 상관관계
def corr_check (data):
    plt.figure(figsize=(12,12))
    a = pd.concat([data.iloc[:,10:18], data.iloc[:,19]], axis=1)
    sub_sample_corr = a.corr()
    mask = np.zeros_like(sub_sample_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(sub_sample_corr, cmap = 'RdYlBu_r', annot = True, square = False, 
                mask=mask, cbar_kws={"shrink": .5}, vmin = -1,vmax = 1)
```


```python
# cross_val_score 
kfold = KFold(n_splits = 5, random_state = 10, shuffle = True)
def cross_socre(model, X_train, y_train):
  cross_score = cross_val_score(model, X_train, y_train, cv = kfold, n_jobs=-1, scoring="recall")
  return cross_score
```


```python
# 베이스 모델 점수 확인 
def base_rate_test (model, X_train, y_train, name):
  score = cross_socre(model, X_train, y_train)
  print(f"The average score of the {name} : {score.mean() :.3f}")
```


```python
# RandomizedSearchCV 
def Random_Grid_CV (model, params, X_train, y_train, x_test):
    random_model = RandomizedSearchCV(model, param_distributions = params, cv = kfold, scoring="recall", n_iter=10)
    random_model.fit(X_train, y_train)
    return random_model

# rf
params_rf_rg = { 
           'n_estimators' : [10, 100],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [8, 12, 18],
           'min_samples_split' : [8, 16, 20]
               }

# xgb
params_xgb_rg = {
          'n_estimators' : [100,200,300,400,500],
          'learning_rate' : [0.01,0.05,0.1,0.15],
          'max_depth' : [3,5,7,10,15],
          'gamma' : [0,1,2,3],
          'colsample_bytree' : [0.8,0.9],
                }

# cat
params_cat_rg = {
          'depth':[3,1,2,6,4,5,7,8,9,10],
          'iterations':[250,100,500,1000],
          'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
          'l2_leaf_reg':[3,1,5,10,100],
          'border_count':[32,5,10,20,50,100,200]
                }

# gbc
param_gbc_rg = {
          'n_estimators':[100, 200, 300, 400, 500],
          'learning_rate':[0.05,0.1, 0.2, 0.3, 0.4]
                }

# lr
params_lr_rg = {
          'C': [0.001, 0.01, 0.1, 1, 10, 100],
          'penalty': ['l1', 'l2']
                }
```


```python
# GridSearchCV 
def Grid_Search_CV (model, params, X_train, y_train, x_test):
    random_model = GridSearchCV(model, param_grid=params, cv=kfold, scoring="recall", n_jobs=-1)
    random_model.fit(X_train, y_train)
    return random_model

# rf
params_rf_gs = { 
           'n_estimators' : [90, 100, 110],
           'max_depth' : [10, 11, 12],
           'min_samples_leaf' : [7, 8, 9],
           'min_samples_split' : [19, 20, 21]
               }

# xgb
params_xgb_gs = {
          'n_estimators' : [490, 500, 510],
          'learning_rate' : [0.04, 0.05, 0.06],
          'max_depth' : [2, 3, 4],
          'gamma' : [1, 2, 3],
          'colsample_bytree' : [0.7, 0.8, 0.9]
                }

# cat
params_cat_gs = {
          'depth':[7, 8, 9],
          'iterations':[990, 1000, 1010],
          'learning_rate':[0.01, 0.02, 0.03], 
          'l2_leaf_reg':[4, 5, 6],
          'border_count':[30, 32, 34]
                }

# gbc
param_gbc_gs = {
          'n_estimators':[390, 400, 410],
          'learning_rate':[0.04, 0.05, 0.06]
                }

# lr
params_lr_gs = {
          'C': [0.001, 0.002, 0.003, 0.2],
          'penalty': ['l1', 'l2']
                }
```


```python
# ROC_CURVE Grape
def roc_curve_graph(X_test, y_test, model_1, model_2, model_3, name_1, name_2, name_3):
    # pred_proba
    y_pred_proba_1 = model_1.predict_proba(X_test)[::,1]
    y_pred_proba_2 = model_2.predict_proba(X_test)[::,1]
    y_pred_proba_3 = model_3.predict_proba(X_test)[::,1]

    # define metrics 
    model_fpr_1, model_tpr_1, _ = roc_curve(y_test, y_pred_proba_1)
    model_fpr_2, model_tpr_2, _ = roc_curve(y_test, y_pred_proba_2)
    model_fpr_3, model_tpr_3, _ = roc_curve(y_test, y_pred_proba_3)

    # score
    auc_1 = roc_auc_score(y_test, y_pred_proba_1)
    auc_2 = roc_auc_score(y_test, y_pred_proba_2)
    auc_3 = roc_auc_score(y_test, y_pred_proba_3)

    # ROC_CURVE Grape setting
    plt.figure(figsize=(15,8))
    plt.title('ROC Curve', fontsize=15)

    # ROC_CURVE Plot
    plt.plot(model_fpr_1, model_tpr_1, label="{} Score = {}".format(name_1, auc_1))
    plt.plot(model_fpr_2, model_tpr_2, label="{} Score = {}".format(name_2, auc_2))
    plt.plot(model_fpr_3, model_tpr_3, label="{} Score = {}".format(name_3, auc_3))
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.legend()
```


```python
# 오차행렬(confusion matrix) Heatmap
def confusion_matrix_heat (y_test, model_pred_1, model_pred_2, model_pred_3, name1, name2, name3):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=True)    
    cm1 = confusion_matrix(y_test, model_pred_1)
    cm2 = confusion_matrix(y_test, model_pred_2)
    cm3 = confusion_matrix(y_test, model_pred_3)
    sns.heatmap( cm1/np.sum(cm1), annot = True, fmt = '0.2%', cmap = 'Reds', ax=ax1)
    ax1.set_title(f"{name1} Confusion Matrix",fontsize=13)
    sns.heatmap( cm2/np.sum(cm2), annot = True, fmt = '0.2%', cmap = 'Reds', ax=ax2)
    ax2.set_title(f"{name2} Confusion Matrix",fontsize=13)
    sns.heatmap( cm3/np.sum(cm3), annot = True, fmt = '0.2%', cmap = 'Reds', ax=ax3)
    ax3.set_title(f"{name3} Confusion Matrix",fontsize=13)
    plt.show()
```


```python
# 머신 러닝 분류 모델 평가 지표 (Classification_report) 
def Classification_report_check (y_test, model_pred_1, model_pred_2, model_pred_3, name_1, name_2, name_3):
  print(f"                {name_1} Classification Report")
  print(classification_report(y_test, model_pred_1)) # 모델 1 평가 지표
  print("*"*50)
  print(f"                  {name_2} Classification Report")
  print(classification_report(y_test, model_pred_2)) # 모델 2 평가 지표
  print("*"*50)
  print(f"                  {name_3} Classification Report")
  print(classification_report(y_test, model_pred_3)) # 모델 3 평가 지표
```


```python
# 변수 중요도 평가 (feature_importances) 그래프
def Feature_import (name, model, X_test):
    feature_importances = model.feature_importances_

    ft_importances = pd.Series(feature_importances, index = X_test.columns)

    plt.figure(figsize=(8, 8))
    ft_importances = ft_importances.sort_values(ascending=True).plot.barh()
    plt.xlabel(f'{name} Feature importance')
    plt.ylabel('Feature')
    plt.show()
```


```python
# 각각의 모델 이름
RF_model = RandomForestClassifier(random_state = 10) # Random Forest
xgb_model = XGBClassifier(random_state = 10) # XGBoost
cat_model = CatBoostClassifier(random_state = 10) # CatBoost
gbc_model = GradientBoostingClassifier(random_state = 10) # GradientBoost
lr_model = LogisticRegression(random_state = 10) # Logistic Regression 
```

### 모델링


가장 많이 발생한 낙상사고를 기준으로 파라미터 조정하여 각 사고에 동일하게 적용


```python
# RandomSearchCV

# Random Forest 
RF_rg = Random_Grid_CV(RF_model, params_rf_rg, X_train_HRFAF_OCRN_CNT, y_train_HRFAF_OCRN_CNT, X_test_HRFAF_OCRN_CNT)

#XGBoost fit
XGB_rg = Random_Grid_CV(xgb_model, params_xgb_rg, X_train_HRFAF_OCRN_CNT, y_train_HRFAF_OCRN_CNT, X_test_HRFAF_OCRN_CNT)

# CatBoost fit
CAT_rg = Random_Grid_CV(cat_model, params_cat_rg, X_train_HRFAF_OCRN_CNT, y_train_HRFAF_OCRN_CNT, X_test_HRFAF_OCRN_CNT)

# GradientBoost fit
GBC_rg = Random_Grid_CV(gbc_model, param_gbc_rg, X_train_HRFAF_OCRN_CNT, y_train_HRFAF_OCRN_CNT, X_test_HRFAF_OCRN_CNT)

# Logistic Regression fit
LR_rg = Random_Grid_CV(lr_model, params_lr_rg, X_train_HRFAF_OCRN_CNT, y_train_HRFAF_OCRN_CNT, X_test_HRFAF_OCRN_CNT)
```

    [1;30;43m스트리밍 출력 내용이 길어서 마지막 5000줄이 삭제되었습니다.[0m
    500:	learn: 0.2082995	total: 523ms	remaining: 521ms
    501:	learn: 0.2081987	total: 524ms	remaining: 520ms
    502:	learn: 0.2081039	total: 525ms	remaining: 519ms
    503:	learn: 0.2079327	total: 526ms	remaining: 518ms
    504:	learn: 0.2072621	total: 527ms	remaining: 516ms
    505:	learn: 0.2071647	total: 528ms	remaining: 515ms
    506:	learn: 0.2069316	total: 529ms	remaining: 515ms
    507:	learn: 0.2067741	total: 530ms	remaining: 514ms
    508:	learn: 0.2066878	total: 532ms	remaining: 513ms
    509:	learn: 0.2065989	total: 534ms	remaining: 513ms
    510:	learn: 0.2064568	total: 536ms	remaining: 513ms
    511:	learn: 0.2063347	total: 539ms	remaining: 514ms
    512:	learn: 0.2061988	total: 541ms	remaining: 514ms
    513:	learn: 0.2060697	total: 543ms	remaining: 513ms
    514:	learn: 0.2059652	total: 545ms	remaining: 513ms
    515:	learn: 0.2058710	total: 548ms	remaining: 514ms
    516:	learn: 0.2057120	total: 549ms	remaining: 513ms
    517:	learn: 0.2051916	total: 551ms	remaining: 513ms
    518:	learn: 0.2049091	total: 553ms	remaining: 512ms
    519:	learn: 0.2046531	total: 554ms	remaining: 511ms
    520:	learn: 0.2046363	total: 556ms	remaining: 511ms
    521:	learn: 0.2045655	total: 557ms	remaining: 510ms
    522:	learn: 0.2045514	total: 558ms	remaining: 509ms
    523:	learn: 0.2044747	total: 559ms	remaining: 508ms
    524:	learn: 0.2044065	total: 560ms	remaining: 507ms
    525:	learn: 0.2043203	total: 561ms	remaining: 506ms
    526:	learn: 0.2042175	total: 562ms	remaining: 505ms
    527:	learn: 0.2040193	total: 563ms	remaining: 503ms
    528:	learn: 0.2039488	total: 564ms	remaining: 502ms
    529:	learn: 0.2038752	total: 565ms	remaining: 501ms
    530:	learn: 0.2038323	total: 566ms	remaining: 500ms
    531:	learn: 0.2035668	total: 567ms	remaining: 499ms
    532:	learn: 0.2035138	total: 568ms	remaining: 498ms
    533:	learn: 0.2033735	total: 569ms	remaining: 497ms
    534:	learn: 0.2032915	total: 570ms	remaining: 496ms
    535:	learn: 0.2030937	total: 571ms	remaining: 495ms
    536:	learn: 0.2028249	total: 572ms	remaining: 493ms
    537:	learn: 0.2025840	total: 573ms	remaining: 492ms
    538:	learn: 0.2025692	total: 574ms	remaining: 491ms
    539:	learn: 0.2024847	total: 575ms	remaining: 490ms
    540:	learn: 0.2020271	total: 576ms	remaining: 489ms
    541:	learn: 0.2016704	total: 577ms	remaining: 488ms
    542:	learn: 0.2016028	total: 578ms	remaining: 487ms
    543:	learn: 0.2013725	total: 579ms	remaining: 486ms
    544:	learn: 0.2013485	total: 580ms	remaining: 484ms
    545:	learn: 0.2012248	total: 581ms	remaining: 483ms
    546:	learn: 0.2011508	total: 582ms	remaining: 482ms
    547:	learn: 0.2010703	total: 583ms	remaining: 481ms
    548:	learn: 0.2009509	total: 584ms	remaining: 480ms
    549:	learn: 0.2008668	total: 585ms	remaining: 479ms
    550:	learn: 0.2007706	total: 586ms	remaining: 478ms
    551:	learn: 0.2006047	total: 587ms	remaining: 477ms
    552:	learn: 0.2005333	total: 588ms	remaining: 475ms
    553:	learn: 0.2002904	total: 589ms	remaining: 474ms
    554:	learn: 0.2001764	total: 590ms	remaining: 473ms
    555:	learn: 0.2001009	total: 591ms	remaining: 472ms
    556:	learn: 0.2000087	total: 592ms	remaining: 471ms
    557:	learn: 0.1999199	total: 593ms	remaining: 470ms
    558:	learn: 0.1997281	total: 594ms	remaining: 469ms
    559:	learn: 0.1996755	total: 595ms	remaining: 468ms
    560:	learn: 0.1996129	total: 597ms	remaining: 467ms
    561:	learn: 0.1996017	total: 598ms	remaining: 466ms
    562:	learn: 0.1994197	total: 599ms	remaining: 465ms
    563:	learn: 0.1992288	total: 600ms	remaining: 464ms
    564:	learn: 0.1991013	total: 601ms	remaining: 463ms
    565:	learn: 0.1990627	total: 602ms	remaining: 462ms
    566:	learn: 0.1989876	total: 603ms	remaining: 461ms
    567:	learn: 0.1988755	total: 604ms	remaining: 460ms
    568:	learn: 0.1987663	total: 605ms	remaining: 459ms
    569:	learn: 0.1987078	total: 607ms	remaining: 458ms
    570:	learn: 0.1985829	total: 608ms	remaining: 457ms
    571:	learn: 0.1984855	total: 609ms	remaining: 455ms
    572:	learn: 0.1984641	total: 610ms	remaining: 454ms
    573:	learn: 0.1984034	total: 611ms	remaining: 453ms
    574:	learn: 0.1983156	total: 612ms	remaining: 452ms
    575:	learn: 0.1982137	total: 613ms	remaining: 451ms
    576:	learn: 0.1980330	total: 614ms	remaining: 450ms
    577:	learn: 0.1976379	total: 615ms	remaining: 449ms
    578:	learn: 0.1975316	total: 616ms	remaining: 448ms
    579:	learn: 0.1974235	total: 617ms	remaining: 447ms
    580:	learn: 0.1968517	total: 618ms	remaining: 446ms
    581:	learn: 0.1967994	total: 619ms	remaining: 445ms
    582:	learn: 0.1966426	total: 620ms	remaining: 444ms
    583:	learn: 0.1966253	total: 621ms	remaining: 442ms
    584:	learn: 0.1964729	total: 622ms	remaining: 441ms
    585:	learn: 0.1963843	total: 623ms	remaining: 440ms
    586:	learn: 0.1962934	total: 624ms	remaining: 439ms
    587:	learn: 0.1961741	total: 625ms	remaining: 438ms
    588:	learn: 0.1960538	total: 626ms	remaining: 437ms
    589:	learn: 0.1959031	total: 627ms	remaining: 436ms
    590:	learn: 0.1957757	total: 628ms	remaining: 434ms
    591:	learn: 0.1956607	total: 629ms	remaining: 433ms
    592:	learn: 0.1955774	total: 630ms	remaining: 432ms
    593:	learn: 0.1954235	total: 631ms	remaining: 431ms
    594:	learn: 0.1953359	total: 632ms	remaining: 430ms
    595:	learn: 0.1953254	total: 633ms	remaining: 429ms
    596:	learn: 0.1951768	total: 634ms	remaining: 428ms
    597:	learn: 0.1950483	total: 634ms	remaining: 427ms
    598:	learn: 0.1949958	total: 635ms	remaining: 425ms
    599:	learn: 0.1948695	total: 636ms	remaining: 424ms
    600:	learn: 0.1947928	total: 638ms	remaining: 423ms
    601:	learn: 0.1946832	total: 639ms	remaining: 422ms
    602:	learn: 0.1946656	total: 640ms	remaining: 421ms
    603:	learn: 0.1945434	total: 641ms	remaining: 420ms
    604:	learn: 0.1944997	total: 642ms	remaining: 419ms
    605:	learn: 0.1944719	total: 643ms	remaining: 418ms
    606:	learn: 0.1944031	total: 644ms	remaining: 417ms
    607:	learn: 0.1940658	total: 645ms	remaining: 416ms
    608:	learn: 0.1939901	total: 646ms	remaining: 414ms
    609:	learn: 0.1939198	total: 646ms	remaining: 413ms
    610:	learn: 0.1938167	total: 647ms	remaining: 412ms
    611:	learn: 0.1937370	total: 648ms	remaining: 411ms
    612:	learn: 0.1936671	total: 649ms	remaining: 410ms
    613:	learn: 0.1936177	total: 650ms	remaining: 409ms
    614:	learn: 0.1935382	total: 651ms	remaining: 408ms
    615:	learn: 0.1933224	total: 652ms	remaining: 406ms
    616:	learn: 0.1932754	total: 653ms	remaining: 405ms
    617:	learn: 0.1931684	total: 654ms	remaining: 404ms
    618:	learn: 0.1929963	total: 655ms	remaining: 403ms
    619:	learn: 0.1928656	total: 656ms	remaining: 402ms
    620:	learn: 0.1928405	total: 657ms	remaining: 401ms
    621:	learn: 0.1927256	total: 658ms	remaining: 400ms
    622:	learn: 0.1927165	total: 659ms	remaining: 399ms
    623:	learn: 0.1926190	total: 660ms	remaining: 398ms
    624:	learn: 0.1925365	total: 661ms	remaining: 396ms
    625:	learn: 0.1924729	total: 662ms	remaining: 395ms
    626:	learn: 0.1923995	total: 663ms	remaining: 394ms
    627:	learn: 0.1920336	total: 664ms	remaining: 393ms
    628:	learn: 0.1919576	total: 665ms	remaining: 392ms
    629:	learn: 0.1918054	total: 666ms	remaining: 391ms
    630:	learn: 0.1916968	total: 667ms	remaining: 390ms
    631:	learn: 0.1915732	total: 668ms	remaining: 389ms
    632:	learn: 0.1914728	total: 668ms	remaining: 388ms
    633:	learn: 0.1913513	total: 669ms	remaining: 386ms
    634:	learn: 0.1911405	total: 670ms	remaining: 385ms
    635:	learn: 0.1909942	total: 671ms	remaining: 384ms
    636:	learn: 0.1908418	total: 672ms	remaining: 383ms
    637:	learn: 0.1906945	total: 673ms	remaining: 382ms
    638:	learn: 0.1906302	total: 674ms	remaining: 381ms
    639:	learn: 0.1905729	total: 675ms	remaining: 380ms
    640:	learn: 0.1905647	total: 676ms	remaining: 379ms
    641:	learn: 0.1904920	total: 677ms	remaining: 378ms
    642:	learn: 0.1903911	total: 678ms	remaining: 377ms
    643:	learn: 0.1899359	total: 679ms	remaining: 376ms
    644:	learn: 0.1897894	total: 680ms	remaining: 374ms
    645:	learn: 0.1896583	total: 681ms	remaining: 373ms
    646:	learn: 0.1894983	total: 682ms	remaining: 372ms
    647:	learn: 0.1893987	total: 683ms	remaining: 371ms
    648:	learn: 0.1893224	total: 684ms	remaining: 370ms
    649:	learn: 0.1892177	total: 685ms	remaining: 369ms
    650:	learn: 0.1890654	total: 686ms	remaining: 368ms
    651:	learn: 0.1887499	total: 687ms	remaining: 367ms
    652:	learn: 0.1886773	total: 688ms	remaining: 366ms
    653:	learn: 0.1886069	total: 689ms	remaining: 364ms
    654:	learn: 0.1884167	total: 690ms	remaining: 364ms
    655:	learn: 0.1883272	total: 691ms	remaining: 362ms
    656:	learn: 0.1881439	total: 692ms	remaining: 361ms
    657:	learn: 0.1880503	total: 693ms	remaining: 360ms
    658:	learn: 0.1878381	total: 694ms	remaining: 359ms
    659:	learn: 0.1876197	total: 695ms	remaining: 358ms
    660:	learn: 0.1875370	total: 696ms	remaining: 357ms
    661:	learn: 0.1874702	total: 697ms	remaining: 356ms
    662:	learn: 0.1873354	total: 698ms	remaining: 355ms
    663:	learn: 0.1872557	total: 699ms	remaining: 353ms
    664:	learn: 0.1871557	total: 699ms	remaining: 352ms
    665:	learn: 0.1870039	total: 700ms	remaining: 351ms
    666:	learn: 0.1869869	total: 701ms	remaining: 350ms
    667:	learn: 0.1868639	total: 702ms	remaining: 349ms
    668:	learn: 0.1867650	total: 707ms	remaining: 350ms
    669:	learn: 0.1866274	total: 708ms	remaining: 349ms
    670:	learn: 0.1865712	total: 718ms	remaining: 352ms
    671:	learn: 0.1864094	total: 723ms	remaining: 353ms
    672:	learn: 0.1861553	total: 724ms	remaining: 352ms
    673:	learn: 0.1860536	total: 725ms	remaining: 351ms
    674:	learn: 0.1860155	total: 726ms	remaining: 350ms
    675:	learn: 0.1858367	total: 728ms	remaining: 349ms
    676:	learn: 0.1857867	total: 729ms	remaining: 348ms
    677:	learn: 0.1857113	total: 730ms	remaining: 347ms
    678:	learn: 0.1856156	total: 731ms	remaining: 345ms
    679:	learn: 0.1855626	total: 732ms	remaining: 344ms
    680:	learn: 0.1854914	total: 733ms	remaining: 343ms
    681:	learn: 0.1854469	total: 734ms	remaining: 342ms
    682:	learn: 0.1853568	total: 735ms	remaining: 341ms
    683:	learn: 0.1852858	total: 736ms	remaining: 340ms
    684:	learn: 0.1851328	total: 736ms	remaining: 339ms
    685:	learn: 0.1850037	total: 737ms	remaining: 338ms
    686:	learn: 0.1849793	total: 738ms	remaining: 336ms
    687:	learn: 0.1848833	total: 739ms	remaining: 335ms
    688:	learn: 0.1847258	total: 740ms	remaining: 334ms
    689:	learn: 0.1846032	total: 741ms	remaining: 333ms
    690:	learn: 0.1845309	total: 742ms	remaining: 332ms
    691:	learn: 0.1844116	total: 743ms	remaining: 331ms
    692:	learn: 0.1843113	total: 744ms	remaining: 330ms
    693:	learn: 0.1841796	total: 745ms	remaining: 328ms
    694:	learn: 0.1840261	total: 746ms	remaining: 327ms
    695:	learn: 0.1838856	total: 747ms	remaining: 326ms
    696:	learn: 0.1838352	total: 748ms	remaining: 325ms
    697:	learn: 0.1837401	total: 749ms	remaining: 324ms
    698:	learn: 0.1833834	total: 750ms	remaining: 323ms
    699:	learn: 0.1832046	total: 751ms	remaining: 322ms
    700:	learn: 0.1831481	total: 751ms	remaining: 321ms
    701:	learn: 0.1830664	total: 752ms	remaining: 319ms
    702:	learn: 0.1829671	total: 753ms	remaining: 318ms
    703:	learn: 0.1828376	total: 754ms	remaining: 317ms
    704:	learn: 0.1828307	total: 755ms	remaining: 316ms
    705:	learn: 0.1827839	total: 756ms	remaining: 315ms
    706:	learn: 0.1825888	total: 757ms	remaining: 314ms
    707:	learn: 0.1825518	total: 758ms	remaining: 313ms
    708:	learn: 0.1825377	total: 759ms	remaining: 312ms
    709:	learn: 0.1824638	total: 760ms	remaining: 311ms
    710:	learn: 0.1823646	total: 761ms	remaining: 310ms
    711:	learn: 0.1823118	total: 762ms	remaining: 308ms
    712:	learn: 0.1821917	total: 763ms	remaining: 307ms
    713:	learn: 0.1820590	total: 764ms	remaining: 306ms
    714:	learn: 0.1820103	total: 765ms	remaining: 305ms
    715:	learn: 0.1819061	total: 766ms	remaining: 304ms
    716:	learn: 0.1818437	total: 767ms	remaining: 303ms
    717:	learn: 0.1817094	total: 768ms	remaining: 302ms
    718:	learn: 0.1816326	total: 769ms	remaining: 301ms
    719:	learn: 0.1815890	total: 770ms	remaining: 299ms
    720:	learn: 0.1814641	total: 771ms	remaining: 298ms
    721:	learn: 0.1814556	total: 772ms	remaining: 297ms
    722:	learn: 0.1813318	total: 773ms	remaining: 296ms
    723:	learn: 0.1811224	total: 774ms	remaining: 295ms
    724:	learn: 0.1810638	total: 775ms	remaining: 294ms
    725:	learn: 0.1810220	total: 776ms	remaining: 293ms
    726:	learn: 0.1809337	total: 777ms	remaining: 292ms
    727:	learn: 0.1808394	total: 778ms	remaining: 291ms
    728:	learn: 0.1807417	total: 779ms	remaining: 289ms
    729:	learn: 0.1806478	total: 779ms	remaining: 288ms
    730:	learn: 0.1805611	total: 780ms	remaining: 287ms
    731:	learn: 0.1804984	total: 781ms	remaining: 286ms
    732:	learn: 0.1803219	total: 781ms	remaining: 285ms
    733:	learn: 0.1802650	total: 782ms	remaining: 283ms
    734:	learn: 0.1802240	total: 782ms	remaining: 282ms
    735:	learn: 0.1800958	total: 783ms	remaining: 281ms
    736:	learn: 0.1800251	total: 784ms	remaining: 280ms
    737:	learn: 0.1799702	total: 785ms	remaining: 279ms
    738:	learn: 0.1798695	total: 786ms	remaining: 278ms
    739:	learn: 0.1796843	total: 787ms	remaining: 276ms
    740:	learn: 0.1794551	total: 788ms	remaining: 275ms
    741:	learn: 0.1793267	total: 789ms	remaining: 274ms
    742:	learn: 0.1792288	total: 790ms	remaining: 273ms
    743:	learn: 0.1791656	total: 791ms	remaining: 272ms
    744:	learn: 0.1790517	total: 792ms	remaining: 271ms
    745:	learn: 0.1783948	total: 793ms	remaining: 270ms
    746:	learn: 0.1781999	total: 794ms	remaining: 269ms
    747:	learn: 0.1781184	total: 795ms	remaining: 268ms
    748:	learn: 0.1780390	total: 796ms	remaining: 267ms
    749:	learn: 0.1779577	total: 797ms	remaining: 266ms
    750:	learn: 0.1778420	total: 798ms	remaining: 265ms
    751:	learn: 0.1777927	total: 799ms	remaining: 263ms
    752:	learn: 0.1777009	total: 800ms	remaining: 262ms
    753:	learn: 0.1776305	total: 801ms	remaining: 261ms
    754:	learn: 0.1776234	total: 803ms	remaining: 260ms
    755:	learn: 0.1774272	total: 804ms	remaining: 259ms
    756:	learn: 0.1773632	total: 805ms	remaining: 258ms
    757:	learn: 0.1773125	total: 806ms	remaining: 257ms
    758:	learn: 0.1771977	total: 807ms	remaining: 256ms
    759:	learn: 0.1770882	total: 808ms	remaining: 255ms
    760:	learn: 0.1769519	total: 809ms	remaining: 254ms
    761:	learn: 0.1768773	total: 810ms	remaining: 253ms
    762:	learn: 0.1767571	total: 811ms	remaining: 252ms
    763:	learn: 0.1766744	total: 812ms	remaining: 251ms
    764:	learn: 0.1766278	total: 813ms	remaining: 250ms
    765:	learn: 0.1765921	total: 814ms	remaining: 249ms
    766:	learn: 0.1765034	total: 815ms	remaining: 247ms
    767:	learn: 0.1763935	total: 816ms	remaining: 246ms
    768:	learn: 0.1763215	total: 816ms	remaining: 245ms
    769:	learn: 0.1762753	total: 817ms	remaining: 244ms
    770:	learn: 0.1761184	total: 818ms	remaining: 243ms
    771:	learn: 0.1760679	total: 820ms	remaining: 242ms
    772:	learn: 0.1759751	total: 821ms	remaining: 241ms
    773:	learn: 0.1759408	total: 822ms	remaining: 240ms
    774:	learn: 0.1758656	total: 823ms	remaining: 239ms
    775:	learn: 0.1757045	total: 824ms	remaining: 238ms
    776:	learn: 0.1756885	total: 825ms	remaining: 237ms
    777:	learn: 0.1756151	total: 826ms	remaining: 236ms
    778:	learn: 0.1754534	total: 827ms	remaining: 235ms
    779:	learn: 0.1754038	total: 828ms	remaining: 233ms
    780:	learn: 0.1753274	total: 829ms	remaining: 232ms
    781:	learn: 0.1752423	total: 830ms	remaining: 231ms
    782:	learn: 0.1751737	total: 831ms	remaining: 230ms
    783:	learn: 0.1751200	total: 832ms	remaining: 229ms
    784:	learn: 0.1750101	total: 833ms	remaining: 228ms
    785:	learn: 0.1749242	total: 834ms	remaining: 227ms
    786:	learn: 0.1748157	total: 835ms	remaining: 226ms
    787:	learn: 0.1747180	total: 836ms	remaining: 225ms
    788:	learn: 0.1746344	total: 837ms	remaining: 224ms
    789:	learn: 0.1744802	total: 838ms	remaining: 223ms
    790:	learn: 0.1744308	total: 839ms	remaining: 222ms
    791:	learn: 0.1743467	total: 840ms	remaining: 221ms
    792:	learn: 0.1741891	total: 841ms	remaining: 220ms
    793:	learn: 0.1741026	total: 842ms	remaining: 219ms
    794:	learn: 0.1740919	total: 843ms	remaining: 217ms
    795:	learn: 0.1740024	total: 844ms	remaining: 216ms
    796:	learn: 0.1739316	total: 846ms	remaining: 215ms
    797:	learn: 0.1738431	total: 847ms	remaining: 214ms
    798:	learn: 0.1737096	total: 848ms	remaining: 213ms
    799:	learn: 0.1736685	total: 849ms	remaining: 212ms
    800:	learn: 0.1736095	total: 850ms	remaining: 211ms
    801:	learn: 0.1734350	total: 851ms	remaining: 210ms
    802:	learn: 0.1733600	total: 852ms	remaining: 209ms
    803:	learn: 0.1732726	total: 853ms	remaining: 208ms
    804:	learn: 0.1731907	total: 854ms	remaining: 207ms
    805:	learn: 0.1731110	total: 855ms	remaining: 206ms
    806:	learn: 0.1730253	total: 856ms	remaining: 205ms
    807:	learn: 0.1727416	total: 858ms	remaining: 204ms
    808:	learn: 0.1725766	total: 859ms	remaining: 203ms
    809:	learn: 0.1725112	total: 860ms	remaining: 202ms
    810:	learn: 0.1724007	total: 862ms	remaining: 201ms
    811:	learn: 0.1723401	total: 864ms	remaining: 200ms
    812:	learn: 0.1723318	total: 865ms	remaining: 199ms
    813:	learn: 0.1722373	total: 867ms	remaining: 198ms
    814:	learn: 0.1721903	total: 867ms	remaining: 197ms
    815:	learn: 0.1721447	total: 869ms	remaining: 196ms
    816:	learn: 0.1720063	total: 870ms	remaining: 195ms
    817:	learn: 0.1719310	total: 871ms	remaining: 194ms
    818:	learn: 0.1718488	total: 872ms	remaining: 193ms
    819:	learn: 0.1717158	total: 873ms	remaining: 192ms
    820:	learn: 0.1716636	total: 875ms	remaining: 191ms
    821:	learn: 0.1715643	total: 876ms	remaining: 190ms
    822:	learn: 0.1715519	total: 877ms	remaining: 189ms
    823:	learn: 0.1714721	total: 878ms	remaining: 188ms
    824:	learn: 0.1714241	total: 879ms	remaining: 186ms
    825:	learn: 0.1713741	total: 880ms	remaining: 185ms
    826:	learn: 0.1712894	total: 881ms	remaining: 184ms
    827:	learn: 0.1712156	total: 882ms	remaining: 183ms
    828:	learn: 0.1710933	total: 883ms	remaining: 182ms
    829:	learn: 0.1709428	total: 884ms	remaining: 181ms
    830:	learn: 0.1707342	total: 887ms	remaining: 180ms
    831:	learn: 0.1706806	total: 888ms	remaining: 179ms
    832:	learn: 0.1706323	total: 890ms	remaining: 178ms
    833:	learn: 0.1705547	total: 892ms	remaining: 178ms
    834:	learn: 0.1704906	total: 894ms	remaining: 177ms
    835:	learn: 0.1704384	total: 898ms	remaining: 176ms
    836:	learn: 0.1703664	total: 900ms	remaining: 175ms
    837:	learn: 0.1703421	total: 901ms	remaining: 174ms
    838:	learn: 0.1702889	total: 904ms	remaining: 173ms
    839:	learn: 0.1702394	total: 905ms	remaining: 172ms
    840:	learn: 0.1701953	total: 906ms	remaining: 171ms
    841:	learn: 0.1700083	total: 907ms	remaining: 170ms
    842:	learn: 0.1699230	total: 908ms	remaining: 169ms
    843:	learn: 0.1698651	total: 909ms	remaining: 168ms
    844:	learn: 0.1698091	total: 910ms	remaining: 167ms
    845:	learn: 0.1696943	total: 911ms	remaining: 166ms
    846:	learn: 0.1696818	total: 913ms	remaining: 165ms
    847:	learn: 0.1696409	total: 914ms	remaining: 164ms
    848:	learn: 0.1695495	total: 915ms	remaining: 163ms
    849:	learn: 0.1693580	total: 916ms	remaining: 162ms
    850:	learn: 0.1693232	total: 917ms	remaining: 161ms
    851:	learn: 0.1692147	total: 918ms	remaining: 160ms
    852:	learn: 0.1691701	total: 919ms	remaining: 158ms
    853:	learn: 0.1691081	total: 921ms	remaining: 157ms
    854:	learn: 0.1690599	total: 922ms	remaining: 156ms
    855:	learn: 0.1689848	total: 923ms	remaining: 155ms
    856:	learn: 0.1689042	total: 924ms	remaining: 154ms
    857:	learn: 0.1688448	total: 925ms	remaining: 153ms
    858:	learn: 0.1686435	total: 926ms	remaining: 152ms
    859:	learn: 0.1684881	total: 927ms	remaining: 151ms
    860:	learn: 0.1684281	total: 928ms	remaining: 150ms
    861:	learn: 0.1683353	total: 929ms	remaining: 149ms
    862:	learn: 0.1682926	total: 930ms	remaining: 148ms
    863:	learn: 0.1682132	total: 931ms	remaining: 147ms
    864:	learn: 0.1681716	total: 932ms	remaining: 145ms
    865:	learn: 0.1680957	total: 933ms	remaining: 144ms
    866:	learn: 0.1679674	total: 934ms	remaining: 143ms
    867:	learn: 0.1678410	total: 935ms	remaining: 142ms
    868:	learn: 0.1677622	total: 936ms	remaining: 141ms
    869:	learn: 0.1676490	total: 937ms	remaining: 140ms
    870:	learn: 0.1675459	total: 938ms	remaining: 139ms
    871:	learn: 0.1673697	total: 939ms	remaining: 138ms
    872:	learn: 0.1672977	total: 940ms	remaining: 137ms
    873:	learn: 0.1672892	total: 941ms	remaining: 136ms
    874:	learn: 0.1671466	total: 942ms	remaining: 135ms
    875:	learn: 0.1671164	total: 943ms	remaining: 134ms
    876:	learn: 0.1670459	total: 944ms	remaining: 132ms
    877:	learn: 0.1669938	total: 945ms	remaining: 131ms
    878:	learn: 0.1669264	total: 946ms	remaining: 130ms
    879:	learn: 0.1668668	total: 947ms	remaining: 129ms
    880:	learn: 0.1667763	total: 948ms	remaining: 128ms
    881:	learn: 0.1666563	total: 949ms	remaining: 127ms
    882:	learn: 0.1665640	total: 950ms	remaining: 126ms
    883:	learn: 0.1664900	total: 951ms	remaining: 125ms
    884:	learn: 0.1664178	total: 952ms	remaining: 124ms
    885:	learn: 0.1663352	total: 953ms	remaining: 123ms
    886:	learn: 0.1662073	total: 954ms	remaining: 122ms
    887:	learn: 0.1661994	total: 955ms	remaining: 120ms
    888:	learn: 0.1661312	total: 956ms	remaining: 119ms
    889:	learn: 0.1660775	total: 957ms	remaining: 118ms
    890:	learn: 0.1660201	total: 958ms	remaining: 117ms
    891:	learn: 0.1659350	total: 959ms	remaining: 116ms
    892:	learn: 0.1658820	total: 960ms	remaining: 115ms
    893:	learn: 0.1658149	total: 961ms	remaining: 114ms
    894:	learn: 0.1656997	total: 962ms	remaining: 113ms
    895:	learn: 0.1655870	total: 962ms	remaining: 112ms
    896:	learn: 0.1655118	total: 963ms	remaining: 111ms
    897:	learn: 0.1654371	total: 964ms	remaining: 110ms
    898:	learn: 0.1654370	total: 965ms	remaining: 108ms
    899:	learn: 0.1653900	total: 966ms	remaining: 107ms
    900:	learn: 0.1652394	total: 967ms	remaining: 106ms
    901:	learn: 0.1651879	total: 968ms	remaining: 105ms
    902:	learn: 0.1651387	total: 969ms	remaining: 104ms
    903:	learn: 0.1650559	total: 970ms	remaining: 103ms
    904:	learn: 0.1648178	total: 971ms	remaining: 102ms
    905:	learn: 0.1647682	total: 972ms	remaining: 101ms
    906:	learn: 0.1647320	total: 973ms	remaining: 99.7ms
    907:	learn: 0.1646537	total: 974ms	remaining: 98.7ms
    908:	learn: 0.1646282	total: 975ms	remaining: 97.6ms
    909:	learn: 0.1644979	total: 976ms	remaining: 96.5ms
    910:	learn: 0.1644212	total: 977ms	remaining: 95.4ms
    911:	learn: 0.1643033	total: 978ms	remaining: 94.3ms
    912:	learn: 0.1642471	total: 979ms	remaining: 93.3ms
    913:	learn: 0.1641662	total: 980ms	remaining: 92.2ms
    914:	learn: 0.1641273	total: 981ms	remaining: 91.1ms
    915:	learn: 0.1640665	total: 982ms	remaining: 90.1ms
    916:	learn: 0.1640066	total: 983ms	remaining: 89ms
    917:	learn: 0.1639955	total: 984ms	remaining: 87.9ms
    918:	learn: 0.1639645	total: 985ms	remaining: 86.8ms
    919:	learn: 0.1639074	total: 986ms	remaining: 85.8ms
    920:	learn: 0.1638246	total: 987ms	remaining: 84.7ms
    921:	learn: 0.1637712	total: 988ms	remaining: 83.6ms
    922:	learn: 0.1636936	total: 989ms	remaining: 82.5ms
    923:	learn: 0.1636872	total: 990ms	remaining: 81.4ms
    924:	learn: 0.1636084	total: 991ms	remaining: 80.4ms
    925:	learn: 0.1635503	total: 992ms	remaining: 79.3ms
    926:	learn: 0.1635113	total: 993ms	remaining: 78.2ms
    927:	learn: 0.1634768	total: 994ms	remaining: 77.1ms
    928:	learn: 0.1633019	total: 995ms	remaining: 76ms
    929:	learn: 0.1632673	total: 996ms	remaining: 75ms
    930:	learn: 0.1632171	total: 997ms	remaining: 73.9ms
    931:	learn: 0.1631208	total: 998ms	remaining: 72.8ms
    932:	learn: 0.1630300	total: 999ms	remaining: 71.7ms
    933:	learn: 0.1629650	total: 1000ms	remaining: 70.6ms
    934:	learn: 0.1628940	total: 1s	remaining: 69.6ms
    935:	learn: 0.1628234	total: 1s	remaining: 68.5ms
    936:	learn: 0.1627803	total: 1s	remaining: 67.4ms
    937:	learn: 0.1626891	total: 1s	remaining: 66.3ms
    938:	learn: 0.1626818	total: 1s	remaining: 65.3ms
    939:	learn: 0.1625753	total: 1s	remaining: 64.2ms
    940:	learn: 0.1625204	total: 1.01s	remaining: 63.1ms
    941:	learn: 0.1624531	total: 1.01s	remaining: 62ms
    942:	learn: 0.1622784	total: 1.01s	remaining: 61ms
    943:	learn: 0.1622338	total: 1.01s	remaining: 59.9ms
    944:	learn: 0.1621829	total: 1.01s	remaining: 58.8ms
    945:	learn: 0.1621297	total: 1.01s	remaining: 57.7ms
    946:	learn: 0.1620339	total: 1.01s	remaining: 56.7ms
    947:	learn: 0.1619775	total: 1.01s	remaining: 55.6ms
    948:	learn: 0.1618670	total: 1.01s	remaining: 54.5ms
    949:	learn: 0.1618028	total: 1.01s	remaining: 53.5ms
    950:	learn: 0.1616877	total: 1.02s	remaining: 52.4ms
    951:	learn: 0.1616714	total: 1.02s	remaining: 51.3ms
    952:	learn: 0.1616291	total: 1.02s	remaining: 50.2ms
    953:	learn: 0.1615295	total: 1.02s	remaining: 49.2ms
    954:	learn: 0.1614419	total: 1.02s	remaining: 48.1ms
    955:	learn: 0.1613796	total: 1.02s	remaining: 47ms
    956:	learn: 0.1613530	total: 1.02s	remaining: 46ms
    957:	learn: 0.1612795	total: 1.02s	remaining: 44.9ms
    958:	learn: 0.1611720	total: 1.03s	remaining: 43.9ms
    959:	learn: 0.1607752	total: 1.03s	remaining: 42.8ms
    960:	learn: 0.1607754	total: 1.03s	remaining: 41.7ms
    961:	learn: 0.1606208	total: 1.03s	remaining: 40.7ms
    962:	learn: 0.1605605	total: 1.03s	remaining: 39.6ms
    963:	learn: 0.1604531	total: 1.03s	remaining: 38.5ms
    964:	learn: 0.1602934	total: 1.03s	remaining: 37.5ms
    965:	learn: 0.1602067	total: 1.03s	remaining: 36.4ms
    966:	learn: 0.1601406	total: 1.03s	remaining: 35.3ms
    967:	learn: 0.1600608	total: 1.04s	remaining: 34.2ms
    968:	learn: 0.1600398	total: 1.04s	remaining: 33.2ms
    969:	learn: 0.1599941	total: 1.04s	remaining: 32.1ms
    970:	learn: 0.1599363	total: 1.04s	remaining: 31.1ms
    971:	learn: 0.1597890	total: 1.04s	remaining: 30ms
    972:	learn: 0.1596706	total: 1.04s	remaining: 28.9ms
    973:	learn: 0.1596682	total: 1.04s	remaining: 27.8ms
    974:	learn: 0.1596288	total: 1.04s	remaining: 26.8ms
    975:	learn: 0.1595062	total: 1.04s	remaining: 25.7ms
    976:	learn: 0.1594299	total: 1.05s	remaining: 24.6ms
    977:	learn: 0.1593667	total: 1.05s	remaining: 23.6ms
    978:	learn: 0.1592811	total: 1.05s	remaining: 22.5ms
    979:	learn: 0.1592507	total: 1.05s	remaining: 21.4ms
    980:	learn: 0.1591316	total: 1.05s	remaining: 20.4ms
    981:	learn: 0.1590438	total: 1.05s	remaining: 19.3ms
    982:	learn: 0.1589998	total: 1.05s	remaining: 18.2ms
    983:	learn: 0.1589930	total: 1.05s	remaining: 17.1ms
    984:	learn: 0.1589428	total: 1.06s	remaining: 16.1ms
    985:	learn: 0.1588995	total: 1.06s	remaining: 15.1ms
    986:	learn: 0.1588384	total: 1.06s	remaining: 14ms
    987:	learn: 0.1587936	total: 1.07s	remaining: 13ms
    988:	learn: 0.1586873	total: 1.07s	remaining: 11.9ms
    989:	learn: 0.1585589	total: 1.07s	remaining: 10.8ms
    990:	learn: 0.1585202	total: 1.07s	remaining: 9.74ms
    991:	learn: 0.1584470	total: 1.08s	remaining: 8.71ms
    992:	learn: 0.1584246	total: 1.08s	remaining: 7.63ms
    993:	learn: 0.1582459	total: 1.08s	remaining: 6.55ms
    994:	learn: 0.1581531	total: 1.09s	remaining: 5.47ms
    995:	learn: 0.1581037	total: 1.09s	remaining: 4.38ms
    996:	learn: 0.1580638	total: 1.09s	remaining: 3.29ms
    997:	learn: 0.1579563	total: 1.09s	remaining: 2.19ms
    998:	learn: 0.1578889	total: 1.09s	remaining: 1.09ms
    999:	learn: 0.1578017	total: 1.09s	remaining: 0us
    0:	learn: 0.5798290	total: 803us	remaining: 802ms
    1:	learn: 0.4911787	total: 2.06ms	remaining: 1.03s
    2:	learn: 0.4392525	total: 3.16ms	remaining: 1.05s
    3:	learn: 0.4181642	total: 4.22ms	remaining: 1.05s
    4:	learn: 0.3971830	total: 5.27ms	remaining: 1.05s
    5:	learn: 0.3843620	total: 6.28ms	remaining: 1.04s
    6:	learn: 0.3734990	total: 7.29ms	remaining: 1.03s
    7:	learn: 0.3643726	total: 8.29ms	remaining: 1.03s
    8:	learn: 0.3595191	total: 9.28ms	remaining: 1.02s
    9:	learn: 0.3566928	total: 10.3ms	remaining: 1.02s
    10:	learn: 0.3546917	total: 11.2ms	remaining: 1.01s
    11:	learn: 0.3534995	total: 12.2ms	remaining: 1.01s
    12:	learn: 0.3510383	total: 13.2ms	remaining: 1s
    13:	learn: 0.3501150	total: 14.2ms	remaining: 1s
    14:	learn: 0.3457974	total: 15.2ms	remaining: 1000ms
    15:	learn: 0.3442256	total: 16.2ms	remaining: 996ms
    16:	learn: 0.3411282	total: 17.2ms	remaining: 994ms
    17:	learn: 0.3389218	total: 18.1ms	remaining: 990ms
    18:	learn: 0.3379843	total: 19.1ms	remaining: 986ms
    19:	learn: 0.3364716	total: 20ms	remaining: 982ms
    20:	learn: 0.3350312	total: 21ms	remaining: 981ms
    21:	learn: 0.3335846	total: 22.1ms	remaining: 982ms
    22:	learn: 0.3328778	total: 22.8ms	remaining: 968ms
    23:	learn: 0.3325098	total: 23.4ms	remaining: 952ms
    24:	learn: 0.3315328	total: 24.1ms	remaining: 939ms
    25:	learn: 0.3310299	total: 24.7ms	remaining: 925ms
    26:	learn: 0.3302585	total: 25.4ms	remaining: 915ms
    27:	learn: 0.3288364	total: 26.1ms	remaining: 905ms
    28:	learn: 0.3275929	total: 26.7ms	remaining: 894ms
    29:	learn: 0.3272197	total: 27.4ms	remaining: 886ms
    30:	learn: 0.3255342	total: 28.4ms	remaining: 888ms
    31:	learn: 0.3246756	total: 29.4ms	remaining: 891ms
    32:	learn: 0.3241342	total: 30.4ms	remaining: 892ms
    33:	learn: 0.3236756	total: 31.6ms	remaining: 898ms
    34:	learn: 0.3230162	total: 32.6ms	remaining: 900ms
    35:	learn: 0.3222978	total: 33.6ms	remaining: 901ms
    36:	learn: 0.3210125	total: 34.6ms	remaining: 901ms
    37:	learn: 0.3204914	total: 35.6ms	remaining: 901ms
    38:	learn: 0.3199926	total: 36.6ms	remaining: 901ms
    39:	learn: 0.3191260	total: 37.8ms	remaining: 908ms
    40:	learn: 0.3185866	total: 38.9ms	remaining: 910ms
    41:	learn: 0.3179929	total: 40ms	remaining: 913ms
    42:	learn: 0.3171480	total: 41.1ms	remaining: 914ms
    43:	learn: 0.3164746	total: 42.2ms	remaining: 916ms
    44:	learn: 0.3155780	total: 43.3ms	remaining: 918ms
    45:	learn: 0.3150048	total: 44.6ms	remaining: 925ms
    46:	learn: 0.3144052	total: 45.8ms	remaining: 928ms
    47:	learn: 0.3139076	total: 46.8ms	remaining: 928ms
    48:	learn: 0.3132359	total: 47.9ms	remaining: 930ms
    49:	learn: 0.3122655	total: 48.9ms	remaining: 930ms
    50:	learn: 0.3111837	total: 50ms	remaining: 931ms
    51:	learn: 0.3106031	total: 51.1ms	remaining: 932ms
    52:	learn: 0.3102073	total: 52.1ms	remaining: 931ms
    53:	learn: 0.3096366	total: 53.4ms	remaining: 935ms
    54:	learn: 0.3089976	total: 54.4ms	remaining: 935ms
    55:	learn: 0.3085924	total: 55.4ms	remaining: 934ms
    56:	learn: 0.3079335	total: 56.6ms	remaining: 937ms
    57:	learn: 0.3066875	total: 57.7ms	remaining: 936ms
    58:	learn: 0.3063326	total: 58.7ms	remaining: 936ms
    59:	learn: 0.3058490	total: 59.7ms	remaining: 935ms
    60:	learn: 0.3056062	total: 60.7ms	remaining: 935ms
    61:	learn: 0.3050180	total: 61.9ms	remaining: 936ms
    62:	learn: 0.3045630	total: 62.9ms	remaining: 936ms
    63:	learn: 0.3042157	total: 63.9ms	remaining: 935ms
    64:	learn: 0.3034486	total: 65ms	remaining: 935ms
    65:	learn: 0.3028659	total: 66ms	remaining: 934ms
    66:	learn: 0.3026005	total: 66.9ms	remaining: 932ms
    67:	learn: 0.3019439	total: 67.9ms	remaining: 931ms
    68:	learn: 0.3013473	total: 68.9ms	remaining: 930ms
    69:	learn: 0.3007241	total: 69.9ms	remaining: 929ms
    70:	learn: 0.3005465	total: 70.9ms	remaining: 927ms
    71:	learn: 0.3002829	total: 71.9ms	remaining: 926ms
    72:	learn: 0.2997346	total: 72.9ms	remaining: 925ms
    73:	learn: 0.2992078	total: 73.9ms	remaining: 925ms
    74:	learn: 0.2989197	total: 74.9ms	remaining: 924ms
    75:	learn: 0.2986224	total: 76ms	remaining: 923ms
    76:	learn: 0.2983551	total: 76.9ms	remaining: 922ms
    77:	learn: 0.2978851	total: 77.9ms	remaining: 921ms
    78:	learn: 0.2971856	total: 78.9ms	remaining: 920ms
    79:	learn: 0.2966931	total: 79.9ms	remaining: 919ms
    80:	learn: 0.2963167	total: 80.9ms	remaining: 918ms
    81:	learn: 0.2959927	total: 81.9ms	remaining: 917ms
    82:	learn: 0.2956356	total: 82.9ms	remaining: 916ms
    83:	learn: 0.2950746	total: 83.9ms	remaining: 915ms
    84:	learn: 0.2948001	total: 84.9ms	remaining: 914ms
    85:	learn: 0.2942520	total: 85.8ms	remaining: 912ms
    86:	learn: 0.2938617	total: 86.8ms	remaining: 911ms
    87:	learn: 0.2934179	total: 87.9ms	remaining: 911ms
    88:	learn: 0.2932122	total: 88.8ms	remaining: 909ms
    89:	learn: 0.2927184	total: 89.8ms	remaining: 908ms
    90:	learn: 0.2923297	total: 90.8ms	remaining: 907ms
    91:	learn: 0.2917114	total: 92.2ms	remaining: 910ms
    92:	learn: 0.2908327	total: 93.3ms	remaining: 910ms
    93:	learn: 0.2904359	total: 94.2ms	remaining: 908ms
    94:	learn: 0.2898252	total: 95.1ms	remaining: 906ms
    95:	learn: 0.2895566	total: 96.9ms	remaining: 912ms
    96:	learn: 0.2890463	total: 97.9ms	remaining: 911ms
    97:	learn: 0.2887044	total: 98.9ms	remaining: 910ms
    98:	learn: 0.2883380	total: 99.8ms	remaining: 909ms
    99:	learn: 0.2876251	total: 101ms	remaining: 908ms
    100:	learn: 0.2874130	total: 102ms	remaining: 906ms
    101:	learn: 0.2871244	total: 103ms	remaining: 905ms
    102:	learn: 0.2869269	total: 104ms	remaining: 904ms
    103:	learn: 0.2867687	total: 107ms	remaining: 918ms
    104:	learn: 0.2863876	total: 107ms	remaining: 916ms
    105:	learn: 0.2860268	total: 108ms	remaining: 914ms
    106:	learn: 0.2857196	total: 110ms	remaining: 920ms
    107:	learn: 0.2852623	total: 112ms	remaining: 924ms
    108:	learn: 0.2847067	total: 114ms	remaining: 928ms
    109:	learn: 0.2843981	total: 115ms	remaining: 933ms
    110:	learn: 0.2837216	total: 117ms	remaining: 937ms
    111:	learn: 0.2833803	total: 119ms	remaining: 941ms
    112:	learn: 0.2830123	total: 120ms	remaining: 946ms
    113:	learn: 0.2824857	total: 122ms	remaining: 950ms
    114:	learn: 0.2819295	total: 124ms	remaining: 953ms
    115:	learn: 0.2814690	total: 125ms	remaining: 949ms
    116:	learn: 0.2812590	total: 126ms	remaining: 948ms
    117:	learn: 0.2809544	total: 127ms	remaining: 947ms
    118:	learn: 0.2807416	total: 128ms	remaining: 945ms
    119:	learn: 0.2803735	total: 129ms	remaining: 943ms
    120:	learn: 0.2799787	total: 130ms	remaining: 942ms
    121:	learn: 0.2795776	total: 131ms	remaining: 942ms
    122:	learn: 0.2792365	total: 132ms	remaining: 940ms
    123:	learn: 0.2790339	total: 133ms	remaining: 938ms
    124:	learn: 0.2786889	total: 134ms	remaining: 937ms
    125:	learn: 0.2784480	total: 135ms	remaining: 937ms
    126:	learn: 0.2781273	total: 136ms	remaining: 935ms
    127:	learn: 0.2776666	total: 137ms	remaining: 934ms
    128:	learn: 0.2774315	total: 138ms	remaining: 933ms
    129:	learn: 0.2769360	total: 139ms	remaining: 932ms
    130:	learn: 0.2766244	total: 140ms	remaining: 930ms
    131:	learn: 0.2763996	total: 141ms	remaining: 928ms
    132:	learn: 0.2761318	total: 142ms	remaining: 926ms
    133:	learn: 0.2757103	total: 143ms	remaining: 925ms
    134:	learn: 0.2752380	total: 144ms	remaining: 923ms
    135:	learn: 0.2749859	total: 145ms	remaining: 922ms
    136:	learn: 0.2747787	total: 146ms	remaining: 921ms
    137:	learn: 0.2745095	total: 147ms	remaining: 919ms
    138:	learn: 0.2741842	total: 148ms	remaining: 918ms
    139:	learn: 0.2738728	total: 149ms	remaining: 916ms
    140:	learn: 0.2737256	total: 150ms	remaining: 914ms
    141:	learn: 0.2733999	total: 151ms	remaining: 913ms
    142:	learn: 0.2729669	total: 152ms	remaining: 912ms
    143:	learn: 0.2724758	total: 153ms	remaining: 911ms
    144:	learn: 0.2722636	total: 154ms	remaining: 910ms
    145:	learn: 0.2719858	total: 155ms	remaining: 909ms
    146:	learn: 0.2717939	total: 156ms	remaining: 907ms
    147:	learn: 0.2715384	total: 157ms	remaining: 906ms
    148:	learn: 0.2712746	total: 158ms	remaining: 904ms
    149:	learn: 0.2710149	total: 159ms	remaining: 902ms
    150:	learn: 0.2708692	total: 160ms	remaining: 900ms
    151:	learn: 0.2704227	total: 162ms	remaining: 902ms
    152:	learn: 0.2701573	total: 163ms	remaining: 901ms
    153:	learn: 0.2699192	total: 164ms	remaining: 900ms
    154:	learn: 0.2696738	total: 165ms	remaining: 898ms
    155:	learn: 0.2691139	total: 166ms	remaining: 897ms
    156:	learn: 0.2688056	total: 167ms	remaining: 895ms
    157:	learn: 0.2684857	total: 168ms	remaining: 894ms
    158:	learn: 0.2683542	total: 169ms	remaining: 892ms
    159:	learn: 0.2682128	total: 170ms	remaining: 891ms
    160:	learn: 0.2679873	total: 171ms	remaining: 889ms
    161:	learn: 0.2676332	total: 172ms	remaining: 888ms
    162:	learn: 0.2674121	total: 173ms	remaining: 886ms
    163:	learn: 0.2667965	total: 174ms	remaining: 885ms
    164:	learn: 0.2663211	total: 175ms	remaining: 883ms
    165:	learn: 0.2659713	total: 175ms	remaining: 881ms
    166:	learn: 0.2657699	total: 176ms	remaining: 880ms
    167:	learn: 0.2655188	total: 177ms	remaining: 878ms
    168:	learn: 0.2653241	total: 178ms	remaining: 877ms
    169:	learn: 0.2651124	total: 179ms	remaining: 875ms
    170:	learn: 0.2648342	total: 180ms	remaining: 874ms
    171:	learn: 0.2646767	total: 181ms	remaining: 872ms
    172:	learn: 0.2645027	total: 182ms	remaining: 870ms
    173:	learn: 0.2642672	total: 183ms	remaining: 869ms
    174:	learn: 0.2640775	total: 184ms	remaining: 867ms
    175:	learn: 0.2639425	total: 185ms	remaining: 866ms
    176:	learn: 0.2637833	total: 186ms	remaining: 864ms
    177:	learn: 0.2636760	total: 187ms	remaining: 863ms
    178:	learn: 0.2633942	total: 188ms	remaining: 862ms
    179:	learn: 0.2631663	total: 189ms	remaining: 860ms
    180:	learn: 0.2630012	total: 190ms	remaining: 859ms
    181:	learn: 0.2628788	total: 191ms	remaining: 858ms
    182:	learn: 0.2622702	total: 192ms	remaining: 856ms
    183:	learn: 0.2619784	total: 192ms	remaining: 853ms
    184:	learn: 0.2618844	total: 193ms	remaining: 851ms
    185:	learn: 0.2616537	total: 194ms	remaining: 850ms
    186:	learn: 0.2614587	total: 196ms	remaining: 850ms
    187:	learn: 0.2610475	total: 197ms	remaining: 849ms
    188:	learn: 0.2606811	total: 198ms	remaining: 848ms
    189:	learn: 0.2604840	total: 199ms	remaining: 847ms
    190:	learn: 0.2603242	total: 200ms	remaining: 845ms
    191:	learn: 0.2601576	total: 201ms	remaining: 844ms
    192:	learn: 0.2600087	total: 202ms	remaining: 843ms
    193:	learn: 0.2597905	total: 202ms	remaining: 841ms
    194:	learn: 0.2595384	total: 203ms	remaining: 840ms
    195:	learn: 0.2590417	total: 204ms	remaining: 838ms
    196:	learn: 0.2588686	total: 206ms	remaining: 838ms
    197:	learn: 0.2586489	total: 207ms	remaining: 838ms
    198:	learn: 0.2583590	total: 208ms	remaining: 837ms
    199:	learn: 0.2582458	total: 209ms	remaining: 836ms
    200:	learn: 0.2580712	total: 210ms	remaining: 835ms
    201:	learn: 0.2577584	total: 211ms	remaining: 834ms
    202:	learn: 0.2575805	total: 212ms	remaining: 833ms
    203:	learn: 0.2572706	total: 213ms	remaining: 832ms
    204:	learn: 0.2570898	total: 214ms	remaining: 831ms
    205:	learn: 0.2568950	total: 215ms	remaining: 830ms
    206:	learn: 0.2564676	total: 217ms	remaining: 830ms
    207:	learn: 0.2561632	total: 218ms	remaining: 829ms
    208:	learn: 0.2559505	total: 219ms	remaining: 828ms
    209:	learn: 0.2557884	total: 220ms	remaining: 827ms
    210:	learn: 0.2556494	total: 221ms	remaining: 826ms
    211:	learn: 0.2554686	total: 222ms	remaining: 824ms
    212:	learn: 0.2552370	total: 223ms	remaining: 824ms
    213:	learn: 0.2550580	total: 224ms	remaining: 821ms
    214:	learn: 0.2548716	total: 225ms	remaining: 820ms
    215:	learn: 0.2547097	total: 226ms	remaining: 819ms
    216:	learn: 0.2546198	total: 227ms	remaining: 818ms
    217:	learn: 0.2544365	total: 228ms	remaining: 817ms
    218:	learn: 0.2539454	total: 229ms	remaining: 816ms
    219:	learn: 0.2537523	total: 230ms	remaining: 814ms
    220:	learn: 0.2535979	total: 231ms	remaining: 814ms
    221:	learn: 0.2532074	total: 232ms	remaining: 812ms
    222:	learn: 0.2530136	total: 233ms	remaining: 811ms
    223:	learn: 0.2525268	total: 234ms	remaining: 810ms
    224:	learn: 0.2523466	total: 235ms	remaining: 809ms
    225:	learn: 0.2517519	total: 236ms	remaining: 809ms
    226:	learn: 0.2514525	total: 237ms	remaining: 809ms
    227:	learn: 0.2510623	total: 239ms	remaining: 808ms
    228:	learn: 0.2507003	total: 240ms	remaining: 807ms
    229:	learn: 0.2502342	total: 241ms	remaining: 806ms
    230:	learn: 0.2500499	total: 242ms	remaining: 805ms
    231:	learn: 0.2497192	total: 243ms	remaining: 804ms
    232:	learn: 0.2495268	total: 244ms	remaining: 803ms
    233:	learn: 0.2494172	total: 245ms	remaining: 803ms
    234:	learn: 0.2492452	total: 246ms	remaining: 801ms
    235:	learn: 0.2491425	total: 247ms	remaining: 800ms
    236:	learn: 0.2488136	total: 248ms	remaining: 799ms
    237:	learn: 0.2485323	total: 249ms	remaining: 799ms
    238:	learn: 0.2482579	total: 250ms	remaining: 797ms
    239:	learn: 0.2479863	total: 251ms	remaining: 796ms
    240:	learn: 0.2477396	total: 252ms	remaining: 795ms
    241:	learn: 0.2473147	total: 253ms	remaining: 794ms
    242:	learn: 0.2471070	total: 255ms	remaining: 793ms
    243:	learn: 0.2469555	total: 256ms	remaining: 792ms
    244:	learn: 0.2466471	total: 257ms	remaining: 791ms
    245:	learn: 0.2465096	total: 258ms	remaining: 790ms
    246:	learn: 0.2461953	total: 259ms	remaining: 789ms
    247:	learn: 0.2460389	total: 260ms	remaining: 787ms
    248:	learn: 0.2458915	total: 261ms	remaining: 786ms
    249:	learn: 0.2457393	total: 262ms	remaining: 785ms
    250:	learn: 0.2455770	total: 263ms	remaining: 785ms
    251:	learn: 0.2452351	total: 264ms	remaining: 784ms
    252:	learn: 0.2450713	total: 265ms	remaining: 783ms
    253:	learn: 0.2447850	total: 266ms	remaining: 783ms
    254:	learn: 0.2446539	total: 268ms	remaining: 782ms
    255:	learn: 0.2444869	total: 269ms	remaining: 781ms
    256:	learn: 0.2443485	total: 270ms	remaining: 780ms
    257:	learn: 0.2440563	total: 271ms	remaining: 779ms
    258:	learn: 0.2438262	total: 272ms	remaining: 778ms
    259:	learn: 0.2436559	total: 273ms	remaining: 777ms
    260:	learn: 0.2434326	total: 274ms	remaining: 776ms
    261:	learn: 0.2433141	total: 275ms	remaining: 775ms
    262:	learn: 0.2422695	total: 277ms	remaining: 777ms
    263:	learn: 0.2421629	total: 278ms	remaining: 776ms
    264:	learn: 0.2420269	total: 280ms	remaining: 777ms
    265:	learn: 0.2418772	total: 281ms	remaining: 775ms
    266:	learn: 0.2417570	total: 282ms	remaining: 774ms
    267:	learn: 0.2415591	total: 283ms	remaining: 773ms
    268:	learn: 0.2411686	total: 284ms	remaining: 771ms
    269:	learn: 0.2409987	total: 293ms	remaining: 791ms
    270:	learn: 0.2407366	total: 294ms	remaining: 790ms
    271:	learn: 0.2405129	total: 295ms	remaining: 789ms
    272:	learn: 0.2404203	total: 296ms	remaining: 788ms
    273:	learn: 0.2402368	total: 297ms	remaining: 787ms
    274:	learn: 0.2401081	total: 298ms	remaining: 785ms
    275:	learn: 0.2399562	total: 299ms	remaining: 784ms
    276:	learn: 0.2396779	total: 300ms	remaining: 783ms
    277:	learn: 0.2396002	total: 301ms	remaining: 781ms
    278:	learn: 0.2395006	total: 302ms	remaining: 780ms
    279:	learn: 0.2393972	total: 303ms	remaining: 779ms
    280:	learn: 0.2391873	total: 304ms	remaining: 777ms
    281:	learn: 0.2389900	total: 305ms	remaining: 776ms
    282:	learn: 0.2388712	total: 306ms	remaining: 775ms
    283:	learn: 0.2387300	total: 307ms	remaining: 773ms
    284:	learn: 0.2386391	total: 308ms	remaining: 772ms
    285:	learn: 0.2384524	total: 309ms	remaining: 771ms
    286:	learn: 0.2382936	total: 310ms	remaining: 769ms
    287:	learn: 0.2380808	total: 311ms	remaining: 768ms
    288:	learn: 0.2379405	total: 312ms	remaining: 766ms
    289:	learn: 0.2378288	total: 313ms	remaining: 765ms
    290:	learn: 0.2374579	total: 313ms	remaining: 764ms
    291:	learn: 0.2373148	total: 314ms	remaining: 762ms
    292:	learn: 0.2370579	total: 315ms	remaining: 761ms
    293:	learn: 0.2369103	total: 316ms	remaining: 760ms
    294:	learn: 0.2368118	total: 317ms	remaining: 758ms
    295:	learn: 0.2363206	total: 318ms	remaining: 757ms
    296:	learn: 0.2362587	total: 319ms	remaining: 756ms
    297:	learn: 0.2361788	total: 320ms	remaining: 755ms
    298:	learn: 0.2358977	total: 321ms	remaining: 753ms
    299:	learn: 0.2357738	total: 322ms	remaining: 752ms
    300:	learn: 0.2354026	total: 323ms	remaining: 750ms
    301:	learn: 0.2351721	total: 324ms	remaining: 749ms
    302:	learn: 0.2349118	total: 325ms	remaining: 748ms
    303:	learn: 0.2347493	total: 326ms	remaining: 746ms
    304:	learn: 0.2346403	total: 327ms	remaining: 745ms
    305:	learn: 0.2345441	total: 328ms	remaining: 743ms
    306:	learn: 0.2344168	total: 329ms	remaining: 742ms
    307:	learn: 0.2339553	total: 330ms	remaining: 741ms
    308:	learn: 0.2337960	total: 331ms	remaining: 739ms
    309:	learn: 0.2336361	total: 332ms	remaining: 738ms
    310:	learn: 0.2334153	total: 333ms	remaining: 737ms
    311:	learn: 0.2326657	total: 334ms	remaining: 736ms
    312:	learn: 0.2325680	total: 335ms	remaining: 734ms
    313:	learn: 0.2324743	total: 336ms	remaining: 733ms
    314:	learn: 0.2323945	total: 337ms	remaining: 732ms
    315:	learn: 0.2322680	total: 338ms	remaining: 731ms
    316:	learn: 0.2320060	total: 339ms	remaining: 730ms
    317:	learn: 0.2317410	total: 340ms	remaining: 729ms
    318:	learn: 0.2316129	total: 341ms	remaining: 728ms
    319:	learn: 0.2312989	total: 342ms	remaining: 727ms
    320:	learn: 0.2311849	total: 343ms	remaining: 725ms
    321:	learn: 0.2310330	total: 344ms	remaining: 724ms
    322:	learn: 0.2309818	total: 345ms	remaining: 723ms
    323:	learn: 0.2308828	total: 346ms	remaining: 722ms
    324:	learn: 0.2307375	total: 347ms	remaining: 720ms
    325:	learn: 0.2305255	total: 348ms	remaining: 719ms
    326:	learn: 0.2303718	total: 349ms	remaining: 718ms
    327:	learn: 0.2302089	total: 350ms	remaining: 716ms
    328:	learn: 0.2298478	total: 351ms	remaining: 715ms
    329:	learn: 0.2296628	total: 352ms	remaining: 714ms
    330:	learn: 0.2296062	total: 353ms	remaining: 713ms
    331:	learn: 0.2287164	total: 354ms	remaining: 711ms
    332:	learn: 0.2285933	total: 355ms	remaining: 710ms
    333:	learn: 0.2284780	total: 356ms	remaining: 709ms
    334:	learn: 0.2283701	total: 356ms	remaining: 708ms
    335:	learn: 0.2281691	total: 357ms	remaining: 706ms
    336:	learn: 0.2280116	total: 358ms	remaining: 705ms
    337:	learn: 0.2278157	total: 359ms	remaining: 704ms
    338:	learn: 0.2275768	total: 360ms	remaining: 702ms
    339:	learn: 0.2274822	total: 361ms	remaining: 701ms
    340:	learn: 0.2273759	total: 362ms	remaining: 700ms
    341:	learn: 0.2273026	total: 363ms	remaining: 698ms
    342:	learn: 0.2271881	total: 364ms	remaining: 697ms
    343:	learn: 0.2270267	total: 365ms	remaining: 697ms
    344:	learn: 0.2268938	total: 366ms	remaining: 696ms
    345:	learn: 0.2265093	total: 367ms	remaining: 694ms
    346:	learn: 0.2264329	total: 368ms	remaining: 693ms
    347:	learn: 0.2260521	total: 369ms	remaining: 692ms
    348:	learn: 0.2259106	total: 370ms	remaining: 690ms
    349:	learn: 0.2256804	total: 371ms	remaining: 689ms
    350:	learn: 0.2255000	total: 372ms	remaining: 688ms
    351:	learn: 0.2253770	total: 373ms	remaining: 686ms
    352:	learn: 0.2251377	total: 374ms	remaining: 685ms
    353:	learn: 0.2249946	total: 375ms	remaining: 684ms
    354:	learn: 0.2247552	total: 376ms	remaining: 683ms
    355:	learn: 0.2246598	total: 377ms	remaining: 682ms
    356:	learn: 0.2244934	total: 378ms	remaining: 681ms
    357:	learn: 0.2243312	total: 379ms	remaining: 680ms
    358:	learn: 0.2241532	total: 380ms	remaining: 678ms
    359:	learn: 0.2235843	total: 381ms	remaining: 677ms
    360:	learn: 0.2234722	total: 382ms	remaining: 676ms
    361:	learn: 0.2233734	total: 383ms	remaining: 675ms
    362:	learn: 0.2228409	total: 384ms	remaining: 673ms
    363:	learn: 0.2226250	total: 385ms	remaining: 672ms
    364:	learn: 0.2221810	total: 386ms	remaining: 671ms
    365:	learn: 0.2220260	total: 387ms	remaining: 670ms
    366:	learn: 0.2218064	total: 387ms	remaining: 668ms
    367:	learn: 0.2214337	total: 388ms	remaining: 667ms
    368:	learn: 0.2213057	total: 389ms	remaining: 666ms
    369:	learn: 0.2211498	total: 390ms	remaining: 665ms
    370:	learn: 0.2211041	total: 391ms	remaining: 663ms
    371:	learn: 0.2209049	total: 392ms	remaining: 662ms
    372:	learn: 0.2207189	total: 393ms	remaining: 661ms
    373:	learn: 0.2205603	total: 394ms	remaining: 660ms
    374:	learn: 0.2204670	total: 395ms	remaining: 659ms
    375:	learn: 0.2202892	total: 396ms	remaining: 657ms
    376:	learn: 0.2201863	total: 397ms	remaining: 656ms
    377:	learn: 0.2200614	total: 398ms	remaining: 655ms
    378:	learn: 0.2199630	total: 399ms	remaining: 654ms
    379:	learn: 0.2196625	total: 400ms	remaining: 652ms
    380:	learn: 0.2195686	total: 401ms	remaining: 651ms
    381:	learn: 0.2193597	total: 402ms	remaining: 650ms
    382:	learn: 0.2191362	total: 403ms	remaining: 649ms
    383:	learn: 0.2190003	total: 404ms	remaining: 648ms
    384:	learn: 0.2188954	total: 405ms	remaining: 646ms
    385:	learn: 0.2188134	total: 406ms	remaining: 646ms
    386:	learn: 0.2185708	total: 407ms	remaining: 644ms
    387:	learn: 0.2184611	total: 408ms	remaining: 643ms
    388:	learn: 0.2184038	total: 409ms	remaining: 642ms
    389:	learn: 0.2182483	total: 410ms	remaining: 641ms
    390:	learn: 0.2177648	total: 411ms	remaining: 640ms
    391:	learn: 0.2176273	total: 412ms	remaining: 639ms
    392:	learn: 0.2174620	total: 413ms	remaining: 638ms
    393:	learn: 0.2173249	total: 414ms	remaining: 637ms
    394:	learn: 0.2170533	total: 415ms	remaining: 635ms
    395:	learn: 0.2169009	total: 416ms	remaining: 634ms
    396:	learn: 0.2167179	total: 417ms	remaining: 633ms
    397:	learn: 0.2165863	total: 418ms	remaining: 632ms
    398:	learn: 0.2164197	total: 418ms	remaining: 630ms
    399:	learn: 0.2162981	total: 420ms	remaining: 630ms
    400:	learn: 0.2161329	total: 421ms	remaining: 629ms
    401:	learn: 0.2158220	total: 422ms	remaining: 627ms
    402:	learn: 0.2154560	total: 423ms	remaining: 627ms
    403:	learn: 0.2153776	total: 424ms	remaining: 625ms
    404:	learn: 0.2153500	total: 425ms	remaining: 624ms
    405:	learn: 0.2152030	total: 426ms	remaining: 623ms
    406:	learn: 0.2149446	total: 427ms	remaining: 622ms
    407:	learn: 0.2146688	total: 428ms	remaining: 621ms
    408:	learn: 0.2145844	total: 429ms	remaining: 620ms
    409:	learn: 0.2144707	total: 430ms	remaining: 619ms
    410:	learn: 0.2142327	total: 431ms	remaining: 617ms
    411:	learn: 0.2141174	total: 432ms	remaining: 616ms
    412:	learn: 0.2140561	total: 433ms	remaining: 615ms
    413:	learn: 0.2139463	total: 434ms	remaining: 614ms
    414:	learn: 0.2138307	total: 435ms	remaining: 613ms
    415:	learn: 0.2137693	total: 436ms	remaining: 612ms
    416:	learn: 0.2136414	total: 436ms	remaining: 610ms
    417:	learn: 0.2131847	total: 437ms	remaining: 609ms
    418:	learn: 0.2130956	total: 438ms	remaining: 608ms
    419:	learn: 0.2129997	total: 440ms	remaining: 607ms
    420:	learn: 0.2128943	total: 441ms	remaining: 606ms
    421:	learn: 0.2127042	total: 442ms	remaining: 605ms
    422:	learn: 0.2126076	total: 443ms	remaining: 604ms
    423:	learn: 0.2124599	total: 444ms	remaining: 603ms
    424:	learn: 0.2123678	total: 445ms	remaining: 602ms
    425:	learn: 0.2123100	total: 446ms	remaining: 601ms
    426:	learn: 0.2122334	total: 447ms	remaining: 599ms
    427:	learn: 0.2121359	total: 447ms	remaining: 598ms
    428:	learn: 0.2120483	total: 448ms	remaining: 596ms
    429:	learn: 0.2119246	total: 449ms	remaining: 595ms
    430:	learn: 0.2118575	total: 450ms	remaining: 594ms
    431:	learn: 0.2116455	total: 451ms	remaining: 593ms
    432:	learn: 0.2115773	total: 452ms	remaining: 591ms
    433:	learn: 0.2113769	total: 453ms	remaining: 590ms
    434:	learn: 0.2111748	total: 454ms	remaining: 589ms
    435:	learn: 0.2110558	total: 455ms	remaining: 588ms
    436:	learn: 0.2109233	total: 455ms	remaining: 587ms
    437:	learn: 0.2107856	total: 456ms	remaining: 586ms
    438:	learn: 0.2106230	total: 457ms	remaining: 584ms
    439:	learn: 0.2104335	total: 458ms	remaining: 583ms
    440:	learn: 0.2103096	total: 459ms	remaining: 582ms
    441:	learn: 0.2101638	total: 462ms	remaining: 583ms
    442:	learn: 0.2099479	total: 463ms	remaining: 582ms
    443:	learn: 0.2098501	total: 465ms	remaining: 582ms
    444:	learn: 0.2097383	total: 466ms	remaining: 581ms
    445:	learn: 0.2095943	total: 469ms	remaining: 582ms
    446:	learn: 0.2094328	total: 470ms	remaining: 581ms
    447:	learn: 0.2092515	total: 471ms	remaining: 580ms
    448:	learn: 0.2090651	total: 475ms	remaining: 583ms
    449:	learn: 0.2088983	total: 477ms	remaining: 583ms
    450:	learn: 0.2087669	total: 480ms	remaining: 584ms
    451:	learn: 0.2086753	total: 482ms	remaining: 584ms
    452:	learn: 0.2086255	total: 483ms	remaining: 583ms
    453:	learn: 0.2085483	total: 483ms	remaining: 581ms
    454:	learn: 0.2084462	total: 484ms	remaining: 580ms
    455:	learn: 0.2083311	total: 485ms	remaining: 578ms
    456:	learn: 0.2082103	total: 486ms	remaining: 577ms
    457:	learn: 0.2080862	total: 487ms	remaining: 576ms
    458:	learn: 0.2080300	total: 488ms	remaining: 575ms
    459:	learn: 0.2079197	total: 489ms	remaining: 574ms
    460:	learn: 0.2077309	total: 490ms	remaining: 573ms
    461:	learn: 0.2075416	total: 491ms	remaining: 571ms
    462:	learn: 0.2074516	total: 492ms	remaining: 570ms
    463:	learn: 0.2072683	total: 493ms	remaining: 569ms
    464:	learn: 0.2071715	total: 494ms	remaining: 568ms
    465:	learn: 0.2070142	total: 495ms	remaining: 567ms
    466:	learn: 0.2069028	total: 496ms	remaining: 566ms
    467:	learn: 0.2067868	total: 497ms	remaining: 565ms
    468:	learn: 0.2066827	total: 500ms	remaining: 566ms
    469:	learn: 0.2065707	total: 507ms	remaining: 571ms
    470:	learn: 0.2064158	total: 508ms	remaining: 570ms
    471:	learn: 0.2063031	total: 508ms	remaining: 569ms
    472:	learn: 0.2062214	total: 510ms	remaining: 568ms
    473:	learn: 0.2061959	total: 511ms	remaining: 567ms
    474:	learn: 0.2060745	total: 512ms	remaining: 566ms
    475:	learn: 0.2059824	total: 513ms	remaining: 565ms
    476:	learn: 0.2058557	total: 514ms	remaining: 564ms
    477:	learn: 0.2056550	total: 515ms	remaining: 563ms
    478:	learn: 0.2054102	total: 516ms	remaining: 562ms
    479:	learn: 0.2052798	total: 517ms	remaining: 561ms
    480:	learn: 0.2052211	total: 518ms	remaining: 559ms
    481:	learn: 0.2050995	total: 519ms	remaining: 558ms
    482:	learn: 0.2049457	total: 520ms	remaining: 557ms
    483:	learn: 0.2048719	total: 521ms	remaining: 556ms
    484:	learn: 0.2048035	total: 522ms	remaining: 555ms
    485:	learn: 0.2046316	total: 523ms	remaining: 554ms
    486:	learn: 0.2043360	total: 524ms	remaining: 552ms
    487:	learn: 0.2042385	total: 525ms	remaining: 551ms
    488:	learn: 0.2041558	total: 526ms	remaining: 550ms
    489:	learn: 0.2040673	total: 528ms	remaining: 549ms
    490:	learn: 0.2037329	total: 529ms	remaining: 548ms
    491:	learn: 0.2036438	total: 530ms	remaining: 547ms
    492:	learn: 0.2035796	total: 531ms	remaining: 546ms
    493:	learn: 0.2034489	total: 532ms	remaining: 545ms
    494:	learn: 0.2033521	total: 533ms	remaining: 543ms
    495:	learn: 0.2030452	total: 534ms	remaining: 542ms
    496:	learn: 0.2029467	total: 535ms	remaining: 541ms
    497:	learn: 0.2028129	total: 536ms	remaining: 540ms
    498:	learn: 0.2027229	total: 537ms	remaining: 539ms
    499:	learn: 0.2026666	total: 538ms	remaining: 538ms
    500:	learn: 0.2025559	total: 539ms	remaining: 537ms
    501:	learn: 0.2024459	total: 540ms	remaining: 535ms
    502:	learn: 0.2023396	total: 541ms	remaining: 534ms
    503:	learn: 0.2021012	total: 542ms	remaining: 533ms
    504:	learn: 0.2019678	total: 543ms	remaining: 532ms
    505:	learn: 0.2018678	total: 544ms	remaining: 531ms
    506:	learn: 0.2017693	total: 545ms	remaining: 530ms
    507:	learn: 0.2016302	total: 546ms	remaining: 528ms
    508:	learn: 0.2015662	total: 547ms	remaining: 527ms
    509:	learn: 0.2014649	total: 548ms	remaining: 526ms
    510:	learn: 0.2013382	total: 549ms	remaining: 525ms
    511:	learn: 0.2011247	total: 550ms	remaining: 524ms
    512:	learn: 0.2010558	total: 551ms	remaining: 523ms
    513:	learn: 0.2009326	total: 552ms	remaining: 521ms
    514:	learn: 0.2008132	total: 553ms	remaining: 520ms
    515:	learn: 0.2006767	total: 554ms	remaining: 519ms
    516:	learn: 0.2002132	total: 555ms	remaining: 518ms
    517:	learn: 0.2001783	total: 556ms	remaining: 517ms
    518:	learn: 0.2000985	total: 557ms	remaining: 516ms
    519:	learn: 0.2000308	total: 558ms	remaining: 515ms
    520:	learn: 0.1999593	total: 558ms	remaining: 513ms
    521:	learn: 0.1997287	total: 559ms	remaining: 512ms
    522:	learn: 0.1996447	total: 561ms	remaining: 511ms
    523:	learn: 0.1993954	total: 562ms	remaining: 510ms
    524:	learn: 0.1993035	total: 563ms	remaining: 509ms
    525:	learn: 0.1991929	total: 564ms	remaining: 508ms
    526:	learn: 0.1989853	total: 565ms	remaining: 507ms
    527:	learn: 0.1988979	total: 566ms	remaining: 506ms
    528:	learn: 0.1987546	total: 567ms	remaining: 505ms
    529:	learn: 0.1986602	total: 568ms	remaining: 504ms
    530:	learn: 0.1985583	total: 569ms	remaining: 503ms
    531:	learn: 0.1985388	total: 570ms	remaining: 502ms
    532:	learn: 0.1984301	total: 571ms	remaining: 500ms
    533:	learn: 0.1982364	total: 572ms	remaining: 499ms
    534:	learn: 0.1981471	total: 573ms	remaining: 498ms
    535:	learn: 0.1979308	total: 574ms	remaining: 497ms
    536:	learn: 0.1978239	total: 575ms	remaining: 496ms
    537:	learn: 0.1977473	total: 576ms	remaining: 495ms
    538:	learn: 0.1976023	total: 577ms	remaining: 494ms
    539:	learn: 0.1975413	total: 578ms	remaining: 492ms
    540:	learn: 0.1974540	total: 579ms	remaining: 491ms
    541:	learn: 0.1973341	total: 580ms	remaining: 490ms
    542:	learn: 0.1972062	total: 581ms	remaining: 489ms
    543:	learn: 0.1971238	total: 582ms	remaining: 488ms
    544:	learn: 0.1970345	total: 583ms	remaining: 487ms
    545:	learn: 0.1969305	total: 584ms	remaining: 486ms
    546:	learn: 0.1968547	total: 585ms	remaining: 484ms
    547:	learn: 0.1968363	total: 586ms	remaining: 483ms
    548:	learn: 0.1967652	total: 587ms	remaining: 482ms
    549:	learn: 0.1967132	total: 588ms	remaining: 481ms
    550:	learn: 0.1965097	total: 589ms	remaining: 480ms
    551:	learn: 0.1962004	total: 590ms	remaining: 479ms
    552:	learn: 0.1961076	total: 591ms	remaining: 478ms
    553:	learn: 0.1960881	total: 592ms	remaining: 476ms
    554:	learn: 0.1959810	total: 593ms	remaining: 475ms
    555:	learn: 0.1958851	total: 594ms	remaining: 474ms
    556:	learn: 0.1958266	total: 595ms	remaining: 473ms
    557:	learn: 0.1956734	total: 596ms	remaining: 472ms
    558:	learn: 0.1956133	total: 597ms	remaining: 471ms
    559:	learn: 0.1955567	total: 598ms	remaining: 470ms
    560:	learn: 0.1955115	total: 599ms	remaining: 468ms
    561:	learn: 0.1953722	total: 599ms	remaining: 467ms
    562:	learn: 0.1952697	total: 600ms	remaining: 466ms
    563:	learn: 0.1952158	total: 601ms	remaining: 465ms
    564:	learn: 0.1950128	total: 602ms	remaining: 464ms
    565:	learn: 0.1949616	total: 603ms	remaining: 463ms
    566:	learn: 0.1948427	total: 604ms	remaining: 462ms
    567:	learn: 0.1947646	total: 605ms	remaining: 460ms
    568:	learn: 0.1946875	total: 606ms	remaining: 459ms
    569:	learn: 0.1945831	total: 607ms	remaining: 458ms
    570:	learn: 0.1945266	total: 608ms	remaining: 457ms
    571:	learn: 0.1944191	total: 609ms	remaining: 456ms
    572:	learn: 0.1941947	total: 610ms	remaining: 454ms
    573:	learn: 0.1941512	total: 611ms	remaining: 454ms
    574:	learn: 0.1940690	total: 612ms	remaining: 453ms
    575:	learn: 0.1939486	total: 613ms	remaining: 451ms
    576:	learn: 0.1938778	total: 614ms	remaining: 450ms
    577:	learn: 0.1937012	total: 615ms	remaining: 449ms
    578:	learn: 0.1936464	total: 616ms	remaining: 448ms
    579:	learn: 0.1935264	total: 617ms	remaining: 447ms
    580:	learn: 0.1934193	total: 618ms	remaining: 446ms
    581:	learn: 0.1933031	total: 619ms	remaining: 445ms
    582:	learn: 0.1931889	total: 620ms	remaining: 443ms
    583:	learn: 0.1931216	total: 621ms	remaining: 442ms
    584:	learn: 0.1930210	total: 622ms	remaining: 441ms
    585:	learn: 0.1929230	total: 623ms	remaining: 440ms
    586:	learn: 0.1927403	total: 624ms	remaining: 439ms
    587:	learn: 0.1926351	total: 625ms	remaining: 438ms
    588:	learn: 0.1925512	total: 626ms	remaining: 437ms
    589:	learn: 0.1923439	total: 627ms	remaining: 435ms
    590:	learn: 0.1922878	total: 628ms	remaining: 434ms
    591:	learn: 0.1920818	total: 628ms	remaining: 433ms
    592:	learn: 0.1920034	total: 629ms	remaining: 432ms
    593:	learn: 0.1918271	total: 630ms	remaining: 431ms
    594:	learn: 0.1917031	total: 631ms	remaining: 430ms
    595:	learn: 0.1915930	total: 632ms	remaining: 429ms
    596:	learn: 0.1914798	total: 633ms	remaining: 427ms
    597:	learn: 0.1913958	total: 634ms	remaining: 426ms
    598:	learn: 0.1913710	total: 635ms	remaining: 425ms
    599:	learn: 0.1912815	total: 637ms	remaining: 425ms
    600:	learn: 0.1912266	total: 638ms	remaining: 424ms
    601:	learn: 0.1911359	total: 640ms	remaining: 423ms
    602:	learn: 0.1910664	total: 641ms	remaining: 422ms
    603:	learn: 0.1909914	total: 644ms	remaining: 422ms
    604:	learn: 0.1908906	total: 645ms	remaining: 421ms
    605:	learn: 0.1907823	total: 647ms	remaining: 421ms
    606:	learn: 0.1907307	total: 648ms	remaining: 420ms
    607:	learn: 0.1906315	total: 649ms	remaining: 419ms
    608:	learn: 0.1905769	total: 650ms	remaining: 417ms
    609:	learn: 0.1905044	total: 650ms	remaining: 416ms
    610:	learn: 0.1904344	total: 651ms	remaining: 415ms
    611:	learn: 0.1903354	total: 652ms	remaining: 413ms
    612:	learn: 0.1901822	total: 653ms	remaining: 412ms
    613:	learn: 0.1900121	total: 654ms	remaining: 411ms
    614:	learn: 0.1899717	total: 655ms	remaining: 410ms
    615:	learn: 0.1898322	total: 656ms	remaining: 409ms
    616:	learn: 0.1897655	total: 657ms	remaining: 408ms
    617:	learn: 0.1897174	total: 658ms	remaining: 407ms
    618:	learn: 0.1896366	total: 659ms	remaining: 405ms
    619:	learn: 0.1895992	total: 660ms	remaining: 404ms
    620:	learn: 0.1895306	total: 661ms	remaining: 403ms
    621:	learn: 0.1894357	total: 662ms	remaining: 402ms
    622:	learn: 0.1893531	total: 662ms	remaining: 401ms
    623:	learn: 0.1892950	total: 663ms	remaining: 400ms
    624:	learn: 0.1892373	total: 664ms	remaining: 399ms
    625:	learn: 0.1890686	total: 665ms	remaining: 397ms
    626:	learn: 0.1889383	total: 667ms	remaining: 397ms
    627:	learn: 0.1888937	total: 668ms	remaining: 395ms
    628:	learn: 0.1888051	total: 669ms	remaining: 394ms
    629:	learn: 0.1885168	total: 670ms	remaining: 393ms
    630:	learn: 0.1884430	total: 670ms	remaining: 392ms
    631:	learn: 0.1883415	total: 671ms	remaining: 391ms
    632:	learn: 0.1880869	total: 672ms	remaining: 390ms
    633:	learn: 0.1879983	total: 674ms	remaining: 389ms
    634:	learn: 0.1879094	total: 674ms	remaining: 388ms
    635:	learn: 0.1878286	total: 675ms	remaining: 387ms
    636:	learn: 0.1876581	total: 676ms	remaining: 385ms
    637:	learn: 0.1875529	total: 678ms	remaining: 385ms
    638:	learn: 0.1874209	total: 679ms	remaining: 383ms
    639:	learn: 0.1873348	total: 680ms	remaining: 382ms
    640:	learn: 0.1871705	total: 681ms	remaining: 381ms
    641:	learn: 0.1869536	total: 682ms	remaining: 380ms
    642:	learn: 0.1868143	total: 683ms	remaining: 379ms
    643:	learn: 0.1867725	total: 684ms	remaining: 378ms
    644:	learn: 0.1867519	total: 685ms	remaining: 377ms
    645:	learn: 0.1866763	total: 686ms	remaining: 376ms
    646:	learn: 0.1865017	total: 687ms	remaining: 375ms
    647:	learn: 0.1864504	total: 688ms	remaining: 374ms
    648:	learn: 0.1863692	total: 689ms	remaining: 373ms
    649:	learn: 0.1862873	total: 690ms	remaining: 372ms
    650:	learn: 0.1862077	total: 691ms	remaining: 370ms
    651:	learn: 0.1857322	total: 692ms	remaining: 369ms
    652:	learn: 0.1857203	total: 693ms	remaining: 368ms
    653:	learn: 0.1856649	total: 694ms	remaining: 367ms
    654:	learn: 0.1855900	total: 695ms	remaining: 366ms
    655:	learn: 0.1854795	total: 696ms	remaining: 365ms
    656:	learn: 0.1854353	total: 697ms	remaining: 364ms
    657:	learn: 0.1853255	total: 698ms	remaining: 363ms
    658:	learn: 0.1852342	total: 699ms	remaining: 362ms
    659:	learn: 0.1851963	total: 700ms	remaining: 361ms
    660:	learn: 0.1851334	total: 701ms	remaining: 359ms
    661:	learn: 0.1849362	total: 702ms	remaining: 358ms
    662:	learn: 0.1848062	total: 703ms	remaining: 357ms
    663:	learn: 0.1846710	total: 704ms	remaining: 356ms
    664:	learn: 0.1846188	total: 705ms	remaining: 355ms
    665:	learn: 0.1845223	total: 706ms	remaining: 354ms
    666:	learn: 0.1844611	total: 707ms	remaining: 353ms
    667:	learn: 0.1843969	total: 707ms	remaining: 352ms
    668:	learn: 0.1842840	total: 708ms	remaining: 350ms
    669:	learn: 0.1842710	total: 709ms	remaining: 349ms
    670:	learn: 0.1841906	total: 710ms	remaining: 348ms
    671:	learn: 0.1841171	total: 711ms	remaining: 347ms
    672:	learn: 0.1840500	total: 712ms	remaining: 346ms
    673:	learn: 0.1838846	total: 713ms	remaining: 345ms
    674:	learn: 0.1838037	total: 714ms	remaining: 344ms
    675:	learn: 0.1835951	total: 715ms	remaining: 343ms
    676:	learn: 0.1835383	total: 716ms	remaining: 342ms
    677:	learn: 0.1834716	total: 717ms	remaining: 341ms
    678:	learn: 0.1833519	total: 718ms	remaining: 339ms
    679:	learn: 0.1832260	total: 719ms	remaining: 338ms
    680:	learn: 0.1831208	total: 720ms	remaining: 337ms
    681:	learn: 0.1830316	total: 721ms	remaining: 336ms
    682:	learn: 0.1830016	total: 722ms	remaining: 335ms
    683:	learn: 0.1829126	total: 723ms	remaining: 334ms
    684:	learn: 0.1827650	total: 724ms	remaining: 333ms
    685:	learn: 0.1827391	total: 725ms	remaining: 332ms
    686:	learn: 0.1824810	total: 726ms	remaining: 331ms
    687:	learn: 0.1824069	total: 727ms	remaining: 330ms
    688:	learn: 0.1823021	total: 728ms	remaining: 329ms
    689:	learn: 0.1822134	total: 729ms	remaining: 327ms
    690:	learn: 0.1821271	total: 729ms	remaining: 326ms
    691:	learn: 0.1820807	total: 730ms	remaining: 325ms
    692:	learn: 0.1819852	total: 731ms	remaining: 324ms
    693:	learn: 0.1819106	total: 732ms	remaining: 323ms
    694:	learn: 0.1817705	total: 733ms	remaining: 322ms
    695:	learn: 0.1816910	total: 734ms	remaining: 321ms
    696:	learn: 0.1815385	total: 735ms	remaining: 320ms
    697:	learn: 0.1812815	total: 736ms	remaining: 318ms
    698:	learn: 0.1812011	total: 737ms	remaining: 317ms
    699:	learn: 0.1811591	total: 738ms	remaining: 316ms
    700:	learn: 0.1810801	total: 739ms	remaining: 315ms
    701:	learn: 0.1810168	total: 740ms	remaining: 314ms
    702:	learn: 0.1809437	total: 741ms	remaining: 313ms
    703:	learn: 0.1808797	total: 742ms	remaining: 312ms
    704:	learn: 0.1808704	total: 743ms	remaining: 311ms
    705:	learn: 0.1808179	total: 744ms	remaining: 310ms
    706:	learn: 0.1808098	total: 745ms	remaining: 309ms
    707:	learn: 0.1807334	total: 746ms	remaining: 308ms
    708:	learn: 0.1807334	total: 747ms	remaining: 306ms
    709:	learn: 0.1806795	total: 748ms	remaining: 305ms
    710:	learn: 0.1805778	total: 749ms	remaining: 304ms
    711:	learn: 0.1805117	total: 750ms	remaining: 303ms
    712:	learn: 0.1804353	total: 751ms	remaining: 302ms
    713:	learn: 0.1803839	total: 753ms	remaining: 301ms
    714:	learn: 0.1803119	total: 754ms	remaining: 301ms
    715:	learn: 0.1802461	total: 755ms	remaining: 299ms
    716:	learn: 0.1801991	total: 756ms	remaining: 298ms
    717:	learn: 0.1800616	total: 757ms	remaining: 297ms
    718:	learn: 0.1795886	total: 758ms	remaining: 296ms
    719:	learn: 0.1794235	total: 759ms	remaining: 295ms
    720:	learn: 0.1793656	total: 760ms	remaining: 294ms
    721:	learn: 0.1792733	total: 761ms	remaining: 293ms
    722:	learn: 0.1791941	total: 763ms	remaining: 292ms
    723:	learn: 0.1790518	total: 764ms	remaining: 291ms
    724:	learn: 0.1790401	total: 765ms	remaining: 290ms
    725:	learn: 0.1789727	total: 766ms	remaining: 289ms
    726:	learn: 0.1788909	total: 767ms	remaining: 288ms
    727:	learn: 0.1788197	total: 768ms	remaining: 287ms
    728:	learn: 0.1787834	total: 769ms	remaining: 286ms
    729:	learn: 0.1786969	total: 770ms	remaining: 285ms
    730:	learn: 0.1785981	total: 771ms	remaining: 284ms
    731:	learn: 0.1785579	total: 772ms	remaining: 283ms
    732:	learn: 0.1784756	total: 773ms	remaining: 282ms
    733:	learn: 0.1784034	total: 774ms	remaining: 280ms
    734:	learn: 0.1783708	total: 775ms	remaining: 279ms
    735:	learn: 0.1782754	total: 776ms	remaining: 278ms
    736:	learn: 0.1782275	total: 777ms	remaining: 277ms
    737:	learn: 0.1781547	total: 778ms	remaining: 276ms
    738:	learn: 0.1781472	total: 779ms	remaining: 275ms
    739:	learn: 0.1780455	total: 780ms	remaining: 274ms
    740:	learn: 0.1780005	total: 781ms	remaining: 273ms
    741:	learn: 0.1779344	total: 782ms	remaining: 272ms
    742:	learn: 0.1778884	total: 783ms	remaining: 271ms
    743:	learn: 0.1778041	total: 784ms	remaining: 270ms
    744:	learn: 0.1776987	total: 785ms	remaining: 269ms
    745:	learn: 0.1776222	total: 786ms	remaining: 268ms
    746:	learn: 0.1775427	total: 787ms	remaining: 267ms
    747:	learn: 0.1775033	total: 788ms	remaining: 266ms
    748:	learn: 0.1774717	total: 790ms	remaining: 265ms
    749:	learn: 0.1774057	total: 791ms	remaining: 264ms
    750:	learn: 0.1773439	total: 792ms	remaining: 263ms
    751:	learn: 0.1772643	total: 793ms	remaining: 261ms
    752:	learn: 0.1772642	total: 794ms	remaining: 260ms
    753:	learn: 0.1771957	total: 795ms	remaining: 259ms
    754:	learn: 0.1771453	total: 796ms	remaining: 258ms
    755:	learn: 0.1770873	total: 797ms	remaining: 257ms
    756:	learn: 0.1770538	total: 798ms	remaining: 256ms
    757:	learn: 0.1769235	total: 799ms	remaining: 255ms
    758:	learn: 0.1768493	total: 800ms	remaining: 254ms
    759:	learn: 0.1767636	total: 801ms	remaining: 253ms
    760:	learn: 0.1766691	total: 802ms	remaining: 252ms
    761:	learn: 0.1765384	total: 803ms	remaining: 251ms
    762:	learn: 0.1763641	total: 804ms	remaining: 250ms
    763:	learn: 0.1762470	total: 805ms	remaining: 249ms
    764:	learn: 0.1761692	total: 806ms	remaining: 248ms
    765:	learn: 0.1761106	total: 807ms	remaining: 246ms
    766:	learn: 0.1760715	total: 808ms	remaining: 245ms
    767:	learn: 0.1759981	total: 809ms	remaining: 244ms
    768:	learn: 0.1758960	total: 810ms	remaining: 243ms
    769:	learn: 0.1758342	total: 811ms	remaining: 242ms
    770:	learn: 0.1757399	total: 812ms	remaining: 241ms
    771:	learn: 0.1754269	total: 813ms	remaining: 240ms
    772:	learn: 0.1753650	total: 814ms	remaining: 239ms
    773:	learn: 0.1752817	total: 815ms	remaining: 238ms
    774:	learn: 0.1752344	total: 817ms	remaining: 237ms
    775:	learn: 0.1751576	total: 817ms	remaining: 236ms
    776:	learn: 0.1751447	total: 818ms	remaining: 235ms
    777:	learn: 0.1750931	total: 820ms	remaining: 234ms
    778:	learn: 0.1749993	total: 823ms	remaining: 234ms
    779:	learn: 0.1749990	total: 825ms	remaining: 233ms
    780:	learn: 0.1749311	total: 827ms	remaining: 232ms
    781:	learn: 0.1748322	total: 829ms	remaining: 231ms
    782:	learn: 0.1747710	total: 830ms	remaining: 230ms
    783:	learn: 0.1747031	total: 832ms	remaining: 229ms
    784:	learn: 0.1746326	total: 834ms	remaining: 228ms
    785:	learn: 0.1745904	total: 836ms	remaining: 228ms
    786:	learn: 0.1745436	total: 837ms	remaining: 227ms
    787:	learn: 0.1745076	total: 839ms	remaining: 226ms
    788:	learn: 0.1744508	total: 841ms	remaining: 225ms
    789:	learn: 0.1743919	total: 843ms	remaining: 224ms
    790:	learn: 0.1743350	total: 845ms	remaining: 223ms
    791:	learn: 0.1742797	total: 849ms	remaining: 223ms
    792:	learn: 0.1742351	total: 850ms	remaining: 222ms
    793:	learn: 0.1742340	total: 851ms	remaining: 221ms
    794:	learn: 0.1742279	total: 852ms	remaining: 220ms
    795:	learn: 0.1741579	total: 853ms	remaining: 219ms
    796:	learn: 0.1740991	total: 854ms	remaining: 218ms
    797:	learn: 0.1739902	total: 855ms	remaining: 216ms
    798:	learn: 0.1739801	total: 856ms	remaining: 215ms
    799:	learn: 0.1738711	total: 857ms	remaining: 214ms
    800:	learn: 0.1738710	total: 858ms	remaining: 213ms
    801:	learn: 0.1738078	total: 859ms	remaining: 212ms
    802:	learn: 0.1736377	total: 860ms	remaining: 211ms
    803:	learn: 0.1735841	total: 861ms	remaining: 210ms
    804:	learn: 0.1735430	total: 862ms	remaining: 209ms
    805:	learn: 0.1734839	total: 863ms	remaining: 208ms
    806:	learn: 0.1733575	total: 864ms	remaining: 207ms
    807:	learn: 0.1731812	total: 865ms	remaining: 206ms
    808:	learn: 0.1731196	total: 866ms	remaining: 204ms
    809:	learn: 0.1730081	total: 867ms	remaining: 203ms
    810:	learn: 0.1729982	total: 868ms	remaining: 202ms
    811:	learn: 0.1729123	total: 869ms	remaining: 201ms
    812:	learn: 0.1728146	total: 870ms	remaining: 200ms
    813:	learn: 0.1727689	total: 871ms	remaining: 199ms
    814:	learn: 0.1726844	total: 872ms	remaining: 198ms
    815:	learn: 0.1726424	total: 873ms	remaining: 197ms
    816:	learn: 0.1725715	total: 874ms	remaining: 196ms
    817:	learn: 0.1725307	total: 875ms	remaining: 195ms
    818:	learn: 0.1724603	total: 876ms	remaining: 194ms
    819:	learn: 0.1723887	total: 877ms	remaining: 193ms
    820:	learn: 0.1723165	total: 878ms	remaining: 191ms
    821:	learn: 0.1722409	total: 879ms	remaining: 190ms
    822:	learn: 0.1722009	total: 880ms	remaining: 189ms
    823:	learn: 0.1721455	total: 881ms	remaining: 188ms
    824:	learn: 0.1720212	total: 882ms	remaining: 187ms
    825:	learn: 0.1717859	total: 884ms	remaining: 186ms
    826:	learn: 0.1717146	total: 887ms	remaining: 186ms
    827:	learn: 0.1716572	total: 892ms	remaining: 185ms
    828:	learn: 0.1715796	total: 893ms	remaining: 184ms
    829:	learn: 0.1715100	total: 894ms	remaining: 183ms
    830:	learn: 0.1713777	total: 895ms	remaining: 182ms
    831:	learn: 0.1713773	total: 896ms	remaining: 181ms
    832:	learn: 0.1711440	total: 897ms	remaining: 180ms
    833:	learn: 0.1710860	total: 899ms	remaining: 179ms
    834:	learn: 0.1710234	total: 900ms	remaining: 178ms
    835:	learn: 0.1709918	total: 901ms	remaining: 177ms
    836:	learn: 0.1709272	total: 902ms	remaining: 176ms
    837:	learn: 0.1708366	total: 903ms	remaining: 174ms
    838:	learn: 0.1707958	total: 904ms	remaining: 173ms
    839:	learn: 0.1706822	total: 905ms	remaining: 172ms
    840:	learn: 0.1706082	total: 906ms	remaining: 171ms
    841:	learn: 0.1705466	total: 907ms	remaining: 170ms
    842:	learn: 0.1704913	total: 908ms	remaining: 169ms
    843:	learn: 0.1704308	total: 909ms	remaining: 168ms
    844:	learn: 0.1703762	total: 910ms	remaining: 167ms
    845:	learn: 0.1703063	total: 911ms	remaining: 166ms
    846:	learn: 0.1702291	total: 912ms	remaining: 165ms
    847:	learn: 0.1701524	total: 913ms	remaining: 164ms
    848:	learn: 0.1701125	total: 914ms	remaining: 163ms
    849:	learn: 0.1700525	total: 915ms	remaining: 162ms
    850:	learn: 0.1700035	total: 916ms	remaining: 160ms
    851:	learn: 0.1699174	total: 917ms	remaining: 159ms
    852:	learn: 0.1698452	total: 918ms	remaining: 158ms
    853:	learn: 0.1697927	total: 919ms	remaining: 157ms
    854:	learn: 0.1696049	total: 920ms	remaining: 156ms
    855:	learn: 0.1694637	total: 921ms	remaining: 155ms
    856:	learn: 0.1693947	total: 922ms	remaining: 154ms
    857:	learn: 0.1693596	total: 923ms	remaining: 153ms
    858:	learn: 0.1693186	total: 924ms	remaining: 152ms
    859:	learn: 0.1691687	total: 925ms	remaining: 151ms
    860:	learn: 0.1690904	total: 927ms	remaining: 150ms
    861:	learn: 0.1690300	total: 928ms	remaining: 149ms
    862:	learn: 0.1689815	total: 929ms	remaining: 147ms
    863:	learn: 0.1689190	total: 930ms	remaining: 146ms
    864:	learn: 0.1688597	total: 931ms	remaining: 145ms
    865:	learn: 0.1686994	total: 932ms	remaining: 144ms
    866:	learn: 0.1685257	total: 933ms	remaining: 143ms
    867:	learn: 0.1684912	total: 934ms	remaining: 142ms
    868:	learn: 0.1684365	total: 935ms	remaining: 141ms
    869:	learn: 0.1683728	total: 936ms	remaining: 140ms
    870:	learn: 0.1683035	total: 937ms	remaining: 139ms
    871:	learn: 0.1682767	total: 938ms	remaining: 138ms
    872:	learn: 0.1682196	total: 939ms	remaining: 137ms
    873:	learn: 0.1680466	total: 940ms	remaining: 135ms
    874:	learn: 0.1679648	total: 941ms	remaining: 134ms
    875:	learn: 0.1679006	total: 942ms	remaining: 133ms
    876:	learn: 0.1678237	total: 943ms	remaining: 132ms
    877:	learn: 0.1677738	total: 944ms	remaining: 131ms
    878:	learn: 0.1676877	total: 945ms	remaining: 130ms
    879:	learn: 0.1676505	total: 946ms	remaining: 129ms
    880:	learn: 0.1675773	total: 947ms	remaining: 128ms
    881:	learn: 0.1675294	total: 948ms	remaining: 127ms
    882:	learn: 0.1674631	total: 949ms	remaining: 126ms
    883:	learn: 0.1673698	total: 950ms	remaining: 125ms
    884:	learn: 0.1673120	total: 951ms	remaining: 124ms
    885:	learn: 0.1672521	total: 952ms	remaining: 123ms
    886:	learn: 0.1671677	total: 953ms	remaining: 121ms
    887:	learn: 0.1670889	total: 954ms	remaining: 120ms
    888:	learn: 0.1670429	total: 955ms	remaining: 119ms
    889:	learn: 0.1669753	total: 956ms	remaining: 118ms
    890:	learn: 0.1669215	total: 957ms	remaining: 117ms
    891:	learn: 0.1668764	total: 958ms	remaining: 116ms
    892:	learn: 0.1667446	total: 959ms	remaining: 115ms
    893:	learn: 0.1666463	total: 960ms	remaining: 114ms
    894:	learn: 0.1666396	total: 961ms	remaining: 113ms
    895:	learn: 0.1665671	total: 962ms	remaining: 112ms
    896:	learn: 0.1665393	total: 963ms	remaining: 111ms
    897:	learn: 0.1664923	total: 964ms	remaining: 110ms
    898:	learn: 0.1664377	total: 965ms	remaining: 108ms
    899:	learn: 0.1663999	total: 966ms	remaining: 107ms
    900:	learn: 0.1663520	total: 967ms	remaining: 106ms
    901:	learn: 0.1662893	total: 968ms	remaining: 105ms
    902:	learn: 0.1662282	total: 969ms	remaining: 104ms
    903:	learn: 0.1661269	total: 970ms	remaining: 103ms
    904:	learn: 0.1661267	total: 971ms	remaining: 102ms
    905:	learn: 0.1660521	total: 972ms	remaining: 101ms
    906:	learn: 0.1660082	total: 973ms	remaining: 99.8ms
    907:	learn: 0.1659372	total: 974ms	remaining: 98.7ms
    908:	learn: 0.1658954	total: 975ms	remaining: 97.6ms
    909:	learn: 0.1657840	total: 976ms	remaining: 96.5ms
    910:	learn: 0.1655490	total: 977ms	remaining: 95.4ms
    911:	learn: 0.1654443	total: 978ms	remaining: 94.3ms
    912:	learn: 0.1653816	total: 979ms	remaining: 93.3ms
    913:	learn: 0.1653092	total: 980ms	remaining: 92.2ms
    914:	learn: 0.1652808	total: 981ms	remaining: 91.1ms
    915:	learn: 0.1652237	total: 982ms	remaining: 90ms
    916:	learn: 0.1651655	total: 983ms	remaining: 88.9ms
    917:	learn: 0.1651065	total: 984ms	remaining: 87.9ms
    918:	learn: 0.1650739	total: 985ms	remaining: 86.8ms
    919:	learn: 0.1649992	total: 985ms	remaining: 85.7ms
    920:	learn: 0.1649334	total: 986ms	remaining: 84.6ms
    921:	learn: 0.1647939	total: 989ms	remaining: 83.6ms
    922:	learn: 0.1647290	total: 990ms	remaining: 82.5ms
    923:	learn: 0.1647149	total: 992ms	remaining: 81.6ms
    924:	learn: 0.1646565	total: 995ms	remaining: 80.7ms
    925:	learn: 0.1645751	total: 997ms	remaining: 79.6ms
    926:	learn: 0.1644853	total: 997ms	remaining: 78.5ms
    927:	learn: 0.1644488	total: 998ms	remaining: 77.5ms
    928:	learn: 0.1644036	total: 999ms	remaining: 76.4ms
    929:	learn: 0.1643675	total: 1s	remaining: 75.6ms
    930:	learn: 0.1643032	total: 1s	remaining: 74.5ms
    931:	learn: 0.1642392	total: 1.01s	remaining: 73.4ms
    932:	learn: 0.1641840	total: 1.01s	remaining: 72.4ms
    933:	learn: 0.1641353	total: 1.01s	remaining: 71.3ms
    934:	learn: 0.1640733	total: 1.01s	remaining: 70.2ms
    935:	learn: 0.1640150	total: 1.01s	remaining: 69.1ms
    936:	learn: 0.1639371	total: 1.01s	remaining: 68ms
    937:	learn: 0.1637889	total: 1.01s	remaining: 66.9ms
    938:	learn: 0.1637196	total: 1.01s	remaining: 65.8ms
    939:	learn: 0.1635805	total: 1.01s	remaining: 64.8ms
    940:	learn: 0.1635216	total: 1.01s	remaining: 63.7ms
    941:	learn: 0.1634507	total: 1.02s	remaining: 62.6ms
    942:	learn: 0.1632295	total: 1.02s	remaining: 61.5ms
    943:	learn: 0.1631579	total: 1.02s	remaining: 60.4ms
    944:	learn: 0.1630617	total: 1.02s	remaining: 59.3ms
    945:	learn: 0.1630048	total: 1.02s	remaining: 58.2ms
    946:	learn: 0.1629472	total: 1.02s	remaining: 57.1ms
    947:	learn: 0.1628978	total: 1.02s	remaining: 56.1ms
    948:	learn: 0.1628866	total: 1.02s	remaining: 55ms
    949:	learn: 0.1627869	total: 1.02s	remaining: 53.9ms
    950:	learn: 0.1627671	total: 1.02s	remaining: 52.8ms
    951:	learn: 0.1626231	total: 1.02s	remaining: 51.7ms
    952:	learn: 0.1625761	total: 1.03s	remaining: 50.6ms
    953:	learn: 0.1624686	total: 1.03s	remaining: 49.6ms
    954:	learn: 0.1623821	total: 1.03s	remaining: 48.5ms
    955:	learn: 0.1623402	total: 1.03s	remaining: 47.4ms
    956:	learn: 0.1623344	total: 1.03s	remaining: 46.3ms
    957:	learn: 0.1622793	total: 1.03s	remaining: 45.2ms
    958:	learn: 0.1622164	total: 1.03s	remaining: 44.2ms
    959:	learn: 0.1622163	total: 1.03s	remaining: 43.1ms
    960:	learn: 0.1621664	total: 1.03s	remaining: 42ms
    961:	learn: 0.1620352	total: 1.03s	remaining: 40.9ms
    962:	learn: 0.1619787	total: 1.04s	remaining: 39.8ms
    963:	learn: 0.1619135	total: 1.04s	remaining: 38.7ms
    964:	learn: 0.1618548	total: 1.04s	remaining: 37.7ms
    965:	learn: 0.1618207	total: 1.04s	remaining: 36.6ms
    966:	learn: 0.1617725	total: 1.04s	remaining: 35.5ms
    967:	learn: 0.1617605	total: 1.04s	remaining: 34.4ms
    968:	learn: 0.1615644	total: 1.04s	remaining: 33.3ms
    969:	learn: 0.1614913	total: 1.04s	remaining: 32.3ms
    970:	learn: 0.1613887	total: 1.04s	remaining: 31.2ms
    971:	learn: 0.1613578	total: 1.04s	remaining: 30.1ms
    972:	learn: 0.1612958	total: 1.05s	remaining: 29ms
    973:	learn: 0.1612345	total: 1.05s	remaining: 28ms
    974:	learn: 0.1611721	total: 1.05s	remaining: 26.9ms
    975:	learn: 0.1611613	total: 1.05s	remaining: 25.8ms
    976:	learn: 0.1610585	total: 1.05s	remaining: 24.7ms
    977:	learn: 0.1610045	total: 1.05s	remaining: 23.6ms
    978:	learn: 0.1608934	total: 1.05s	remaining: 22.6ms
    979:	learn: 0.1608528	total: 1.05s	remaining: 21.5ms
    980:	learn: 0.1608338	total: 1.05s	remaining: 20.4ms
    981:	learn: 0.1607894	total: 1.05s	remaining: 19.3ms
    982:	learn: 0.1606880	total: 1.06s	remaining: 18.3ms
    983:	learn: 0.1606268	total: 1.06s	remaining: 17.2ms
    984:	learn: 0.1605707	total: 1.06s	remaining: 16.1ms
    985:	learn: 0.1604322	total: 1.06s	remaining: 15ms
    986:	learn: 0.1603796	total: 1.06s	remaining: 14ms
    987:	learn: 0.1603454	total: 1.06s	remaining: 12.9ms
    988:	learn: 0.1602666	total: 1.06s	remaining: 11.8ms
    989:	learn: 0.1601603	total: 1.06s	remaining: 10.7ms
    990:	learn: 0.1600971	total: 1.06s	remaining: 9.66ms
    991:	learn: 0.1599760	total: 1.06s	remaining: 8.59ms
    992:	learn: 0.1598838	total: 1.07s	remaining: 7.51ms
    993:	learn: 0.1598492	total: 1.07s	remaining: 6.44ms
    994:	learn: 0.1598115	total: 1.07s	remaining: 5.37ms
    995:	learn: 0.1598113	total: 1.07s	remaining: 4.29ms
    996:	learn: 0.1597605	total: 1.07s	remaining: 3.22ms
    997:	learn: 0.1596599	total: 1.07s	remaining: 2.15ms
    998:	learn: 0.1596047	total: 1.07s	remaining: 1.07ms
    999:	learn: 0.1595435	total: 1.07s	remaining: 0us
    0:	learn: 0.5800704	total: 817us	remaining: 817ms
    1:	learn: 0.5052183	total: 1.52ms	remaining: 757ms
    2:	learn: 0.4515178	total: 2.61ms	remaining: 868ms
    3:	learn: 0.4213866	total: 3.72ms	remaining: 926ms
    4:	learn: 0.3997630	total: 4.76ms	remaining: 948ms
    5:	learn: 0.3872194	total: 5.8ms	remaining: 960ms
    6:	learn: 0.3754497	total: 6.74ms	remaining: 956ms
    7:	learn: 0.3631909	total: 7.68ms	remaining: 953ms
    8:	learn: 0.3583815	total: 8.53ms	remaining: 939ms
    9:	learn: 0.3530560	total: 9.46ms	remaining: 936ms
    10:	learn: 0.3482312	total: 11ms	remaining: 986ms
    11:	learn: 0.3466111	total: 11.9ms	remaining: 981ms
    12:	learn: 0.3443715	total: 12.9ms	remaining: 980ms
    13:	learn: 0.3429841	total: 14.2ms	remaining: 999ms
    14:	learn: 0.3410287	total: 15.2ms	remaining: 1000ms
    15:	learn: 0.3384510	total: 16.3ms	remaining: 1s
    16:	learn: 0.3380134	total: 17.3ms	remaining: 1s
    17:	learn: 0.3373043	total: 18.4ms	remaining: 1s
    18:	learn: 0.3360914	total: 19.5ms	remaining: 1.01s
    19:	learn: 0.3355725	total: 20.5ms	remaining: 1.01s
    20:	learn: 0.3347449	total: 21.6ms	remaining: 1s
    21:	learn: 0.3332413	total: 22.6ms	remaining: 1s
    22:	learn: 0.3325097	total: 23.7ms	remaining: 1.01s
    23:	learn: 0.3323640	total: 24.8ms	remaining: 1.01s
    24:	learn: 0.3310687	total: 25.9ms	remaining: 1.01s
    25:	learn: 0.3291119	total: 27ms	remaining: 1.01s
    26:	learn: 0.3286319	total: 28ms	remaining: 1.01s
    27:	learn: 0.3284337	total: 28.7ms	remaining: 995ms
    28:	learn: 0.3277603	total: 29.7ms	remaining: 995ms
    29:	learn: 0.3263002	total: 30.7ms	remaining: 992ms
    30:	learn: 0.3257369	total: 31.7ms	remaining: 991ms
    31:	learn: 0.3252872	total: 32.7ms	remaining: 990ms
    32:	learn: 0.3244329	total: 34ms	remaining: 995ms
    33:	learn: 0.3238628	total: 35.1ms	remaining: 996ms
    34:	learn: 0.3234408	total: 36.1ms	remaining: 996ms
    35:	learn: 0.3221518	total: 37.1ms	remaining: 994ms
    36:	learn: 0.3216229	total: 38.2ms	remaining: 994ms
    37:	learn: 0.3204724	total: 39.2ms	remaining: 992ms
    38:	learn: 0.3202225	total: 40.2ms	remaining: 991ms
    39:	learn: 0.3192186	total: 41.3ms	remaining: 991ms
    40:	learn: 0.3184807	total: 42.4ms	remaining: 991ms
    41:	learn: 0.3178094	total: 43.4ms	remaining: 991ms
    42:	learn: 0.3173971	total: 44.5ms	remaining: 990ms
    43:	learn: 0.3171160	total: 45.5ms	remaining: 988ms
    44:	learn: 0.3163064	total: 46.7ms	remaining: 990ms
    45:	learn: 0.3160252	total: 47.8ms	remaining: 991ms
    46:	learn: 0.3144429	total: 48.8ms	remaining: 989ms
    47:	learn: 0.3142506	total: 49.8ms	remaining: 987ms
    48:	learn: 0.3133807	total: 50.8ms	remaining: 985ms
    49:	learn: 0.3123928	total: 51.7ms	remaining: 983ms
    50:	learn: 0.3116586	total: 52.9ms	remaining: 984ms
    51:	learn: 0.3108302	total: 53.9ms	remaining: 982ms
    52:	learn: 0.3099331	total: 54.9ms	remaining: 980ms
    53:	learn: 0.3092234	total: 55.8ms	remaining: 978ms
    54:	learn: 0.3083542	total: 56.8ms	remaining: 976ms
    55:	learn: 0.3078555	total: 57.7ms	remaining: 973ms
    56:	learn: 0.3072272	total: 58.6ms	remaining: 970ms
    57:	learn: 0.3068450	total: 60.6ms	remaining: 984ms
    58:	learn: 0.3062964	total: 61.6ms	remaining: 982ms
    59:	learn: 0.3056636	total: 62.4ms	remaining: 978ms
    60:	learn: 0.3051323	total: 63.3ms	remaining: 974ms
    61:	learn: 0.3045331	total: 64.1ms	remaining: 970ms
    62:	learn: 0.3024986	total: 73.5ms	remaining: 1.09s
    63:	learn: 0.3019093	total: 75.3ms	remaining: 1.1s
    64:	learn: 0.3013471	total: 76.2ms	remaining: 1.09s
    65:	learn: 0.3008789	total: 77ms	remaining: 1.09s
    66:	learn: 0.3004061	total: 80.5ms	remaining: 1.12s
    67:	learn: 0.2999790	total: 81.2ms	remaining: 1.11s
    68:	learn: 0.2992956	total: 84.2ms	remaining: 1.14s
    69:	learn: 0.2989653	total: 84.8ms	remaining: 1.13s
    70:	learn: 0.2983558	total: 85.4ms	remaining: 1.12s
    71:	learn: 0.2979214	total: 86.1ms	remaining: 1.11s
    72:	learn: 0.2974696	total: 86.6ms	remaining: 1.1s
    73:	learn: 0.2971021	total: 87.3ms	remaining: 1.09s
    74:	learn: 0.2967809	total: 87.9ms	remaining: 1.08s
    75:	learn: 0.2963702	total: 88.5ms	remaining: 1.08s
    76:	learn: 0.2959737	total: 91.2ms	remaining: 1.09s
    77:	learn: 0.2956173	total: 92.2ms	remaining: 1.09s
    78:	learn: 0.2950815	total: 93.4ms	remaining: 1.09s
    79:	learn: 0.2947093	total: 94ms	remaining: 1.08s
    80:	learn: 0.2942807	total: 94.6ms	remaining: 1.07s
    81:	learn: 0.2938752	total: 95.9ms	remaining: 1.07s
    82:	learn: 0.2933174	total: 97ms	remaining: 1.07s
    83:	learn: 0.2931349	total: 97.6ms	remaining: 1.06s
    84:	learn: 0.2928577	total: 98.6ms	remaining: 1.06s
    85:	learn: 0.2926586	total: 99.6ms	remaining: 1.06s
    86:	learn: 0.2920827	total: 101ms	remaining: 1.05s
    87:	learn: 0.2918193	total: 101ms	remaining: 1.05s
    88:	learn: 0.2911530	total: 102ms	remaining: 1.05s
    89:	learn: 0.2909441	total: 103ms	remaining: 1.04s
    90:	learn: 0.2906448	total: 104ms	remaining: 1.04s
    91:	learn: 0.2903269	total: 105ms	remaining: 1.04s
    92:	learn: 0.2898714	total: 106ms	remaining: 1.03s
    93:	learn: 0.2894011	total: 107ms	remaining: 1.03s
    94:	learn: 0.2888780	total: 108ms	remaining: 1.03s
    95:	learn: 0.2884891	total: 109ms	remaining: 1.03s
    96:	learn: 0.2880902	total: 110ms	remaining: 1.02s
    97:	learn: 0.2877611	total: 111ms	remaining: 1.02s
    98:	learn: 0.2871830	total: 112ms	remaining: 1.02s
    99:	learn: 0.2868984	total: 113ms	remaining: 1.02s
    100:	learn: 0.2863430	total: 114ms	remaining: 1.01s
    101:	learn: 0.2859670	total: 115ms	remaining: 1.01s
    102:	learn: 0.2856482	total: 116ms	remaining: 1.01s
    103:	learn: 0.2852566	total: 117ms	remaining: 1s
    104:	learn: 0.2845543	total: 118ms	remaining: 1s
    105:	learn: 0.2843051	total: 119ms	remaining: 1s
    106:	learn: 0.2839660	total: 120ms	remaining: 998ms
    107:	learn: 0.2836566	total: 121ms	remaining: 996ms
    108:	learn: 0.2832458	total: 122ms	remaining: 994ms
    109:	learn: 0.2831568	total: 123ms	remaining: 991ms
    110:	learn: 0.2827501	total: 123ms	remaining: 989ms
    111:	learn: 0.2820599	total: 124ms	remaining: 987ms
    112:	learn: 0.2816441	total: 125ms	remaining: 984ms
    113:	learn: 0.2814081	total: 126ms	remaining: 982ms
    114:	learn: 0.2807955	total: 127ms	remaining: 979ms
    115:	learn: 0.2804903	total: 128ms	remaining: 977ms
    116:	learn: 0.2802199	total: 129ms	remaining: 975ms
    117:	learn: 0.2799264	total: 130ms	remaining: 973ms
    118:	learn: 0.2796458	total: 131ms	remaining: 971ms
    119:	learn: 0.2792496	total: 132ms	remaining: 968ms
    120:	learn: 0.2789399	total: 133ms	remaining: 965ms
    121:	learn: 0.2784223	total: 134ms	remaining: 962ms
    122:	learn: 0.2781337	total: 135ms	remaining: 959ms
    123:	learn: 0.2774355	total: 135ms	remaining: 957ms
    124:	learn: 0.2771497	total: 136ms	remaining: 955ms
    125:	learn: 0.2769462	total: 137ms	remaining: 952ms
    126:	learn: 0.2766480	total: 138ms	remaining: 950ms
    127:	learn: 0.2763325	total: 139ms	remaining: 948ms
    128:	learn: 0.2758102	total: 141ms	remaining: 951ms
    129:	learn: 0.2757444	total: 142ms	remaining: 947ms
    130:	learn: 0.2755179	total: 142ms	remaining: 943ms
    131:	learn: 0.2749824	total: 143ms	remaining: 941ms
    132:	learn: 0.2745668	total: 144ms	remaining: 939ms
    133:	learn: 0.2742261	total: 146ms	remaining: 940ms
    134:	learn: 0.2738417	total: 146ms	remaining: 938ms
    135:	learn: 0.2737247	total: 147ms	remaining: 937ms
    136:	learn: 0.2733926	total: 148ms	remaining: 935ms
    137:	learn: 0.2730498	total: 149ms	remaining: 933ms
    138:	learn: 0.2728477	total: 150ms	remaining: 932ms
    139:	learn: 0.2725340	total: 151ms	remaining: 930ms
    140:	learn: 0.2722446	total: 152ms	remaining: 928ms
    141:	learn: 0.2719866	total: 153ms	remaining: 927ms
    142:	learn: 0.2718610	total: 154ms	remaining: 925ms
    143:	learn: 0.2713696	total: 155ms	remaining: 923ms
    144:	learn: 0.2710631	total: 156ms	remaining: 921ms
    145:	learn: 0.2708951	total: 157ms	remaining: 919ms
    146:	learn: 0.2705064	total: 158ms	remaining: 917ms
    147:	learn: 0.2703080	total: 159ms	remaining: 915ms
    148:	learn: 0.2698515	total: 160ms	remaining: 912ms
    149:	learn: 0.2691599	total: 161ms	remaining: 911ms
    150:	learn: 0.2680554	total: 162ms	remaining: 911ms
    151:	learn: 0.2677166	total: 163ms	remaining: 909ms
    152:	learn: 0.2672107	total: 164ms	remaining: 907ms
    153:	learn: 0.2669196	total: 165ms	remaining: 905ms
    154:	learn: 0.2666391	total: 166ms	remaining: 904ms
    155:	learn: 0.2663646	total: 167ms	remaining: 902ms
    156:	learn: 0.2660724	total: 168ms	remaining: 901ms
    157:	learn: 0.2659101	total: 169ms	remaining: 899ms
    158:	learn: 0.2656139	total: 170ms	remaining: 898ms
    159:	learn: 0.2651898	total: 171ms	remaining: 896ms
    160:	learn: 0.2648378	total: 172ms	remaining: 894ms
    161:	learn: 0.2645281	total: 173ms	remaining: 892ms
    162:	learn: 0.2642911	total: 177ms	remaining: 909ms
    163:	learn: 0.2640636	total: 178ms	remaining: 906ms
    164:	learn: 0.2638242	total: 179ms	remaining: 906ms
    165:	learn: 0.2637190	total: 180ms	remaining: 905ms
    166:	learn: 0.2633486	total: 181ms	remaining: 904ms
    167:	learn: 0.2631329	total: 182ms	remaining: 904ms
    168:	learn: 0.2629764	total: 184ms	remaining: 903ms
    169:	learn: 0.2628143	total: 185ms	remaining: 901ms
    170:	learn: 0.2626075	total: 186ms	remaining: 900ms
    171:	learn: 0.2624910	total: 187ms	remaining: 898ms
    172:	learn: 0.2622134	total: 188ms	remaining: 897ms
    173:	learn: 0.2619458	total: 189ms	remaining: 895ms
    174:	learn: 0.2615583	total: 189ms	remaining: 893ms
    175:	learn: 0.2613523	total: 190ms	remaining: 891ms
    176:	learn: 0.2611334	total: 191ms	remaining: 889ms
    177:	learn: 0.2609104	total: 192ms	remaining: 887ms
    178:	learn: 0.2606569	total: 194ms	remaining: 888ms
    179:	learn: 0.2603777	total: 195ms	remaining: 886ms
    180:	learn: 0.2599936	total: 195ms	remaining: 884ms
    181:	learn: 0.2597671	total: 196ms	remaining: 883ms
    182:	learn: 0.2595738	total: 197ms	remaining: 881ms
    183:	learn: 0.2593030	total: 198ms	remaining: 879ms
    184:	learn: 0.2589107	total: 199ms	remaining: 878ms
    185:	learn: 0.2585514	total: 201ms	remaining: 881ms
    186:	learn: 0.2583762	total: 202ms	remaining: 878ms
    187:	learn: 0.2579012	total: 202ms	remaining: 875ms
    188:	learn: 0.2576339	total: 203ms	remaining: 872ms
    189:	learn: 0.2573955	total: 204ms	remaining: 870ms
    190:	learn: 0.2571931	total: 205ms	remaining: 867ms
    191:	learn: 0.2569250	total: 205ms	remaining: 864ms
    192:	learn: 0.2565973	total: 206ms	remaining: 862ms
    193:	learn: 0.2563299	total: 207ms	remaining: 859ms
    194:	learn: 0.2561671	total: 207ms	remaining: 857ms
    195:	learn: 0.2556185	total: 208ms	remaining: 854ms
    196:	learn: 0.2554350	total: 209ms	remaining: 852ms
    197:	learn: 0.2551141	total: 210ms	remaining: 850ms
    198:	learn: 0.2549473	total: 211ms	remaining: 848ms
    199:	learn: 0.2548561	total: 212ms	remaining: 848ms
    200:	learn: 0.2543312	total: 213ms	remaining: 847ms
    201:	learn: 0.2541765	total: 214ms	remaining: 845ms
    202:	learn: 0.2540456	total: 215ms	remaining: 844ms
    203:	learn: 0.2539760	total: 216ms	remaining: 843ms
    204:	learn: 0.2537449	total: 217ms	remaining: 841ms
    205:	learn: 0.2536069	total: 218ms	remaining: 840ms
    206:	learn: 0.2535323	total: 219ms	remaining: 839ms
    207:	learn: 0.2534423	total: 220ms	remaining: 838ms
    208:	learn: 0.2530460	total: 221ms	remaining: 837ms
    209:	learn: 0.2528624	total: 222ms	remaining: 835ms
    210:	learn: 0.2526883	total: 223ms	remaining: 835ms
    211:	learn: 0.2523906	total: 224ms	remaining: 833ms
    212:	learn: 0.2521682	total: 225ms	remaining: 832ms
    213:	learn: 0.2519041	total: 226ms	remaining: 830ms
    214:	learn: 0.2517261	total: 227ms	remaining: 829ms
    215:	learn: 0.2514328	total: 228ms	remaining: 828ms
    216:	learn: 0.2512595	total: 229ms	remaining: 826ms
    217:	learn: 0.2510593	total: 230ms	remaining: 825ms
    218:	learn: 0.2509242	total: 231ms	remaining: 823ms
    219:	learn: 0.2507275	total: 232ms	remaining: 822ms
    220:	learn: 0.2505220	total: 233ms	remaining: 821ms
    221:	learn: 0.2502588	total: 234ms	remaining: 819ms
    222:	learn: 0.2499170	total: 235ms	remaining: 818ms
    223:	learn: 0.2496528	total: 236ms	remaining: 817ms
    224:	learn: 0.2493143	total: 237ms	remaining: 817ms
    225:	learn: 0.2490762	total: 238ms	remaining: 816ms
    226:	learn: 0.2488575	total: 239ms	remaining: 816ms
    227:	learn: 0.2486036	total: 249ms	remaining: 844ms
    228:	learn: 0.2484448	total: 250ms	remaining: 842ms
    229:	learn: 0.2481968	total: 251ms	remaining: 841ms
    230:	learn: 0.2479475	total: 252ms	remaining: 839ms
    231:	learn: 0.2477875	total: 253ms	remaining: 837ms
    232:	learn: 0.2475829	total: 254ms	remaining: 836ms
    233:	learn: 0.2475382	total: 255ms	remaining: 835ms
    234:	learn: 0.2472427	total: 256ms	remaining: 833ms
    235:	learn: 0.2470481	total: 257ms	remaining: 832ms
    236:	learn: 0.2468988	total: 258ms	remaining: 831ms
    237:	learn: 0.2467166	total: 259ms	remaining: 829ms
    238:	learn: 0.2465500	total: 260ms	remaining: 829ms
    239:	learn: 0.2463567	total: 261ms	remaining: 827ms
    240:	learn: 0.2460826	total: 262ms	remaining: 826ms
    241:	learn: 0.2459361	total: 263ms	remaining: 825ms
    242:	learn: 0.2458093	total: 264ms	remaining: 823ms
    243:	learn: 0.2455885	total: 265ms	remaining: 822ms
    244:	learn: 0.2451369	total: 266ms	remaining: 820ms
    245:	learn: 0.2450739	total: 267ms	remaining: 819ms
    246:	learn: 0.2448642	total: 268ms	remaining: 817ms
    247:	learn: 0.2445342	total: 269ms	remaining: 816ms
    248:	learn: 0.2443085	total: 270ms	remaining: 814ms
    249:	learn: 0.2440080	total: 271ms	remaining: 813ms
    250:	learn: 0.2439020	total: 272ms	remaining: 812ms
    251:	learn: 0.2437365	total: 273ms	remaining: 810ms
    252:	learn: 0.2435282	total: 274ms	remaining: 809ms
    253:	learn: 0.2432643	total: 275ms	remaining: 807ms
    254:	learn: 0.2431386	total: 276ms	remaining: 806ms
    255:	learn: 0.2429485	total: 277ms	remaining: 804ms
    256:	learn: 0.2428664	total: 278ms	remaining: 803ms
    257:	learn: 0.2425396	total: 279ms	remaining: 802ms
    258:	learn: 0.2422991	total: 280ms	remaining: 800ms
    259:	learn: 0.2421329	total: 281ms	remaining: 799ms
    260:	learn: 0.2419334	total: 282ms	remaining: 797ms
    261:	learn: 0.2417880	total: 283ms	remaining: 796ms
    262:	learn: 0.2415913	total: 284ms	remaining: 794ms
    263:	learn: 0.2415522	total: 284ms	remaining: 793ms
    264:	learn: 0.2413519	total: 285ms	remaining: 791ms
    265:	learn: 0.2410580	total: 286ms	remaining: 790ms
    266:	learn: 0.2409060	total: 287ms	remaining: 789ms
    267:	learn: 0.2407277	total: 288ms	remaining: 787ms
    268:	learn: 0.2405567	total: 289ms	remaining: 786ms
    269:	learn: 0.2403501	total: 290ms	remaining: 784ms
    270:	learn: 0.2401403	total: 291ms	remaining: 783ms
    271:	learn: 0.2399514	total: 292ms	remaining: 781ms
    272:	learn: 0.2399256	total: 293ms	remaining: 780ms
    273:	learn: 0.2397122	total: 294ms	remaining: 778ms
    274:	learn: 0.2395144	total: 295ms	remaining: 777ms
    275:	learn: 0.2391830	total: 296ms	remaining: 776ms
    276:	learn: 0.2390318	total: 297ms	remaining: 774ms
    277:	learn: 0.2386739	total: 298ms	remaining: 773ms
    278:	learn: 0.2385481	total: 299ms	remaining: 772ms
    279:	learn: 0.2383946	total: 300ms	remaining: 770ms
    280:	learn: 0.2382426	total: 301ms	remaining: 769ms
    281:	learn: 0.2379218	total: 301ms	remaining: 768ms
    282:	learn: 0.2377526	total: 302ms	remaining: 766ms
    283:	learn: 0.2376951	total: 303ms	remaining: 765ms
    284:	learn: 0.2375404	total: 304ms	remaining: 763ms
    285:	learn: 0.2371824	total: 305ms	remaining: 762ms
    286:	learn: 0.2369984	total: 306ms	remaining: 761ms
    287:	learn: 0.2368923	total: 307ms	remaining: 759ms
    288:	learn: 0.2367558	total: 308ms	remaining: 758ms
    289:	learn: 0.2365759	total: 309ms	remaining: 757ms
    290:	learn: 0.2363691	total: 310ms	remaining: 755ms
    291:	learn: 0.2361354	total: 311ms	remaining: 754ms
    292:	learn: 0.2356210	total: 312ms	remaining: 753ms
    293:	learn: 0.2354132	total: 313ms	remaining: 752ms
    294:	learn: 0.2353850	total: 314ms	remaining: 751ms
    295:	learn: 0.2352143	total: 315ms	remaining: 750ms
    296:	learn: 0.2350327	total: 316ms	remaining: 748ms
    297:	learn: 0.2348783	total: 317ms	remaining: 747ms
    298:	learn: 0.2346964	total: 318ms	remaining: 746ms
    299:	learn: 0.2345881	total: 319ms	remaining: 744ms
    300:	learn: 0.2344249	total: 320ms	remaining: 743ms
    301:	learn: 0.2343017	total: 321ms	remaining: 742ms
    302:	learn: 0.2338062	total: 322ms	remaining: 740ms
    303:	learn: 0.2337875	total: 323ms	remaining: 739ms
    304:	learn: 0.2334262	total: 324ms	remaining: 738ms
    305:	learn: 0.2332785	total: 325ms	remaining: 737ms
    306:	learn: 0.2331738	total: 326ms	remaining: 736ms
    307:	learn: 0.2330594	total: 327ms	remaining: 734ms
    308:	learn: 0.2328867	total: 328ms	remaining: 733ms
    309:	learn: 0.2324902	total: 329ms	remaining: 732ms
    310:	learn: 0.2322885	total: 330ms	remaining: 730ms
    311:	learn: 0.2320785	total: 331ms	remaining: 729ms
    312:	learn: 0.2320005	total: 331ms	remaining: 728ms
    313:	learn: 0.2318195	total: 333ms	remaining: 726ms
    314:	learn: 0.2315745	total: 333ms	remaining: 725ms
    315:	learn: 0.2314114	total: 334ms	remaining: 724ms
    316:	learn: 0.2312671	total: 335ms	remaining: 722ms
    317:	learn: 0.2311251	total: 336ms	remaining: 721ms
    318:	learn: 0.2310111	total: 337ms	remaining: 720ms
    319:	learn: 0.2308495	total: 338ms	remaining: 718ms
    320:	learn: 0.2308289	total: 339ms	remaining: 717ms
    321:	learn: 0.2306895	total: 340ms	remaining: 716ms
    322:	learn: 0.2305316	total: 341ms	remaining: 715ms
    323:	learn: 0.2303099	total: 342ms	remaining: 713ms
    324:	learn: 0.2301639	total: 343ms	remaining: 712ms
    325:	learn: 0.2301475	total: 344ms	remaining: 711ms
    326:	learn: 0.2300594	total: 345ms	remaining: 709ms
    327:	learn: 0.2298678	total: 346ms	remaining: 708ms
    328:	learn: 0.2297077	total: 347ms	remaining: 707ms
    329:	learn: 0.2295043	total: 347ms	remaining: 705ms
    330:	learn: 0.2291556	total: 348ms	remaining: 704ms
    331:	learn: 0.2289520	total: 349ms	remaining: 702ms
    332:	learn: 0.2289317	total: 350ms	remaining: 701ms
    333:	learn: 0.2287776	total: 351ms	remaining: 700ms
    334:	learn: 0.2286424	total: 352ms	remaining: 699ms
    335:	learn: 0.2284547	total: 353ms	remaining: 698ms
    336:	learn: 0.2283069	total: 354ms	remaining: 696ms
    337:	learn: 0.2281806	total: 355ms	remaining: 695ms
    338:	learn: 0.2280262	total: 356ms	remaining: 694ms
    339:	learn: 0.2277605	total: 357ms	remaining: 692ms
    340:	learn: 0.2275899	total: 358ms	remaining: 691ms
    341:	learn: 0.2273680	total: 359ms	remaining: 690ms
    342:	learn: 0.2271880	total: 359ms	remaining: 689ms
    343:	learn: 0.2270935	total: 360ms	remaining: 687ms
    344:	learn: 0.2269839	total: 361ms	remaining: 686ms
    345:	learn: 0.2268004	total: 362ms	remaining: 685ms
    346:	learn: 0.2266160	total: 363ms	remaining: 684ms
    347:	learn: 0.2264616	total: 364ms	remaining: 682ms
    348:	learn: 0.2261398	total: 365ms	remaining: 681ms
    349:	learn: 0.2259667	total: 366ms	remaining: 680ms
    350:	learn: 0.2257357	total: 367ms	remaining: 679ms
    351:	learn: 0.2256004	total: 368ms	remaining: 677ms
    352:	learn: 0.2255141	total: 369ms	remaining: 676ms
    353:	learn: 0.2253843	total: 370ms	remaining: 675ms
    354:	learn: 0.2251357	total: 371ms	remaining: 674ms
    355:	learn: 0.2249306	total: 372ms	remaining: 672ms
    356:	learn: 0.2248067	total: 373ms	remaining: 671ms
    357:	learn: 0.2246362	total: 374ms	remaining: 670ms
    358:	learn: 0.2245897	total: 375ms	remaining: 669ms
    359:	learn: 0.2244599	total: 375ms	remaining: 667ms
    360:	learn: 0.2243792	total: 376ms	remaining: 666ms
    361:	learn: 0.2241426	total: 377ms	remaining: 665ms
    362:	learn: 0.2238617	total: 378ms	remaining: 664ms
    363:	learn: 0.2237669	total: 379ms	remaining: 662ms
    364:	learn: 0.2236135	total: 380ms	remaining: 661ms
    365:	learn: 0.2234745	total: 381ms	remaining: 660ms
    366:	learn: 0.2234328	total: 382ms	remaining: 660ms
    367:	learn: 0.2233555	total: 383ms	remaining: 658ms
    368:	learn: 0.2232226	total: 384ms	remaining: 657ms
    369:	learn: 0.2231326	total: 385ms	remaining: 656ms
    370:	learn: 0.2230404	total: 386ms	remaining: 655ms
    371:	learn: 0.2229538	total: 387ms	remaining: 654ms
    372:	learn: 0.2227777	total: 388ms	remaining: 653ms
    373:	learn: 0.2225795	total: 389ms	remaining: 652ms
    374:	learn: 0.2225710	total: 390ms	remaining: 650ms
    375:	learn: 0.2224615	total: 391ms	remaining: 649ms
    376:	learn: 0.2223421	total: 392ms	remaining: 648ms
    377:	learn: 0.2222571	total: 393ms	remaining: 647ms
    378:	learn: 0.2221305	total: 394ms	remaining: 646ms
    379:	learn: 0.2219196	total: 395ms	remaining: 645ms
    380:	learn: 0.2218034	total: 396ms	remaining: 643ms
    381:	learn: 0.2216943	total: 397ms	remaining: 642ms
    382:	learn: 0.2215015	total: 398ms	remaining: 641ms
    383:	learn: 0.2213971	total: 399ms	remaining: 640ms
    384:	learn: 0.2212120	total: 400ms	remaining: 639ms
    385:	learn: 0.2211060	total: 401ms	remaining: 638ms
    386:	learn: 0.2210524	total: 402ms	remaining: 636ms
    387:	learn: 0.2209084	total: 403ms	remaining: 635ms
    388:	learn: 0.2207930	total: 404ms	remaining: 634ms
    389:	learn: 0.2207173	total: 405ms	remaining: 633ms
    390:	learn: 0.2205771	total: 406ms	remaining: 632ms
    391:	learn: 0.2204144	total: 407ms	remaining: 631ms
    392:	learn: 0.2202577	total: 408ms	remaining: 630ms
    393:	learn: 0.2201388	total: 409ms	remaining: 628ms
    394:	learn: 0.2200203	total: 409ms	remaining: 627ms
    395:	learn: 0.2198401	total: 410ms	remaining: 626ms
    396:	learn: 0.2197172	total: 412ms	remaining: 625ms
    397:	learn: 0.2196006	total: 413ms	remaining: 624ms
    398:	learn: 0.2195505	total: 414ms	remaining: 623ms
    399:	learn: 0.2193129	total: 415ms	remaining: 622ms
    400:	learn: 0.2191898	total: 416ms	remaining: 621ms
    401:	learn: 0.2190684	total: 416ms	remaining: 620ms
    402:	learn: 0.2185711	total: 417ms	remaining: 618ms
    403:	learn: 0.2184490	total: 418ms	remaining: 617ms
    404:	learn: 0.2181881	total: 419ms	remaining: 616ms
    405:	learn: 0.2180388	total: 420ms	remaining: 615ms
    406:	learn: 0.2178893	total: 421ms	remaining: 614ms
    407:	learn: 0.2177664	total: 422ms	remaining: 612ms
    408:	learn: 0.2176388	total: 423ms	remaining: 611ms
    409:	learn: 0.2175148	total: 425ms	remaining: 611ms
    410:	learn: 0.2173359	total: 426ms	remaining: 610ms
    411:	learn: 0.2172823	total: 427ms	remaining: 610ms
    412:	learn: 0.2170855	total: 428ms	remaining: 608ms
    413:	learn: 0.2170486	total: 432ms	remaining: 611ms
    414:	learn: 0.2169021	total: 435ms	remaining: 614ms
    415:	learn: 0.2167779	total: 437ms	remaining: 613ms
    416:	learn: 0.2166595	total: 438ms	remaining: 612ms
    417:	learn: 0.2165589	total: 440ms	remaining: 613ms
    418:	learn: 0.2162123	total: 442ms	remaining: 613ms
    419:	learn: 0.2159959	total: 443ms	remaining: 612ms
    420:	learn: 0.2159250	total: 444ms	remaining: 611ms
    421:	learn: 0.2157002	total: 445ms	remaining: 610ms
    422:	learn: 0.2155720	total: 446ms	remaining: 608ms
    423:	learn: 0.2154450	total: 447ms	remaining: 607ms
    424:	learn: 0.2152065	total: 448ms	remaining: 606ms
    425:	learn: 0.2150882	total: 449ms	remaining: 605ms
    426:	learn: 0.2150843	total: 450ms	remaining: 604ms
    427:	learn: 0.2149688	total: 451ms	remaining: 602ms
    428:	learn: 0.2147750	total: 452ms	remaining: 601ms
    429:	learn: 0.2147471	total: 453ms	remaining: 600ms
    430:	learn: 0.2146307	total: 454ms	remaining: 599ms
    431:	learn: 0.2144846	total: 454ms	remaining: 598ms
    432:	learn: 0.2142725	total: 455ms	remaining: 596ms
    433:	learn: 0.2141822	total: 456ms	remaining: 595ms
    434:	learn: 0.2140604	total: 457ms	remaining: 594ms
    435:	learn: 0.2138816	total: 458ms	remaining: 593ms
    436:	learn: 0.2138534	total: 459ms	remaining: 592ms
    437:	learn: 0.2137614	total: 460ms	remaining: 591ms
    438:	learn: 0.2137410	total: 461ms	remaining: 589ms
    439:	learn: 0.2136554	total: 462ms	remaining: 588ms
    440:	learn: 0.2135383	total: 463ms	remaining: 587ms
    441:	learn: 0.2133715	total: 464ms	remaining: 586ms
    442:	learn: 0.2132178	total: 465ms	remaining: 584ms
    443:	learn: 0.2130669	total: 466ms	remaining: 583ms
    444:	learn: 0.2126712	total: 467ms	remaining: 582ms
    445:	learn: 0.2125858	total: 467ms	remaining: 581ms
    446:	learn: 0.2124369	total: 468ms	remaining: 579ms
    447:	learn: 0.2123195	total: 469ms	remaining: 578ms
    448:	learn: 0.2121662	total: 470ms	remaining: 577ms
    449:	learn: 0.2121556	total: 471ms	remaining: 576ms
    450:	learn: 0.2119840	total: 472ms	remaining: 575ms
    451:	learn: 0.2118470	total: 473ms	remaining: 574ms
    452:	learn: 0.2118273	total: 474ms	remaining: 572ms
    453:	learn: 0.2113979	total: 475ms	remaining: 571ms
    454:	learn: 0.2112511	total: 476ms	remaining: 570ms
    455:	learn: 0.2112442	total: 477ms	remaining: 569ms
    456:	learn: 0.2111097	total: 478ms	remaining: 567ms
    457:	learn: 0.2110142	total: 479ms	remaining: 566ms
    458:	learn: 0.2109091	total: 479ms	remaining: 565ms
    459:	learn: 0.2108025	total: 480ms	remaining: 564ms
    460:	learn: 0.2106833	total: 481ms	remaining: 563ms
    461:	learn: 0.2104752	total: 482ms	remaining: 562ms
    462:	learn: 0.2103895	total: 483ms	remaining: 560ms
    463:	learn: 0.2102639	total: 484ms	remaining: 559ms
    464:	learn: 0.2101352	total: 485ms	remaining: 558ms
    465:	learn: 0.2100612	total: 486ms	remaining: 557ms
    466:	learn: 0.2099500	total: 487ms	remaining: 556ms
    467:	learn: 0.2098378	total: 488ms	remaining: 555ms
    468:	learn: 0.2096559	total: 489ms	remaining: 553ms
    469:	learn: 0.2095356	total: 490ms	remaining: 552ms
    470:	learn: 0.2094151	total: 491ms	remaining: 551ms
    471:	learn: 0.2093134	total: 492ms	remaining: 550ms
    472:	learn: 0.2091256	total: 493ms	remaining: 549ms
    473:	learn: 0.2089976	total: 494ms	remaining: 548ms
    474:	learn: 0.2088789	total: 495ms	remaining: 547ms
    475:	learn: 0.2088169	total: 495ms	remaining: 545ms
    476:	learn: 0.2086850	total: 496ms	remaining: 544ms
    477:	learn: 0.2085121	total: 497ms	remaining: 543ms
    478:	learn: 0.2084195	total: 498ms	remaining: 542ms
    479:	learn: 0.2082497	total: 499ms	remaining: 541ms
    480:	learn: 0.2081580	total: 500ms	remaining: 540ms
    481:	learn: 0.2080403	total: 501ms	remaining: 539ms
    482:	learn: 0.2078111	total: 502ms	remaining: 537ms
    483:	learn: 0.2077049	total: 503ms	remaining: 536ms
    484:	learn: 0.2075767	total: 504ms	remaining: 535ms
    485:	learn: 0.2074531	total: 505ms	remaining: 534ms
    486:	learn: 0.2074306	total: 506ms	remaining: 533ms
    487:	learn: 0.2072755	total: 507ms	remaining: 532ms
    488:	learn: 0.2071573	total: 508ms	remaining: 531ms
    489:	learn: 0.2070761	total: 509ms	remaining: 530ms
    490:	learn: 0.2069686	total: 510ms	remaining: 528ms
    491:	learn: 0.2068897	total: 510ms	remaining: 527ms
    492:	learn: 0.2067770	total: 511ms	remaining: 525ms
    493:	learn: 0.2065505	total: 512ms	remaining: 524ms
    494:	learn: 0.2064604	total: 512ms	remaining: 523ms
    495:	learn: 0.2062906	total: 513ms	remaining: 521ms
    496:	learn: 0.2060950	total: 513ms	remaining: 520ms
    497:	learn: 0.2059199	total: 514ms	remaining: 518ms
    498:	learn: 0.2059056	total: 515ms	remaining: 517ms
    499:	learn: 0.2058137	total: 515ms	remaining: 515ms
    500:	learn: 0.2057029	total: 516ms	remaining: 514ms
    501:	learn: 0.2053805	total: 517ms	remaining: 513ms
    502:	learn: 0.2052319	total: 517ms	remaining: 511ms
    503:	learn: 0.2051395	total: 518ms	remaining: 510ms
    504:	learn: 0.2051363	total: 519ms	remaining: 509ms
    505:	learn: 0.2050301	total: 520ms	remaining: 507ms
    506:	learn: 0.2049033	total: 521ms	remaining: 506ms
    507:	learn: 0.2048109	total: 522ms	remaining: 505ms
    508:	learn: 0.2048033	total: 523ms	remaining: 504ms
    509:	learn: 0.2046647	total: 524ms	remaining: 503ms
    510:	learn: 0.2045519	total: 525ms	remaining: 502ms
    511:	learn: 0.2044151	total: 525ms	remaining: 501ms
    512:	learn: 0.2042737	total: 526ms	remaining: 500ms
    513:	learn: 0.2041367	total: 527ms	remaining: 499ms
    514:	learn: 0.2041287	total: 528ms	remaining: 497ms
    515:	learn: 0.2040045	total: 529ms	remaining: 497ms
    516:	learn: 0.2033267	total: 530ms	remaining: 496ms
    517:	learn: 0.2032239	total: 531ms	remaining: 495ms
    518:	learn: 0.2031169	total: 532ms	remaining: 493ms
    519:	learn: 0.2029977	total: 533ms	remaining: 492ms
    520:	learn: 0.2028880	total: 534ms	remaining: 491ms
    521:	learn: 0.2027474	total: 535ms	remaining: 490ms
    522:	learn: 0.2026522	total: 536ms	remaining: 489ms
    523:	learn: 0.2025374	total: 537ms	remaining: 488ms
    524:	learn: 0.2024582	total: 538ms	remaining: 487ms
    525:	learn: 0.2019911	total: 539ms	remaining: 486ms
    526:	learn: 0.2018507	total: 540ms	remaining: 485ms
    527:	learn: 0.2017697	total: 541ms	remaining: 484ms
    528:	learn: 0.2015989	total: 542ms	remaining: 482ms
    529:	learn: 0.2015126	total: 543ms	remaining: 481ms
    530:	learn: 0.2014501	total: 544ms	remaining: 480ms
    531:	learn: 0.2013071	total: 545ms	remaining: 479ms
    532:	learn: 0.2012508	total: 546ms	remaining: 478ms
    533:	learn: 0.2011443	total: 547ms	remaining: 477ms
    534:	learn: 0.2010263	total: 548ms	remaining: 476ms
    535:	learn: 0.2006591	total: 549ms	remaining: 475ms
    536:	learn: 0.2005646	total: 550ms	remaining: 474ms
    537:	learn: 0.2004676	total: 551ms	remaining: 473ms
    538:	learn: 0.2002575	total: 552ms	remaining: 472ms
    539:	learn: 0.2001328	total: 553ms	remaining: 471ms
    540:	learn: 0.2000240	total: 553ms	remaining: 470ms
    541:	learn: 0.1999429	total: 554ms	remaining: 469ms
    542:	learn: 0.1998127	total: 555ms	remaining: 467ms
    543:	learn: 0.1994973	total: 556ms	remaining: 466ms
    544:	learn: 0.1993866	total: 557ms	remaining: 465ms
    545:	learn: 0.1990902	total: 558ms	remaining: 464ms
    546:	learn: 0.1989183	total: 559ms	remaining: 463ms
    547:	learn: 0.1988315	total: 560ms	remaining: 462ms
    548:	learn: 0.1987225	total: 561ms	remaining: 461ms
    549:	learn: 0.1986285	total: 562ms	remaining: 460ms
    550:	learn: 0.1985354	total: 563ms	remaining: 459ms
    551:	learn: 0.1984243	total: 564ms	remaining: 458ms
    552:	learn: 0.1983255	total: 565ms	remaining: 457ms
    553:	learn: 0.1982113	total: 566ms	remaining: 456ms
    554:	learn: 0.1980990	total: 567ms	remaining: 455ms
    555:	learn: 0.1979883	total: 568ms	remaining: 453ms
    556:	learn: 0.1978893	total: 569ms	remaining: 452ms
    557:	learn: 0.1978144	total: 570ms	remaining: 451ms
    558:	learn: 0.1977396	total: 571ms	remaining: 450ms
    559:	learn: 0.1976605	total: 572ms	remaining: 449ms
    560:	learn: 0.1975942	total: 573ms	remaining: 448ms
    561:	learn: 0.1975099	total: 574ms	remaining: 447ms
    562:	learn: 0.1974248	total: 575ms	remaining: 446ms
    563:	learn: 0.1973457	total: 576ms	remaining: 445ms
    564:	learn: 0.1972657	total: 577ms	remaining: 444ms
    565:	learn: 0.1971782	total: 578ms	remaining: 443ms
    566:	learn: 0.1970391	total: 578ms	remaining: 442ms
    567:	learn: 0.1969424	total: 579ms	remaining: 441ms
    568:	learn: 0.1969357	total: 580ms	remaining: 440ms
    569:	learn: 0.1968430	total: 582ms	remaining: 439ms
    570:	learn: 0.1967684	total: 583ms	remaining: 438ms
    571:	learn: 0.1966822	total: 584ms	remaining: 437ms
    572:	learn: 0.1965696	total: 585ms	remaining: 436ms
    573:	learn: 0.1964616	total: 586ms	remaining: 435ms
    574:	learn: 0.1963811	total: 587ms	remaining: 434ms
    575:	learn: 0.1962320	total: 588ms	remaining: 433ms
    576:	learn: 0.1958831	total: 589ms	remaining: 432ms
    577:	learn: 0.1956067	total: 590ms	remaining: 431ms
    578:	learn: 0.1955136	total: 591ms	remaining: 430ms
    579:	learn: 0.1955125	total: 592ms	remaining: 429ms
    580:	learn: 0.1953587	total: 593ms	remaining: 428ms
    581:	learn: 0.1952657	total: 594ms	remaining: 427ms
    582:	learn: 0.1951009	total: 595ms	remaining: 426ms
    583:	learn: 0.1948040	total: 596ms	remaining: 424ms
    584:	learn: 0.1947285	total: 597ms	remaining: 423ms
    585:	learn: 0.1946251	total: 598ms	remaining: 422ms
    586:	learn: 0.1945507	total: 599ms	remaining: 421ms
    587:	learn: 0.1944856	total: 600ms	remaining: 420ms
    588:	learn: 0.1944325	total: 601ms	remaining: 419ms
    589:	learn: 0.1943405	total: 602ms	remaining: 418ms
    590:	learn: 0.1942489	total: 605ms	remaining: 419ms
    591:	learn: 0.1941334	total: 606ms	remaining: 417ms
    592:	learn: 0.1940434	total: 607ms	remaining: 416ms
    593:	learn: 0.1939450	total: 607ms	remaining: 415ms
    594:	learn: 0.1938912	total: 611ms	remaining: 416ms
    595:	learn: 0.1937825	total: 611ms	remaining: 414ms
    596:	learn: 0.1937018	total: 612ms	remaining: 413ms
    597:	learn: 0.1936342	total: 615ms	remaining: 414ms
    598:	learn: 0.1935265	total: 617ms	remaining: 413ms
    599:	learn: 0.1934105	total: 620ms	remaining: 413ms
    600:	learn: 0.1933075	total: 620ms	remaining: 412ms
    601:	learn: 0.1931673	total: 621ms	remaining: 411ms
    602:	learn: 0.1930847	total: 627ms	remaining: 413ms
    603:	learn: 0.1930141	total: 629ms	remaining: 412ms
    604:	learn: 0.1927793	total: 634ms	remaining: 414ms
    605:	learn: 0.1926125	total: 636ms	remaining: 413ms
    606:	learn: 0.1924593	total: 638ms	remaining: 413ms
    607:	learn: 0.1924517	total: 644ms	remaining: 415ms
    608:	learn: 0.1923216	total: 647ms	remaining: 415ms
    609:	learn: 0.1922442	total: 650ms	remaining: 415ms
    610:	learn: 0.1921236	total: 652ms	remaining: 415ms
    611:	learn: 0.1920690	total: 654ms	remaining: 415ms
    612:	learn: 0.1919110	total: 655ms	remaining: 414ms
    613:	learn: 0.1918496	total: 656ms	remaining: 412ms
    614:	learn: 0.1917402	total: 657ms	remaining: 411ms
    615:	learn: 0.1915432	total: 658ms	remaining: 410ms
    616:	learn: 0.1914472	total: 659ms	remaining: 409ms
    617:	learn: 0.1912251	total: 660ms	remaining: 408ms
    618:	learn: 0.1911529	total: 661ms	remaining: 407ms
    619:	learn: 0.1910185	total: 662ms	remaining: 406ms
    620:	learn: 0.1909418	total: 663ms	remaining: 405ms
    621:	learn: 0.1908532	total: 664ms	remaining: 403ms
    622:	learn: 0.1904388	total: 665ms	remaining: 402ms
    623:	learn: 0.1902792	total: 666ms	remaining: 401ms
    624:	learn: 0.1901845	total: 667ms	remaining: 400ms
    625:	learn: 0.1900785	total: 668ms	remaining: 399ms
    626:	learn: 0.1899628	total: 669ms	remaining: 398ms
    627:	learn: 0.1898756	total: 670ms	remaining: 397ms
    628:	learn: 0.1897863	total: 671ms	remaining: 395ms
    629:	learn: 0.1896754	total: 672ms	remaining: 394ms
    630:	learn: 0.1894193	total: 672ms	remaining: 393ms
    631:	learn: 0.1892377	total: 673ms	remaining: 392ms
    632:	learn: 0.1891410	total: 674ms	remaining: 391ms
    633:	learn: 0.1888903	total: 675ms	remaining: 390ms
    634:	learn: 0.1887829	total: 676ms	remaining: 389ms
    635:	learn: 0.1887542	total: 677ms	remaining: 387ms
    636:	learn: 0.1886735	total: 678ms	remaining: 386ms
    637:	learn: 0.1885492	total: 679ms	remaining: 385ms
    638:	learn: 0.1884770	total: 680ms	remaining: 384ms
    639:	learn: 0.1884109	total: 681ms	remaining: 383ms
    640:	learn: 0.1882786	total: 682ms	remaining: 382ms
    641:	learn: 0.1881768	total: 683ms	remaining: 381ms
    642:	learn: 0.1880884	total: 684ms	remaining: 380ms
    643:	learn: 0.1879927	total: 685ms	remaining: 379ms
    644:	learn: 0.1879027	total: 686ms	remaining: 377ms
    645:	learn: 0.1878034	total: 687ms	remaining: 376ms
    646:	learn: 0.1877975	total: 688ms	remaining: 375ms
    647:	learn: 0.1877155	total: 689ms	remaining: 374ms
    648:	learn: 0.1876333	total: 690ms	remaining: 373ms
    649:	learn: 0.1874673	total: 690ms	remaining: 372ms
    650:	learn: 0.1870861	total: 691ms	remaining: 371ms
    651:	learn: 0.1870073	total: 692ms	remaining: 370ms
    652:	learn: 0.1869254	total: 693ms	remaining: 368ms
    653:	learn: 0.1868586	total: 694ms	remaining: 367ms
    654:	learn: 0.1867995	total: 696ms	remaining: 366ms
    655:	learn: 0.1867113	total: 696ms	remaining: 365ms
    656:	learn: 0.1863974	total: 697ms	remaining: 364ms
    657:	learn: 0.1863053	total: 697ms	remaining: 362ms
    658:	learn: 0.1861746	total: 698ms	remaining: 361ms
    659:	learn: 0.1860585	total: 699ms	remaining: 360ms
    660:	learn: 0.1857097	total: 699ms	remaining: 359ms
    661:	learn: 0.1856374	total: 700ms	remaining: 357ms
    662:	learn: 0.1855283	total: 701ms	remaining: 356ms
    663:	learn: 0.1854665	total: 701ms	remaining: 355ms
    664:	learn: 0.1853634	total: 702ms	remaining: 354ms
    665:	learn: 0.1853605	total: 703ms	remaining: 352ms
    666:	learn: 0.1852821	total: 703ms	remaining: 351ms
    667:	learn: 0.1852106	total: 704ms	remaining: 350ms
    668:	learn: 0.1850995	total: 704ms	remaining: 349ms
    669:	learn: 0.1850994	total: 705ms	remaining: 347ms
    670:	learn: 0.1850366	total: 706ms	remaining: 346ms
    671:	learn: 0.1849987	total: 706ms	remaining: 345ms
    672:	learn: 0.1849187	total: 707ms	remaining: 344ms
    673:	learn: 0.1849138	total: 708ms	remaining: 342ms
    674:	learn: 0.1848128	total: 709ms	remaining: 341ms
    675:	learn: 0.1847491	total: 709ms	remaining: 340ms
    676:	learn: 0.1846651	total: 710ms	remaining: 339ms
    677:	learn: 0.1845456	total: 711ms	remaining: 338ms
    678:	learn: 0.1843402	total: 711ms	remaining: 336ms
    679:	learn: 0.1842748	total: 712ms	remaining: 335ms
    680:	learn: 0.1841999	total: 713ms	remaining: 334ms
    681:	learn: 0.1840819	total: 713ms	remaining: 333ms
    682:	learn: 0.1839082	total: 714ms	remaining: 331ms
    683:	learn: 0.1837519	total: 715ms	remaining: 330ms
    684:	learn: 0.1836844	total: 715ms	remaining: 329ms
    685:	learn: 0.1835158	total: 716ms	remaining: 328ms
    686:	learn: 0.1834517	total: 717ms	remaining: 327ms
    687:	learn: 0.1833788	total: 718ms	remaining: 325ms
    688:	learn: 0.1832985	total: 718ms	remaining: 324ms
    689:	learn: 0.1831089	total: 719ms	remaining: 323ms
    690:	learn: 0.1830221	total: 720ms	remaining: 322ms
    691:	learn: 0.1829702	total: 721ms	remaining: 321ms
    692:	learn: 0.1829153	total: 722ms	remaining: 320ms
    693:	learn: 0.1828055	total: 723ms	remaining: 319ms
    694:	learn: 0.1827486	total: 724ms	remaining: 318ms
    695:	learn: 0.1826888	total: 725ms	remaining: 317ms
    696:	learn: 0.1824099	total: 726ms	remaining: 316ms
    697:	learn: 0.1821891	total: 727ms	remaining: 314ms
    698:	learn: 0.1821210	total: 728ms	remaining: 313ms
    699:	learn: 0.1820514	total: 729ms	remaining: 312ms
    700:	learn: 0.1819620	total: 730ms	remaining: 311ms
    701:	learn: 0.1818591	total: 730ms	remaining: 310ms
    702:	learn: 0.1817549	total: 732ms	remaining: 309ms
    703:	learn: 0.1816751	total: 733ms	remaining: 308ms
    704:	learn: 0.1815677	total: 734ms	remaining: 307ms
    705:	learn: 0.1814738	total: 735ms	remaining: 306ms
    706:	learn: 0.1814039	total: 736ms	remaining: 305ms
    707:	learn: 0.1810639	total: 737ms	remaining: 304ms
    708:	learn: 0.1810093	total: 737ms	remaining: 303ms
    709:	learn: 0.1809975	total: 738ms	remaining: 302ms
    710:	learn: 0.1807188	total: 740ms	remaining: 301ms
    711:	learn: 0.1806177	total: 741ms	remaining: 300ms
    712:	learn: 0.1805303	total: 742ms	remaining: 299ms
    713:	learn: 0.1805280	total: 743ms	remaining: 297ms
    714:	learn: 0.1804194	total: 743ms	remaining: 296ms
    715:	learn: 0.1804150	total: 744ms	remaining: 295ms
    716:	learn: 0.1803476	total: 745ms	remaining: 294ms
    717:	learn: 0.1802941	total: 746ms	remaining: 293ms
    718:	learn: 0.1802108	total: 747ms	remaining: 292ms
    719:	learn: 0.1799460	total: 748ms	remaining: 291ms
    720:	learn: 0.1798780	total: 749ms	remaining: 290ms
    721:	learn: 0.1798027	total: 750ms	remaining: 289ms
    722:	learn: 0.1797400	total: 751ms	remaining: 288ms
    723:	learn: 0.1795433	total: 752ms	remaining: 287ms
    724:	learn: 0.1791901	total: 753ms	remaining: 286ms
    725:	learn: 0.1791089	total: 754ms	remaining: 285ms
    726:	learn: 0.1791064	total: 755ms	remaining: 284ms
    727:	learn: 0.1790647	total: 756ms	remaining: 282ms
    728:	learn: 0.1786856	total: 757ms	remaining: 281ms
    729:	learn: 0.1785616	total: 758ms	remaining: 280ms
    730:	learn: 0.1784636	total: 759ms	remaining: 279ms
    731:	learn: 0.1783832	total: 760ms	remaining: 278ms
    732:	learn: 0.1781909	total: 761ms	remaining: 277ms
    733:	learn: 0.1780896	total: 762ms	remaining: 276ms
    734:	learn: 0.1779685	total: 762ms	remaining: 275ms
    735:	learn: 0.1779179	total: 763ms	remaining: 274ms
    736:	learn: 0.1779133	total: 764ms	remaining: 272ms
    737:	learn: 0.1777319	total: 764ms	remaining: 271ms
    738:	learn: 0.1776470	total: 765ms	remaining: 270ms
    739:	learn: 0.1775755	total: 765ms	remaining: 269ms
    740:	learn: 0.1775016	total: 766ms	remaining: 268ms
    741:	learn: 0.1774151	total: 769ms	remaining: 267ms
    742:	learn: 0.1773669	total: 770ms	remaining: 266ms
    743:	learn: 0.1772881	total: 771ms	remaining: 265ms
    744:	learn: 0.1772838	total: 772ms	remaining: 264ms
    745:	learn: 0.1770440	total: 773ms	remaining: 263ms
    746:	learn: 0.1769835	total: 773ms	remaining: 262ms
    747:	learn: 0.1769098	total: 775ms	remaining: 261ms
    748:	learn: 0.1767049	total: 775ms	remaining: 260ms
    749:	learn: 0.1765878	total: 776ms	remaining: 259ms
    750:	learn: 0.1765822	total: 777ms	remaining: 258ms
    751:	learn: 0.1765103	total: 778ms	remaining: 257ms
    752:	learn: 0.1764484	total: 779ms	remaining: 256ms
    753:	learn: 0.1763940	total: 780ms	remaining: 255ms
    754:	learn: 0.1761889	total: 781ms	remaining: 253ms
    755:	learn: 0.1759999	total: 782ms	remaining: 252ms
    756:	learn: 0.1759973	total: 784ms	remaining: 252ms
    757:	learn: 0.1759568	total: 785ms	remaining: 250ms
    758:	learn: 0.1758005	total: 786ms	remaining: 250ms
    759:	learn: 0.1757578	total: 787ms	remaining: 248ms
    760:	learn: 0.1756972	total: 788ms	remaining: 247ms
    761:	learn: 0.1755597	total: 789ms	remaining: 246ms
    762:	learn: 0.1755312	total: 795ms	remaining: 247ms
    763:	learn: 0.1755264	total: 796ms	remaining: 246ms
    764:	learn: 0.1754830	total: 796ms	remaining: 245ms
    765:	learn: 0.1754255	total: 799ms	remaining: 244ms
    766:	learn: 0.1753172	total: 800ms	remaining: 243ms
    767:	learn: 0.1753099	total: 801ms	remaining: 242ms
    768:	learn: 0.1751747	total: 802ms	remaining: 241ms
    769:	learn: 0.1751030	total: 804ms	remaining: 240ms
    770:	learn: 0.1749252	total: 805ms	remaining: 239ms
    771:	learn: 0.1747708	total: 806ms	remaining: 238ms
    772:	learn: 0.1746147	total: 807ms	remaining: 237ms
    773:	learn: 0.1744211	total: 808ms	remaining: 236ms
    774:	learn: 0.1744132	total: 809ms	remaining: 235ms
    775:	learn: 0.1742430	total: 810ms	remaining: 234ms
    776:	learn: 0.1741439	total: 811ms	remaining: 233ms
    777:	learn: 0.1739721	total: 812ms	remaining: 232ms
    778:	learn: 0.1738163	total: 812ms	remaining: 230ms
    779:	learn: 0.1736888	total: 813ms	remaining: 229ms
    780:	learn: 0.1736117	total: 814ms	remaining: 228ms
    781:	learn: 0.1734937	total: 815ms	remaining: 227ms
    782:	learn: 0.1734449	total: 816ms	remaining: 226ms
    783:	learn: 0.1733318	total: 817ms	remaining: 225ms
    784:	learn: 0.1730706	total: 818ms	remaining: 224ms
    785:	learn: 0.1729538	total: 819ms	remaining: 223ms
    786:	learn: 0.1726609	total: 820ms	remaining: 222ms
    787:	learn: 0.1725770	total: 821ms	remaining: 221ms
    788:	learn: 0.1723687	total: 822ms	remaining: 220ms
    789:	learn: 0.1723044	total: 823ms	remaining: 219ms
    790:	learn: 0.1722186	total: 824ms	remaining: 218ms
    791:	learn: 0.1721431	total: 825ms	remaining: 217ms
    792:	learn: 0.1719981	total: 826ms	remaining: 216ms
    793:	learn: 0.1717370	total: 827ms	remaining: 215ms
    794:	learn: 0.1716214	total: 828ms	remaining: 214ms
    795:	learn: 0.1715593	total: 830ms	remaining: 213ms
    796:	learn: 0.1715076	total: 831ms	remaining: 212ms
    797:	learn: 0.1711546	total: 831ms	remaining: 210ms
    798:	learn: 0.1710725	total: 832ms	remaining: 209ms
    799:	learn: 0.1708792	total: 833ms	remaining: 208ms
    800:	learn: 0.1708235	total: 834ms	remaining: 207ms
    801:	learn: 0.1707461	total: 835ms	remaining: 206ms
    802:	learn: 0.1706587	total: 836ms	remaining: 205ms
    803:	learn: 0.1705851	total: 837ms	remaining: 204ms
    804:	learn: 0.1705000	total: 838ms	remaining: 203ms
    805:	learn: 0.1704617	total: 839ms	remaining: 202ms
    806:	learn: 0.1704095	total: 840ms	remaining: 201ms
    807:	learn: 0.1703226	total: 841ms	remaining: 200ms
    808:	learn: 0.1702385	total: 842ms	remaining: 199ms
    809:	learn: 0.1700250	total: 843ms	remaining: 198ms
    810:	learn: 0.1699521	total: 843ms	remaining: 197ms
    811:	learn: 0.1698580	total: 844ms	remaining: 195ms
    812:	learn: 0.1697915	total: 845ms	remaining: 194ms
    813:	learn: 0.1697252	total: 846ms	remaining: 193ms
    814:	learn: 0.1696603	total: 847ms	remaining: 192ms
    815:	learn: 0.1696028	total: 848ms	remaining: 191ms
    816:	learn: 0.1695323	total: 849ms	remaining: 190ms
    817:	learn: 0.1694610	total: 850ms	remaining: 189ms
    818:	learn: 0.1693249	total: 851ms	remaining: 188ms
    819:	learn: 0.1692594	total: 852ms	remaining: 187ms
    820:	learn: 0.1692187	total: 853ms	remaining: 186ms
    821:	learn: 0.1691224	total: 854ms	remaining: 185ms
    822:	learn: 0.1690796	total: 855ms	remaining: 184ms
    823:	learn: 0.1688541	total: 856ms	remaining: 183ms
    824:	learn: 0.1686675	total: 857ms	remaining: 182ms
    825:	learn: 0.1685098	total: 858ms	remaining: 181ms
    826:	learn: 0.1684234	total: 859ms	remaining: 180ms
    827:	learn: 0.1682507	total: 860ms	remaining: 179ms
    828:	learn: 0.1681944	total: 860ms	remaining: 177ms
    829:	learn: 0.1681491	total: 861ms	remaining: 176ms
    830:	learn: 0.1680674	total: 862ms	remaining: 175ms
    831:	learn: 0.1679987	total: 863ms	remaining: 174ms
    832:	learn: 0.1679080	total: 864ms	remaining: 173ms
    833:	learn: 0.1676489	total: 865ms	remaining: 172ms
    834:	learn: 0.1674079	total: 866ms	remaining: 171ms
    835:	learn: 0.1673343	total: 867ms	remaining: 170ms
    836:	learn: 0.1672563	total: 868ms	remaining: 169ms
    837:	learn: 0.1671904	total: 869ms	remaining: 168ms
    838:	learn: 0.1671294	total: 870ms	remaining: 167ms
    839:	learn: 0.1667489	total: 871ms	remaining: 166ms
    840:	learn: 0.1666692	total: 872ms	remaining: 165ms
    841:	learn: 0.1665769	total: 873ms	remaining: 164ms
    842:	learn: 0.1664967	total: 874ms	remaining: 163ms
    843:	learn: 0.1664817	total: 875ms	remaining: 162ms
    844:	learn: 0.1664029	total: 876ms	remaining: 161ms
    845:	learn: 0.1663719	total: 877ms	remaining: 160ms
    846:	learn: 0.1662809	total: 878ms	remaining: 159ms
    847:	learn: 0.1661726	total: 879ms	remaining: 158ms
    848:	learn: 0.1660928	total: 880ms	remaining: 156ms
    849:	learn: 0.1660355	total: 881ms	remaining: 155ms
    850:	learn: 0.1659614	total: 882ms	remaining: 154ms
    851:	learn: 0.1659193	total: 883ms	remaining: 153ms
    852:	learn: 0.1658226	total: 883ms	remaining: 152ms
    853:	learn: 0.1656748	total: 884ms	remaining: 151ms
    854:	learn: 0.1654622	total: 885ms	remaining: 150ms
    855:	learn: 0.1652934	total: 886ms	remaining: 149ms
    856:	learn: 0.1652008	total: 887ms	remaining: 148ms
    857:	learn: 0.1650767	total: 888ms	remaining: 147ms
    858:	learn: 0.1649885	total: 889ms	remaining: 146ms
    859:	learn: 0.1648402	total: 890ms	remaining: 145ms
    860:	learn: 0.1647802	total: 891ms	remaining: 144ms
    861:	learn: 0.1644632	total: 892ms	remaining: 143ms
    862:	learn: 0.1643738	total: 893ms	remaining: 142ms
    863:	learn: 0.1643135	total: 894ms	remaining: 141ms
    864:	learn: 0.1642569	total: 895ms	remaining: 140ms
    865:	learn: 0.1642057	total: 896ms	remaining: 139ms
    866:	learn: 0.1641411	total: 897ms	remaining: 138ms
    867:	learn: 0.1641380	total: 898ms	remaining: 137ms
    868:	learn: 0.1640932	total: 899ms	remaining: 136ms
    869:	learn: 0.1640304	total: 900ms	remaining: 134ms
    870:	learn: 0.1638754	total: 901ms	remaining: 133ms
    871:	learn: 0.1638093	total: 902ms	remaining: 132ms
    872:	learn: 0.1637771	total: 903ms	remaining: 131ms
    873:	learn: 0.1637037	total: 904ms	remaining: 130ms
    874:	learn: 0.1636449	total: 905ms	remaining: 129ms
    875:	learn: 0.1635008	total: 905ms	remaining: 128ms
    876:	learn: 0.1634532	total: 906ms	remaining: 127ms
    877:	learn: 0.1632099	total: 907ms	remaining: 126ms
    878:	learn: 0.1632073	total: 908ms	remaining: 125ms
    879:	learn: 0.1631366	total: 909ms	remaining: 124ms
    880:	learn: 0.1629487	total: 910ms	remaining: 123ms
    881:	learn: 0.1628915	total: 911ms	remaining: 122ms
    882:	learn: 0.1628179	total: 912ms	remaining: 121ms
    883:	learn: 0.1626574	total: 913ms	remaining: 120ms
    884:	learn: 0.1625345	total: 914ms	remaining: 119ms
    885:	learn: 0.1624363	total: 915ms	remaining: 118ms
    886:	learn: 0.1623650	total: 916ms	remaining: 117ms
    887:	learn: 0.1623120	total: 917ms	remaining: 116ms
    888:	learn: 0.1622107	total: 918ms	remaining: 115ms
    889:	learn: 0.1621043	total: 919ms	remaining: 114ms
    890:	learn: 0.1620909	total: 919ms	remaining: 112ms
    891:	learn: 0.1620204	total: 920ms	remaining: 111ms
    892:	learn: 0.1619067	total: 922ms	remaining: 110ms
    893:	learn: 0.1618697	total: 923ms	remaining: 109ms
    894:	learn: 0.1617966	total: 923ms	remaining: 108ms
    895:	learn: 0.1617332	total: 924ms	remaining: 107ms
    896:	learn: 0.1615584	total: 925ms	remaining: 106ms
    897:	learn: 0.1615021	total: 925ms	remaining: 105ms
    898:	learn: 0.1614150	total: 926ms	remaining: 104ms
    899:	learn: 0.1611916	total: 927ms	remaining: 103ms
    900:	learn: 0.1611093	total: 927ms	remaining: 102ms
    901:	learn: 0.1609790	total: 928ms	remaining: 101ms
    902:	learn: 0.1609266	total: 928ms	remaining: 99.7ms
    903:	learn: 0.1607741	total: 929ms	remaining: 98.7ms
    904:	learn: 0.1607086	total: 930ms	remaining: 97.6ms
    905:	learn: 0.1606110	total: 930ms	remaining: 96.5ms
    906:	learn: 0.1605755	total: 931ms	remaining: 95.5ms
    907:	learn: 0.1603414	total: 932ms	remaining: 94.4ms
    908:	learn: 0.1602738	total: 932ms	remaining: 93.3ms
    909:	learn: 0.1602231	total: 933ms	remaining: 92.3ms
    910:	learn: 0.1601619	total: 934ms	remaining: 91.2ms
    911:	learn: 0.1601048	total: 935ms	remaining: 90.2ms
    912:	learn: 0.1600491	total: 935ms	remaining: 89.1ms
    913:	learn: 0.1600470	total: 936ms	remaining: 88.1ms
    914:	learn: 0.1600063	total: 937ms	remaining: 87.1ms
    915:	learn: 0.1600044	total: 938ms	remaining: 86.1ms
    916:	learn: 0.1598592	total: 939ms	remaining: 85ms
    917:	learn: 0.1597990	total: 940ms	remaining: 84ms
    918:	learn: 0.1597369	total: 941ms	remaining: 83ms
    919:	learn: 0.1596631	total: 942ms	remaining: 81.9ms
    920:	learn: 0.1596144	total: 943ms	remaining: 80.9ms
    921:	learn: 0.1594794	total: 945ms	remaining: 79.9ms
    922:	learn: 0.1594288	total: 945ms	remaining: 78.9ms
    923:	learn: 0.1593637	total: 946ms	remaining: 77.8ms
    924:	learn: 0.1593326	total: 948ms	remaining: 76.9ms
    925:	learn: 0.1592584	total: 949ms	remaining: 75.8ms
    926:	learn: 0.1592583	total: 950ms	remaining: 74.8ms
    927:	learn: 0.1591825	total: 951ms	remaining: 73.8ms
    928:	learn: 0.1591206	total: 952ms	remaining: 72.8ms
    929:	learn: 0.1590345	total: 953ms	remaining: 71.7ms
    930:	learn: 0.1589750	total: 954ms	remaining: 70.7ms
    931:	learn: 0.1589295	total: 955ms	remaining: 69.7ms
    932:	learn: 0.1587834	total: 956ms	remaining: 68.6ms
    933:	learn: 0.1586981	total: 957ms	remaining: 67.6ms
    934:	learn: 0.1586040	total: 958ms	remaining: 66.6ms
    935:	learn: 0.1584331	total: 959ms	remaining: 65.5ms
    936:	learn: 0.1583698	total: 960ms	remaining: 64.5ms
    937:	learn: 0.1583170	total: 961ms	remaining: 63.5ms
    938:	learn: 0.1582314	total: 961ms	remaining: 62.5ms
    939:	learn: 0.1580722	total: 963ms	remaining: 61.5ms
    940:	learn: 0.1579852	total: 964ms	remaining: 60.4ms
    941:	learn: 0.1578740	total: 965ms	remaining: 59.4ms
    942:	learn: 0.1577883	total: 967ms	remaining: 58.4ms
    943:	learn: 0.1577600	total: 967ms	remaining: 57.4ms
    944:	learn: 0.1577245	total: 968ms	remaining: 56.4ms
    945:	learn: 0.1576874	total: 969ms	remaining: 55.3ms
    946:	learn: 0.1575349	total: 978ms	remaining: 54.7ms
    947:	learn: 0.1573821	total: 980ms	remaining: 53.7ms
    948:	learn: 0.1573478	total: 981ms	remaining: 52.7ms
    949:	learn: 0.1572702	total: 983ms	remaining: 51.7ms
    950:	learn: 0.1572310	total: 984ms	remaining: 50.7ms
    951:	learn: 0.1571759	total: 986ms	remaining: 49.7ms
    952:	learn: 0.1571224	total: 987ms	remaining: 48.7ms
    953:	learn: 0.1570914	total: 988ms	remaining: 47.7ms
    954:	learn: 0.1570653	total: 989ms	remaining: 46.6ms
    955:	learn: 0.1569144	total: 990ms	remaining: 45.6ms
    956:	learn: 0.1568531	total: 991ms	remaining: 44.5ms
    957:	learn: 0.1567843	total: 992ms	remaining: 43.5ms
    958:	learn: 0.1567185	total: 993ms	remaining: 42.5ms
    959:	learn: 0.1567045	total: 994ms	remaining: 41.4ms
    960:	learn: 0.1566529	total: 995ms	remaining: 40.4ms
    961:	learn: 0.1565938	total: 996ms	remaining: 39.4ms
    962:	learn: 0.1565398	total: 997ms	remaining: 38.3ms
    963:	learn: 0.1564815	total: 998ms	remaining: 37.3ms
    964:	learn: 0.1563640	total: 999ms	remaining: 36.2ms
    965:	learn: 0.1562444	total: 1s	remaining: 35.2ms
    966:	learn: 0.1561808	total: 1s	remaining: 34.2ms
    967:	learn: 0.1561323	total: 1s	remaining: 33.1ms
    968:	learn: 0.1560700	total: 1s	remaining: 32.1ms
    969:	learn: 0.1559488	total: 1s	remaining: 31ms
    970:	learn: 0.1558942	total: 1s	remaining: 30ms
    971:	learn: 0.1557938	total: 1s	remaining: 29ms
    972:	learn: 0.1557071	total: 1.01s	remaining: 27.9ms
    973:	learn: 0.1556606	total: 1.01s	remaining: 26.9ms
    974:	learn: 0.1555015	total: 1.01s	remaining: 25.9ms
    975:	learn: 0.1553081	total: 1.01s	remaining: 24.8ms
    976:	learn: 0.1552328	total: 1.01s	remaining: 23.8ms
    977:	learn: 0.1551797	total: 1.01s	remaining: 22.8ms
    978:	learn: 0.1551151	total: 1.01s	remaining: 21.7ms
    979:	learn: 0.1549891	total: 1.01s	remaining: 20.7ms
    980:	learn: 0.1549061	total: 1.01s	remaining: 19.6ms
    981:	learn: 0.1548527	total: 1.01s	remaining: 18.6ms
    982:	learn: 0.1547748	total: 1.02s	remaining: 17.6ms
    983:	learn: 0.1547365	total: 1.02s	remaining: 16.5ms
    984:	learn: 0.1546178	total: 1.02s	remaining: 15.5ms
    985:	learn: 0.1544933	total: 1.02s	remaining: 14.5ms
    986:	learn: 0.1544910	total: 1.02s	remaining: 13.4ms
    987:	learn: 0.1543874	total: 1.02s	remaining: 12.4ms
    988:	learn: 0.1542915	total: 1.02s	remaining: 11.4ms
    989:	learn: 0.1542346	total: 1.02s	remaining: 10.3ms
    990:	learn: 0.1541279	total: 1.02s	remaining: 9.3ms
    991:	learn: 0.1541198	total: 1.02s	remaining: 8.27ms
    992:	learn: 0.1540265	total: 1.03s	remaining: 7.23ms
    993:	learn: 0.1539442	total: 1.03s	remaining: 6.2ms
    994:	learn: 0.1539109	total: 1.03s	remaining: 5.17ms
    995:	learn: 0.1538594	total: 1.03s	remaining: 4.13ms
    996:	learn: 0.1538593	total: 1.03s	remaining: 3.1ms
    997:	learn: 0.1538249	total: 1.03s	remaining: 2.06ms
    998:	learn: 0.1535424	total: 1.03s	remaining: 1.03ms
    999:	learn: 0.1535424	total: 1.03s	remaining: 0us
    0:	learn: 0.5685542	total: 757us	remaining: 757ms
    1:	learn: 0.4959451	total: 1.5ms	remaining: 747ms
    2:	learn: 0.4574516	total: 2.63ms	remaining: 875ms
    3:	learn: 0.4141575	total: 3.74ms	remaining: 931ms
    4:	learn: 0.3974872	total: 4.8ms	remaining: 954ms
    5:	learn: 0.3705207	total: 5.83ms	remaining: 966ms
    6:	learn: 0.3601860	total: 6.81ms	remaining: 966ms
    7:	learn: 0.3551007	total: 7.81ms	remaining: 968ms
    8:	learn: 0.3495584	total: 8.79ms	remaining: 968ms
    9:	learn: 0.3447114	total: 9.77ms	remaining: 968ms
    10:	learn: 0.3431003	total: 10.7ms	remaining: 966ms
    11:	learn: 0.3420607	total: 11.8ms	remaining: 969ms
    12:	learn: 0.3369543	total: 12.8ms	remaining: 973ms
    13:	learn: 0.3345273	total: 13.8ms	remaining: 969ms
    14:	learn: 0.3333714	total: 14.7ms	remaining: 966ms
    15:	learn: 0.3325031	total: 15.6ms	remaining: 962ms
    16:	learn: 0.3321777	total: 16.6ms	remaining: 961ms
    17:	learn: 0.3318116	total: 17.6ms	remaining: 960ms
    18:	learn: 0.3311497	total: 18.5ms	remaining: 957ms
    19:	learn: 0.3308805	total: 19.5ms	remaining: 957ms
    20:	learn: 0.3289918	total: 20.5ms	remaining: 955ms
    21:	learn: 0.3286439	total: 21.4ms	remaining: 953ms
    22:	learn: 0.3278350	total: 22.3ms	remaining: 949ms
    23:	learn: 0.3269175	total: 23.2ms	remaining: 942ms
    24:	learn: 0.3266557	total: 24ms	remaining: 936ms
    25:	learn: 0.3262100	total: 24.8ms	remaining: 929ms
    26:	learn: 0.3243955	total: 25.8ms	remaining: 929ms
    27:	learn: 0.3239494	total: 26.6ms	remaining: 922ms
    28:	learn: 0.3218381	total: 27.2ms	remaining: 909ms
    29:	learn: 0.3214472	total: 28.1ms	remaining: 908ms
    30:	learn: 0.3213240	total: 29.5ms	remaining: 922ms
    31:	learn: 0.3211058	total: 30.5ms	remaining: 921ms
    32:	learn: 0.3201028	total: 31.5ms	remaining: 923ms
    33:	learn: 0.3196102	total: 32.4ms	remaining: 920ms
    34:	learn: 0.3191946	total: 33.3ms	remaining: 918ms
    35:	learn: 0.3178192	total: 34.2ms	remaining: 916ms
    36:	learn: 0.3166600	total: 35.2ms	remaining: 915ms
    37:	learn: 0.3154598	total: 36.2ms	remaining: 917ms
    38:	learn: 0.3150629	total: 37.2ms	remaining: 917ms
    39:	learn: 0.3132048	total: 38.3ms	remaining: 918ms
    40:	learn: 0.3129834	total: 39.2ms	remaining: 917ms
    41:	learn: 0.3122003	total: 40.2ms	remaining: 916ms
    42:	learn: 0.3117371	total: 41.1ms	remaining: 915ms
    43:	learn: 0.3109573	total: 42.1ms	remaining: 914ms
    44:	learn: 0.3101913	total: 43ms	remaining: 913ms
    45:	learn: 0.3097216	total: 44ms	remaining: 912ms
    46:	learn: 0.3092015	total: 44.8ms	remaining: 909ms
    47:	learn: 0.3084144	total: 45.8ms	remaining: 908ms
    48:	learn: 0.3077250	total: 46.8ms	remaining: 908ms
    49:	learn: 0.3064210	total: 47.7ms	remaining: 906ms
    50:	learn: 0.3057280	total: 48.6ms	remaining: 905ms
    51:	learn: 0.3048548	total: 49.6ms	remaining: 904ms
    52:	learn: 0.3042946	total: 50.5ms	remaining: 903ms
    53:	learn: 0.3036216	total: 51.5ms	remaining: 902ms
    54:	learn: 0.3023601	total: 52.5ms	remaining: 902ms
    55:	learn: 0.3019487	total: 53.5ms	remaining: 902ms
    56:	learn: 0.3014452	total: 54.1ms	remaining: 896ms
    57:	learn: 0.3009154	total: 54.9ms	remaining: 891ms
    58:	learn: 0.3003318	total: 55.6ms	remaining: 887ms
    59:	learn: 0.2999837	total: 56.6ms	remaining: 887ms
    60:	learn: 0.2996496	total: 57.6ms	remaining: 886ms
    61:	learn: 0.2986099	total: 58.5ms	remaining: 886ms
    62:	learn: 0.2984550	total: 59.5ms	remaining: 885ms
    63:	learn: 0.2980780	total: 60.4ms	remaining: 884ms
    64:	learn: 0.2969774	total: 61.4ms	remaining: 884ms
    65:	learn: 0.2964327	total: 62.3ms	remaining: 882ms
    66:	learn: 0.2960496	total: 63.3ms	remaining: 882ms
    67:	learn: 0.2952942	total: 64.2ms	remaining: 880ms
    68:	learn: 0.2945451	total: 65.1ms	remaining: 878ms
    69:	learn: 0.2941334	total: 66ms	remaining: 877ms
    70:	learn: 0.2937963	total: 67ms	remaining: 876ms
    71:	learn: 0.2937115	total: 67.9ms	remaining: 876ms
    72:	learn: 0.2929363	total: 68.8ms	remaining: 874ms
    73:	learn: 0.2923244	total: 69.7ms	remaining: 872ms
    74:	learn: 0.2918678	total: 70.6ms	remaining: 871ms
    75:	learn: 0.2915710	total: 71.5ms	remaining: 870ms
    76:	learn: 0.2911143	total: 73.1ms	remaining: 876ms
    77:	learn: 0.2906865	total: 74ms	remaining: 874ms
    78:	learn: 0.2902471	total: 74.8ms	remaining: 872ms
    79:	learn: 0.2896861	total: 78.5ms	remaining: 903ms
    80:	learn: 0.2890449	total: 80.6ms	remaining: 914ms
    81:	learn: 0.2886875	total: 84.5ms	remaining: 946ms
    82:	learn: 0.2884350	total: 85.7ms	remaining: 947ms
    83:	learn: 0.2876188	total: 88.7ms	remaining: 967ms
    84:	learn: 0.2871134	total: 90.6ms	remaining: 976ms
    85:	learn: 0.2864210	total: 91.4ms	remaining: 971ms
    86:	learn: 0.2859158	total: 92.1ms	remaining: 966ms
    87:	learn: 0.2853852	total: 92.7ms	remaining: 961ms
    88:	learn: 0.2851747	total: 93.4ms	remaining: 956ms
    89:	learn: 0.2846738	total: 94.1ms	remaining: 951ms
    90:	learn: 0.2841073	total: 94.7ms	remaining: 946ms
    91:	learn: 0.2837212	total: 95.4ms	remaining: 941ms
    92:	learn: 0.2834508	total: 96ms	remaining: 936ms
    93:	learn: 0.2823726	total: 98ms	remaining: 945ms
    94:	learn: 0.2819703	total: 99.1ms	remaining: 944ms
    95:	learn: 0.2813830	total: 100ms	remaining: 943ms
    96:	learn: 0.2806747	total: 101ms	remaining: 942ms
    97:	learn: 0.2805565	total: 102ms	remaining: 940ms
    98:	learn: 0.2804671	total: 104ms	remaining: 946ms
    99:	learn: 0.2800276	total: 105ms	remaining: 945ms
    100:	learn: 0.2796484	total: 106ms	remaining: 945ms
    101:	learn: 0.2792221	total: 107ms	remaining: 944ms
    102:	learn: 0.2789048	total: 108ms	remaining: 943ms
    103:	learn: 0.2784266	total: 109ms	remaining: 942ms
    104:	learn: 0.2780539	total: 110ms	remaining: 940ms
    105:	learn: 0.2777319	total: 111ms	remaining: 939ms
    106:	learn: 0.2771106	total: 112ms	remaining: 937ms
    107:	learn: 0.2768482	total: 113ms	remaining: 937ms
    108:	learn: 0.2760715	total: 115ms	remaining: 936ms
    109:	learn: 0.2758513	total: 116ms	remaining: 935ms
    110:	learn: 0.2756247	total: 117ms	remaining: 935ms
    111:	learn: 0.2751236	total: 118ms	remaining: 934ms
    112:	learn: 0.2746302	total: 119ms	remaining: 932ms
    113:	learn: 0.2745489	total: 120ms	remaining: 931ms
    114:	learn: 0.2742877	total: 121ms	remaining: 929ms
    115:	learn: 0.2739388	total: 122ms	remaining: 926ms
    116:	learn: 0.2737025	total: 122ms	remaining: 924ms
    117:	learn: 0.2735013	total: 123ms	remaining: 922ms
    118:	learn: 0.2733379	total: 125ms	remaining: 925ms
    119:	learn: 0.2731680	total: 126ms	remaining: 923ms
    120:	learn: 0.2728843	total: 127ms	remaining: 922ms
    121:	learn: 0.2726245	total: 128ms	remaining: 920ms
    122:	learn: 0.2722860	total: 129ms	remaining: 918ms
    123:	learn: 0.2719026	total: 130ms	remaining: 916ms
    124:	learn: 0.2717138	total: 131ms	remaining: 915ms
    125:	learn: 0.2714392	total: 132ms	remaining: 914ms
    126:	learn: 0.2709610	total: 133ms	remaining: 912ms
    127:	learn: 0.2703596	total: 134ms	remaining: 910ms
    128:	learn: 0.2699564	total: 135ms	remaining: 909ms
    129:	learn: 0.2698796	total: 136ms	remaining: 908ms
    130:	learn: 0.2696115	total: 137ms	remaining: 907ms
    131:	learn: 0.2691692	total: 138ms	remaining: 905ms
    132:	learn: 0.2688702	total: 139ms	remaining: 906ms
    133:	learn: 0.2683692	total: 140ms	remaining: 905ms
    134:	learn: 0.2679605	total: 141ms	remaining: 904ms
    135:	learn: 0.2674710	total: 142ms	remaining: 903ms
    136:	learn: 0.2670669	total: 143ms	remaining: 901ms
    137:	learn: 0.2667296	total: 144ms	remaining: 900ms
    138:	learn: 0.2664905	total: 145ms	remaining: 898ms
    139:	learn: 0.2661342	total: 146ms	remaining: 897ms
    140:	learn: 0.2658312	total: 147ms	remaining: 896ms
    141:	learn: 0.2651692	total: 148ms	remaining: 894ms
    142:	learn: 0.2650120	total: 149ms	remaining: 893ms
    143:	learn: 0.2648102	total: 150ms	remaining: 892ms
    144:	learn: 0.2646146	total: 151ms	remaining: 890ms
    145:	learn: 0.2640458	total: 152ms	remaining: 889ms
    146:	learn: 0.2637253	total: 153ms	remaining: 888ms
    147:	learn: 0.2633183	total: 154ms	remaining: 886ms
    148:	learn: 0.2630276	total: 155ms	remaining: 885ms
    149:	learn: 0.2628972	total: 156ms	remaining: 883ms
    150:	learn: 0.2626333	total: 157ms	remaining: 882ms
    151:	learn: 0.2623329	total: 158ms	remaining: 881ms
    152:	learn: 0.2621277	total: 159ms	remaining: 879ms
    153:	learn: 0.2618087	total: 160ms	remaining: 878ms
    154:	learn: 0.2615887	total: 161ms	remaining: 876ms
    155:	learn: 0.2614075	total: 162ms	remaining: 875ms
    156:	learn: 0.2607894	total: 163ms	remaining: 873ms
    157:	learn: 0.2604858	total: 164ms	remaining: 872ms
    158:	learn: 0.2603014	total: 165ms	remaining: 870ms
    159:	learn: 0.2600427	total: 165ms	remaining: 869ms
    160:	learn: 0.2597267	total: 166ms	remaining: 867ms
    161:	learn: 0.2596608	total: 167ms	remaining: 866ms
    162:	learn: 0.2595173	total: 168ms	remaining: 865ms
    163:	learn: 0.2592812	total: 169ms	remaining: 863ms
    164:	learn: 0.2586553	total: 170ms	remaining: 862ms
    165:	learn: 0.2582921	total: 171ms	remaining: 861ms
    166:	learn: 0.2580159	total: 172ms	remaining: 860ms
    167:	learn: 0.2576296	total: 173ms	remaining: 859ms
    168:	learn: 0.2574555	total: 174ms	remaining: 858ms
    169:	learn: 0.2574031	total: 175ms	remaining: 857ms
    170:	learn: 0.2571717	total: 176ms	remaining: 855ms
    171:	learn: 0.2568994	total: 179ms	remaining: 864ms
    172:	learn: 0.2564267	total: 180ms	remaining: 862ms
    173:	learn: 0.2562552	total: 185ms	remaining: 876ms
    174:	learn: 0.2562090	total: 185ms	remaining: 874ms
    175:	learn: 0.2560141	total: 186ms	remaining: 871ms
    176:	learn: 0.2558354	total: 187ms	remaining: 870ms
    177:	learn: 0.2555601	total: 188ms	remaining: 868ms
    178:	learn: 0.2553863	total: 189ms	remaining: 866ms
    179:	learn: 0.2550198	total: 190ms	remaining: 866ms
    180:	learn: 0.2547481	total: 191ms	remaining: 865ms
    181:	learn: 0.2545333	total: 192ms	remaining: 864ms
    182:	learn: 0.2543279	total: 193ms	remaining: 863ms
    183:	learn: 0.2542007	total: 194ms	remaining: 861ms
    184:	learn: 0.2538101	total: 195ms	remaining: 860ms
    185:	learn: 0.2534392	total: 196ms	remaining: 859ms
    186:	learn: 0.2528680	total: 197ms	remaining: 858ms
    187:	learn: 0.2527330	total: 198ms	remaining: 856ms
    188:	learn: 0.2525249	total: 199ms	remaining: 855ms
    189:	learn: 0.2522079	total: 200ms	remaining: 853ms
    190:	learn: 0.2519200	total: 201ms	remaining: 853ms
    191:	learn: 0.2517422	total: 202ms	remaining: 852ms
    192:	learn: 0.2512815	total: 203ms	remaining: 851ms
    193:	learn: 0.2511738	total: 204ms	remaining: 849ms
    194:	learn: 0.2508495	total: 205ms	remaining: 848ms
    195:	learn: 0.2505260	total: 206ms	remaining: 847ms
    196:	learn: 0.2503971	total: 207ms	remaining: 845ms
    197:	learn: 0.2500344	total: 208ms	remaining: 844ms
    198:	learn: 0.2496997	total: 209ms	remaining: 842ms
    199:	learn: 0.2493403	total: 210ms	remaining: 841ms
    200:	learn: 0.2492067	total: 211ms	remaining: 840ms
    201:	learn: 0.2490390	total: 212ms	remaining: 838ms
    202:	learn: 0.2487791	total: 213ms	remaining: 837ms
    203:	learn: 0.2485999	total: 214ms	remaining: 835ms
    204:	learn: 0.2481272	total: 215ms	remaining: 835ms
    205:	learn: 0.2479030	total: 217ms	remaining: 835ms
    206:	learn: 0.2478628	total: 218ms	remaining: 833ms
    207:	learn: 0.2475662	total: 218ms	remaining: 832ms
    208:	learn: 0.2474450	total: 220ms	remaining: 831ms
    209:	learn: 0.2470553	total: 220ms	remaining: 829ms
    210:	learn: 0.2468504	total: 222ms	remaining: 829ms
    211:	learn: 0.2463647	total: 223ms	remaining: 828ms
    212:	learn: 0.2462072	total: 224ms	remaining: 827ms
    213:	learn: 0.2460839	total: 225ms	remaining: 826ms
    214:	learn: 0.2456478	total: 226ms	remaining: 825ms
    215:	learn: 0.2448751	total: 227ms	remaining: 824ms
    216:	learn: 0.2446329	total: 228ms	remaining: 823ms
    217:	learn: 0.2445931	total: 229ms	remaining: 821ms
    218:	learn: 0.2444588	total: 230ms	remaining: 820ms
    219:	learn: 0.2440315	total: 231ms	remaining: 819ms
    220:	learn: 0.2437886	total: 232ms	remaining: 818ms
    221:	learn: 0.2435428	total: 233ms	remaining: 816ms
    222:	learn: 0.2434288	total: 234ms	remaining: 815ms
    223:	learn: 0.2431253	total: 235ms	remaining: 814ms
    224:	learn: 0.2429960	total: 236ms	remaining: 812ms
    225:	learn: 0.2426850	total: 237ms	remaining: 811ms
    226:	learn: 0.2426441	total: 238ms	remaining: 810ms
    227:	learn: 0.2426047	total: 239ms	remaining: 808ms
    228:	learn: 0.2423073	total: 240ms	remaining: 807ms
    229:	learn: 0.2421514	total: 241ms	remaining: 806ms
    230:	learn: 0.2420484	total: 242ms	remaining: 805ms
    231:	learn: 0.2418918	total: 243ms	remaining: 803ms
    232:	learn: 0.2414121	total: 244ms	remaining: 802ms
    233:	learn: 0.2411553	total: 245ms	remaining: 801ms
    234:	learn: 0.2408066	total: 246ms	remaining: 799ms
    235:	learn: 0.2405886	total: 247ms	remaining: 798ms
    236:	learn: 0.2403837	total: 248ms	remaining: 797ms
    237:	learn: 0.2401566	total: 248ms	remaining: 795ms
    238:	learn: 0.2399210	total: 249ms	remaining: 794ms
    239:	learn: 0.2395761	total: 250ms	remaining: 793ms
    240:	learn: 0.2391612	total: 251ms	remaining: 792ms
    241:	learn: 0.2389240	total: 252ms	remaining: 791ms
    242:	learn: 0.2386742	total: 253ms	remaining: 789ms
    243:	learn: 0.2385706	total: 254ms	remaining: 788ms
    244:	learn: 0.2383734	total: 257ms	remaining: 791ms
    245:	learn: 0.2382884	total: 257ms	remaining: 789ms
    246:	learn: 0.2380067	total: 258ms	remaining: 787ms
    247:	learn: 0.2377607	total: 260ms	remaining: 788ms
    248:	learn: 0.2377232	total: 261ms	remaining: 787ms
    249:	learn: 0.2373985	total: 262ms	remaining: 785ms
    250:	learn: 0.2373070	total: 273ms	remaining: 816ms
    251:	learn: 0.2369573	total: 275ms	remaining: 815ms
    252:	learn: 0.2368283	total: 276ms	remaining: 814ms
    253:	learn: 0.2366708	total: 277ms	remaining: 813ms
    254:	learn: 0.2365171	total: 278ms	remaining: 811ms
    255:	learn: 0.2364262	total: 279ms	remaining: 811ms
    256:	learn: 0.2362216	total: 280ms	remaining: 810ms
    257:	learn: 0.2359323	total: 281ms	remaining: 809ms
    258:	learn: 0.2356891	total: 282ms	remaining: 807ms
    259:	learn: 0.2352679	total: 283ms	remaining: 806ms
    260:	learn: 0.2347541	total: 284ms	remaining: 805ms
    261:	learn: 0.2345822	total: 285ms	remaining: 803ms
    262:	learn: 0.2342967	total: 286ms	remaining: 802ms
    263:	learn: 0.2341810	total: 287ms	remaining: 801ms
    264:	learn: 0.2338450	total: 288ms	remaining: 799ms
    265:	learn: 0.2337293	total: 289ms	remaining: 798ms
    266:	learn: 0.2336426	total: 290ms	remaining: 797ms
    267:	learn: 0.2334159	total: 291ms	remaining: 795ms
    268:	learn: 0.2332043	total: 292ms	remaining: 794ms
    269:	learn: 0.2329058	total: 293ms	remaining: 792ms
    270:	learn: 0.2326751	total: 294ms	remaining: 791ms
    271:	learn: 0.2326033	total: 295ms	remaining: 790ms
    272:	learn: 0.2323781	total: 296ms	remaining: 789ms
    273:	learn: 0.2321476	total: 297ms	remaining: 788ms
    274:	learn: 0.2319054	total: 299ms	remaining: 787ms
    275:	learn: 0.2317532	total: 301ms	remaining: 789ms
    276:	learn: 0.2316496	total: 302ms	remaining: 789ms
    277:	learn: 0.2314883	total: 303ms	remaining: 788ms
    278:	learn: 0.2312156	total: 304ms	remaining: 786ms
    279:	learn: 0.2309550	total: 305ms	remaining: 785ms
    280:	learn: 0.2307100	total: 306ms	remaining: 783ms
    281:	learn: 0.2305455	total: 307ms	remaining: 782ms
    282:	learn: 0.2303951	total: 308ms	remaining: 781ms
    283:	learn: 0.2302022	total: 309ms	remaining: 779ms
    284:	learn: 0.2300458	total: 310ms	remaining: 778ms
    285:	learn: 0.2298936	total: 311ms	remaining: 776ms
    286:	learn: 0.2297862	total: 312ms	remaining: 775ms
    287:	learn: 0.2291539	total: 313ms	remaining: 773ms
    288:	learn: 0.2290002	total: 314ms	remaining: 772ms
    289:	learn: 0.2289051	total: 315ms	remaining: 771ms
    290:	learn: 0.2286153	total: 316ms	remaining: 769ms
    291:	learn: 0.2284380	total: 317ms	remaining: 768ms
    292:	learn: 0.2282066	total: 318ms	remaining: 766ms
    293:	learn: 0.2281223	total: 319ms	remaining: 765ms
    294:	learn: 0.2277664	total: 320ms	remaining: 764ms
    295:	learn: 0.2277092	total: 321ms	remaining: 762ms
    296:	learn: 0.2275191	total: 321ms	remaining: 761ms
    297:	learn: 0.2274455	total: 322ms	remaining: 760ms
    298:	learn: 0.2271102	total: 323ms	remaining: 758ms
    299:	learn: 0.2269389	total: 324ms	remaining: 757ms
    300:	learn: 0.2268768	total: 325ms	remaining: 755ms
    301:	learn: 0.2267573	total: 326ms	remaining: 754ms
    302:	learn: 0.2263592	total: 327ms	remaining: 753ms
    303:	learn: 0.2261847	total: 328ms	remaining: 751ms
    304:	learn: 0.2259337	total: 329ms	remaining: 750ms
    305:	learn: 0.2258425	total: 330ms	remaining: 748ms
    306:	learn: 0.2256023	total: 331ms	remaining: 747ms
    307:	learn: 0.2255247	total: 332ms	remaining: 746ms
    308:	learn: 0.2252752	total: 333ms	remaining: 745ms
    309:	learn: 0.2250942	total: 334ms	remaining: 744ms
    310:	learn: 0.2249470	total: 335ms	remaining: 742ms
    311:	learn: 0.2246349	total: 336ms	remaining: 741ms
    312:	learn: 0.2245029	total: 337ms	remaining: 740ms
    313:	learn: 0.2243868	total: 338ms	remaining: 739ms
    314:	learn: 0.2242153	total: 339ms	remaining: 737ms
    315:	learn: 0.2240333	total: 340ms	remaining: 736ms
    316:	learn: 0.2239497	total: 341ms	remaining: 735ms
    317:	learn: 0.2239154	total: 342ms	remaining: 733ms
    318:	learn: 0.2237749	total: 343ms	remaining: 732ms
    319:	learn: 0.2237111	total: 344ms	remaining: 731ms
    320:	learn: 0.2235744	total: 345ms	remaining: 729ms
    321:	learn: 0.2235166	total: 346ms	remaining: 728ms
    322:	learn: 0.2233172	total: 347ms	remaining: 727ms
    323:	learn: 0.2231433	total: 348ms	remaining: 725ms
    324:	learn: 0.2228634	total: 349ms	remaining: 724ms
    325:	learn: 0.2227434	total: 350ms	remaining: 723ms
    326:	learn: 0.2226287	total: 351ms	remaining: 722ms
    327:	learn: 0.2223178	total: 352ms	remaining: 721ms
    328:	learn: 0.2222855	total: 353ms	remaining: 720ms
    329:	learn: 0.2221745	total: 354ms	remaining: 719ms
    330:	learn: 0.2221504	total: 355ms	remaining: 717ms
    331:	learn: 0.2218896	total: 356ms	remaining: 716ms
    332:	learn: 0.2218213	total: 357ms	remaining: 715ms
    333:	learn: 0.2217417	total: 358ms	remaining: 714ms
    334:	learn: 0.2215725	total: 359ms	remaining: 712ms
    335:	learn: 0.2215512	total: 360ms	remaining: 711ms
    336:	learn: 0.2214006	total: 361ms	remaining: 710ms
    337:	learn: 0.2212762	total: 362ms	remaining: 708ms
    338:	learn: 0.2210530	total: 363ms	remaining: 707ms
    339:	learn: 0.2210303	total: 364ms	remaining: 706ms
    340:	learn: 0.2208477	total: 365ms	remaining: 705ms
    341:	learn: 0.2207255	total: 366ms	remaining: 703ms
    342:	learn: 0.2205969	total: 367ms	remaining: 702ms
    343:	learn: 0.2204194	total: 368ms	remaining: 701ms
    344:	learn: 0.2201852	total: 369ms	remaining: 700ms
    345:	learn: 0.2201640	total: 370ms	remaining: 698ms
    346:	learn: 0.2200260	total: 370ms	remaining: 697ms
    347:	learn: 0.2196634	total: 371ms	remaining: 696ms
    348:	learn: 0.2196090	total: 372ms	remaining: 695ms
    349:	learn: 0.2195449	total: 373ms	remaining: 693ms
    350:	learn: 0.2192661	total: 374ms	remaining: 692ms
    351:	learn: 0.2190230	total: 375ms	remaining: 691ms
    352:	learn: 0.2188664	total: 377ms	remaining: 690ms
    353:	learn: 0.2187304	total: 378ms	remaining: 689ms
    354:	learn: 0.2185957	total: 379ms	remaining: 688ms
    355:	learn: 0.2185093	total: 380ms	remaining: 687ms
    356:	learn: 0.2184040	total: 381ms	remaining: 686ms
    357:	learn: 0.2182120	total: 382ms	remaining: 685ms
    358:	learn: 0.2181555	total: 383ms	remaining: 683ms
    359:	learn: 0.2180743	total: 384ms	remaining: 682ms
    360:	learn: 0.2179018	total: 385ms	remaining: 681ms
    361:	learn: 0.2178045	total: 385ms	remaining: 679ms
    362:	learn: 0.2175761	total: 386ms	remaining: 678ms
    363:	learn: 0.2175492	total: 387ms	remaining: 677ms
    364:	learn: 0.2174669	total: 388ms	remaining: 675ms
    365:	learn: 0.2171947	total: 389ms	remaining: 674ms
    366:	learn: 0.2170580	total: 390ms	remaining: 673ms
    367:	learn: 0.2166713	total: 391ms	remaining: 672ms
    368:	learn: 0.2165531	total: 392ms	remaining: 670ms
    369:	learn: 0.2164280	total: 393ms	remaining: 669ms
    370:	learn: 0.2163653	total: 394ms	remaining: 668ms
    371:	learn: 0.2160837	total: 395ms	remaining: 667ms
    372:	learn: 0.2159703	total: 396ms	remaining: 665ms
    373:	learn: 0.2156456	total: 397ms	remaining: 664ms
    374:	learn: 0.2154815	total: 398ms	remaining: 663ms
    375:	learn: 0.2153775	total: 399ms	remaining: 662ms
    376:	learn: 0.2153121	total: 400ms	remaining: 660ms
    377:	learn: 0.2152108	total: 401ms	remaining: 659ms
    378:	learn: 0.2149364	total: 401ms	remaining: 658ms
    379:	learn: 0.2147509	total: 402ms	remaining: 657ms
    380:	learn: 0.2145573	total: 404ms	remaining: 656ms
    381:	learn: 0.2141959	total: 405ms	remaining: 655ms
    382:	learn: 0.2140320	total: 406ms	remaining: 654ms
    383:	learn: 0.2138394	total: 407ms	remaining: 653ms
    384:	learn: 0.2137931	total: 408ms	remaining: 652ms
    385:	learn: 0.2137672	total: 409ms	remaining: 651ms
    386:	learn: 0.2135497	total: 410ms	remaining: 649ms
    387:	learn: 0.2134593	total: 411ms	remaining: 648ms
    388:	learn: 0.2134432	total: 412ms	remaining: 647ms
    389:	learn: 0.2132734	total: 413ms	remaining: 646ms
    390:	learn: 0.2131471	total: 414ms	remaining: 644ms
    391:	learn: 0.2129096	total: 415ms	remaining: 643ms
    392:	learn: 0.2127060	total: 416ms	remaining: 642ms
    393:	learn: 0.2125420	total: 417ms	remaining: 642ms
    394:	learn: 0.2121630	total: 418ms	remaining: 640ms
    395:	learn: 0.2119652	total: 419ms	remaining: 639ms
    396:	learn: 0.2118615	total: 420ms	remaining: 638ms
    397:	learn: 0.2117788	total: 421ms	remaining: 637ms
    398:	learn: 0.2116107	total: 422ms	remaining: 636ms
    399:	learn: 0.2114569	total: 423ms	remaining: 635ms
    400:	learn: 0.2113373	total: 424ms	remaining: 634ms
    401:	learn: 0.2112022	total: 425ms	remaining: 632ms
    402:	learn: 0.2109917	total: 426ms	remaining: 632ms
    403:	learn: 0.2108278	total: 427ms	remaining: 631ms
    404:	learn: 0.2106857	total: 428ms	remaining: 629ms
    405:	learn: 0.2106084	total: 429ms	remaining: 628ms
    406:	learn: 0.2104620	total: 430ms	remaining: 627ms
    407:	learn: 0.2103159	total: 431ms	remaining: 626ms
    408:	learn: 0.2098575	total: 432ms	remaining: 625ms
    409:	learn: 0.2096904	total: 433ms	remaining: 624ms
    410:	learn: 0.2094049	total: 434ms	remaining: 623ms
    411:	learn: 0.2092461	total: 435ms	remaining: 621ms
    412:	learn: 0.2089077	total: 437ms	remaining: 621ms
    413:	learn: 0.2087411	total: 438ms	remaining: 620ms
    414:	learn: 0.2085709	total: 439ms	remaining: 618ms
    415:	learn: 0.2085036	total: 440ms	remaining: 617ms
    416:	learn: 0.2083783	total: 442ms	remaining: 618ms
    417:	learn: 0.2082799	total: 443ms	remaining: 616ms
    418:	learn: 0.2082116	total: 445ms	remaining: 618ms
    419:	learn: 0.2080658	total: 448ms	remaining: 619ms
    420:	learn: 0.2079671	total: 449ms	remaining: 618ms
    421:	learn: 0.2078777	total: 450ms	remaining: 617ms
    422:	learn: 0.2077556	total: 461ms	remaining: 629ms
    423:	learn: 0.2077372	total: 465ms	remaining: 631ms
    424:	learn: 0.2077029	total: 470ms	remaining: 636ms
    425:	learn: 0.2076409	total: 472ms	remaining: 637ms
    426:	learn: 0.2074809	total: 475ms	remaining: 638ms
    427:	learn: 0.2073352	total: 480ms	remaining: 641ms
    428:	learn: 0.2072223	total: 481ms	remaining: 640ms
    429:	learn: 0.2071455	total: 482ms	remaining: 639ms
    430:	learn: 0.2070662	total: 483ms	remaining: 638ms
    431:	learn: 0.2068328	total: 484ms	remaining: 637ms
    432:	learn: 0.2065982	total: 485ms	remaining: 635ms
    433:	learn: 0.2065782	total: 486ms	remaining: 634ms
    434:	learn: 0.2064109	total: 487ms	remaining: 633ms
    435:	learn: 0.2062666	total: 488ms	remaining: 632ms
    436:	learn: 0.2061710	total: 489ms	remaining: 630ms
    437:	learn: 0.2060459	total: 490ms	remaining: 629ms
    438:	learn: 0.2059355	total: 491ms	remaining: 628ms
    439:	learn: 0.2059195	total: 492ms	remaining: 627ms
    440:	learn: 0.2059044	total: 493ms	remaining: 625ms
    441:	learn: 0.2058903	total: 494ms	remaining: 624ms
    442:	learn: 0.2058094	total: 495ms	remaining: 623ms
    443:	learn: 0.2055710	total: 496ms	remaining: 621ms
    444:	learn: 0.2054465	total: 497ms	remaining: 620ms
    445:	learn: 0.2052717	total: 498ms	remaining: 619ms
    446:	learn: 0.2051704	total: 499ms	remaining: 618ms
    447:	learn: 0.2050230	total: 501ms	remaining: 617ms
    448:	learn: 0.2048923	total: 502ms	remaining: 615ms
    449:	learn: 0.2044983	total: 503ms	remaining: 614ms
    450:	learn: 0.2043253	total: 504ms	remaining: 613ms
    451:	learn: 0.2040418	total: 505ms	remaining: 612ms
    452:	learn: 0.2038840	total: 506ms	remaining: 611ms
    453:	learn: 0.2037859	total: 508ms	remaining: 611ms
    454:	learn: 0.2036510	total: 510ms	remaining: 610ms
    455:	learn: 0.2035590	total: 511ms	remaining: 609ms
    456:	learn: 0.2034686	total: 512ms	remaining: 608ms
    457:	learn: 0.2033160	total: 513ms	remaining: 607ms
    458:	learn: 0.2033014	total: 514ms	remaining: 606ms
    459:	learn: 0.2029835	total: 515ms	remaining: 605ms
    460:	learn: 0.2028588	total: 516ms	remaining: 604ms
    461:	learn: 0.2027656	total: 518ms	remaining: 603ms
    462:	learn: 0.2026195	total: 519ms	remaining: 602ms
    463:	learn: 0.2024341	total: 520ms	remaining: 601ms
    464:	learn: 0.2023500	total: 521ms	remaining: 599ms
    465:	learn: 0.2022501	total: 522ms	remaining: 598ms
    466:	learn: 0.2022010	total: 523ms	remaining: 597ms
    467:	learn: 0.2020473	total: 524ms	remaining: 595ms
    468:	learn: 0.2017376	total: 525ms	remaining: 595ms
    469:	learn: 0.2015885	total: 526ms	remaining: 594ms
    470:	learn: 0.2013956	total: 527ms	remaining: 592ms
    471:	learn: 0.2013304	total: 529ms	remaining: 591ms
    472:	learn: 0.2012707	total: 530ms	remaining: 590ms
    473:	learn: 0.2010371	total: 531ms	remaining: 589ms
    474:	learn: 0.2008905	total: 532ms	remaining: 588ms
    475:	learn: 0.2007306	total: 533ms	remaining: 587ms
    476:	learn: 0.2005356	total: 534ms	remaining: 586ms
    477:	learn: 0.2001679	total: 535ms	remaining: 584ms
    478:	learn: 0.2000986	total: 536ms	remaining: 583ms
    479:	learn: 0.2000628	total: 537ms	remaining: 582ms
    480:	learn: 0.1999095	total: 538ms	remaining: 581ms
    481:	learn: 0.1997430	total: 539ms	remaining: 580ms
    482:	learn: 0.1996594	total: 540ms	remaining: 578ms
    483:	learn: 0.1995381	total: 541ms	remaining: 577ms
    484:	learn: 0.1993708	total: 542ms	remaining: 576ms
    485:	learn: 0.1992424	total: 543ms	remaining: 575ms
    486:	learn: 0.1991240	total: 544ms	remaining: 573ms
    487:	learn: 0.1989853	total: 545ms	remaining: 572ms
    488:	learn: 0.1988515	total: 546ms	remaining: 571ms
    489:	learn: 0.1986479	total: 547ms	remaining: 570ms
    490:	learn: 0.1985996	total: 548ms	remaining: 568ms
    491:	learn: 0.1984229	total: 549ms	remaining: 567ms
    492:	learn: 0.1983387	total: 550ms	remaining: 566ms
    493:	learn: 0.1981010	total: 551ms	remaining: 564ms
    494:	learn: 0.1980551	total: 552ms	remaining: 563ms
    495:	learn: 0.1979928	total: 553ms	remaining: 562ms
    496:	learn: 0.1977814	total: 554ms	remaining: 561ms
    497:	learn: 0.1975544	total: 555ms	remaining: 560ms
    498:	learn: 0.1974640	total: 556ms	remaining: 558ms
    499:	learn: 0.1973246	total: 557ms	remaining: 557ms
    500:	learn: 0.1972615	total: 557ms	remaining: 555ms
    501:	learn: 0.1971443	total: 558ms	remaining: 554ms
    502:	learn: 0.1971069	total: 559ms	remaining: 552ms
    503:	learn: 0.1969612	total: 560ms	remaining: 551ms
    504:	learn: 0.1968863	total: 560ms	remaining: 549ms
    505:	learn: 0.1966053	total: 561ms	remaining: 548ms
    506:	learn: 0.1965460	total: 562ms	remaining: 546ms
    507:	learn: 0.1964280	total: 562ms	remaining: 545ms
    508:	learn: 0.1963744	total: 563ms	remaining: 543ms
    509:	learn: 0.1962508	total: 564ms	remaining: 542ms
    510:	learn: 0.1961379	total: 564ms	remaining: 540ms
    511:	learn: 0.1960067	total: 565ms	remaining: 539ms
    512:	learn: 0.1959015	total: 566ms	remaining: 537ms
    513:	learn: 0.1958273	total: 567ms	remaining: 536ms
    514:	learn: 0.1956265	total: 568ms	remaining: 535ms
    515:	learn: 0.1954493	total: 569ms	remaining: 534ms
    516:	learn: 0.1953799	total: 570ms	remaining: 533ms
    517:	learn: 0.1952615	total: 571ms	remaining: 532ms
    518:	learn: 0.1951648	total: 572ms	remaining: 530ms
    519:	learn: 0.1951504	total: 573ms	remaining: 529ms
    520:	learn: 0.1950464	total: 574ms	remaining: 528ms
    521:	learn: 0.1948707	total: 575ms	remaining: 526ms
    522:	learn: 0.1947060	total: 576ms	remaining: 525ms
    523:	learn: 0.1946886	total: 577ms	remaining: 524ms
    524:	learn: 0.1943902	total: 578ms	remaining: 523ms
    525:	learn: 0.1942823	total: 579ms	remaining: 522ms
    526:	learn: 0.1941789	total: 580ms	remaining: 520ms
    527:	learn: 0.1940626	total: 581ms	remaining: 519ms
    528:	learn: 0.1939673	total: 582ms	remaining: 518ms
    529:	learn: 0.1938082	total: 583ms	remaining: 517ms
    530:	learn: 0.1936521	total: 584ms	remaining: 516ms
    531:	learn: 0.1936128	total: 585ms	remaining: 514ms
    532:	learn: 0.1935599	total: 586ms	remaining: 513ms
    533:	learn: 0.1934229	total: 587ms	remaining: 512ms
    534:	learn: 0.1933827	total: 588ms	remaining: 511ms
    535:	learn: 0.1933174	total: 589ms	remaining: 510ms
    536:	learn: 0.1932192	total: 590ms	remaining: 509ms
    537:	learn: 0.1931999	total: 592ms	remaining: 508ms
    538:	learn: 0.1931507	total: 593ms	remaining: 507ms
    539:	learn: 0.1930327	total: 594ms	remaining: 506ms
    540:	learn: 0.1926413	total: 595ms	remaining: 505ms
    541:	learn: 0.1925486	total: 596ms	remaining: 503ms
    542:	learn: 0.1924127	total: 597ms	remaining: 502ms
    543:	learn: 0.1922698	total: 598ms	remaining: 501ms
    544:	learn: 0.1921957	total: 599ms	remaining: 500ms
    545:	learn: 0.1920664	total: 600ms	remaining: 499ms
    546:	learn: 0.1920025	total: 601ms	remaining: 498ms
    547:	learn: 0.1917137	total: 602ms	remaining: 496ms
    548:	learn: 0.1915415	total: 603ms	remaining: 495ms
    549:	learn: 0.1914497	total: 604ms	remaining: 494ms
    550:	learn: 0.1914414	total: 605ms	remaining: 493ms
    551:	learn: 0.1914287	total: 606ms	remaining: 492ms
    552:	learn: 0.1913906	total: 607ms	remaining: 491ms
    553:	learn: 0.1913073	total: 608ms	remaining: 489ms
    554:	learn: 0.1912573	total: 609ms	remaining: 488ms
    555:	learn: 0.1910888	total: 610ms	remaining: 487ms
    556:	learn: 0.1909550	total: 611ms	remaining: 486ms
    557:	learn: 0.1908790	total: 612ms	remaining: 485ms
    558:	learn: 0.1907933	total: 613ms	remaining: 483ms
    559:	learn: 0.1906664	total: 613ms	remaining: 482ms
    560:	learn: 0.1905896	total: 615ms	remaining: 482ms
    561:	learn: 0.1905428	total: 617ms	remaining: 481ms
    562:	learn: 0.1903831	total: 618ms	remaining: 480ms
    563:	learn: 0.1902841	total: 619ms	remaining: 479ms
    564:	learn: 0.1902399	total: 620ms	remaining: 478ms
    565:	learn: 0.1901219	total: 621ms	remaining: 476ms
    566:	learn: 0.1900788	total: 622ms	remaining: 475ms
    567:	learn: 0.1900640	total: 623ms	remaining: 474ms
    568:	learn: 0.1893295	total: 625ms	remaining: 474ms
    569:	learn: 0.1891256	total: 626ms	remaining: 472ms
    570:	learn: 0.1890912	total: 628ms	remaining: 472ms
    571:	learn: 0.1890711	total: 631ms	remaining: 472ms
    572:	learn: 0.1887377	total: 634ms	remaining: 472ms
    573:	learn: 0.1885920	total: 635ms	remaining: 471ms
    574:	learn: 0.1884822	total: 636ms	remaining: 470ms
    575:	learn: 0.1884190	total: 637ms	remaining: 469ms
    576:	learn: 0.1883473	total: 638ms	remaining: 468ms
    577:	learn: 0.1882602	total: 639ms	remaining: 467ms
    578:	learn: 0.1882263	total: 640ms	remaining: 465ms
    579:	learn: 0.1881031	total: 641ms	remaining: 464ms
    580:	learn: 0.1879612	total: 642ms	remaining: 463ms
    581:	learn: 0.1879019	total: 643ms	remaining: 462ms
    582:	learn: 0.1878897	total: 644ms	remaining: 461ms
    583:	learn: 0.1878798	total: 645ms	remaining: 460ms
    584:	learn: 0.1877806	total: 646ms	remaining: 458ms
    585:	learn: 0.1876927	total: 647ms	remaining: 457ms
    586:	learn: 0.1875728	total: 648ms	remaining: 456ms
    587:	learn: 0.1874220	total: 649ms	remaining: 455ms
    588:	learn: 0.1872791	total: 650ms	remaining: 454ms
    589:	learn: 0.1872138	total: 652ms	remaining: 453ms
    590:	learn: 0.1871540	total: 653ms	remaining: 452ms
    591:	learn: 0.1870674	total: 654ms	remaining: 451ms
    592:	learn: 0.1869793	total: 655ms	remaining: 450ms
    593:	learn: 0.1868410	total: 657ms	remaining: 449ms
    594:	learn: 0.1865465	total: 658ms	remaining: 448ms
    595:	learn: 0.1864606	total: 659ms	remaining: 447ms
    596:	learn: 0.1863313	total: 660ms	remaining: 445ms
    597:	learn: 0.1862650	total: 661ms	remaining: 444ms
    598:	learn: 0.1861631	total: 662ms	remaining: 443ms
    599:	learn: 0.1859565	total: 663ms	remaining: 442ms
    600:	learn: 0.1858571	total: 664ms	remaining: 441ms
    601:	learn: 0.1858214	total: 665ms	remaining: 440ms
    602:	learn: 0.1857629	total: 667ms	remaining: 439ms
    603:	learn: 0.1856651	total: 668ms	remaining: 438ms
    604:	learn: 0.1854559	total: 669ms	remaining: 437ms
    605:	learn: 0.1853551	total: 670ms	remaining: 435ms
    606:	learn: 0.1852990	total: 671ms	remaining: 434ms
    607:	learn: 0.1851593	total: 672ms	remaining: 433ms
    608:	learn: 0.1851230	total: 673ms	remaining: 432ms
    609:	learn: 0.1850223	total: 674ms	remaining: 431ms
    610:	learn: 0.1849590	total: 675ms	remaining: 430ms
    611:	learn: 0.1849456	total: 676ms	remaining: 428ms
    612:	learn: 0.1849109	total: 677ms	remaining: 427ms
    613:	learn: 0.1841837	total: 678ms	remaining: 426ms
    614:	learn: 0.1841100	total: 679ms	remaining: 425ms
    615:	learn: 0.1841020	total: 680ms	remaining: 424ms
    616:	learn: 0.1840735	total: 681ms	remaining: 423ms
    617:	learn: 0.1840385	total: 682ms	remaining: 422ms
    618:	learn: 0.1840274	total: 683ms	remaining: 420ms
    619:	learn: 0.1839429	total: 684ms	remaining: 419ms
    620:	learn: 0.1839125	total: 685ms	remaining: 418ms
    621:	learn: 0.1839027	total: 686ms	remaining: 417ms
    622:	learn: 0.1838159	total: 687ms	remaining: 416ms
    623:	learn: 0.1836036	total: 688ms	remaining: 415ms
    624:	learn: 0.1835376	total: 689ms	remaining: 413ms
    625:	learn: 0.1835238	total: 690ms	remaining: 412ms
    626:	learn: 0.1834912	total: 691ms	remaining: 411ms
    627:	learn: 0.1832900	total: 692ms	remaining: 410ms
    628:	learn: 0.1832228	total: 693ms	remaining: 409ms
    629:	learn: 0.1831139	total: 694ms	remaining: 408ms
    630:	learn: 0.1830033	total: 695ms	remaining: 407ms
    631:	learn: 0.1829948	total: 697ms	remaining: 406ms
    632:	learn: 0.1829179	total: 698ms	remaining: 404ms
    633:	learn: 0.1828379	total: 699ms	remaining: 403ms
    634:	learn: 0.1827787	total: 700ms	remaining: 402ms
    635:	learn: 0.1827232	total: 701ms	remaining: 401ms
    636:	learn: 0.1825676	total: 702ms	remaining: 400ms
    637:	learn: 0.1825564	total: 703ms	remaining: 399ms
    638:	learn: 0.1824087	total: 704ms	remaining: 398ms
    639:	learn: 0.1820913	total: 705ms	remaining: 397ms
    640:	learn: 0.1819853	total: 706ms	remaining: 395ms
    641:	learn: 0.1818762	total: 707ms	remaining: 394ms
    642:	learn: 0.1818052	total: 708ms	remaining: 393ms
    643:	learn: 0.1817017	total: 709ms	remaining: 392ms
    644:	learn: 0.1816196	total: 710ms	remaining: 391ms
    645:	learn: 0.1814928	total: 711ms	remaining: 390ms
    646:	learn: 0.1814013	total: 712ms	remaining: 389ms
    647:	learn: 0.1813235	total: 717ms	remaining: 389ms
    648:	learn: 0.1809884	total: 717ms	remaining: 388ms
    649:	learn: 0.1806994	total: 718ms	remaining: 387ms
    650:	learn: 0.1805080	total: 719ms	remaining: 386ms
    651:	learn: 0.1804256	total: 720ms	remaining: 384ms
    652:	learn: 0.1803153	total: 721ms	remaining: 383ms
    653:	learn: 0.1803005	total: 722ms	remaining: 382ms
    654:	learn: 0.1801885	total: 722ms	remaining: 380ms
    655:	learn: 0.1801548	total: 723ms	remaining: 379ms
    656:	learn: 0.1800725	total: 724ms	remaining: 378ms
    657:	learn: 0.1798497	total: 724ms	remaining: 377ms
    658:	learn: 0.1797611	total: 725ms	remaining: 375ms
    659:	learn: 0.1796755	total: 726ms	remaining: 374ms
    660:	learn: 0.1796011	total: 726ms	remaining: 373ms
    661:	learn: 0.1794794	total: 727ms	remaining: 371ms
    662:	learn: 0.1794262	total: 728ms	remaining: 370ms
    663:	learn: 0.1793590	total: 728ms	remaining: 369ms
    664:	learn: 0.1793316	total: 729ms	remaining: 367ms
    665:	learn: 0.1793217	total: 730ms	remaining: 366ms
    666:	learn: 0.1791735	total: 730ms	remaining: 365ms
    667:	learn: 0.1789614	total: 731ms	remaining: 363ms
    668:	learn: 0.1788652	total: 732ms	remaining: 362ms
    669:	learn: 0.1787772	total: 732ms	remaining: 361ms
    670:	learn: 0.1786707	total: 733ms	remaining: 359ms
    671:	learn: 0.1786617	total: 734ms	remaining: 358ms
    672:	learn: 0.1785496	total: 734ms	remaining: 357ms
    673:	learn: 0.1784636	total: 735ms	remaining: 355ms
    674:	learn: 0.1783580	total: 736ms	remaining: 354ms
    675:	learn: 0.1782529	total: 736ms	remaining: 353ms
    676:	learn: 0.1782308	total: 737ms	remaining: 352ms
    677:	learn: 0.1781232	total: 738ms	remaining: 350ms
    678:	learn: 0.1781136	total: 738ms	remaining: 349ms
    679:	learn: 0.1780596	total: 739ms	remaining: 348ms
    680:	learn: 0.1779676	total: 740ms	remaining: 346ms
    681:	learn: 0.1778733	total: 740ms	remaining: 345ms
    682:	learn: 0.1776646	total: 741ms	remaining: 344ms
    683:	learn: 0.1776084	total: 742ms	remaining: 343ms
    684:	learn: 0.1775528	total: 742ms	remaining: 341ms
    685:	learn: 0.1775002	total: 743ms	remaining: 340ms
    686:	learn: 0.1773977	total: 744ms	remaining: 339ms
    687:	learn: 0.1773227	total: 744ms	remaining: 338ms
    688:	learn: 0.1772458	total: 745ms	remaining: 336ms
    689:	learn: 0.1771706	total: 746ms	remaining: 335ms
    690:	learn: 0.1770130	total: 747ms	remaining: 334ms
    691:	learn: 0.1767884	total: 748ms	remaining: 333ms
    692:	learn: 0.1767486	total: 749ms	remaining: 332ms
    693:	learn: 0.1767368	total: 750ms	remaining: 331ms
    694:	learn: 0.1766562	total: 751ms	remaining: 329ms
    695:	learn: 0.1765606	total: 752ms	remaining: 328ms
    696:	learn: 0.1763363	total: 753ms	remaining: 327ms
    697:	learn: 0.1762548	total: 754ms	remaining: 326ms
    698:	learn: 0.1761977	total: 755ms	remaining: 325ms
    699:	learn: 0.1761165	total: 756ms	remaining: 324ms
    700:	learn: 0.1760482	total: 756ms	remaining: 323ms
    701:	learn: 0.1759326	total: 758ms	remaining: 322ms
    702:	learn: 0.1758557	total: 759ms	remaining: 320ms
    703:	learn: 0.1757944	total: 760ms	remaining: 319ms
    704:	learn: 0.1756271	total: 760ms	remaining: 318ms
    705:	learn: 0.1752310	total: 761ms	remaining: 317ms
    706:	learn: 0.1751678	total: 762ms	remaining: 316ms
    707:	learn: 0.1751375	total: 763ms	remaining: 315ms
    708:	learn: 0.1750561	total: 764ms	remaining: 314ms
    709:	learn: 0.1750466	total: 765ms	remaining: 313ms
    710:	learn: 0.1750387	total: 766ms	remaining: 311ms
    711:	learn: 0.1748242	total: 767ms	remaining: 310ms
    712:	learn: 0.1747981	total: 768ms	remaining: 309ms
    713:	learn: 0.1746028	total: 769ms	remaining: 308ms
    714:	learn: 0.1744981	total: 770ms	remaining: 307ms
    715:	learn: 0.1744171	total: 771ms	remaining: 306ms
    716:	learn: 0.1743122	total: 772ms	remaining: 305ms
    717:	learn: 0.1741422	total: 773ms	remaining: 303ms
    718:	learn: 0.1739595	total: 774ms	remaining: 302ms
    719:	learn: 0.1734515	total: 775ms	remaining: 301ms
    720:	learn: 0.1733880	total: 776ms	remaining: 300ms
    721:	learn: 0.1732807	total: 777ms	remaining: 299ms
    722:	learn: 0.1732361	total: 778ms	remaining: 298ms
    723:	learn: 0.1730341	total: 779ms	remaining: 297ms
    724:	learn: 0.1729546	total: 779ms	remaining: 296ms
    725:	learn: 0.1728731	total: 780ms	remaining: 295ms
    726:	learn: 0.1726532	total: 781ms	remaining: 293ms
    727:	learn: 0.1725025	total: 782ms	remaining: 292ms
    728:	learn: 0.1723554	total: 783ms	remaining: 291ms
    729:	learn: 0.1722644	total: 784ms	remaining: 290ms
    730:	learn: 0.1722225	total: 785ms	remaining: 289ms
    731:	learn: 0.1721449	total: 786ms	remaining: 288ms
    732:	learn: 0.1720881	total: 787ms	remaining: 287ms
    733:	learn: 0.1719476	total: 788ms	remaining: 286ms
    734:	learn: 0.1718765	total: 789ms	remaining: 285ms
    735:	learn: 0.1718069	total: 790ms	remaining: 283ms
    736:	learn: 0.1717466	total: 792ms	remaining: 283ms
    737:	learn: 0.1717117	total: 793ms	remaining: 281ms
    738:	learn: 0.1715959	total: 794ms	remaining: 281ms
    739:	learn: 0.1715036	total: 796ms	remaining: 280ms
    740:	learn: 0.1714399	total: 798ms	remaining: 279ms
    741:	learn: 0.1713092	total: 801ms	remaining: 278ms
    742:	learn: 0.1711821	total: 803ms	remaining: 278ms
    743:	learn: 0.1711243	total: 804ms	remaining: 277ms
    744:	learn: 0.1710042	total: 805ms	remaining: 276ms
    745:	learn: 0.1709129	total: 806ms	remaining: 274ms
    746:	learn: 0.1708271	total: 808ms	remaining: 274ms
    747:	learn: 0.1706126	total: 810ms	remaining: 273ms
    748:	learn: 0.1704585	total: 811ms	remaining: 272ms
    749:	learn: 0.1703788	total: 811ms	remaining: 270ms
    750:	learn: 0.1702324	total: 815ms	remaining: 270ms
    751:	learn: 0.1701749	total: 818ms	remaining: 270ms
    752:	learn: 0.1700596	total: 819ms	remaining: 269ms
    753:	learn: 0.1698077	total: 820ms	remaining: 267ms
    754:	learn: 0.1697614	total: 820ms	remaining: 266ms
    755:	learn: 0.1695841	total: 822ms	remaining: 265ms
    756:	learn: 0.1695132	total: 823ms	remaining: 264ms
    757:	learn: 0.1694722	total: 824ms	remaining: 263ms
    758:	learn: 0.1693963	total: 825ms	remaining: 262ms
    759:	learn: 0.1693763	total: 826ms	remaining: 261ms
    760:	learn: 0.1692974	total: 827ms	remaining: 260ms
    761:	learn: 0.1692884	total: 828ms	remaining: 259ms
    762:	learn: 0.1692121	total: 829ms	remaining: 258ms
    763:	learn: 0.1691522	total: 830ms	remaining: 256ms
    764:	learn: 0.1691083	total: 831ms	remaining: 255ms
    765:	learn: 0.1690581	total: 832ms	remaining: 254ms
    766:	learn: 0.1690515	total: 833ms	remaining: 253ms
    767:	learn: 0.1689803	total: 834ms	remaining: 252ms
    768:	learn: 0.1689349	total: 835ms	remaining: 251ms
    769:	learn: 0.1688165	total: 836ms	remaining: 250ms
    770:	learn: 0.1687907	total: 837ms	remaining: 249ms
    771:	learn: 0.1686665	total: 838ms	remaining: 248ms
    772:	learn: 0.1684628	total: 839ms	remaining: 246ms
    773:	learn: 0.1683633	total: 840ms	remaining: 245ms
    774:	learn: 0.1683075	total: 841ms	remaining: 244ms
    775:	learn: 0.1682442	total: 842ms	remaining: 243ms
    776:	learn: 0.1681256	total: 843ms	remaining: 242ms
    777:	learn: 0.1679821	total: 844ms	remaining: 241ms
    778:	learn: 0.1679131	total: 847ms	remaining: 240ms
    779:	learn: 0.1678832	total: 848ms	remaining: 239ms
    780:	learn: 0.1677655	total: 848ms	remaining: 238ms
    781:	learn: 0.1677189	total: 849ms	remaining: 237ms
    782:	learn: 0.1676265	total: 851ms	remaining: 236ms
    783:	learn: 0.1676014	total: 852ms	remaining: 235ms
    784:	learn: 0.1675265	total: 853ms	remaining: 234ms
    785:	learn: 0.1674372	total: 854ms	remaining: 232ms
    786:	learn: 0.1673224	total: 855ms	remaining: 231ms
    787:	learn: 0.1672155	total: 856ms	remaining: 230ms
    788:	learn: 0.1671666	total: 857ms	remaining: 229ms
    789:	learn: 0.1669552	total: 858ms	remaining: 228ms
    790:	learn: 0.1668270	total: 859ms	remaining: 227ms
    791:	learn: 0.1667311	total: 860ms	remaining: 226ms
    792:	learn: 0.1666519	total: 861ms	remaining: 225ms
    793:	learn: 0.1666441	total: 862ms	remaining: 224ms
    794:	learn: 0.1666364	total: 863ms	remaining: 223ms
    795:	learn: 0.1665946	total: 864ms	remaining: 221ms
    796:	learn: 0.1664870	total: 865ms	remaining: 220ms
    797:	learn: 0.1664391	total: 866ms	remaining: 219ms
    798:	learn: 0.1663756	total: 867ms	remaining: 218ms
    799:	learn: 0.1662973	total: 868ms	remaining: 217ms
    800:	learn: 0.1662073	total: 869ms	remaining: 216ms
    801:	learn: 0.1661304	total: 871ms	remaining: 215ms
    802:	learn: 0.1660642	total: 872ms	remaining: 214ms
    803:	learn: 0.1659773	total: 873ms	remaining: 213ms
    804:	learn: 0.1659019	total: 874ms	remaining: 212ms
    805:	learn: 0.1658047	total: 875ms	remaining: 211ms
    806:	learn: 0.1657136	total: 876ms	remaining: 210ms
    807:	learn: 0.1656745	total: 877ms	remaining: 208ms
    808:	learn: 0.1655625	total: 878ms	remaining: 207ms
    809:	learn: 0.1654784	total: 879ms	remaining: 206ms
    810:	learn: 0.1653551	total: 881ms	remaining: 205ms
    811:	learn: 0.1653471	total: 882ms	remaining: 204ms
    812:	learn: 0.1652016	total: 883ms	remaining: 203ms
    813:	learn: 0.1651467	total: 884ms	remaining: 202ms
    814:	learn: 0.1650590	total: 885ms	remaining: 201ms
    815:	learn: 0.1649483	total: 886ms	remaining: 200ms
    816:	learn: 0.1647129	total: 887ms	remaining: 199ms
    817:	learn: 0.1646143	total: 888ms	remaining: 198ms
    818:	learn: 0.1644890	total: 889ms	remaining: 196ms
    819:	learn: 0.1643832	total: 890ms	remaining: 195ms
    820:	learn: 0.1641083	total: 891ms	remaining: 194ms
    821:	learn: 0.1640528	total: 892ms	remaining: 193ms
    822:	learn: 0.1639973	total: 893ms	remaining: 192ms
    823:	learn: 0.1639208	total: 894ms	remaining: 191ms
    824:	learn: 0.1638613	total: 896ms	remaining: 190ms
    825:	learn: 0.1638107	total: 897ms	remaining: 189ms
    826:	learn: 0.1637260	total: 898ms	remaining: 188ms
    827:	learn: 0.1636532	total: 899ms	remaining: 187ms
    828:	learn: 0.1635610	total: 900ms	remaining: 186ms
    829:	learn: 0.1634940	total: 900ms	remaining: 184ms
    830:	learn: 0.1633598	total: 901ms	remaining: 183ms
    831:	learn: 0.1633161	total: 902ms	remaining: 182ms
    832:	learn: 0.1632587	total: 904ms	remaining: 181ms
    833:	learn: 0.1631977	total: 905ms	remaining: 180ms
    834:	learn: 0.1631266	total: 908ms	remaining: 179ms
    835:	learn: 0.1630700	total: 909ms	remaining: 178ms
    836:	learn: 0.1630002	total: 910ms	remaining: 177ms
    837:	learn: 0.1629451	total: 911ms	remaining: 176ms
    838:	learn: 0.1628518	total: 912ms	remaining: 175ms
    839:	learn: 0.1627831	total: 913ms	remaining: 174ms
    840:	learn: 0.1627753	total: 914ms	remaining: 173ms
    841:	learn: 0.1627686	total: 915ms	remaining: 172ms
    842:	learn: 0.1627617	total: 916ms	remaining: 171ms
    843:	learn: 0.1626651	total: 917ms	remaining: 170ms
    844:	learn: 0.1625613	total: 918ms	remaining: 168ms
    845:	learn: 0.1625152	total: 919ms	remaining: 167ms
    846:	learn: 0.1624636	total: 921ms	remaining: 166ms
    847:	learn: 0.1623313	total: 922ms	remaining: 165ms
    848:	learn: 0.1622520	total: 923ms	remaining: 164ms
    849:	learn: 0.1620599	total: 924ms	remaining: 163ms
    850:	learn: 0.1619873	total: 924ms	remaining: 162ms
    851:	learn: 0.1618443	total: 925ms	remaining: 161ms
    852:	learn: 0.1617706	total: 926ms	remaining: 160ms
    853:	learn: 0.1616348	total: 927ms	remaining: 159ms
    854:	learn: 0.1615206	total: 928ms	remaining: 157ms
    855:	learn: 0.1614785	total: 929ms	remaining: 156ms
    856:	learn: 0.1614213	total: 930ms	remaining: 155ms
    857:	learn: 0.1613269	total: 931ms	remaining: 154ms
    858:	learn: 0.1612713	total: 932ms	remaining: 153ms
    859:	learn: 0.1612387	total: 933ms	remaining: 152ms
    860:	learn: 0.1610797	total: 934ms	remaining: 151ms
    861:	learn: 0.1610358	total: 935ms	remaining: 150ms
    862:	learn: 0.1609737	total: 936ms	remaining: 149ms
    863:	learn: 0.1608773	total: 937ms	remaining: 148ms
    864:	learn: 0.1608304	total: 938ms	remaining: 146ms
    865:	learn: 0.1607236	total: 939ms	remaining: 145ms
    866:	learn: 0.1606666	total: 941ms	remaining: 144ms
    867:	learn: 0.1606422	total: 942ms	remaining: 143ms
    868:	learn: 0.1606355	total: 943ms	remaining: 142ms
    869:	learn: 0.1606067	total: 944ms	remaining: 141ms
    870:	learn: 0.1605449	total: 945ms	remaining: 140ms
    871:	learn: 0.1604523	total: 946ms	remaining: 139ms
    872:	learn: 0.1603733	total: 947ms	remaining: 138ms
    873:	learn: 0.1602951	total: 948ms	remaining: 137ms
    874:	learn: 0.1602286	total: 949ms	remaining: 136ms
    875:	learn: 0.1601371	total: 950ms	remaining: 134ms
    876:	learn: 0.1599518	total: 952ms	remaining: 133ms
    877:	learn: 0.1599061	total: 953ms	remaining: 132ms
    878:	learn: 0.1598491	total: 954ms	remaining: 131ms
    879:	learn: 0.1597910	total: 955ms	remaining: 130ms
    880:	learn: 0.1596836	total: 956ms	remaining: 129ms
    881:	learn: 0.1595919	total: 957ms	remaining: 128ms
    882:	learn: 0.1594643	total: 958ms	remaining: 127ms
    883:	learn: 0.1594314	total: 959ms	remaining: 126ms
    884:	learn: 0.1593283	total: 960ms	remaining: 125ms
    885:	learn: 0.1592921	total: 961ms	remaining: 124ms
    886:	learn: 0.1592404	total: 962ms	remaining: 123ms
    887:	learn: 0.1590566	total: 963ms	remaining: 121ms
    888:	learn: 0.1588650	total: 964ms	remaining: 120ms
    889:	learn: 0.1588202	total: 965ms	remaining: 119ms
    890:	learn: 0.1587751	total: 967ms	remaining: 118ms
    891:	learn: 0.1586572	total: 968ms	remaining: 117ms
    892:	learn: 0.1585945	total: 969ms	remaining: 116ms
    893:	learn: 0.1584443	total: 970ms	remaining: 115ms
    894:	learn: 0.1584125	total: 971ms	remaining: 114ms
    895:	learn: 0.1582947	total: 972ms	remaining: 113ms
    896:	learn: 0.1582263	total: 974ms	remaining: 112ms
    897:	learn: 0.1581468	total: 975ms	remaining: 111ms
    898:	learn: 0.1580977	total: 977ms	remaining: 110ms
    899:	learn: 0.1580108	total: 980ms	remaining: 109ms
    900:	learn: 0.1579442	total: 985ms	remaining: 108ms
    901:	learn: 0.1576377	total: 986ms	remaining: 107ms
    902:	learn: 0.1575368	total: 989ms	remaining: 106ms
    903:	learn: 0.1575281	total: 991ms	remaining: 105ms
    904:	learn: 0.1575206	total: 992ms	remaining: 104ms
    905:	learn: 0.1574960	total: 993ms	remaining: 103ms
    906:	learn: 0.1574372	total: 994ms	remaining: 102ms
    907:	learn: 0.1573191	total: 995ms	remaining: 101ms
    908:	learn: 0.1571884	total: 996ms	remaining: 99.7ms
    909:	learn: 0.1571265	total: 997ms	remaining: 98.6ms
    910:	learn: 0.1571009	total: 998ms	remaining: 97.5ms
    911:	learn: 0.1570690	total: 999ms	remaining: 96.4ms
    912:	learn: 0.1569357	total: 1s	remaining: 95.3ms
    913:	learn: 0.1568281	total: 1s	remaining: 94.2ms
    914:	learn: 0.1567682	total: 1s	remaining: 93.1ms
    915:	learn: 0.1566366	total: 1s	remaining: 92ms
    916:	learn: 0.1566282	total: 1s	remaining: 90.9ms
    917:	learn: 0.1566008	total: 1s	remaining: 89.8ms
    918:	learn: 0.1565486	total: 1.01s	remaining: 88.7ms
    919:	learn: 0.1564961	total: 1.01s	remaining: 87.6ms
    920:	learn: 0.1563984	total: 1.01s	remaining: 86.5ms
    921:	learn: 0.1563241	total: 1.01s	remaining: 85.4ms
    922:	learn: 0.1562751	total: 1.01s	remaining: 84.3ms
    923:	learn: 0.1562398	total: 1.01s	remaining: 83.2ms
    924:	learn: 0.1561376	total: 1.01s	remaining: 82.1ms
    925:	learn: 0.1560204	total: 1.01s	remaining: 81ms
    926:	learn: 0.1559204	total: 1.01s	remaining: 79.9ms
    927:	learn: 0.1557904	total: 1.02s	remaining: 78.8ms
    928:	learn: 0.1557394	total: 1.02s	remaining: 77.7ms
    929:	learn: 0.1556866	total: 1.02s	remaining: 76.6ms
    930:	learn: 0.1556218	total: 1.02s	remaining: 75.5ms
    931:	learn: 0.1556029	total: 1.02s	remaining: 74.4ms
    932:	learn: 0.1555474	total: 1.02s	remaining: 73.3ms
    933:	learn: 0.1555007	total: 1.02s	remaining: 72.2ms
    934:	learn: 0.1554459	total: 1.02s	remaining: 71.1ms
    935:	learn: 0.1553747	total: 1.02s	remaining: 70ms
    936:	learn: 0.1551199	total: 1.02s	remaining: 68.9ms
    937:	learn: 0.1550480	total: 1.03s	remaining: 67.8ms
    938:	learn: 0.1549344	total: 1.03s	remaining: 66.7ms
    939:	learn: 0.1549289	total: 1.03s	remaining: 65.6ms
    940:	learn: 0.1549233	total: 1.03s	remaining: 64.5ms
    941:	learn: 0.1548619	total: 1.03s	remaining: 63.5ms
    942:	learn: 0.1547575	total: 1.03s	remaining: 62.4ms
    943:	learn: 0.1547027	total: 1.03s	remaining: 61.3ms
    944:	learn: 0.1546366	total: 1.03s	remaining: 60.2ms
    945:	learn: 0.1544659	total: 1.03s	remaining: 59.1ms
    946:	learn: 0.1543766	total: 1.03s	remaining: 58ms
    947:	learn: 0.1543203	total: 1.04s	remaining: 56.9ms
    948:	learn: 0.1542126	total: 1.04s	remaining: 55.8ms
    949:	learn: 0.1541169	total: 1.04s	remaining: 54.7ms
    950:	learn: 0.1540308	total: 1.04s	remaining: 53.6ms
    951:	learn: 0.1539359	total: 1.04s	remaining: 52.5ms
    952:	learn: 0.1538893	total: 1.04s	remaining: 51.4ms
    953:	learn: 0.1538825	total: 1.04s	remaining: 50.3ms
    954:	learn: 0.1538222	total: 1.04s	remaining: 49.2ms
    955:	learn: 0.1538158	total: 1.04s	remaining: 48.1ms
    956:	learn: 0.1536864	total: 1.04s	remaining: 47ms
    957:	learn: 0.1536542	total: 1.05s	remaining: 45.9ms
    958:	learn: 0.1535823	total: 1.05s	remaining: 44.8ms
    959:	learn: 0.1534891	total: 1.05s	remaining: 43.7ms
    960:	learn: 0.1534509	total: 1.05s	remaining: 42.6ms
    961:	learn: 0.1533540	total: 1.05s	remaining: 41.5ms
    962:	learn: 0.1532722	total: 1.05s	remaining: 40.4ms
    963:	learn: 0.1531790	total: 1.05s	remaining: 39.3ms
    964:	learn: 0.1530538	total: 1.05s	remaining: 38.2ms
    965:	learn: 0.1529324	total: 1.05s	remaining: 37.1ms
    966:	learn: 0.1527861	total: 1.05s	remaining: 36ms
    967:	learn: 0.1527583	total: 1.06s	remaining: 34.9ms
    968:	learn: 0.1526699	total: 1.06s	remaining: 33.8ms
    969:	learn: 0.1525238	total: 1.06s	remaining: 32.7ms
    970:	learn: 0.1525056	total: 1.06s	remaining: 31.6ms
    971:	learn: 0.1524043	total: 1.06s	remaining: 30.5ms
    972:	learn: 0.1523644	total: 1.06s	remaining: 29.4ms
    973:	learn: 0.1523024	total: 1.06s	remaining: 28.4ms
    974:	learn: 0.1522743	total: 1.06s	remaining: 27.3ms
    975:	learn: 0.1522095	total: 1.06s	remaining: 26.2ms
    976:	learn: 0.1521183	total: 1.06s	remaining: 25.1ms
    977:	learn: 0.1520472	total: 1.07s	remaining: 24ms
    978:	learn: 0.1518772	total: 1.07s	remaining: 22.9ms
    979:	learn: 0.1517791	total: 1.07s	remaining: 21.8ms
    980:	learn: 0.1517617	total: 1.07s	remaining: 20.7ms
    981:	learn: 0.1515831	total: 1.07s	remaining: 19.6ms
    982:	learn: 0.1515258	total: 1.07s	remaining: 18.5ms
    983:	learn: 0.1515113	total: 1.07s	remaining: 17.4ms
    984:	learn: 0.1513040	total: 1.07s	remaining: 16.3ms
    985:	learn: 0.1510875	total: 1.07s	remaining: 15.3ms
    986:	learn: 0.1509226	total: 1.07s	remaining: 14.2ms
    987:	learn: 0.1508306	total: 1.08s	remaining: 13.1ms
    988:	learn: 0.1507973	total: 1.08s	remaining: 12ms
    989:	learn: 0.1507310	total: 1.08s	remaining: 10.9ms
    990:	learn: 0.1506610	total: 1.08s	remaining: 9.8ms
    991:	learn: 0.1506324	total: 1.08s	remaining: 8.71ms
    992:	learn: 0.1504706	total: 1.08s	remaining: 7.62ms
    993:	learn: 0.1503786	total: 1.08s	remaining: 6.53ms
    994:	learn: 0.1502000	total: 1.08s	remaining: 5.44ms
    995:	learn: 0.1501279	total: 1.08s	remaining: 4.35ms
    996:	learn: 0.1500849	total: 1.08s	remaining: 3.27ms
    997:	learn: 0.1500717	total: 1.09s	remaining: 2.18ms
    998:	learn: 0.1500141	total: 1.09s	remaining: 1.09ms
    999:	learn: 0.1499760	total: 1.09s	remaining: 0us
    0:	learn: 0.6160455	total: 2.94ms	remaining: 731ms
    1:	learn: 0.5519616	total: 6.57ms	remaining: 815ms
    2:	learn: 0.5143418	total: 8.28ms	remaining: 681ms
    3:	learn: 0.4786104	total: 11.3ms	remaining: 696ms
    4:	learn: 0.4514147	total: 14.8ms	remaining: 725ms
    5:	learn: 0.4224463	total: 20.2ms	remaining: 820ms
    6:	learn: 0.4048565	total: 25.7ms	remaining: 893ms
    7:	learn: 0.3888092	total: 31ms	remaining: 937ms
    8:	learn: 0.3721394	total: 36.2ms	remaining: 968ms
    9:	learn: 0.3612886	total: 40.4ms	remaining: 971ms
    10:	learn: 0.3484773	total: 49.5ms	remaining: 1.08s
    11:	learn: 0.3414557	total: 59.6ms	remaining: 1.18s
    12:	learn: 0.3326290	total: 63.6ms	remaining: 1.16s
    13:	learn: 0.3229436	total: 66.4ms	remaining: 1.12s
    14:	learn: 0.3148072	total: 69.3ms	remaining: 1.08s
    15:	learn: 0.3059291	total: 72.4ms	remaining: 1.06s
    16:	learn: 0.2997127	total: 75.5ms	remaining: 1.03s
    17:	learn: 0.2940592	total: 78.7ms	remaining: 1.01s
    18:	learn: 0.2899830	total: 81.8ms	remaining: 995ms
    19:	learn: 0.2857724	total: 84.9ms	remaining: 976ms
    20:	learn: 0.2827697	total: 88.1ms	remaining: 961ms
    21:	learn: 0.2799993	total: 91.2ms	remaining: 946ms
    22:	learn: 0.2782467	total: 94.3ms	remaining: 930ms
    23:	learn: 0.2757556	total: 97.3ms	remaining: 917ms
    24:	learn: 0.2724064	total: 100ms	remaining: 904ms
    25:	learn: 0.2691188	total: 104ms	remaining: 892ms
    26:	learn: 0.2668827	total: 107ms	remaining: 881ms
    27:	learn: 0.2641400	total: 110ms	remaining: 870ms
    28:	learn: 0.2621337	total: 113ms	remaining: 861ms
    29:	learn: 0.2617246	total: 115ms	remaining: 842ms
    30:	learn: 0.2591477	total: 118ms	remaining: 833ms
    31:	learn: 0.2577591	total: 121ms	remaining: 825ms
    32:	learn: 0.2541788	total: 124ms	remaining: 818ms
    33:	learn: 0.2484644	total: 128ms	remaining: 812ms
    34:	learn: 0.2473039	total: 131ms	remaining: 805ms
    35:	learn: 0.2434838	total: 134ms	remaining: 798ms
    36:	learn: 0.2387344	total: 138ms	remaining: 792ms
    37:	learn: 0.2367292	total: 141ms	remaining: 785ms
    38:	learn: 0.2346037	total: 144ms	remaining: 778ms
    39:	learn: 0.2316968	total: 147ms	remaining: 772ms
    40:	learn: 0.2301499	total: 150ms	remaining: 765ms
    41:	learn: 0.2261027	total: 153ms	remaining: 759ms
    42:	learn: 0.2229674	total: 156ms	remaining: 752ms
    43:	learn: 0.2190738	total: 159ms	remaining: 746ms
    44:	learn: 0.2167461	total: 162ms	remaining: 740ms
    45:	learn: 0.2126653	total: 166ms	remaining: 734ms
    46:	learn: 0.2092375	total: 169ms	remaining: 729ms
    47:	learn: 0.2065552	total: 172ms	remaining: 724ms
    48:	learn: 0.2043525	total: 175ms	remaining: 718ms
    49:	learn: 0.2014857	total: 178ms	remaining: 714ms
    50:	learn: 0.1997260	total: 181ms	remaining: 708ms
    51:	learn: 0.1963163	total: 184ms	remaining: 702ms
    52:	learn: 0.1954478	total: 188ms	remaining: 697ms
    53:	learn: 0.1932843	total: 191ms	remaining: 692ms
    54:	learn: 0.1918851	total: 194ms	remaining: 688ms
    55:	learn: 0.1898602	total: 197ms	remaining: 683ms
    56:	learn: 0.1874605	total: 200ms	remaining: 679ms
    57:	learn: 0.1849330	total: 204ms	remaining: 674ms
    58:	learn: 0.1828089	total: 211ms	remaining: 684ms
    59:	learn: 0.1812267	total: 218ms	remaining: 691ms
    60:	learn: 0.1789936	total: 230ms	remaining: 712ms
    61:	learn: 0.1778275	total: 240ms	remaining: 728ms
    62:	learn: 0.1764701	total: 245ms	remaining: 727ms
    63:	learn: 0.1741801	total: 248ms	remaining: 720ms
    64:	learn: 0.1720621	total: 251ms	remaining: 714ms
    65:	learn: 0.1694361	total: 254ms	remaining: 708ms
    66:	learn: 0.1679567	total: 257ms	remaining: 703ms
    67:	learn: 0.1653600	total: 261ms	remaining: 698ms
    68:	learn: 0.1633139	total: 265ms	remaining: 695ms
    69:	learn: 0.1622743	total: 268ms	remaining: 690ms
    70:	learn: 0.1605303	total: 274ms	remaining: 690ms
    71:	learn: 0.1592067	total: 278ms	remaining: 688ms
    72:	learn: 0.1569746	total: 282ms	remaining: 683ms
    73:	learn: 0.1537184	total: 285ms	remaining: 678ms
    74:	learn: 0.1518205	total: 289ms	remaining: 674ms
    75:	learn: 0.1503016	total: 293ms	remaining: 670ms
    76:	learn: 0.1473012	total: 297ms	remaining: 667ms
    77:	learn: 0.1454821	total: 300ms	remaining: 662ms
    78:	learn: 0.1427923	total: 304ms	remaining: 658ms
    79:	learn: 0.1411677	total: 308ms	remaining: 654ms
    80:	learn: 0.1398584	total: 311ms	remaining: 649ms
    81:	learn: 0.1387914	total: 314ms	remaining: 644ms
    82:	learn: 0.1365188	total: 317ms	remaining: 639ms
    83:	learn: 0.1342978	total: 321ms	remaining: 634ms
    84:	learn: 0.1320962	total: 324ms	remaining: 628ms
    85:	learn: 0.1306233	total: 327ms	remaining: 623ms
    86:	learn: 0.1273179	total: 330ms	remaining: 618ms
    87:	learn: 0.1258176	total: 333ms	remaining: 614ms
    88:	learn: 0.1247669	total: 337ms	remaining: 609ms
    89:	learn: 0.1230144	total: 340ms	remaining: 605ms
    90:	learn: 0.1220496	total: 344ms	remaining: 600ms
    91:	learn: 0.1209220	total: 347ms	remaining: 596ms
    92:	learn: 0.1198800	total: 351ms	remaining: 592ms
    93:	learn: 0.1178129	total: 354ms	remaining: 588ms
    94:	learn: 0.1162588	total: 358ms	remaining: 584ms
    95:	learn: 0.1146910	total: 361ms	remaining: 580ms
    96:	learn: 0.1133186	total: 365ms	remaining: 576ms
    97:	learn: 0.1124848	total: 368ms	remaining: 571ms
    98:	learn: 0.1105563	total: 371ms	remaining: 567ms
    99:	learn: 0.1095440	total: 375ms	remaining: 563ms
    100:	learn: 0.1082472	total: 378ms	remaining: 558ms
    101:	learn: 0.1070136	total: 381ms	remaining: 554ms
    102:	learn: 0.1060070	total: 385ms	remaining: 549ms
    103:	learn: 0.1049137	total: 388ms	remaining: 545ms
    104:	learn: 0.1036947	total: 392ms	remaining: 541ms
    105:	learn: 0.1027930	total: 396ms	remaining: 537ms
    106:	learn: 0.1006944	total: 399ms	remaining: 533ms
    107:	learn: 0.0996606	total: 402ms	remaining: 529ms
    108:	learn: 0.0981078	total: 405ms	remaining: 525ms
    109:	learn: 0.0970607	total: 409ms	remaining: 521ms
    110:	learn: 0.0953970	total: 413ms	remaining: 517ms
    111:	learn: 0.0942722	total: 418ms	remaining: 515ms
    112:	learn: 0.0932732	total: 429ms	remaining: 521ms
    113:	learn: 0.0914226	total: 433ms	remaining: 516ms
    114:	learn: 0.0904913	total: 436ms	remaining: 512ms
    115:	learn: 0.0899780	total: 439ms	remaining: 508ms
    116:	learn: 0.0891337	total: 443ms	remaining: 503ms
    117:	learn: 0.0883723	total: 447ms	remaining: 500ms
    118:	learn: 0.0875859	total: 451ms	remaining: 496ms
    119:	learn: 0.0867198	total: 454ms	remaining: 492ms
    120:	learn: 0.0855739	total: 458ms	remaining: 488ms
    121:	learn: 0.0844130	total: 462ms	remaining: 484ms
    122:	learn: 0.0833391	total: 465ms	remaining: 480ms
    123:	learn: 0.0827234	total: 469ms	remaining: 476ms
    124:	learn: 0.0820061	total: 472ms	remaining: 472ms
    125:	learn: 0.0811995	total: 476ms	remaining: 468ms
    126:	learn: 0.0801220	total: 479ms	remaining: 464ms
    127:	learn: 0.0786852	total: 483ms	remaining: 460ms
    128:	learn: 0.0778452	total: 486ms	remaining: 456ms
    129:	learn: 0.0769445	total: 490ms	remaining: 452ms
    130:	learn: 0.0760460	total: 493ms	remaining: 448ms
    131:	learn: 0.0748215	total: 497ms	remaining: 444ms
    132:	learn: 0.0740532	total: 500ms	remaining: 440ms
    133:	learn: 0.0735114	total: 503ms	remaining: 436ms
    134:	learn: 0.0724012	total: 507ms	remaining: 432ms
    135:	learn: 0.0706882	total: 510ms	remaining: 428ms
    136:	learn: 0.0694624	total: 513ms	remaining: 424ms
    137:	learn: 0.0685101	total: 519ms	remaining: 421ms
    138:	learn: 0.0672514	total: 526ms	remaining: 420ms
    139:	learn: 0.0663569	total: 529ms	remaining: 416ms
    140:	learn: 0.0656746	total: 533ms	remaining: 412ms
    141:	learn: 0.0652120	total: 536ms	remaining: 408ms
    142:	learn: 0.0644016	total: 539ms	remaining: 404ms
    143:	learn: 0.0635113	total: 542ms	remaining: 399ms
    144:	learn: 0.0625180	total: 546ms	remaining: 395ms
    145:	learn: 0.0618363	total: 549ms	remaining: 391ms
    146:	learn: 0.0610980	total: 552ms	remaining: 387ms
    147:	learn: 0.0605313	total: 555ms	remaining: 383ms
    148:	learn: 0.0600882	total: 558ms	remaining: 379ms
    149:	learn: 0.0598501	total: 562ms	remaining: 374ms
    150:	learn: 0.0590868	total: 565ms	remaining: 370ms
    151:	learn: 0.0584390	total: 568ms	remaining: 366ms
    152:	learn: 0.0580562	total: 571ms	remaining: 362ms
    153:	learn: 0.0576279	total: 574ms	remaining: 358ms
    154:	learn: 0.0569018	total: 577ms	remaining: 354ms
    155:	learn: 0.0559620	total: 580ms	remaining: 350ms
    156:	learn: 0.0556380	total: 584ms	remaining: 346ms
    157:	learn: 0.0551240	total: 587ms	remaining: 342ms
    158:	learn: 0.0545686	total: 590ms	remaining: 338ms
    159:	learn: 0.0539492	total: 594ms	remaining: 334ms
    160:	learn: 0.0528380	total: 599ms	remaining: 331ms
    161:	learn: 0.0521404	total: 603ms	remaining: 327ms
    162:	learn: 0.0517537	total: 605ms	remaining: 323ms
    163:	learn: 0.0511920	total: 608ms	remaining: 319ms
    164:	learn: 0.0506250	total: 617ms	remaining: 318ms
    165:	learn: 0.0499600	total: 625ms	remaining: 316ms
    166:	learn: 0.0496919	total: 632ms	remaining: 314ms
    167:	learn: 0.0487964	total: 635ms	remaining: 310ms
    168:	learn: 0.0484219	total: 638ms	remaining: 306ms
    169:	learn: 0.0480387	total: 641ms	remaining: 302ms
    170:	learn: 0.0476759	total: 644ms	remaining: 298ms
    171:	learn: 0.0472035	total: 647ms	remaining: 294ms
    172:	learn: 0.0465537	total: 650ms	remaining: 289ms
    173:	learn: 0.0461253	total: 653ms	remaining: 285ms
    174:	learn: 0.0458853	total: 656ms	remaining: 281ms
    175:	learn: 0.0451743	total: 661ms	remaining: 278ms
    176:	learn: 0.0446650	total: 667ms	remaining: 275ms
    177:	learn: 0.0441611	total: 673ms	remaining: 272ms
    178:	learn: 0.0438187	total: 680ms	remaining: 270ms
    179:	learn: 0.0433399	total: 689ms	remaining: 268ms
    180:	learn: 0.0428256	total: 698ms	remaining: 266ms
    181:	learn: 0.0424263	total: 705ms	remaining: 263ms
    182:	learn: 0.0421742	total: 708ms	remaining: 259ms
    183:	learn: 0.0419699	total: 711ms	remaining: 255ms
    184:	learn: 0.0415429	total: 714ms	remaining: 251ms
    185:	learn: 0.0413772	total: 717ms	remaining: 247ms
    186:	learn: 0.0409633	total: 721ms	remaining: 243ms
    187:	learn: 0.0406013	total: 724ms	remaining: 239ms
    188:	learn: 0.0401256	total: 727ms	remaining: 235ms
    189:	learn: 0.0396237	total: 730ms	remaining: 231ms
    190:	learn: 0.0394027	total: 733ms	remaining: 226ms
    191:	learn: 0.0391123	total: 737ms	remaining: 223ms
    192:	learn: 0.0388674	total: 740ms	remaining: 218ms
    193:	learn: 0.0384977	total: 743ms	remaining: 214ms
    194:	learn: 0.0381490	total: 746ms	remaining: 210ms
    195:	learn: 0.0379459	total: 749ms	remaining: 206ms
    196:	learn: 0.0376317	total: 752ms	remaining: 202ms
    197:	learn: 0.0372271	total: 755ms	remaining: 198ms
    198:	learn: 0.0367769	total: 759ms	remaining: 194ms
    199:	learn: 0.0364726	total: 762ms	remaining: 190ms
    200:	learn: 0.0361750	total: 765ms	remaining: 186ms
    201:	learn: 0.0358361	total: 768ms	remaining: 183ms
    202:	learn: 0.0352830	total: 771ms	remaining: 179ms
    203:	learn: 0.0349770	total: 775ms	remaining: 175ms
    204:	learn: 0.0347718	total: 778ms	remaining: 171ms
    205:	learn: 0.0343735	total: 781ms	remaining: 167ms
    206:	learn: 0.0339981	total: 785ms	remaining: 163ms
    207:	learn: 0.0337922	total: 788ms	remaining: 159ms
    208:	learn: 0.0334289	total: 792ms	remaining: 155ms
    209:	learn: 0.0326743	total: 795ms	remaining: 151ms
    210:	learn: 0.0322735	total: 803ms	remaining: 148ms
    211:	learn: 0.0319980	total: 810ms	remaining: 145ms
    212:	learn: 0.0316606	total: 814ms	remaining: 141ms
    213:	learn: 0.0313589	total: 817ms	remaining: 137ms
    214:	learn: 0.0311399	total: 820ms	remaining: 134ms
    215:	learn: 0.0308792	total: 823ms	remaining: 130ms
    216:	learn: 0.0305704	total: 827ms	remaining: 126ms
    217:	learn: 0.0302707	total: 830ms	remaining: 122ms
    218:	learn: 0.0298549	total: 833ms	remaining: 118ms
    219:	learn: 0.0296445	total: 836ms	remaining: 114ms
    220:	learn: 0.0292918	total: 840ms	remaining: 110ms
    221:	learn: 0.0290420	total: 843ms	remaining: 106ms
    222:	learn: 0.0288738	total: 846ms	remaining: 102ms
    223:	learn: 0.0286785	total: 850ms	remaining: 98.7ms
    224:	learn: 0.0284590	total: 853ms	remaining: 94.8ms
    225:	learn: 0.0282321	total: 857ms	remaining: 91ms
    226:	learn: 0.0280481	total: 860ms	remaining: 87.1ms
    227:	learn: 0.0276536	total: 863ms	remaining: 83.3ms
    228:	learn: 0.0274960	total: 867ms	remaining: 79.5ms
    229:	learn: 0.0272862	total: 870ms	remaining: 75.6ms
    230:	learn: 0.0270218	total: 873ms	remaining: 71.8ms
    231:	learn: 0.0268161	total: 877ms	remaining: 68ms
    232:	learn: 0.0266580	total: 880ms	remaining: 64.2ms
    233:	learn: 0.0264622	total: 883ms	remaining: 60.4ms
    234:	learn: 0.0263247	total: 887ms	remaining: 56.6ms
    235:	learn: 0.0259974	total: 890ms	remaining: 52.8ms
    236:	learn: 0.0258766	total: 893ms	remaining: 49ms
    237:	learn: 0.0256992	total: 896ms	remaining: 45.2ms
    238:	learn: 0.0255404	total: 900ms	remaining: 41.4ms
    239:	learn: 0.0253526	total: 903ms	remaining: 37.6ms
    240:	learn: 0.0251425	total: 906ms	remaining: 33.8ms
    241:	learn: 0.0249644	total: 909ms	remaining: 30.1ms
    242:	learn: 0.0246985	total: 912ms	remaining: 26.3ms
    243:	learn: 0.0245207	total: 915ms	remaining: 22.5ms
    244:	learn: 0.0243999	total: 918ms	remaining: 18.7ms
    245:	learn: 0.0241764	total: 922ms	remaining: 15ms
    246:	learn: 0.0239338	total: 925ms	remaining: 11.2ms
    247:	learn: 0.0236816	total: 928ms	remaining: 7.48ms
    248:	learn: 0.0235957	total: 931ms	remaining: 3.74ms
    249:	learn: 0.0234841	total: 934ms	remaining: 0us
    0:	learn: 0.6194013	total: 2.96ms	remaining: 737ms
    1:	learn: 0.5497488	total: 6.14ms	remaining: 762ms
    2:	learn: 0.5170714	total: 7.78ms	remaining: 641ms
    3:	learn: 0.4771949	total: 10.8ms	remaining: 661ms
    4:	learn: 0.4455738	total: 13.8ms	remaining: 678ms
    5:	learn: 0.4167931	total: 17.8ms	remaining: 726ms
    6:	learn: 0.3985933	total: 22.3ms	remaining: 775ms
    7:	learn: 0.3836735	total: 28.7ms	remaining: 867ms
    8:	learn: 0.3690926	total: 32.9ms	remaining: 880ms
    9:	learn: 0.3594903	total: 37.4ms	remaining: 898ms
    10:	learn: 0.3474878	total: 42.6ms	remaining: 926ms
    11:	learn: 0.3414331	total: 46.5ms	remaining: 923ms
    12:	learn: 0.3325339	total: 49.8ms	remaining: 907ms
    13:	learn: 0.3240738	total: 52.9ms	remaining: 892ms
    14:	learn: 0.3176142	total: 56ms	remaining: 877ms
    15:	learn: 0.3124348	total: 59.2ms	remaining: 866ms
    16:	learn: 0.3055652	total: 62.4ms	remaining: 856ms
    17:	learn: 0.2997726	total: 65.5ms	remaining: 845ms
    18:	learn: 0.2964150	total: 68.7ms	remaining: 835ms
    19:	learn: 0.2912174	total: 71.9ms	remaining: 827ms
    20:	learn: 0.2869091	total: 75.1ms	remaining: 819ms
    21:	learn: 0.2828816	total: 78.1ms	remaining: 809ms
    22:	learn: 0.2794748	total: 81.3ms	remaining: 803ms
    23:	learn: 0.2755365	total: 84.4ms	remaining: 795ms
    24:	learn: 0.2741319	total: 87.7ms	remaining: 789ms
    25:	learn: 0.2697425	total: 90.7ms	remaining: 782ms
    26:	learn: 0.2672185	total: 93.8ms	remaining: 774ms
    27:	learn: 0.2620129	total: 97ms	remaining: 769ms
    28:	learn: 0.2604862	total: 100ms	remaining: 762ms
    29:	learn: 0.2588973	total: 103ms	remaining: 758ms
    30:	learn: 0.2548119	total: 107ms	remaining: 753ms
    31:	learn: 0.2521281	total: 110ms	remaining: 748ms
    32:	learn: 0.2481886	total: 113ms	remaining: 745ms
    33:	learn: 0.2441827	total: 116ms	remaining: 740ms
    34:	learn: 0.2424403	total: 120ms	remaining: 734ms
    35:	learn: 0.2372806	total: 123ms	remaining: 729ms
    36:	learn: 0.2339956	total: 126ms	remaining: 724ms
    37:	learn: 0.2328067	total: 129ms	remaining: 719ms
    38:	learn: 0.2289317	total: 132ms	remaining: 715ms
    39:	learn: 0.2265157	total: 135ms	remaining: 711ms
    40:	learn: 0.2232050	total: 138ms	remaining: 706ms
    41:	learn: 0.2208147	total: 142ms	remaining: 701ms
    42:	learn: 0.2175803	total: 145ms	remaining: 697ms
    43:	learn: 0.2167345	total: 148ms	remaining: 692ms
    44:	learn: 0.2135405	total: 151ms	remaining: 689ms
    45:	learn: 0.2118203	total: 154ms	remaining: 685ms
    46:	learn: 0.2101534	total: 158ms	remaining: 680ms
    47:	learn: 0.2077852	total: 161ms	remaining: 676ms
    48:	learn: 0.2066462	total: 164ms	remaining: 672ms
    49:	learn: 0.2045610	total: 170ms	remaining: 679ms
    50:	learn: 0.2026668	total: 174ms	remaining: 678ms
    51:	learn: 0.2001569	total: 181ms	remaining: 688ms
    52:	learn: 0.1984234	total: 187ms	remaining: 696ms
    53:	learn: 0.1965138	total: 191ms	remaining: 694ms
    54:	learn: 0.1950770	total: 194ms	remaining: 688ms
    55:	learn: 0.1932339	total: 197ms	remaining: 683ms
    56:	learn: 0.1914829	total: 200ms	remaining: 677ms
    57:	learn: 0.1880085	total: 206ms	remaining: 683ms
    58:	learn: 0.1852862	total: 212ms	remaining: 687ms
    59:	learn: 0.1839013	total: 219ms	remaining: 695ms
    60:	learn: 0.1814132	total: 224ms	remaining: 693ms
    61:	learn: 0.1781206	total: 227ms	remaining: 688ms
    62:	learn: 0.1769318	total: 230ms	remaining: 683ms
    63:	learn: 0.1758824	total: 233ms	remaining: 678ms
    64:	learn: 0.1742810	total: 236ms	remaining: 673ms
    65:	learn: 0.1734868	total: 239ms	remaining: 668ms
    66:	learn: 0.1715109	total: 243ms	remaining: 663ms
    67:	learn: 0.1692464	total: 246ms	remaining: 658ms
    68:	learn: 0.1680468	total: 249ms	remaining: 653ms
    69:	learn: 0.1660024	total: 252ms	remaining: 648ms
    70:	learn: 0.1641725	total: 255ms	remaining: 643ms
    71:	learn: 0.1615597	total: 258ms	remaining: 638ms
    72:	learn: 0.1599843	total: 261ms	remaining: 634ms
    73:	learn: 0.1571231	total: 264ms	remaining: 629ms
    74:	learn: 0.1551016	total: 268ms	remaining: 624ms
    75:	learn: 0.1525881	total: 271ms	remaining: 620ms
    76:	learn: 0.1497691	total: 274ms	remaining: 616ms
    77:	learn: 0.1490311	total: 277ms	remaining: 611ms
    78:	learn: 0.1471599	total: 280ms	remaining: 606ms
    79:	learn: 0.1449745	total: 283ms	remaining: 602ms
    80:	learn: 0.1409991	total: 287ms	remaining: 598ms
    81:	learn: 0.1395533	total: 289ms	remaining: 593ms
    82:	learn: 0.1384212	total: 293ms	remaining: 589ms
    83:	learn: 0.1364312	total: 296ms	remaining: 584ms
    84:	learn: 0.1342288	total: 299ms	remaining: 580ms
    85:	learn: 0.1334791	total: 302ms	remaining: 576ms
    86:	learn: 0.1318690	total: 305ms	remaining: 572ms
    87:	learn: 0.1304817	total: 308ms	remaining: 567ms
    88:	learn: 0.1284834	total: 311ms	remaining: 562ms
    89:	learn: 0.1273819	total: 314ms	remaining: 557ms
    90:	learn: 0.1260613	total: 317ms	remaining: 554ms
    91:	learn: 0.1238084	total: 320ms	remaining: 550ms
    92:	learn: 0.1225944	total: 323ms	remaining: 546ms
    93:	learn: 0.1207053	total: 326ms	remaining: 541ms
    94:	learn: 0.1193048	total: 329ms	remaining: 537ms
    95:	learn: 0.1181205	total: 333ms	remaining: 533ms
    96:	learn: 0.1173229	total: 336ms	remaining: 530ms
    97:	learn: 0.1147709	total: 339ms	remaining: 526ms
    98:	learn: 0.1125516	total: 342ms	remaining: 522ms
    99:	learn: 0.1111605	total: 345ms	remaining: 518ms
    100:	learn: 0.1098454	total: 348ms	remaining: 514ms
    101:	learn: 0.1086862	total: 352ms	remaining: 510ms
    102:	learn: 0.1069537	total: 355ms	remaining: 507ms
    103:	learn: 0.1049497	total: 358ms	remaining: 503ms
    104:	learn: 0.1033309	total: 361ms	remaining: 499ms
    105:	learn: 0.1026627	total: 364ms	remaining: 495ms
    106:	learn: 0.1015414	total: 367ms	remaining: 491ms
    107:	learn: 0.1003169	total: 370ms	remaining: 487ms
    108:	learn: 0.0988514	total: 373ms	remaining: 483ms
    109:	learn: 0.0976505	total: 376ms	remaining: 479ms
    110:	learn: 0.0966543	total: 379ms	remaining: 475ms
    111:	learn: 0.0960129	total: 382ms	remaining: 471ms
    112:	learn: 0.0948064	total: 386ms	remaining: 467ms
    113:	learn: 0.0937284	total: 389ms	remaining: 463ms
    114:	learn: 0.0924295	total: 392ms	remaining: 460ms
    115:	learn: 0.0909618	total: 395ms	remaining: 456ms
    116:	learn: 0.0901601	total: 399ms	remaining: 453ms
    117:	learn: 0.0893360	total: 403ms	remaining: 451ms
    118:	learn: 0.0887384	total: 411ms	remaining: 452ms
    119:	learn: 0.0877912	total: 423ms	remaining: 458ms
    120:	learn: 0.0867619	total: 428ms	remaining: 457ms
    121:	learn: 0.0858349	total: 433ms	remaining: 454ms
    122:	learn: 0.0847869	total: 436ms	remaining: 450ms
    123:	learn: 0.0843582	total: 439ms	remaining: 446ms
    124:	learn: 0.0833061	total: 442ms	remaining: 442ms
    125:	learn: 0.0824791	total: 446ms	remaining: 438ms
    126:	learn: 0.0812409	total: 449ms	remaining: 434ms
    127:	learn: 0.0801716	total: 455ms	remaining: 434ms
    128:	learn: 0.0794959	total: 458ms	remaining: 430ms
    129:	learn: 0.0791623	total: 461ms	remaining: 426ms
    130:	learn: 0.0780898	total: 464ms	remaining: 422ms
    131:	learn: 0.0771682	total: 467ms	remaining: 418ms
    132:	learn: 0.0762380	total: 470ms	remaining: 414ms
    133:	learn: 0.0751469	total: 474ms	remaining: 410ms
    134:	learn: 0.0744059	total: 477ms	remaining: 406ms
    135:	learn: 0.0735313	total: 480ms	remaining: 402ms
    136:	learn: 0.0725527	total: 483ms	remaining: 398ms
    137:	learn: 0.0717822	total: 486ms	remaining: 395ms
    138:	learn: 0.0710404	total: 489ms	remaining: 391ms
    139:	learn: 0.0705722	total: 492ms	remaining: 387ms
    140:	learn: 0.0698836	total: 496ms	remaining: 383ms
    141:	learn: 0.0692528	total: 499ms	remaining: 379ms
    142:	learn: 0.0686683	total: 502ms	remaining: 376ms
    143:	learn: 0.0675112	total: 505ms	remaining: 372ms
    144:	learn: 0.0670528	total: 509ms	remaining: 368ms
    145:	learn: 0.0663535	total: 512ms	remaining: 365ms
    146:	learn: 0.0658307	total: 515ms	remaining: 361ms
    147:	learn: 0.0647984	total: 519ms	remaining: 357ms
    148:	learn: 0.0639983	total: 522ms	remaining: 354ms
    149:	learn: 0.0635247	total: 525ms	remaining: 350ms
    150:	learn: 0.0625917	total: 529ms	remaining: 347ms
    151:	learn: 0.0618432	total: 532ms	remaining: 343ms
    152:	learn: 0.0615489	total: 535ms	remaining: 339ms
    153:	learn: 0.0607827	total: 538ms	remaining: 336ms
    154:	learn: 0.0600090	total: 542ms	remaining: 332ms
    155:	learn: 0.0593876	total: 545ms	remaining: 328ms
    156:	learn: 0.0589847	total: 549ms	remaining: 325ms
    157:	learn: 0.0583512	total: 552ms	remaining: 322ms
    158:	learn: 0.0580404	total: 555ms	remaining: 318ms
    159:	learn: 0.0575998	total: 559ms	remaining: 314ms
    160:	learn: 0.0572129	total: 562ms	remaining: 310ms
    161:	learn: 0.0569413	total: 563ms	remaining: 306ms
    162:	learn: 0.0566487	total: 568ms	remaining: 303ms
    163:	learn: 0.0561695	total: 571ms	remaining: 299ms
    164:	learn: 0.0556256	total: 574ms	remaining: 296ms
    165:	learn: 0.0549740	total: 577ms	remaining: 292ms
    166:	learn: 0.0540508	total: 580ms	remaining: 288ms
    167:	learn: 0.0536415	total: 583ms	remaining: 284ms
    168:	learn: 0.0533792	total: 586ms	remaining: 281ms
    169:	learn: 0.0531208	total: 589ms	remaining: 277ms
    170:	learn: 0.0528030	total: 593ms	remaining: 274ms
    171:	learn: 0.0523889	total: 598ms	remaining: 271ms
    172:	learn: 0.0520567	total: 601ms	remaining: 268ms
    173:	learn: 0.0514050	total: 604ms	remaining: 264ms
    174:	learn: 0.0511382	total: 607ms	remaining: 260ms
    175:	learn: 0.0504940	total: 610ms	remaining: 256ms
    176:	learn: 0.0501155	total: 613ms	remaining: 253ms
    177:	learn: 0.0496385	total: 616ms	remaining: 249ms
    178:	learn: 0.0493840	total: 619ms	remaining: 245ms
    179:	learn: 0.0488764	total: 622ms	remaining: 242ms
    180:	learn: 0.0486082	total: 625ms	remaining: 238ms
    181:	learn: 0.0480460	total: 628ms	remaining: 235ms
    182:	learn: 0.0477154	total: 632ms	remaining: 231ms
    183:	learn: 0.0474448	total: 635ms	remaining: 228ms
    184:	learn: 0.0470126	total: 638ms	remaining: 224ms
    185:	learn: 0.0467009	total: 641ms	remaining: 221ms
    186:	learn: 0.0461774	total: 644ms	remaining: 217ms
    187:	learn: 0.0458508	total: 647ms	remaining: 213ms
    188:	learn: 0.0454503	total: 650ms	remaining: 210ms
    189:	learn: 0.0450114	total: 654ms	remaining: 206ms
    190:	learn: 0.0446893	total: 657ms	remaining: 203ms
    191:	learn: 0.0444028	total: 664ms	remaining: 201ms
    192:	learn: 0.0441720	total: 666ms	remaining: 197ms
    193:	learn: 0.0438420	total: 669ms	remaining: 193ms
    194:	learn: 0.0435752	total: 672ms	remaining: 189ms
    195:	learn: 0.0431706	total: 675ms	remaining: 186ms
    196:	learn: 0.0426811	total: 678ms	remaining: 182ms
    197:	learn: 0.0423201	total: 681ms	remaining: 179ms
    198:	learn: 0.0417604	total: 684ms	remaining: 175ms
    199:	learn: 0.0413379	total: 687ms	remaining: 172ms
    200:	learn: 0.0411294	total: 690ms	remaining: 168ms
    201:	learn: 0.0408781	total: 693ms	remaining: 165ms
    202:	learn: 0.0404938	total: 697ms	remaining: 161ms
    203:	learn: 0.0398802	total: 700ms	remaining: 158ms
    204:	learn: 0.0395538	total: 703ms	remaining: 154ms
    205:	learn: 0.0393110	total: 706ms	remaining: 151ms
    206:	learn: 0.0387888	total: 709ms	remaining: 147ms
    207:	learn: 0.0384518	total: 712ms	remaining: 144ms
    208:	learn: 0.0380242	total: 716ms	remaining: 140ms
    209:	learn: 0.0377754	total: 719ms	remaining: 137ms
    210:	learn: 0.0373118	total: 722ms	remaining: 134ms
    211:	learn: 0.0369299	total: 725ms	remaining: 130ms
    212:	learn: 0.0367495	total: 729ms	remaining: 127ms
    213:	learn: 0.0364540	total: 732ms	remaining: 123ms
    214:	learn: 0.0360568	total: 736ms	remaining: 120ms
    215:	learn: 0.0357284	total: 739ms	remaining: 116ms
    216:	learn: 0.0354142	total: 742ms	remaining: 113ms
    217:	learn: 0.0351350	total: 745ms	remaining: 109ms
    218:	learn: 0.0348655	total: 749ms	remaining: 106ms
    219:	learn: 0.0347200	total: 752ms	remaining: 103ms
    220:	learn: 0.0344248	total: 755ms	remaining: 99.1ms
    221:	learn: 0.0342512	total: 758ms	remaining: 95.6ms
    222:	learn: 0.0337933	total: 762ms	remaining: 92.2ms
    223:	learn: 0.0335825	total: 765ms	remaining: 88.8ms
    224:	learn: 0.0332816	total: 768ms	remaining: 85.3ms
    225:	learn: 0.0329744	total: 771ms	remaining: 81.9ms
    226:	learn: 0.0327361	total: 776ms	remaining: 78.6ms
    227:	learn: 0.0326167	total: 782ms	remaining: 75.4ms
    228:	learn: 0.0324313	total: 785ms	remaining: 72ms
    229:	learn: 0.0321921	total: 789ms	remaining: 68.6ms
    230:	learn: 0.0317766	total: 792ms	remaining: 65.2ms
    231:	learn: 0.0315009	total: 796ms	remaining: 61.7ms
    232:	learn: 0.0311857	total: 799ms	remaining: 58.3ms
    233:	learn: 0.0307415	total: 802ms	remaining: 54.8ms
    234:	learn: 0.0305502	total: 805ms	remaining: 51.4ms
    235:	learn: 0.0303656	total: 807ms	remaining: 47.9ms
    236:	learn: 0.0302507	total: 810ms	remaining: 44.4ms
    237:	learn: 0.0300751	total: 813ms	remaining: 41ms
    238:	learn: 0.0297452	total: 815ms	remaining: 37.5ms
    239:	learn: 0.0295569	total: 818ms	remaining: 34.1ms
    240:	learn: 0.0293273	total: 821ms	remaining: 30.7ms
    241:	learn: 0.0290785	total: 824ms	remaining: 27.2ms
    242:	learn: 0.0288692	total: 828ms	remaining: 23.8ms
    243:	learn: 0.0285804	total: 831ms	remaining: 20.4ms
    244:	learn: 0.0284158	total: 834ms	remaining: 17ms
    245:	learn: 0.0283152	total: 838ms	remaining: 13.6ms
    246:	learn: 0.0282023	total: 841ms	remaining: 10.2ms
    247:	learn: 0.0280157	total: 845ms	remaining: 6.81ms
    248:	learn: 0.0277558	total: 848ms	remaining: 3.4ms
    249:	learn: 0.0275146	total: 851ms	remaining: 0us
    0:	learn: 0.6092706	total: 2.91ms	remaining: 725ms
    1:	learn: 0.5483092	total: 6.1ms	remaining: 756ms
    2:	learn: 0.5118250	total: 7.59ms	remaining: 625ms
    3:	learn: 0.4734270	total: 10.5ms	remaining: 649ms
    4:	learn: 0.4432262	total: 13.5ms	remaining: 662ms
    5:	learn: 0.4161619	total: 16.6ms	remaining: 676ms
    6:	learn: 0.3962615	total: 19.8ms	remaining: 688ms
    7:	learn: 0.3801844	total: 22.9ms	remaining: 692ms
    8:	learn: 0.3650893	total: 25.9ms	remaining: 695ms
    9:	learn: 0.3475730	total: 29.2ms	remaining: 700ms
    10:	learn: 0.3359669	total: 32.3ms	remaining: 701ms
    11:	learn: 0.3290911	total: 35.3ms	remaining: 701ms
    12:	learn: 0.3215560	total: 38.4ms	remaining: 700ms
    13:	learn: 0.3147133	total: 41.7ms	remaining: 703ms
    14:	learn: 0.3064492	total: 44.8ms	remaining: 702ms
    15:	learn: 0.3010187	total: 47.9ms	remaining: 700ms
    16:	learn: 0.2925860	total: 51.1ms	remaining: 700ms
    17:	learn: 0.2856652	total: 54.5ms	remaining: 702ms
    18:	learn: 0.2833366	total: 57.7ms	remaining: 702ms
    19:	learn: 0.2802679	total: 60.9ms	remaining: 700ms
    20:	learn: 0.2765276	total: 65.4ms	remaining: 713ms
    21:	learn: 0.2736542	total: 73ms	remaining: 757ms
    22:	learn: 0.2696157	total: 80.9ms	remaining: 799ms
    23:	learn: 0.2661379	total: 83.7ms	remaining: 788ms
    24:	learn: 0.2631349	total: 86.4ms	remaining: 778ms
    25:	learn: 0.2590635	total: 89.2ms	remaining: 768ms
    26:	learn: 0.2543081	total: 92.2ms	remaining: 762ms
    27:	learn: 0.2517621	total: 95.5ms	remaining: 758ms
    28:	learn: 0.2493425	total: 98.6ms	remaining: 752ms
    29:	learn: 0.2472433	total: 102ms	remaining: 749ms
    30:	learn: 0.2451757	total: 105ms	remaining: 742ms
    31:	learn: 0.2421092	total: 109ms	remaining: 740ms
    32:	learn: 0.2388969	total: 112ms	remaining: 740ms
    33:	learn: 0.2357063	total: 116ms	remaining: 736ms
    34:	learn: 0.2339942	total: 119ms	remaining: 731ms
    35:	learn: 0.2308683	total: 122ms	remaining: 726ms
    36:	learn: 0.2271874	total: 125ms	remaining: 721ms
    37:	learn: 0.2241411	total: 129ms	remaining: 719ms
    38:	learn: 0.2229340	total: 132ms	remaining: 712ms
    39:	learn: 0.2200625	total: 134ms	remaining: 705ms
    40:	learn: 0.2163081	total: 137ms	remaining: 697ms
    41:	learn: 0.2139391	total: 139ms	remaining: 690ms
    42:	learn: 0.2101920	total: 143ms	remaining: 687ms
    43:	learn: 0.2085320	total: 146ms	remaining: 683ms
    44:	learn: 0.2067197	total: 149ms	remaining: 680ms
    45:	learn: 0.2048137	total: 152ms	remaining: 676ms
    46:	learn: 0.2018549	total: 156ms	remaining: 673ms
    47:	learn: 0.1995428	total: 159ms	remaining: 668ms
    48:	learn: 0.1986209	total: 163ms	remaining: 667ms
    49:	learn: 0.1957396	total: 167ms	remaining: 670ms
    50:	learn: 0.1939618	total: 172ms	remaining: 671ms
    51:	learn: 0.1926382	total: 175ms	remaining: 668ms
    52:	learn: 0.1907973	total: 178ms	remaining: 661ms
    53:	learn: 0.1884270	total: 181ms	remaining: 657ms
    54:	learn: 0.1861139	total: 184ms	remaining: 654ms
    55:	learn: 0.1851019	total: 188ms	remaining: 650ms
    56:	learn: 0.1834382	total: 191ms	remaining: 646ms
    57:	learn: 0.1825239	total: 194ms	remaining: 642ms
    58:	learn: 0.1813251	total: 198ms	remaining: 642ms
    59:	learn: 0.1799729	total: 202ms	remaining: 640ms
    60:	learn: 0.1774509	total: 207ms	remaining: 640ms
    61:	learn: 0.1748408	total: 215ms	remaining: 651ms
    62:	learn: 0.1728666	total: 223ms	remaining: 661ms
    63:	learn: 0.1709538	total: 225ms	remaining: 655ms
    64:	learn: 0.1674536	total: 228ms	remaining: 649ms
    65:	learn: 0.1640022	total: 231ms	remaining: 644ms
    66:	learn: 0.1626121	total: 235ms	remaining: 641ms
    67:	learn: 0.1603429	total: 238ms	remaining: 637ms
    68:	learn: 0.1581798	total: 241ms	remaining: 632ms
    69:	learn: 0.1565309	total: 244ms	remaining: 628ms
    70:	learn: 0.1539081	total: 247ms	remaining: 624ms
    71:	learn: 0.1524306	total: 252ms	remaining: 623ms
    72:	learn: 0.1505649	total: 260ms	remaining: 630ms
    73:	learn: 0.1477454	total: 266ms	remaining: 632ms
    74:	learn: 0.1456470	total: 276ms	remaining: 643ms
    75:	learn: 0.1436327	total: 281ms	remaining: 643ms
    76:	learn: 0.1420938	total: 284ms	remaining: 638ms
    77:	learn: 0.1396748	total: 286ms	remaining: 631ms
    78:	learn: 0.1374965	total: 289ms	remaining: 626ms
    79:	learn: 0.1353485	total: 292ms	remaining: 621ms
    80:	learn: 0.1339746	total: 295ms	remaining: 616ms
    81:	learn: 0.1319270	total: 298ms	remaining: 611ms
    82:	learn: 0.1304866	total: 302ms	remaining: 607ms
    83:	learn: 0.1289112	total: 305ms	remaining: 602ms
    84:	learn: 0.1256760	total: 308ms	remaining: 598ms
    85:	learn: 0.1226680	total: 311ms	remaining: 593ms
    86:	learn: 0.1209215	total: 314ms	remaining: 589ms
    87:	learn: 0.1198400	total: 317ms	remaining: 584ms
    88:	learn: 0.1178356	total: 321ms	remaining: 581ms
    89:	learn: 0.1163493	total: 324ms	remaining: 577ms
    90:	learn: 0.1156463	total: 328ms	remaining: 572ms
    91:	learn: 0.1145688	total: 331ms	remaining: 568ms
    92:	learn: 0.1110849	total: 334ms	remaining: 563ms
    93:	learn: 0.1090365	total: 337ms	remaining: 560ms
    94:	learn: 0.1082680	total: 341ms	remaining: 556ms
    95:	learn: 0.1052198	total: 344ms	remaining: 552ms
    96:	learn: 0.1035675	total: 347ms	remaining: 547ms
    97:	learn: 0.1016416	total: 350ms	remaining: 543ms
    98:	learn: 0.1004460	total: 353ms	remaining: 539ms
    99:	learn: 0.0990316	total: 356ms	remaining: 535ms
    100:	learn: 0.0970890	total: 360ms	remaining: 530ms
    101:	learn: 0.0962174	total: 363ms	remaining: 526ms
    102:	learn: 0.0946088	total: 366ms	remaining: 523ms
    103:	learn: 0.0937930	total: 369ms	remaining: 518ms
    104:	learn: 0.0928641	total: 373ms	remaining: 514ms
    105:	learn: 0.0919037	total: 376ms	remaining: 510ms
    106:	learn: 0.0905119	total: 379ms	remaining: 506ms
    107:	learn: 0.0895944	total: 382ms	remaining: 502ms
    108:	learn: 0.0886780	total: 385ms	remaining: 498ms
    109:	learn: 0.0875779	total: 388ms	remaining: 494ms
    110:	learn: 0.0867305	total: 391ms	remaining: 490ms
    111:	learn: 0.0861047	total: 394ms	remaining: 486ms
    112:	learn: 0.0852467	total: 397ms	remaining: 482ms
    113:	learn: 0.0848492	total: 401ms	remaining: 478ms
    114:	learn: 0.0839011	total: 404ms	remaining: 474ms
    115:	learn: 0.0830048	total: 407ms	remaining: 470ms
    116:	learn: 0.0821459	total: 410ms	remaining: 466ms
    117:	learn: 0.0812613	total: 413ms	remaining: 462ms
    118:	learn: 0.0805315	total: 416ms	remaining: 458ms
    119:	learn: 0.0792158	total: 419ms	remaining: 454ms
    120:	learn: 0.0783946	total: 422ms	remaining: 450ms
    121:	learn: 0.0774049	total: 425ms	remaining: 446ms
    122:	learn: 0.0766134	total: 428ms	remaining: 442ms
    123:	learn: 0.0758759	total: 432ms	remaining: 439ms
    124:	learn: 0.0745749	total: 435ms	remaining: 435ms
    125:	learn: 0.0740838	total: 439ms	remaining: 432ms
    126:	learn: 0.0735201	total: 444ms	remaining: 430ms
    127:	learn: 0.0727286	total: 453ms	remaining: 432ms
    128:	learn: 0.0717105	total: 457ms	remaining: 428ms
    129:	learn: 0.0713752	total: 460ms	remaining: 424ms
    130:	learn: 0.0703963	total: 463ms	remaining: 420ms
    131:	learn: 0.0693434	total: 466ms	remaining: 417ms
    132:	learn: 0.0688542	total: 470ms	remaining: 413ms
    133:	learn: 0.0676210	total: 473ms	remaining: 409ms
    134:	learn: 0.0669467	total: 476ms	remaining: 406ms
    135:	learn: 0.0664904	total: 479ms	remaining: 402ms
    136:	learn: 0.0658145	total: 483ms	remaining: 398ms
    137:	learn: 0.0652735	total: 486ms	remaining: 394ms
    138:	learn: 0.0643717	total: 489ms	remaining: 390ms
    139:	learn: 0.0634535	total: 492ms	remaining: 386ms
    140:	learn: 0.0622686	total: 495ms	remaining: 383ms
    141:	learn: 0.0617023	total: 498ms	remaining: 379ms
    142:	learn: 0.0612296	total: 501ms	remaining: 375ms
    143:	learn: 0.0606484	total: 505ms	remaining: 371ms
    144:	learn: 0.0601298	total: 508ms	remaining: 368ms
    145:	learn: 0.0597766	total: 511ms	remaining: 364ms
    146:	learn: 0.0593091	total: 514ms	remaining: 360ms
    147:	learn: 0.0587103	total: 517ms	remaining: 356ms
    148:	learn: 0.0581103	total: 520ms	remaining: 353ms
    149:	learn: 0.0572176	total: 523ms	remaining: 349ms
    150:	learn: 0.0567370	total: 527ms	remaining: 345ms
    151:	learn: 0.0560814	total: 530ms	remaining: 342ms
    152:	learn: 0.0557531	total: 533ms	remaining: 338ms
    153:	learn: 0.0551419	total: 536ms	remaining: 334ms
    154:	learn: 0.0548095	total: 540ms	remaining: 331ms
    155:	learn: 0.0541516	total: 544ms	remaining: 328ms
    156:	learn: 0.0538633	total: 546ms	remaining: 324ms
    157:	learn: 0.0534005	total: 552ms	remaining: 322ms
    158:	learn: 0.0532417	total: 556ms	remaining: 318ms
    159:	learn: 0.0526091	total: 560ms	remaining: 315ms
    160:	learn: 0.0520897	total: 565ms	remaining: 313ms
    161:	learn: 0.0515189	total: 570ms	remaining: 310ms
    162:	learn: 0.0511297	total: 574ms	remaining: 307ms
    163:	learn: 0.0505832	total: 579ms	remaining: 304ms
    164:	learn: 0.0498615	total: 583ms	remaining: 300ms
    165:	learn: 0.0491964	total: 588ms	remaining: 298ms
    166:	learn: 0.0486214	total: 592ms	remaining: 294ms
    167:	learn: 0.0481271	total: 595ms	remaining: 290ms
    168:	learn: 0.0476399	total: 598ms	remaining: 286ms
    169:	learn: 0.0474114	total: 601ms	remaining: 283ms
    170:	learn: 0.0470895	total: 604ms	remaining: 279ms
    171:	learn: 0.0467435	total: 607ms	remaining: 275ms
    172:	learn: 0.0463516	total: 610ms	remaining: 271ms
    173:	learn: 0.0458161	total: 613ms	remaining: 268ms
    174:	learn: 0.0454505	total: 616ms	remaining: 264ms
    175:	learn: 0.0451269	total: 619ms	remaining: 260ms
    176:	learn: 0.0447898	total: 622ms	remaining: 257ms
    177:	learn: 0.0444317	total: 628ms	remaining: 254ms
    178:	learn: 0.0439177	total: 635ms	remaining: 252ms
    179:	learn: 0.0435715	total: 646ms	remaining: 251ms
    180:	learn: 0.0432301	total: 654ms	remaining: 249ms
    181:	learn: 0.0427220	total: 657ms	remaining: 245ms
    182:	learn: 0.0423660	total: 660ms	remaining: 242ms
    183:	learn: 0.0420203	total: 663ms	remaining: 238ms
    184:	learn: 0.0413804	total: 667ms	remaining: 234ms
    185:	learn: 0.0411249	total: 670ms	remaining: 230ms
    186:	learn: 0.0406743	total: 673ms	remaining: 227ms
    187:	learn: 0.0402570	total: 676ms	remaining: 223ms
    188:	learn: 0.0396490	total: 679ms	remaining: 219ms
    189:	learn: 0.0392963	total: 683ms	remaining: 216ms
    190:	learn: 0.0389794	total: 687ms	remaining: 212ms
    191:	learn: 0.0386448	total: 690ms	remaining: 209ms
    192:	learn: 0.0383705	total: 693ms	remaining: 205ms
    193:	learn: 0.0381095	total: 696ms	remaining: 201ms
    194:	learn: 0.0379001	total: 700ms	remaining: 197ms
    195:	learn: 0.0374682	total: 703ms	remaining: 194ms
    196:	learn: 0.0370828	total: 706ms	remaining: 190ms
    197:	learn: 0.0365417	total: 709ms	remaining: 186ms
    198:	learn: 0.0363562	total: 712ms	remaining: 182ms
    199:	learn: 0.0360777	total: 715ms	remaining: 179ms
    200:	learn: 0.0357400	total: 719ms	remaining: 175ms
    201:	learn: 0.0351980	total: 722ms	remaining: 172ms
    202:	learn: 0.0349813	total: 725ms	remaining: 168ms
    203:	learn: 0.0346097	total: 729ms	remaining: 164ms
    204:	learn: 0.0343302	total: 732ms	remaining: 161ms
    205:	learn: 0.0340361	total: 735ms	remaining: 157ms
    206:	learn: 0.0337483	total: 738ms	remaining: 153ms
    207:	learn: 0.0335658	total: 743ms	remaining: 150ms
    208:	learn: 0.0332899	total: 745ms	remaining: 146ms
    209:	learn: 0.0328396	total: 748ms	remaining: 143ms
    210:	learn: 0.0324736	total: 752ms	remaining: 139ms
    211:	learn: 0.0321053	total: 755ms	remaining: 135ms
    212:	learn: 0.0318850	total: 759ms	remaining: 132ms
    213:	learn: 0.0315530	total: 762ms	remaining: 128ms
    214:	learn: 0.0313276	total: 765ms	remaining: 125ms
    215:	learn: 0.0310175	total: 769ms	remaining: 121ms
    216:	learn: 0.0308188	total: 772ms	remaining: 117ms
    217:	learn: 0.0306007	total: 775ms	remaining: 114ms
    218:	learn: 0.0302203	total: 778ms	remaining: 110ms
    219:	learn: 0.0300805	total: 782ms	remaining: 107ms
    220:	learn: 0.0297982	total: 785ms	remaining: 103ms
    221:	learn: 0.0295076	total: 788ms	remaining: 99.4ms
    222:	learn: 0.0291921	total: 791ms	remaining: 95.8ms
    223:	learn: 0.0289806	total: 795ms	remaining: 92.2ms
    224:	learn: 0.0288423	total: 798ms	remaining: 88.6ms
    225:	learn: 0.0285599	total: 802ms	remaining: 85.2ms
    226:	learn: 0.0282576	total: 805ms	remaining: 81.6ms
    227:	learn: 0.0279053	total: 809ms	remaining: 78ms
    228:	learn: 0.0276480	total: 812ms	remaining: 74.5ms
    229:	learn: 0.0275081	total: 815ms	remaining: 70.9ms
    230:	learn: 0.0272193	total: 818ms	remaining: 67.3ms
    231:	learn: 0.0269598	total: 822ms	remaining: 63.8ms
    232:	learn: 0.0267049	total: 830ms	remaining: 60.6ms
    233:	learn: 0.0265291	total: 837ms	remaining: 57.2ms
    234:	learn: 0.0263663	total: 840ms	remaining: 53.6ms
    235:	learn: 0.0261421	total: 844ms	remaining: 50.1ms
    236:	learn: 0.0258372	total: 847ms	remaining: 46.5ms
    237:	learn: 0.0256715	total: 851ms	remaining: 42.9ms
    238:	learn: 0.0255286	total: 854ms	remaining: 39.3ms
    239:	learn: 0.0254143	total: 857ms	remaining: 35.7ms
    240:	learn: 0.0252756	total: 861ms	remaining: 32.2ms
    241:	learn: 0.0251442	total: 864ms	remaining: 28.6ms
    242:	learn: 0.0250168	total: 867ms	remaining: 25ms
    243:	learn: 0.0248567	total: 870ms	remaining: 21.4ms
    244:	learn: 0.0246678	total: 874ms	remaining: 17.8ms
    245:	learn: 0.0245471	total: 877ms	remaining: 14.3ms
    246:	learn: 0.0244924	total: 880ms	remaining: 10.7ms
    247:	learn: 0.0243485	total: 883ms	remaining: 7.12ms
    248:	learn: 0.0241502	total: 886ms	remaining: 3.56ms
    249:	learn: 0.0239829	total: 889ms	remaining: 0us
    0:	learn: 0.6162246	total: 2.83ms	remaining: 704ms
    1:	learn: 0.5514114	total: 6.08ms	remaining: 754ms
    2:	learn: 0.5148502	total: 7.73ms	remaining: 637ms
    3:	learn: 0.4692805	total: 10.8ms	remaining: 664ms
    4:	learn: 0.4431430	total: 13.1ms	remaining: 642ms
    5:	learn: 0.4191936	total: 16.2ms	remaining: 660ms
    6:	learn: 0.3886652	total: 19.3ms	remaining: 669ms
    7:	learn: 0.3719025	total: 22.7ms	remaining: 688ms
    8:	learn: 0.3558449	total: 25.8ms	remaining: 691ms
    9:	learn: 0.3426368	total: 28.8ms	remaining: 692ms
    10:	learn: 0.3302953	total: 32ms	remaining: 696ms
    11:	learn: 0.3206197	total: 35.1ms	remaining: 697ms
    12:	learn: 0.3089425	total: 38.2ms	remaining: 697ms
    13:	learn: 0.3022381	total: 41.3ms	remaining: 696ms
    14:	learn: 0.2962251	total: 44.5ms	remaining: 697ms
    15:	learn: 0.2922688	total: 47.6ms	remaining: 696ms
    16:	learn: 0.2903936	total: 50.6ms	remaining: 693ms
    17:	learn: 0.2865927	total: 53.8ms	remaining: 694ms
    18:	learn: 0.2841276	total: 56.9ms	remaining: 692ms
    19:	learn: 0.2802842	total: 60ms	remaining: 690ms
    20:	learn: 0.2739147	total: 63.4ms	remaining: 691ms
    21:	learn: 0.2702267	total: 66.5ms	remaining: 689ms
    22:	learn: 0.2672637	total: 69.6ms	remaining: 687ms
    23:	learn: 0.2635820	total: 72.7ms	remaining: 684ms
    24:	learn: 0.2588160	total: 76ms	remaining: 684ms
    25:	learn: 0.2550561	total: 79.4ms	remaining: 684ms
    26:	learn: 0.2534523	total: 82.6ms	remaining: 682ms
    27:	learn: 0.2525073	total: 85.7ms	remaining: 680ms
    28:	learn: 0.2495608	total: 88.8ms	remaining: 677ms
    29:	learn: 0.2470196	total: 92.1ms	remaining: 676ms
    30:	learn: 0.2442681	total: 99ms	remaining: 699ms
    31:	learn: 0.2418144	total: 106ms	remaining: 722ms
    32:	learn: 0.2403594	total: 110ms	remaining: 726ms
    33:	learn: 0.2393434	total: 114ms	remaining: 725ms
    34:	learn: 0.2357503	total: 117ms	remaining: 720ms
    35:	learn: 0.2342391	total: 120ms	remaining: 715ms
    36:	learn: 0.2327154	total: 124ms	remaining: 711ms
    37:	learn: 0.2287540	total: 127ms	remaining: 707ms
    38:	learn: 0.2255673	total: 130ms	remaining: 702ms
    39:	learn: 0.2240016	total: 133ms	remaining: 698ms
    40:	learn: 0.2215796	total: 136ms	remaining: 693ms
    41:	learn: 0.2180947	total: 139ms	remaining: 689ms
    42:	learn: 0.2164989	total: 142ms	remaining: 685ms
    43:	learn: 0.2131846	total: 145ms	remaining: 680ms
    44:	learn: 0.2097750	total: 148ms	remaining: 676ms
    45:	learn: 0.2062942	total: 152ms	remaining: 676ms
    46:	learn: 0.2003533	total: 157ms	remaining: 677ms
    47:	learn: 0.1987937	total: 159ms	remaining: 671ms
    48:	learn: 0.1942804	total: 162ms	remaining: 665ms
    49:	learn: 0.1923811	total: 165ms	remaining: 659ms
    50:	learn: 0.1907809	total: 167ms	remaining: 654ms
    51:	learn: 0.1872866	total: 172ms	remaining: 655ms
    52:	learn: 0.1848611	total: 177ms	remaining: 659ms
    53:	learn: 0.1825787	total: 181ms	remaining: 658ms
    54:	learn: 0.1807250	total: 184ms	remaining: 654ms
    55:	learn: 0.1785611	total: 188ms	remaining: 650ms
    56:	learn: 0.1766277	total: 191ms	remaining: 648ms
    57:	learn: 0.1750681	total: 195ms	remaining: 644ms
    58:	learn: 0.1732887	total: 198ms	remaining: 640ms
    59:	learn: 0.1715710	total: 201ms	remaining: 637ms
    60:	learn: 0.1706018	total: 205ms	remaining: 634ms
    61:	learn: 0.1692853	total: 214ms	remaining: 648ms
    62:	learn: 0.1668150	total: 226ms	remaining: 670ms
    63:	learn: 0.1647861	total: 229ms	remaining: 665ms
    64:	learn: 0.1643450	total: 231ms	remaining: 657ms
    65:	learn: 0.1628431	total: 234ms	remaining: 652ms
    66:	learn: 0.1608214	total: 237ms	remaining: 647ms
    67:	learn: 0.1596660	total: 240ms	remaining: 642ms
    68:	learn: 0.1563846	total: 243ms	remaining: 637ms
    69:	learn: 0.1552277	total: 246ms	remaining: 633ms
    70:	learn: 0.1543707	total: 249ms	remaining: 628ms
    71:	learn: 0.1525865	total: 252ms	remaining: 623ms
    72:	learn: 0.1491564	total: 255ms	remaining: 619ms
    73:	learn: 0.1472222	total: 258ms	remaining: 614ms
    74:	learn: 0.1454513	total: 261ms	remaining: 610ms
    75:	learn: 0.1428399	total: 264ms	remaining: 605ms
    76:	learn: 0.1414372	total: 267ms	remaining: 601ms
    77:	learn: 0.1390266	total: 271ms	remaining: 598ms
    78:	learn: 0.1374465	total: 274ms	remaining: 593ms
    79:	learn: 0.1346493	total: 277ms	remaining: 589ms
    80:	learn: 0.1339616	total: 281ms	remaining: 586ms
    81:	learn: 0.1322387	total: 285ms	remaining: 584ms
    82:	learn: 0.1310060	total: 292ms	remaining: 587ms
    83:	learn: 0.1295449	total: 302ms	remaining: 597ms
    84:	learn: 0.1277615	total: 305ms	remaining: 592ms
    85:	learn: 0.1270793	total: 309ms	remaining: 589ms
    86:	learn: 0.1257512	total: 312ms	remaining: 585ms
    87:	learn: 0.1240928	total: 316ms	remaining: 581ms
    88:	learn: 0.1221052	total: 319ms	remaining: 578ms
    89:	learn: 0.1209733	total: 323ms	remaining: 574ms
    90:	learn: 0.1198353	total: 326ms	remaining: 569ms
    91:	learn: 0.1171784	total: 329ms	remaining: 565ms
    92:	learn: 0.1146405	total: 332ms	remaining: 561ms
    93:	learn: 0.1120638	total: 335ms	remaining: 557ms
    94:	learn: 0.1106487	total: 339ms	remaining: 553ms
    95:	learn: 0.1097683	total: 342ms	remaining: 549ms
    96:	learn: 0.1083583	total: 345ms	remaining: 545ms
    97:	learn: 0.1069018	total: 349ms	remaining: 541ms
    98:	learn: 0.1051963	total: 353ms	remaining: 538ms
    99:	learn: 0.1037920	total: 356ms	remaining: 534ms
    100:	learn: 0.1021890	total: 359ms	remaining: 530ms
    101:	learn: 0.1011822	total: 363ms	remaining: 526ms
    102:	learn: 0.0993853	total: 366ms	remaining: 522ms
    103:	learn: 0.0977390	total: 369ms	remaining: 519ms
    104:	learn: 0.0960495	total: 373ms	remaining: 515ms
    105:	learn: 0.0950484	total: 376ms	remaining: 511ms
    106:	learn: 0.0934815	total: 379ms	remaining: 507ms
    107:	learn: 0.0920099	total: 383ms	remaining: 504ms
    108:	learn: 0.0906425	total: 386ms	remaining: 500ms
    109:	learn: 0.0887715	total: 390ms	remaining: 496ms
    110:	learn: 0.0877283	total: 393ms	remaining: 492ms
    111:	learn: 0.0867776	total: 396ms	remaining: 488ms
    112:	learn: 0.0860398	total: 399ms	remaining: 484ms
    113:	learn: 0.0852944	total: 402ms	remaining: 480ms
    114:	learn: 0.0842480	total: 405ms	remaining: 476ms
    115:	learn: 0.0833998	total: 408ms	remaining: 472ms
    116:	learn: 0.0822040	total: 412ms	remaining: 468ms
    117:	learn: 0.0810358	total: 415ms	remaining: 464ms
    118:	learn: 0.0803953	total: 418ms	remaining: 460ms
    119:	learn: 0.0793723	total: 421ms	remaining: 456ms
    120:	learn: 0.0782876	total: 424ms	remaining: 452ms
    121:	learn: 0.0768139	total: 428ms	remaining: 449ms
    122:	learn: 0.0762389	total: 431ms	remaining: 445ms
    123:	learn: 0.0752420	total: 434ms	remaining: 441ms
    124:	learn: 0.0743193	total: 437ms	remaining: 437ms
    125:	learn: 0.0735733	total: 440ms	remaining: 433ms
    126:	learn: 0.0722586	total: 443ms	remaining: 429ms
    127:	learn: 0.0714450	total: 447ms	remaining: 426ms
    128:	learn: 0.0704842	total: 450ms	remaining: 422ms
    129:	learn: 0.0695973	total: 453ms	remaining: 418ms
    130:	learn: 0.0691582	total: 456ms	remaining: 415ms
    131:	learn: 0.0678199	total: 460ms	remaining: 411ms
    132:	learn: 0.0671418	total: 463ms	remaining: 407ms
    133:	learn: 0.0662192	total: 466ms	remaining: 403ms
    134:	learn: 0.0656363	total: 469ms	remaining: 399ms
    135:	learn: 0.0648522	total: 472ms	remaining: 396ms
    136:	learn: 0.0640969	total: 476ms	remaining: 393ms
    137:	learn: 0.0634931	total: 483ms	remaining: 392ms
    138:	learn: 0.0631306	total: 489ms	remaining: 391ms
    139:	learn: 0.0623700	total: 492ms	remaining: 387ms
    140:	learn: 0.0614934	total: 495ms	remaining: 382ms
    141:	learn: 0.0610302	total: 498ms	remaining: 378ms
    142:	learn: 0.0598174	total: 501ms	remaining: 375ms
    143:	learn: 0.0591575	total: 504ms	remaining: 371ms
    144:	learn: 0.0585755	total: 507ms	remaining: 367ms
    145:	learn: 0.0578970	total: 510ms	remaining: 363ms
    146:	learn: 0.0571132	total: 513ms	remaining: 360ms
    147:	learn: 0.0566183	total: 517ms	remaining: 356ms
    148:	learn: 0.0561365	total: 520ms	remaining: 352ms
    149:	learn: 0.0555363	total: 523ms	remaining: 349ms
    150:	learn: 0.0548783	total: 526ms	remaining: 345ms
    151:	learn: 0.0546090	total: 529ms	remaining: 341ms
    152:	learn: 0.0540168	total: 532ms	remaining: 337ms
    153:	learn: 0.0536140	total: 535ms	remaining: 333ms
    154:	learn: 0.0531358	total: 538ms	remaining: 330ms
    155:	learn: 0.0524868	total: 541ms	remaining: 326ms
    156:	learn: 0.0519878	total: 545ms	remaining: 323ms
    157:	learn: 0.0511583	total: 549ms	remaining: 320ms
    158:	learn: 0.0507140	total: 552ms	remaining: 316ms
    159:	learn: 0.0503982	total: 555ms	remaining: 312ms
    160:	learn: 0.0498693	total: 558ms	remaining: 309ms
    161:	learn: 0.0494282	total: 561ms	remaining: 305ms
    162:	learn: 0.0490264	total: 566ms	remaining: 302ms
    163:	learn: 0.0487051	total: 569ms	remaining: 298ms
    164:	learn: 0.0481749	total: 572ms	remaining: 295ms
    165:	learn: 0.0478258	total: 575ms	remaining: 291ms
    166:	learn: 0.0474863	total: 578ms	remaining: 287ms
    167:	learn: 0.0470545	total: 581ms	remaining: 284ms
    168:	learn: 0.0466702	total: 585ms	remaining: 280ms
    169:	learn: 0.0459386	total: 588ms	remaining: 277ms
    170:	learn: 0.0454282	total: 591ms	remaining: 273ms
    171:	learn: 0.0449687	total: 594ms	remaining: 269ms
    172:	learn: 0.0444148	total: 597ms	remaining: 266ms
    173:	learn: 0.0439676	total: 600ms	remaining: 262ms
    174:	learn: 0.0436566	total: 603ms	remaining: 258ms
    175:	learn: 0.0432161	total: 606ms	remaining: 255ms
    176:	learn: 0.0429097	total: 609ms	remaining: 251ms
    177:	learn: 0.0425104	total: 612ms	remaining: 248ms
    178:	learn: 0.0420525	total: 615ms	remaining: 244ms
    179:	learn: 0.0417556	total: 619ms	remaining: 241ms
    180:	learn: 0.0413841	total: 622ms	remaining: 237ms
    181:	learn: 0.0409621	total: 624ms	remaining: 233ms
    182:	learn: 0.0405653	total: 627ms	remaining: 230ms
    183:	learn: 0.0401925	total: 631ms	remaining: 226ms
    184:	learn: 0.0397852	total: 634ms	remaining: 223ms
    185:	learn: 0.0393994	total: 637ms	remaining: 219ms
    186:	learn: 0.0389972	total: 640ms	remaining: 216ms
    187:	learn: 0.0386339	total: 643ms	remaining: 212ms
    188:	learn: 0.0383227	total: 646ms	remaining: 209ms
    189:	learn: 0.0379880	total: 650ms	remaining: 205ms
    190:	learn: 0.0376761	total: 653ms	remaining: 202ms
    191:	learn: 0.0374346	total: 656ms	remaining: 198ms
    192:	learn: 0.0370418	total: 659ms	remaining: 195ms
    193:	learn: 0.0366575	total: 663ms	remaining: 191ms
    194:	learn: 0.0362668	total: 666ms	remaining: 188ms
    195:	learn: 0.0359512	total: 674ms	remaining: 186ms
    196:	learn: 0.0357273	total: 680ms	remaining: 183ms
    197:	learn: 0.0354568	total: 688ms	remaining: 181ms
    198:	learn: 0.0351691	total: 692ms	remaining: 177ms
    199:	learn: 0.0349131	total: 695ms	remaining: 174ms
    200:	learn: 0.0345280	total: 699ms	remaining: 171ms
    201:	learn: 0.0342573	total: 706ms	remaining: 168ms
    202:	learn: 0.0340710	total: 710ms	remaining: 164ms
    203:	learn: 0.0338849	total: 712ms	remaining: 161ms
    204:	learn: 0.0336256	total: 715ms	remaining: 157ms
    205:	learn: 0.0334631	total: 719ms	remaining: 153ms
    206:	learn: 0.0331489	total: 722ms	remaining: 150ms
    207:	learn: 0.0328855	total: 725ms	remaining: 146ms
    208:	learn: 0.0325189	total: 728ms	remaining: 143ms
    209:	learn: 0.0323075	total: 731ms	remaining: 139ms
    210:	learn: 0.0320097	total: 734ms	remaining: 136ms
    211:	learn: 0.0317556	total: 738ms	remaining: 132ms
    212:	learn: 0.0313963	total: 741ms	remaining: 129ms
    213:	learn: 0.0312505	total: 744ms	remaining: 125ms
    214:	learn: 0.0309041	total: 748ms	remaining: 122ms
    215:	learn: 0.0307203	total: 751ms	remaining: 118ms
    216:	learn: 0.0305014	total: 754ms	remaining: 115ms
    217:	learn: 0.0299857	total: 757ms	remaining: 111ms
    218:	learn: 0.0296546	total: 760ms	remaining: 108ms
    219:	learn: 0.0293241	total: 763ms	remaining: 104ms
    220:	learn: 0.0292238	total: 766ms	remaining: 101ms
    221:	learn: 0.0289673	total: 769ms	remaining: 97.1ms
    222:	learn: 0.0287105	total: 773ms	remaining: 93.6ms
    223:	learn: 0.0285229	total: 776ms	remaining: 90.1ms
    224:	learn: 0.0281959	total: 779ms	remaining: 86.5ms
    225:	learn: 0.0278538	total: 782ms	remaining: 83ms
    226:	learn: 0.0275127	total: 785ms	remaining: 79.5ms
    227:	learn: 0.0271436	total: 788ms	remaining: 76ms
    228:	learn: 0.0269759	total: 791ms	remaining: 72.5ms
    229:	learn: 0.0268025	total: 794ms	remaining: 69.1ms
    230:	learn: 0.0265461	total: 797ms	remaining: 65.6ms
    231:	learn: 0.0263919	total: 800ms	remaining: 62.1ms
    232:	learn: 0.0261763	total: 803ms	remaining: 58.6ms
    233:	learn: 0.0259244	total: 807ms	remaining: 55.2ms
    234:	learn: 0.0258140	total: 810ms	remaining: 51.7ms
    235:	learn: 0.0256665	total: 813ms	remaining: 48.2ms
    236:	learn: 0.0254568	total: 816ms	remaining: 44.8ms
    237:	learn: 0.0250972	total: 819ms	remaining: 41.3ms
    238:	learn: 0.0249250	total: 823ms	remaining: 37.9ms
    239:	learn: 0.0245982	total: 826ms	remaining: 34.4ms
    240:	learn: 0.0244540	total: 829ms	remaining: 31ms
    241:	learn: 0.0242473	total: 832ms	remaining: 27.5ms
    242:	learn: 0.0240056	total: 835ms	remaining: 24.1ms
    243:	learn: 0.0238766	total: 838ms	remaining: 20.6ms
    244:	learn: 0.0236726	total: 841ms	remaining: 17.2ms
    245:	learn: 0.0234968	total: 844ms	remaining: 13.7ms
    246:	learn: 0.0233636	total: 848ms	remaining: 10.3ms
    247:	learn: 0.0232473	total: 851ms	remaining: 6.86ms
    248:	learn: 0.0230630	total: 854ms	remaining: 3.43ms
    249:	learn: 0.0228726	total: 857ms	remaining: 0us
    0:	learn: 0.6148838	total: 2.83ms	remaining: 705ms
    1:	learn: 0.5477614	total: 6.38ms	remaining: 791ms
    2:	learn: 0.5129463	total: 7.96ms	remaining: 655ms
    3:	learn: 0.4667790	total: 11ms	remaining: 676ms
    4:	learn: 0.4391139	total: 13.3ms	remaining: 650ms
    5:	learn: 0.4173599	total: 16.3ms	remaining: 663ms
    6:	learn: 0.3931244	total: 19.3ms	remaining: 671ms
    7:	learn: 0.3758897	total: 22.4ms	remaining: 678ms
    8:	learn: 0.3594499	total: 25.4ms	remaining: 680ms
    9:	learn: 0.3467225	total: 28.7ms	remaining: 689ms
    10:	learn: 0.3357549	total: 32.1ms	remaining: 698ms
    11:	learn: 0.3265852	total: 35.6ms	remaining: 706ms
    12:	learn: 0.3188165	total: 37.5ms	remaining: 684ms
    13:	learn: 0.3077856	total: 40.9ms	remaining: 689ms
    14:	learn: 0.3012092	total: 44.1ms	remaining: 691ms
    15:	learn: 0.2967395	total: 47.2ms	remaining: 691ms
    16:	learn: 0.2921410	total: 50.4ms	remaining: 690ms
    17:	learn: 0.2865858	total: 53.5ms	remaining: 689ms
    18:	learn: 0.2812118	total: 56.8ms	remaining: 690ms
    19:	learn: 0.2781306	total: 59.9ms	remaining: 689ms
    20:	learn: 0.2705275	total: 63.2ms	remaining: 689ms
    21:	learn: 0.2641733	total: 66.3ms	remaining: 687ms
    22:	learn: 0.2601627	total: 69.6ms	remaining: 687ms
    23:	learn: 0.2548389	total: 72.7ms	remaining: 685ms
    24:	learn: 0.2518465	total: 76ms	remaining: 684ms
    25:	learn: 0.2504752	total: 79.1ms	remaining: 682ms
    26:	learn: 0.2472132	total: 82.1ms	remaining: 678ms
    27:	learn: 0.2439564	total: 85.3ms	remaining: 676ms
    28:	learn: 0.2408602	total: 88.6ms	remaining: 675ms
    29:	learn: 0.2383202	total: 91.6ms	remaining: 672ms
    30:	learn: 0.2358368	total: 94.7ms	remaining: 669ms
    31:	learn: 0.2334302	total: 97.7ms	remaining: 666ms
    32:	learn: 0.2302524	total: 101ms	remaining: 664ms
    33:	learn: 0.2273804	total: 104ms	remaining: 662ms
    34:	learn: 0.2262065	total: 107ms	remaining: 660ms
    35:	learn: 0.2227723	total: 110ms	remaining: 656ms
    36:	learn: 0.2212087	total: 113ms	remaining: 652ms
    37:	learn: 0.2205636	total: 116ms	remaining: 649ms
    38:	learn: 0.2176224	total: 119ms	remaining: 646ms
    39:	learn: 0.2162306	total: 122ms	remaining: 643ms
    40:	learn: 0.2144650	total: 125ms	remaining: 639ms
    41:	learn: 0.2128141	total: 129ms	remaining: 637ms
    42:	learn: 0.2120597	total: 132ms	remaining: 634ms
    43:	learn: 0.2113190	total: 135ms	remaining: 630ms
    44:	learn: 0.2094854	total: 138ms	remaining: 627ms
    45:	learn: 0.2055465	total: 141ms	remaining: 624ms
    46:	learn: 0.2027422	total: 144ms	remaining: 622ms
    47:	learn: 0.2013226	total: 148ms	remaining: 621ms
    48:	learn: 0.1981153	total: 151ms	remaining: 619ms
    49:	learn: 0.1979732	total: 153ms	remaining: 612ms
    50:	learn: 0.1958094	total: 156ms	remaining: 610ms
    51:	learn: 0.1932614	total: 160ms	remaining: 607ms
    52:	learn: 0.1923904	total: 163ms	remaining: 606ms
    53:	learn: 0.1912849	total: 166ms	remaining: 604ms
    54:	learn: 0.1873073	total: 170ms	remaining: 603ms
    55:	learn: 0.1856878	total: 173ms	remaining: 600ms
    56:	learn: 0.1835815	total: 176ms	remaining: 597ms
    57:	learn: 0.1806117	total: 179ms	remaining: 594ms
    58:	learn: 0.1794720	total: 183ms	remaining: 592ms
    59:	learn: 0.1774689	total: 187ms	remaining: 591ms
    60:	learn: 0.1747918	total: 191ms	remaining: 593ms
    61:	learn: 0.1736615	total: 204ms	remaining: 618ms
    62:	learn: 0.1736236	total: 208ms	remaining: 616ms
    63:	learn: 0.1724602	total: 211ms	remaining: 614ms
    64:	learn: 0.1704067	total: 215ms	remaining: 611ms
    65:	learn: 0.1688919	total: 218ms	remaining: 608ms
    66:	learn: 0.1669775	total: 227ms	remaining: 619ms
    67:	learn: 0.1636748	total: 231ms	remaining: 618ms
    68:	learn: 0.1604854	total: 234ms	remaining: 613ms
    69:	learn: 0.1582924	total: 237ms	remaining: 609ms
    70:	learn: 0.1569306	total: 240ms	remaining: 605ms
    71:	learn: 0.1551198	total: 243ms	remaining: 601ms
    72:	learn: 0.1517015	total: 249ms	remaining: 604ms
    73:	learn: 0.1499741	total: 254ms	remaining: 605ms
    74:	learn: 0.1483015	total: 266ms	remaining: 621ms
    75:	learn: 0.1472566	total: 269ms	remaining: 615ms
    76:	learn: 0.1457791	total: 272ms	remaining: 610ms
    77:	learn: 0.1446618	total: 275ms	remaining: 606ms
    78:	learn: 0.1433263	total: 278ms	remaining: 602ms
    79:	learn: 0.1421515	total: 281ms	remaining: 597ms
    80:	learn: 0.1408311	total: 284ms	remaining: 593ms
    81:	learn: 0.1383843	total: 288ms	remaining: 589ms
    82:	learn: 0.1362150	total: 290ms	remaining: 584ms
    83:	learn: 0.1340770	total: 294ms	remaining: 581ms
    84:	learn: 0.1320885	total: 297ms	remaining: 576ms
    85:	learn: 0.1299566	total: 300ms	remaining: 572ms
    86:	learn: 0.1283585	total: 303ms	remaining: 568ms
    87:	learn: 0.1253789	total: 306ms	remaining: 563ms
    88:	learn: 0.1238640	total: 309ms	remaining: 559ms
    89:	learn: 0.1227218	total: 312ms	remaining: 555ms
    90:	learn: 0.1195295	total: 315ms	remaining: 551ms
    91:	learn: 0.1180271	total: 318ms	remaining: 546ms
    92:	learn: 0.1166764	total: 321ms	remaining: 542ms
    93:	learn: 0.1156382	total: 324ms	remaining: 538ms
    94:	learn: 0.1141578	total: 328ms	remaining: 534ms
    95:	learn: 0.1127702	total: 331ms	remaining: 531ms
    96:	learn: 0.1111850	total: 334ms	remaining: 527ms
    97:	learn: 0.1091322	total: 337ms	remaining: 523ms
    98:	learn: 0.1075709	total: 341ms	remaining: 520ms
    99:	learn: 0.1066286	total: 345ms	remaining: 517ms
    100:	learn: 0.1055166	total: 348ms	remaining: 513ms
    101:	learn: 0.1045509	total: 351ms	remaining: 510ms
    102:	learn: 0.1037719	total: 354ms	remaining: 506ms
    103:	learn: 0.1015849	total: 357ms	remaining: 502ms
    104:	learn: 0.0997786	total: 361ms	remaining: 498ms
    105:	learn: 0.0980375	total: 364ms	remaining: 494ms
    106:	learn: 0.0970699	total: 368ms	remaining: 491ms
    107:	learn: 0.0956435	total: 371ms	remaining: 488ms
    108:	learn: 0.0945097	total: 374ms	remaining: 484ms
    109:	learn: 0.0933791	total: 377ms	remaining: 480ms
    110:	learn: 0.0918158	total: 385ms	remaining: 482ms
    111:	learn: 0.0908634	total: 390ms	remaining: 480ms
    112:	learn: 0.0884116	total: 393ms	remaining: 476ms
    113:	learn: 0.0877035	total: 395ms	remaining: 472ms
    114:	learn: 0.0868489	total: 399ms	remaining: 468ms
    115:	learn: 0.0854155	total: 402ms	remaining: 464ms
    116:	learn: 0.0848627	total: 405ms	remaining: 461ms
    117:	learn: 0.0831940	total: 409ms	remaining: 457ms
    118:	learn: 0.0823668	total: 412ms	remaining: 453ms
    119:	learn: 0.0814462	total: 415ms	remaining: 449ms
    120:	learn: 0.0805860	total: 418ms	remaining: 445ms
    121:	learn: 0.0795224	total: 421ms	remaining: 441ms
    122:	learn: 0.0784286	total: 424ms	remaining: 438ms
    123:	learn: 0.0777405	total: 427ms	remaining: 434ms
    124:	learn: 0.0770562	total: 430ms	remaining: 430ms
    125:	learn: 0.0759909	total: 433ms	remaining: 426ms
    126:	learn: 0.0751964	total: 436ms	remaining: 423ms
    127:	learn: 0.0743053	total: 439ms	remaining: 419ms
    128:	learn: 0.0737686	total: 443ms	remaining: 415ms
    129:	learn: 0.0732238	total: 446ms	remaining: 411ms
    130:	learn: 0.0718328	total: 449ms	remaining: 407ms
    131:	learn: 0.0711284	total: 452ms	remaining: 404ms
    132:	learn: 0.0703505	total: 455ms	remaining: 400ms
    133:	learn: 0.0691060	total: 458ms	remaining: 397ms
    134:	learn: 0.0684488	total: 462ms	remaining: 393ms
    135:	learn: 0.0675683	total: 465ms	remaining: 390ms
    136:	learn: 0.0668542	total: 468ms	remaining: 386ms
    137:	learn: 0.0661180	total: 471ms	remaining: 383ms
    138:	learn: 0.0655698	total: 475ms	remaining: 379ms
    139:	learn: 0.0649068	total: 480ms	remaining: 377ms
    140:	learn: 0.0642974	total: 484ms	remaining: 374ms
    141:	learn: 0.0637278	total: 488ms	remaining: 371ms
    142:	learn: 0.0625120	total: 491ms	remaining: 367ms
    143:	learn: 0.0616062	total: 494ms	remaining: 364ms
    144:	learn: 0.0603164	total: 498ms	remaining: 360ms
    145:	learn: 0.0596115	total: 501ms	remaining: 357ms
    146:	learn: 0.0585673	total: 505ms	remaining: 354ms
    147:	learn: 0.0581108	total: 508ms	remaining: 350ms
    148:	learn: 0.0573514	total: 511ms	remaining: 346ms
    149:	learn: 0.0569819	total: 514ms	remaining: 343ms
    150:	learn: 0.0565607	total: 518ms	remaining: 339ms
    151:	learn: 0.0560526	total: 521ms	remaining: 336ms
    152:	learn: 0.0552590	total: 524ms	remaining: 332ms
    153:	learn: 0.0548235	total: 527ms	remaining: 329ms
    154:	learn: 0.0543834	total: 530ms	remaining: 325ms
    155:	learn: 0.0540739	total: 533ms	remaining: 321ms
    156:	learn: 0.0533398	total: 537ms	remaining: 318ms
    157:	learn: 0.0528854	total: 540ms	remaining: 314ms
    158:	learn: 0.0521715	total: 543ms	remaining: 311ms
    159:	learn: 0.0516616	total: 547ms	remaining: 308ms
    160:	learn: 0.0508687	total: 551ms	remaining: 304ms
    161:	learn: 0.0504697	total: 554ms	remaining: 301ms
    162:	learn: 0.0499438	total: 557ms	remaining: 297ms
    163:	learn: 0.0493840	total: 560ms	remaining: 294ms
    164:	learn: 0.0490540	total: 564ms	remaining: 290ms
    165:	learn: 0.0484086	total: 567ms	remaining: 287ms
    166:	learn: 0.0478879	total: 573ms	remaining: 285ms
    167:	learn: 0.0472228	total: 578ms	remaining: 282ms
    168:	learn: 0.0468257	total: 583ms	remaining: 279ms
    169:	learn: 0.0463351	total: 588ms	remaining: 277ms
    170:	learn: 0.0459169	total: 591ms	remaining: 273ms
    171:	learn: 0.0450083	total: 594ms	remaining: 269ms
    172:	learn: 0.0445714	total: 601ms	remaining: 268ms
    173:	learn: 0.0440311	total: 607ms	remaining: 265ms
    174:	learn: 0.0433053	total: 611ms	remaining: 262ms
    175:	learn: 0.0430221	total: 614ms	remaining: 258ms
    176:	learn: 0.0425773	total: 617ms	remaining: 254ms
    177:	learn: 0.0420688	total: 620ms	remaining: 251ms
    178:	learn: 0.0416380	total: 623ms	remaining: 247ms
    179:	learn: 0.0413409	total: 626ms	remaining: 243ms
    180:	learn: 0.0409804	total: 629ms	remaining: 240ms
    181:	learn: 0.0406453	total: 633ms	remaining: 236ms
    182:	learn: 0.0402961	total: 636ms	remaining: 233ms
    183:	learn: 0.0399654	total: 639ms	remaining: 229ms
    184:	learn: 0.0394654	total: 642ms	remaining: 226ms
    185:	learn: 0.0388885	total: 645ms	remaining: 222ms
    186:	learn: 0.0384945	total: 648ms	remaining: 218ms
    187:	learn: 0.0381636	total: 652ms	remaining: 215ms
    188:	learn: 0.0379528	total: 655ms	remaining: 211ms
    189:	learn: 0.0376244	total: 659ms	remaining: 208ms
    190:	learn: 0.0372323	total: 662ms	remaining: 205ms
    191:	learn: 0.0365875	total: 665ms	remaining: 201ms
    192:	learn: 0.0363578	total: 668ms	remaining: 197ms
    193:	learn: 0.0360690	total: 672ms	remaining: 194ms
    194:	learn: 0.0357088	total: 675ms	remaining: 190ms
    195:	learn: 0.0354449	total: 678ms	remaining: 187ms
    196:	learn: 0.0352193	total: 681ms	remaining: 183ms
    197:	learn: 0.0350264	total: 684ms	remaining: 180ms
    198:	learn: 0.0347002	total: 688ms	remaining: 176ms
    199:	learn: 0.0343640	total: 691ms	remaining: 173ms
    200:	learn: 0.0341183	total: 694ms	remaining: 169ms
    201:	learn: 0.0336501	total: 697ms	remaining: 166ms
    202:	learn: 0.0332832	total: 701ms	remaining: 162ms
    203:	learn: 0.0330222	total: 704ms	remaining: 159ms
    204:	learn: 0.0327269	total: 707ms	remaining: 155ms
    205:	learn: 0.0324396	total: 710ms	remaining: 152ms
    206:	learn: 0.0322344	total: 713ms	remaining: 148ms
    207:	learn: 0.0318079	total: 716ms	remaining: 145ms
    208:	learn: 0.0315432	total: 720ms	remaining: 141ms
    209:	learn: 0.0311746	total: 723ms	remaining: 138ms
    210:	learn: 0.0309505	total: 726ms	remaining: 134ms
    211:	learn: 0.0307374	total: 729ms	remaining: 131ms
    212:	learn: 0.0302623	total: 732ms	remaining: 127ms
    213:	learn: 0.0299782	total: 735ms	remaining: 124ms
    214:	learn: 0.0297582	total: 739ms	remaining: 120ms
    215:	learn: 0.0293860	total: 742ms	remaining: 117ms
    216:	learn: 0.0291410	total: 745ms	remaining: 113ms
    217:	learn: 0.0288181	total: 748ms	remaining: 110ms
    218:	learn: 0.0287171	total: 751ms	remaining: 106ms
    219:	learn: 0.0284638	total: 755ms	remaining: 103ms
    220:	learn: 0.0281733	total: 758ms	remaining: 99.4ms
    221:	learn: 0.0279277	total: 764ms	remaining: 96.4ms
    222:	learn: 0.0277248	total: 775ms	remaining: 93.8ms
    223:	learn: 0.0276317	total: 781ms	remaining: 90.6ms
    224:	learn: 0.0273948	total: 784ms	remaining: 87.1ms
    225:	learn: 0.0272174	total: 787ms	remaining: 83.6ms
    226:	learn: 0.0270849	total: 791ms	remaining: 80.1ms
    227:	learn: 0.0268107	total: 794ms	remaining: 76.6ms
    228:	learn: 0.0266055	total: 797ms	remaining: 73.1ms
    229:	learn: 0.0264007	total: 801ms	remaining: 69.7ms
    230:	learn: 0.0262961	total: 804ms	remaining: 66.1ms
    231:	learn: 0.0260678	total: 807ms	remaining: 62.6ms
    232:	learn: 0.0259090	total: 810ms	remaining: 59.1ms
    233:	learn: 0.0257168	total: 814ms	remaining: 55.6ms
    234:	learn: 0.0255660	total: 817ms	remaining: 52.1ms
    235:	learn: 0.0254141	total: 820ms	remaining: 48.6ms
    236:	learn: 0.0252782	total: 823ms	remaining: 45.1ms
    237:	learn: 0.0252302	total: 825ms	remaining: 41.6ms
    238:	learn: 0.0249886	total: 828ms	remaining: 38.1ms
    239:	learn: 0.0248088	total: 831ms	remaining: 34.6ms
    240:	learn: 0.0243839	total: 835ms	remaining: 31.2ms
    241:	learn: 0.0241391	total: 838ms	remaining: 27.7ms
    242:	learn: 0.0239487	total: 841ms	remaining: 24.2ms
    243:	learn: 0.0238601	total: 844ms	remaining: 20.8ms
    244:	learn: 0.0237650	total: 847ms	remaining: 17.3ms
    245:	learn: 0.0235408	total: 850ms	remaining: 13.8ms
    246:	learn: 0.0233032	total: 853ms	remaining: 10.4ms
    247:	learn: 0.0230985	total: 856ms	remaining: 6.91ms
    248:	learn: 0.0228684	total: 860ms	remaining: 3.45ms
    249:	learn: 0.0227937	total: 863ms	remaining: 0us
    0:	learn: 0.6178980	total: 2.91ms	remaining: 725ms
    1:	learn: 0.5537802	total: 5.89ms	remaining: 730ms
    2:	learn: 0.5170379	total: 7.6ms	remaining: 626ms
    3:	learn: 0.4758274	total: 11ms	remaining: 675ms
    4:	learn: 0.4447746	total: 14.1ms	remaining: 691ms
    5:	learn: 0.4211806	total: 17.3ms	remaining: 705ms
    6:	learn: 0.3950070	total: 20.6ms	remaining: 715ms
    7:	learn: 0.3748930	total: 23.9ms	remaining: 723ms
    8:	learn: 0.3639845	total: 27.6ms	remaining: 739ms
    9:	learn: 0.3525138	total: 31ms	remaining: 743ms
    10:	learn: 0.3431550	total: 34.3ms	remaining: 746ms
    11:	learn: 0.3369194	total: 37.7ms	remaining: 747ms
    12:	learn: 0.3295031	total: 41.1ms	remaining: 749ms
    13:	learn: 0.3210653	total: 49.7ms	remaining: 837ms
    14:	learn: 0.3136320	total: 55.2ms	remaining: 865ms
    15:	learn: 0.3101789	total: 58.5ms	remaining: 856ms
    16:	learn: 0.3059390	total: 61.9ms	remaining: 849ms
    17:	learn: 0.3012878	total: 65.1ms	remaining: 839ms
    18:	learn: 0.2970932	total: 68.3ms	remaining: 830ms
    19:	learn: 0.2922639	total: 71.4ms	remaining: 821ms
    20:	learn: 0.2885129	total: 74.5ms	remaining: 813ms
    21:	learn: 0.2844804	total: 77.7ms	remaining: 805ms
    22:	learn: 0.2785747	total: 80.8ms	remaining: 798ms
    23:	learn: 0.2761373	total: 84.2ms	remaining: 793ms
    24:	learn: 0.2696812	total: 87.5ms	remaining: 787ms
    25:	learn: 0.2667467	total: 90.7ms	remaining: 781ms
    26:	learn: 0.2641879	total: 93.7ms	remaining: 774ms
    27:	learn: 0.2613196	total: 97ms	remaining: 769ms
    28:	learn: 0.2576400	total: 100ms	remaining: 764ms
    29:	learn: 0.2556394	total: 104ms	remaining: 759ms
    30:	learn: 0.2525371	total: 107ms	remaining: 754ms
    31:	learn: 0.2511393	total: 110ms	remaining: 749ms
    32:	learn: 0.2510329	total: 111ms	remaining: 732ms
    33:	learn: 0.2476666	total: 115ms	remaining: 729ms
    34:	learn: 0.2449453	total: 118ms	remaining: 723ms
    35:	learn: 0.2427016	total: 121ms	remaining: 719ms
    36:	learn: 0.2409121	total: 124ms	remaining: 715ms
    37:	learn: 0.2381342	total: 128ms	remaining: 714ms
    38:	learn: 0.2361751	total: 131ms	remaining: 710ms
    39:	learn: 0.2326686	total: 135ms	remaining: 707ms
    40:	learn: 0.2314317	total: 138ms	remaining: 703ms
    41:	learn: 0.2278072	total: 141ms	remaining: 699ms
    42:	learn: 0.2265188	total: 144ms	remaining: 694ms
    43:	learn: 0.2224080	total: 148ms	remaining: 691ms
    44:	learn: 0.2207955	total: 151ms	remaining: 687ms
    45:	learn: 0.2170922	total: 154ms	remaining: 684ms
    46:	learn: 0.2156379	total: 157ms	remaining: 680ms
    47:	learn: 0.2129137	total: 161ms	remaining: 676ms
    48:	learn: 0.2116710	total: 164ms	remaining: 673ms
    49:	learn: 0.2105308	total: 167ms	remaining: 669ms
    50:	learn: 0.2088351	total: 171ms	remaining: 666ms
    51:	learn: 0.2064516	total: 174ms	remaining: 662ms
    52:	learn: 0.2039588	total: 177ms	remaining: 660ms
    53:	learn: 0.2010997	total: 181ms	remaining: 657ms
    54:	learn: 0.1984069	total: 184ms	remaining: 653ms
    55:	learn: 0.1962804	total: 187ms	remaining: 649ms
    56:	learn: 0.1947686	total: 191ms	remaining: 645ms
    57:	learn: 0.1920801	total: 194ms	remaining: 642ms
    58:	learn: 0.1903945	total: 197ms	remaining: 638ms
    59:	learn: 0.1901363	total: 199ms	remaining: 630ms
    60:	learn: 0.1883070	total: 202ms	remaining: 627ms
    61:	learn: 0.1861776	total: 206ms	remaining: 624ms
    62:	learn: 0.1851266	total: 210ms	remaining: 623ms
    63:	learn: 0.1829740	total: 213ms	remaining: 619ms
    64:	learn: 0.1805558	total: 216ms	remaining: 616ms
    65:	learn: 0.1796002	total: 220ms	remaining: 612ms
    66:	learn: 0.1775660	total: 223ms	remaining: 610ms
    67:	learn: 0.1763558	total: 227ms	remaining: 607ms
    68:	learn: 0.1749103	total: 230ms	remaining: 603ms
    69:	learn: 0.1729003	total: 233ms	remaining: 600ms
    70:	learn: 0.1700147	total: 239ms	remaining: 602ms
    71:	learn: 0.1683127	total: 250ms	remaining: 617ms
    72:	learn: 0.1665078	total: 257ms	remaining: 623ms
    73:	learn: 0.1657670	total: 260ms	remaining: 619ms
    74:	learn: 0.1636233	total: 263ms	remaining: 615ms
    75:	learn: 0.1616965	total: 267ms	remaining: 611ms
    76:	learn: 0.1608279	total: 270ms	remaining: 607ms
    77:	learn: 0.1594658	total: 274ms	remaining: 603ms
    78:	learn: 0.1569567	total: 280ms	remaining: 607ms
    79:	learn: 0.1538547	total: 284ms	remaining: 603ms
    80:	learn: 0.1533260	total: 287ms	remaining: 599ms
    81:	learn: 0.1521623	total: 291ms	remaining: 595ms
    82:	learn: 0.1506115	total: 296ms	remaining: 595ms
    83:	learn: 0.1472127	total: 300ms	remaining: 593ms
    84:	learn: 0.1452120	total: 304ms	remaining: 589ms
    85:	learn: 0.1438965	total: 307ms	remaining: 585ms
    86:	learn: 0.1429644	total: 310ms	remaining: 581ms
    87:	learn: 0.1417679	total: 314ms	remaining: 578ms
    88:	learn: 0.1403799	total: 317ms	remaining: 574ms
    89:	learn: 0.1394889	total: 321ms	remaining: 570ms
    90:	learn: 0.1375880	total: 324ms	remaining: 567ms
    91:	learn: 0.1361230	total: 327ms	remaining: 562ms
    92:	learn: 0.1351771	total: 331ms	remaining: 558ms
    93:	learn: 0.1323127	total: 334ms	remaining: 554ms
    94:	learn: 0.1313761	total: 337ms	remaining: 550ms
    95:	learn: 0.1301427	total: 341ms	remaining: 546ms
    96:	learn: 0.1291329	total: 344ms	remaining: 543ms
    97:	learn: 0.1274232	total: 347ms	remaining: 539ms
    98:	learn: 0.1261439	total: 351ms	remaining: 535ms
    99:	learn: 0.1247679	total: 354ms	remaining: 531ms
    100:	learn: 0.1230184	total: 357ms	remaining: 527ms
    101:	learn: 0.1211676	total: 361ms	remaining: 523ms
    102:	learn: 0.1203480	total: 364ms	remaining: 519ms
    103:	learn: 0.1194109	total: 367ms	remaining: 515ms
    104:	learn: 0.1163845	total: 370ms	remaining: 511ms
    105:	learn: 0.1155801	total: 374ms	remaining: 507ms
    106:	learn: 0.1144288	total: 377ms	remaining: 504ms
    107:	learn: 0.1133921	total: 380ms	remaining: 500ms
    108:	learn: 0.1119661	total: 384ms	remaining: 496ms
    109:	learn: 0.1105369	total: 387ms	remaining: 492ms
    110:	learn: 0.1093079	total: 390ms	remaining: 488ms
    111:	learn: 0.1079283	total: 394ms	remaining: 485ms
    112:	learn: 0.1070162	total: 397ms	remaining: 481ms
    113:	learn: 0.1060392	total: 400ms	remaining: 478ms
    114:	learn: 0.1046895	total: 404ms	remaining: 474ms
    115:	learn: 0.1031513	total: 407ms	remaining: 470ms
    116:	learn: 0.1023907	total: 410ms	remaining: 467ms
    117:	learn: 0.1016388	total: 414ms	remaining: 463ms
    118:	learn: 0.1006788	total: 417ms	remaining: 459ms
    119:	learn: 0.0992544	total: 426ms	remaining: 461ms
    120:	learn: 0.0982651	total: 431ms	remaining: 460ms
    121:	learn: 0.0970468	total: 434ms	remaining: 455ms
    122:	learn: 0.0954436	total: 437ms	remaining: 452ms
    123:	learn: 0.0943420	total: 441ms	remaining: 448ms
    124:	learn: 0.0935760	total: 444ms	remaining: 444ms
    125:	learn: 0.0916780	total: 448ms	remaining: 440ms
    126:	learn: 0.0909769	total: 451ms	remaining: 437ms
    127:	learn: 0.0900086	total: 454ms	remaining: 433ms
    128:	learn: 0.0889100	total: 457ms	remaining: 429ms
    129:	learn: 0.0877134	total: 461ms	remaining: 425ms
    130:	learn: 0.0870200	total: 464ms	remaining: 422ms
    131:	learn: 0.0860046	total: 467ms	remaining: 418ms
    132:	learn: 0.0851082	total: 471ms	remaining: 414ms
    133:	learn: 0.0845576	total: 474ms	remaining: 410ms
    134:	learn: 0.0837164	total: 477ms	remaining: 406ms
    135:	learn: 0.0827429	total: 480ms	remaining: 402ms
    136:	learn: 0.0814491	total: 484ms	remaining: 399ms
    137:	learn: 0.0809318	total: 487ms	remaining: 395ms
    138:	learn: 0.0799080	total: 490ms	remaining: 392ms
    139:	learn: 0.0790837	total: 494ms	remaining: 388ms
    140:	learn: 0.0782885	total: 497ms	remaining: 384ms
    141:	learn: 0.0777004	total: 500ms	remaining: 380ms
    142:	learn: 0.0771213	total: 503ms	remaining: 376ms
    143:	learn: 0.0762729	total: 506ms	remaining: 373ms
    144:	learn: 0.0753692	total: 509ms	remaining: 369ms
    145:	learn: 0.0741630	total: 513ms	remaining: 365ms
    146:	learn: 0.0730975	total: 516ms	remaining: 361ms
    147:	learn: 0.0722481	total: 519ms	remaining: 358ms
    148:	learn: 0.0719109	total: 522ms	remaining: 354ms
    149:	learn: 0.0711478	total: 525ms	remaining: 350ms
    150:	learn: 0.0706261	total: 528ms	remaining: 346ms
    151:	learn: 0.0699810	total: 531ms	remaining: 343ms
    152:	learn: 0.0693041	total: 535ms	remaining: 339ms
    153:	learn: 0.0689159	total: 538ms	remaining: 336ms
    154:	learn: 0.0686991	total: 542ms	remaining: 332ms
    155:	learn: 0.0679573	total: 545ms	remaining: 328ms
    156:	learn: 0.0671342	total: 548ms	remaining: 325ms
    157:	learn: 0.0667319	total: 552ms	remaining: 321ms
    158:	learn: 0.0661791	total: 556ms	remaining: 318ms
    159:	learn: 0.0651785	total: 560ms	remaining: 315ms
    160:	learn: 0.0646100	total: 564ms	remaining: 312ms
    161:	learn: 0.0640433	total: 567ms	remaining: 308ms
    162:	learn: 0.0635686	total: 571ms	remaining: 305ms
    163:	learn: 0.0633211	total: 574ms	remaining: 301ms
    164:	learn: 0.0626234	total: 577ms	remaining: 297ms
    165:	learn: 0.0619965	total: 581ms	remaining: 294ms
    166:	learn: 0.0614753	total: 584ms	remaining: 290ms
    167:	learn: 0.0610278	total: 587ms	remaining: 287ms
    168:	learn: 0.0604490	total: 590ms	remaining: 283ms
    169:	learn: 0.0598282	total: 594ms	remaining: 280ms
    170:	learn: 0.0593063	total: 597ms	remaining: 276ms
    171:	learn: 0.0586943	total: 601ms	remaining: 272ms
    172:	learn: 0.0577849	total: 604ms	remaining: 269ms
    173:	learn: 0.0570660	total: 607ms	remaining: 265ms
    174:	learn: 0.0567316	total: 611ms	remaining: 262ms
    175:	learn: 0.0562683	total: 616ms	remaining: 259ms
    176:	learn: 0.0557726	total: 624ms	remaining: 257ms
    177:	learn: 0.0551913	total: 631ms	remaining: 255ms
    178:	learn: 0.0545096	total: 635ms	remaining: 252ms
    179:	learn: 0.0537660	total: 638ms	remaining: 248ms
    180:	learn: 0.0533283	total: 641ms	remaining: 244ms
    181:	learn: 0.0530926	total: 644ms	remaining: 241ms
    182:	learn: 0.0526903	total: 648ms	remaining: 237ms
    183:	learn: 0.0520033	total: 651ms	remaining: 234ms
    184:	learn: 0.0516160	total: 654ms	remaining: 230ms
    185:	learn: 0.0510541	total: 658ms	remaining: 226ms
    186:	learn: 0.0506280	total: 661ms	remaining: 223ms
    187:	learn: 0.0499835	total: 664ms	remaining: 219ms
    188:	learn: 0.0494844	total: 667ms	remaining: 215ms
    189:	learn: 0.0489612	total: 675ms	remaining: 213ms
    190:	learn: 0.0486046	total: 678ms	remaining: 209ms
    191:	learn: 0.0482813	total: 681ms	remaining: 206ms
    192:	learn: 0.0479923	total: 684ms	remaining: 202ms
    193:	learn: 0.0475455	total: 687ms	remaining: 198ms
    194:	learn: 0.0471721	total: 690ms	remaining: 195ms
    195:	learn: 0.0466699	total: 693ms	remaining: 191ms
    196:	learn: 0.0463654	total: 696ms	remaining: 187ms
    197:	learn: 0.0463217	total: 698ms	remaining: 183ms
    198:	learn: 0.0460428	total: 701ms	remaining: 180ms
    199:	learn: 0.0456187	total: 705ms	remaining: 176ms
    200:	learn: 0.0454104	total: 708ms	remaining: 173ms
    201:	learn: 0.0450075	total: 712ms	remaining: 169ms
    202:	learn: 0.0445366	total: 715ms	remaining: 166ms
    203:	learn: 0.0439487	total: 718ms	remaining: 162ms
    204:	learn: 0.0434631	total: 722ms	remaining: 158ms
    205:	learn: 0.0431981	total: 725ms	remaining: 155ms
    206:	learn: 0.0429133	total: 728ms	remaining: 151ms
    207:	learn: 0.0427118	total: 732ms	remaining: 148ms
    208:	learn: 0.0422119	total: 735ms	remaining: 144ms
    209:	learn: 0.0420675	total: 738ms	remaining: 141ms
    210:	learn: 0.0417700	total: 741ms	remaining: 137ms
    211:	learn: 0.0414062	total: 744ms	remaining: 133ms
    212:	learn: 0.0411873	total: 748ms	remaining: 130ms
    213:	learn: 0.0408804	total: 751ms	remaining: 126ms
    214:	learn: 0.0405137	total: 754ms	remaining: 123ms
    215:	learn: 0.0400964	total: 757ms	remaining: 119ms
    216:	learn: 0.0398400	total: 760ms	remaining: 116ms
    217:	learn: 0.0395610	total: 764ms	remaining: 112ms
    218:	learn: 0.0391608	total: 767ms	remaining: 109ms
    219:	learn: 0.0388035	total: 770ms	remaining: 105ms
    220:	learn: 0.0384118	total: 773ms	remaining: 101ms
    221:	learn: 0.0383299	total: 776ms	remaining: 97.9ms
    222:	learn: 0.0380641	total: 780ms	remaining: 94.4ms
    223:	learn: 0.0378997	total: 783ms	remaining: 90.8ms
    224:	learn: 0.0377406	total: 785ms	remaining: 87.3ms
    225:	learn: 0.0374964	total: 788ms	remaining: 83.7ms
    226:	learn: 0.0372463	total: 791ms	remaining: 80.2ms
    227:	learn: 0.0370262	total: 794ms	remaining: 76.6ms
    228:	learn: 0.0367118	total: 797ms	remaining: 73.1ms
    229:	learn: 0.0364085	total: 804ms	remaining: 69.9ms
    230:	learn: 0.0361841	total: 813ms	remaining: 66.8ms
    231:	learn: 0.0359630	total: 817ms	remaining: 63.4ms
    232:	learn: 0.0357493	total: 820ms	remaining: 59.8ms
    233:	learn: 0.0354115	total: 823ms	remaining: 56.3ms
    234:	learn: 0.0350627	total: 826ms	remaining: 52.7ms
    235:	learn: 0.0347839	total: 830ms	remaining: 49.2ms
    236:	learn: 0.0345117	total: 833ms	remaining: 45.7ms
    237:	learn: 0.0342068	total: 837ms	remaining: 42.2ms
    238:	learn: 0.0339281	total: 840ms	remaining: 38.7ms
    239:	learn: 0.0337224	total: 843ms	remaining: 35.1ms
    240:	learn: 0.0335157	total: 847ms	remaining: 31.6ms
    241:	learn: 0.0332715	total: 850ms	remaining: 28.1ms
    242:	learn: 0.0329909	total: 854ms	remaining: 24.6ms
    243:	learn: 0.0326493	total: 857ms	remaining: 21.1ms
    244:	learn: 0.0324681	total: 860ms	remaining: 17.6ms
    245:	learn: 0.0322613	total: 864ms	remaining: 14ms
    246:	learn: 0.0321345	total: 867ms	remaining: 10.5ms
    247:	learn: 0.0319629	total: 871ms	remaining: 7.02ms
    248:	learn: 0.0317017	total: 874ms	remaining: 3.51ms
    249:	learn: 0.0315664	total: 877ms	remaining: 0us
    


```python
# RandomSearchCV result
model_list = [RF_rg, XGB_rg, CAT_rg, GBC_rg, LR_rg]
for model in model_list :
    print(model.best_score_)
    print(model.best_params_)
    print("**"*50)
```

    0.8268213049243942
    {'n_estimators': 10, 'min_samples_split': 20, 'min_samples_leaf': 8, 'max_depth': 6}
    ****************************************************************************************************
    0.8415855576025656
    {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'gamma': 2, 'colsample_bytree': 0.9}
    ****************************************************************************************************
    0.8362475710297647
    {'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 250, 'depth': 7, 'border_count': 200}
    ****************************************************************************************************
    0.8402764650213452
    {'n_estimators': 100, 'learning_rate': 0.1}
    ****************************************************************************************************
    0.0
    {'penalty': 'l2', 'C': 0.1}
    ****************************************************************************************************
    


```python
from sklearn.model_selection import GridSearchCV
```

GridSearchCV이 시간이 오래 걸리는 관계로 결과값을 각 모델에 직접적용


```python
# GridSearchCV

# Random Forest 
# RF_gs = Grid_Search_CV(RF_model, params_rf_gs, X_train_HRFAF_OCRN_CNT, y_train_HRFAF_OCRN_CNT, X_test_HRFAF_OCRN_CNT)

# XGBoost fit
# XGB_gs = Grid_Search_CV(xgb_model, params_xgb_gs, X_train_HRFAF_OCRN_CNT, y_train_HRFAF_OCRN_CNT, X_test_HRFAF_OCRN_CNT)

# CatBoost fit
# CAT_gs = Grid_Search_CV(cat_model, params_cat_gs, X_train_HRFAF_OCRN_CNT, y_train_HRFAF_OCRN_CNT, X_test_HRFAF_OCRN_CNT)

# GradientBoost fit
# GBC_gs = Grid_Search_CV(gbc_model, param_gbc_gs, X_train_HRFAF_OCRN_CNT, y_train_HRFAF_OCRN_CNT, X_test_HRFAF_OCRN_CNT)

# Logistic Regression fit
# LR_gs = Grid_Search_CV(lr_model, params_lr_gs, X_train_HRFAF_OCRN_CNT, y_train_HRFAF_OCRN_CNT, X_test_HRFAF_OCRN_CNT)
```


```python
## GridSearchCV result
# model_list = [RF_gs, XGB_gs, CAT_gs, GBC_gs, LR_gs]
# for model in model_list :
#     print(model.best_score_)
#     print(model.best_params_)
#     print("**"*50)
```

####파라미터 조정결과


```python
# 파라미터 조정 결과
# Random Forest fit
def Random_Forset_model (X_train, y_train, X_test):
    params = {
              "n_estimators" : 100,
              "criterion" : 'gini',
              "max_depth" : None,
              "min_samples_split" : 2,
              "min_samples_leaf" : 1,
              "max_features" : "auto"
             }
    RFM = RandomForestClassifier(random_state=10, n_jobs=-1, **params)
    RFM.fit(X_train, y_train)
    return RFM 

# XGBoost fit
def XGBoost_model(X_train, y_train, X_test):
    params ={
              "max_depth" : 6,
              "learning_rate" : 0.3, 
              'n_estimators' : 100,
              "colsample_bytree" : 1,
              "colsample_bylevel" : 1,
              "colsample_bynode" : 1,
              "reg_lambda" : 1,
              "reg_alpha" : 0,
              'subsample' : 1,
              'min_child_weight' : 1,
              "gamma" :0
             }
    XGBCM = XGBClassifier(n_jobs = -1, 
                         random_state = 10, 
                         use_label_encoder = False, 
                         eval_metric = "error",
                         **params )
    XGBCM.fit(X_train, y_train)
    return XGBCM 

# CatBoost fit
def CatBoost_model(X_train, y_train, X_test):
    params = {
            "iterations" : 1000,
            "learning_rate" : 0.002,
            "depth" : 6,
            "l2_leaf_reg" : 3,
            "model_size_reg" : 0.5,
            "rsm" : 1,
            "loss_function" : "Logloss",
            "border_count" : 254,
            "feature_border_type" : "GreedyLogSum",
            "leaf_estimation_iterations" : 10,
            "leaf_estimation_method" : 'Newton',
            "class_weights" :None,
            "random_strength" :1,
            "eval_metric" : "Logloss",
            "boosting_type" :"Plain",
            "task_type" :"CPU",
            "subsample" : 0.8,
            "grow_policy" : "SymmetricTree",
            "min_data_in_leaf" :1,
            "max_leaves" :64,
             }
    CBCM = CatBoostClassifier(**params, random_state = 10)
    CBCM.fit(X_train, y_train)
    return CBCM

# GradientBoost fit
def GradientBoost_model(X_train, y_train, X_test):
    params = {
            "learning_rate" : 0.1, 
            "max_depth" : 3,
            "n_estimators" : 100,
             }
    GBCM = GradientBoostingClassifier(**params, random_state = 10)
    GBCM.fit(X_train, y_train)
    return GBCM

# Logistic Regression fit
def Logistic_Regression_model(X_train, y_train, X_test):
    params = {
                "fit_intercept" : True,
                "intercept_scaling" : 1,
                "max_iter" : 1000,
                "multi_class" : 'auto',
                "penalty" : 'l2',
                "solver" : 'lbfgs' 
             }
    LRM = LogisticRegression(**params, n_jobs = -1, random_state = 10)
    LRM.fit(X_train, y_train)
    return LRM
```


```python
# model fit
for i in col:
     globals()["Random_Forset_model_{}".format(i)] = Random_Forset_model( globals()["X_train_{}".format(i)], globals()["y_train_{}".format(i)], globals()["X_test_{}".format(i)])
     globals()["Random_Forset_model_y_pred_{}".format(i)] = globals()["Random_Forset_model_{}".format(i)].predict(globals()["X_test_{}".format(i)])
     globals()["Random_Forset_model_score_{}".format(i)] = f1_score(globals()["y_test_{}".format(i)], globals()["Random_Forset_model_y_pred_{}".format(i)])

     globals()["XGBoost_model_{}".format(i)] = XGBoost_model( globals()["X_train_{}".format(i)], globals()["y_train_{}".format(i)], globals()["X_test_{}".format(i)])
     globals()["XGBoost_model_y_pred_{}".format(i)] = globals()["XGBoost_model_{}".format(i)].predict(globals()["X_test_{}".format(i)])
     globals()["XGBoost_model_score_{}".format(i)] = f1_score(globals()["y_test_{}".format(i)], globals()["XGBoost_model_y_pred_{}".format(i)])

     globals()["CatBoost_model_{}".format(i)] = CatBoost_model( globals()["X_train_{}".format(i)], globals()["y_train_{}".format(i)], globals()["X_test_{}".format(i)])
     globals()["CatBoost_model_y_pred_{}".format(i)] = globals()["CatBoost_model_{}".format(i)].predict(globals()["X_test_{}".format(i)])
     globals()["CatBoost_model_score_{}".format(i)] = f1_score(globals()["y_test_{}".format(i)], globals()["CatBoost_model_y_pred_{}".format(i)])

     globals()["GradientBoost_model_{}".format(i)] = CatBoost_model( globals()["X_train_{}".format(i)], globals()["y_train_{}".format(i)], globals()["X_test_{}".format(i)])
     globals()["GradientBoost_model_y_pred_{}".format(i)] = globals()["GradientBoost_model_{}".format(i)].predict(globals()["X_test_{}".format(i)])
     globals()["GradientBoost_model_score_{}".format(i)] = f1_score(globals()["y_test_{}".format(i)], globals()["GradientBoost_model_y_pred_{}".format(i)])

     globals()["Logistic_Regression_model_{}".format(i)] = Logistic_Regression_model( globals()["X_train_{}".format(i)], globals()["y_train_{}".format(i)], globals()["X_test_{}".format(i)])
     globals()["Logistic_Regression_model_y_pred_{}".format(i)] = globals()["Logistic_Regression_model_{}".format(i)].predict(globals()["X_test_{}".format(i)])
     globals()["Logistic_Regression_model_score_{}".format(i)] = f1_score(globals()["y_test_{}".format(i)], globals()["Logistic_Regression_model_y_pred_{}".format(i)])

```

    [1;30;43m스트리밍 출력 내용이 길어서 마지막 5000줄이 삭제되었습니다.[0m
    0:	learn: 0.6916004	total: 4.12ms	remaining: 4.12s
    1:	learn: 0.6903150	total: 7.48ms	remaining: 3.73s
    2:	learn: 0.6882891	total: 9.37ms	remaining: 3.11s
    3:	learn: 0.6866363	total: 19.8ms	remaining: 4.94s
    4:	learn: 0.6848510	total: 21.2ms	remaining: 4.21s
    5:	learn: 0.6833290	total: 36.8ms	remaining: 6.1s
    6:	learn: 0.6814711	total: 38.4ms	remaining: 5.45s
    7:	learn: 0.6795799	total: 55ms	remaining: 6.81s
    8:	learn: 0.6777166	total: 64.2ms	remaining: 7.07s
    9:	learn: 0.6761632	total: 73.5ms	remaining: 7.28s
    10:	learn: 0.6745601	total: 82.4ms	remaining: 7.41s
    11:	learn: 0.6726128	total: 90.1ms	remaining: 7.42s
    12:	learn: 0.6711535	total: 97.3ms	remaining: 7.38s
    13:	learn: 0.6693834	total: 106ms	remaining: 7.49s
    14:	learn: 0.6676363	total: 114ms	remaining: 7.49s
    15:	learn: 0.6660630	total: 121ms	remaining: 7.47s
    16:	learn: 0.6644992	total: 123ms	remaining: 7.13s
    17:	learn: 0.6629606	total: 131ms	remaining: 7.16s
    18:	learn: 0.6612448	total: 140ms	remaining: 7.21s
    19:	learn: 0.6595728	total: 141ms	remaining: 6.91s
    20:	learn: 0.6580681	total: 142ms	remaining: 6.64s
    21:	learn: 0.6567718	total: 144ms	remaining: 6.39s
    22:	learn: 0.6549222	total: 146ms	remaining: 6.21s
    23:	learn: 0.6538005	total: 148ms	remaining: 6.03s
    24:	learn: 0.6527090	total: 154ms	remaining: 5.99s
    25:	learn: 0.6512616	total: 155ms	remaining: 5.81s
    26:	learn: 0.6496262	total: 162ms	remaining: 5.84s
    27:	learn: 0.6483066	total: 163ms	remaining: 5.66s
    28:	learn: 0.6464154	total: 164ms	remaining: 5.5s
    29:	learn: 0.6446429	total: 170ms	remaining: 5.5s
    30:	learn: 0.6430586	total: 171ms	remaining: 5.36s
    31:	learn: 0.6414484	total: 175ms	remaining: 5.3s
    32:	learn: 0.6396850	total: 183ms	remaining: 5.37s
    33:	learn: 0.6384143	total: 185ms	remaining: 5.24s
    34:	learn: 0.6368098	total: 186ms	remaining: 5.13s
    35:	learn: 0.6354758	total: 187ms	remaining: 5.01s
    36:	learn: 0.6343140	total: 188ms	remaining: 4.9s
    37:	learn: 0.6327967	total: 190ms	remaining: 4.8s
    38:	learn: 0.6316350	total: 191ms	remaining: 4.7s
    39:	learn: 0.6300072	total: 193ms	remaining: 4.64s
    40:	learn: 0.6285985	total: 196ms	remaining: 4.58s
    41:	learn: 0.6272060	total: 198ms	remaining: 4.52s
    42:	learn: 0.6258754	total: 200ms	remaining: 4.46s
    43:	learn: 0.6244082	total: 202ms	remaining: 4.38s
    44:	learn: 0.6225656	total: 208ms	remaining: 4.41s
    45:	learn: 0.6212388	total: 209ms	remaining: 4.33s
    46:	learn: 0.6200774	total: 210ms	remaining: 4.26s
    47:	learn: 0.6187593	total: 227ms	remaining: 4.49s
    48:	learn: 0.6173784	total: 228ms	remaining: 4.42s
    49:	learn: 0.6160128	total: 230ms	remaining: 4.37s
    50:	learn: 0.6143278	total: 238ms	remaining: 4.42s
    51:	learn: 0.6127739	total: 239ms	remaining: 4.36s
    52:	learn: 0.6112475	total: 240ms	remaining: 4.29s
    53:	learn: 0.6094364	total: 242ms	remaining: 4.23s
    54:	learn: 0.6082174	total: 243ms	remaining: 4.17s
    55:	learn: 0.6073356	total: 245ms	remaining: 4.13s
    56:	learn: 0.6062183	total: 246ms	remaining: 4.06s
    57:	learn: 0.6048525	total: 247ms	remaining: 4.01s
    58:	learn: 0.6033874	total: 254ms	remaining: 4.06s
    59:	learn: 0.6020164	total: 256ms	remaining: 4s
    60:	learn: 0.6007589	total: 257ms	remaining: 3.95s
    61:	learn: 0.5996000	total: 259ms	remaining: 3.91s
    62:	learn: 0.5983093	total: 264ms	remaining: 3.92s
    63:	learn: 0.5971637	total: 270ms	remaining: 3.95s
    64:	learn: 0.5960783	total: 271ms	remaining: 3.9s
    65:	learn: 0.5947439	total: 273ms	remaining: 3.86s
    66:	learn: 0.5934368	total: 274ms	remaining: 3.81s
    67:	learn: 0.5921190	total: 275ms	remaining: 3.77s
    68:	learn: 0.5905971	total: 276ms	remaining: 3.73s
    69:	learn: 0.5889469	total: 282ms	remaining: 3.75s
    70:	learn: 0.5875091	total: 284ms	remaining: 3.72s
    71:	learn: 0.5864849	total: 286ms	remaining: 3.69s
    72:	learn: 0.5850185	total: 288ms	remaining: 3.66s
    73:	learn: 0.5836672	total: 307ms	remaining: 3.84s
    74:	learn: 0.5824565	total: 316ms	remaining: 3.9s
    75:	learn: 0.5812975	total: 322ms	remaining: 3.91s
    76:	learn: 0.5798943	total: 342ms	remaining: 4.1s
    77:	learn: 0.5784122	total: 346ms	remaining: 4.09s
    78:	learn: 0.5770322	total: 356ms	remaining: 4.15s
    79:	learn: 0.5759222	total: 358ms	remaining: 4.12s
    80:	learn: 0.5744130	total: 368ms	remaining: 4.17s
    81:	learn: 0.5732428	total: 375ms	remaining: 4.2s
    82:	learn: 0.5720468	total: 380ms	remaining: 4.2s
    83:	learn: 0.5707699	total: 384ms	remaining: 4.19s
    84:	learn: 0.5694750	total: 398ms	remaining: 4.29s
    85:	learn: 0.5682983	total: 409ms	remaining: 4.34s
    86:	learn: 0.5671711	total: 413ms	remaining: 4.34s
    87:	learn: 0.5659427	total: 425ms	remaining: 4.4s
    88:	learn: 0.5647276	total: 430ms	remaining: 4.4s
    89:	learn: 0.5637862	total: 433ms	remaining: 4.38s
    90:	learn: 0.5625423	total: 443ms	remaining: 4.42s
    91:	learn: 0.5608811	total: 445ms	remaining: 4.39s
    92:	learn: 0.5599613	total: 449ms	remaining: 4.38s
    93:	learn: 0.5586899	total: 459ms	remaining: 4.43s
    94:	learn: 0.5573977	total: 462ms	remaining: 4.4s
    95:	learn: 0.5564105	total: 478ms	remaining: 4.5s
    96:	learn: 0.5552762	total: 490ms	remaining: 4.56s
    97:	learn: 0.5540946	total: 494ms	remaining: 4.55s
    98:	learn: 0.5527510	total: 503ms	remaining: 4.58s
    99:	learn: 0.5517017	total: 506ms	remaining: 4.56s
    100:	learn: 0.5504806	total: 511ms	remaining: 4.54s
    101:	learn: 0.5492397	total: 524ms	remaining: 4.61s
    102:	learn: 0.5481115	total: 534ms	remaining: 4.65s
    103:	learn: 0.5471638	total: 536ms	remaining: 4.62s
    104:	learn: 0.5462307	total: 539ms	remaining: 4.6s
    105:	learn: 0.5450873	total: 547ms	remaining: 4.61s
    106:	learn: 0.5440179	total: 551ms	remaining: 4.6s
    107:	learn: 0.5429412	total: 561ms	remaining: 4.63s
    108:	learn: 0.5419914	total: 565ms	remaining: 4.62s
    109:	learn: 0.5409663	total: 569ms	remaining: 4.6s
    110:	learn: 0.5397617	total: 579ms	remaining: 4.63s
    111:	learn: 0.5387660	total: 586ms	remaining: 4.64s
    112:	learn: 0.5377717	total: 595ms	remaining: 4.67s
    113:	learn: 0.5368395	total: 597ms	remaining: 4.64s
    114:	learn: 0.5359460	total: 601ms	remaining: 4.62s
    115:	learn: 0.5348159	total: 611ms	remaining: 4.66s
    116:	learn: 0.5339123	total: 614ms	remaining: 4.63s
    117:	learn: 0.5329590	total: 618ms	remaining: 4.62s
    118:	learn: 0.5319580	total: 629ms	remaining: 4.65s
    119:	learn: 0.5308427	total: 631ms	remaining: 4.63s
    120:	learn: 0.5296286	total: 635ms	remaining: 4.62s
    121:	learn: 0.5288139	total: 647ms	remaining: 4.65s
    122:	learn: 0.5278808	total: 649ms	remaining: 4.63s
    123:	learn: 0.5267278	total: 653ms	remaining: 4.61s
    124:	learn: 0.5255742	total: 659ms	remaining: 4.62s
    125:	learn: 0.5243342	total: 662ms	remaining: 4.59s
    126:	learn: 0.5233028	total: 671ms	remaining: 4.61s
    127:	learn: 0.5224017	total: 680ms	remaining: 4.63s
    128:	learn: 0.5216878	total: 683ms	remaining: 4.61s
    129:	learn: 0.5205872	total: 692ms	remaining: 4.63s
    130:	learn: 0.5194733	total: 695ms	remaining: 4.61s
    131:	learn: 0.5183374	total: 699ms	remaining: 4.59s
    132:	learn: 0.5174467	total: 701ms	remaining: 4.57s
    133:	learn: 0.5164937	total: 702ms	remaining: 4.54s
    134:	learn: 0.5154824	total: 706ms	remaining: 4.52s
    135:	learn: 0.5145889	total: 709ms	remaining: 4.5s
    136:	learn: 0.5136641	total: 714ms	remaining: 4.5s
    137:	learn: 0.5127560	total: 716ms	remaining: 4.47s
    138:	learn: 0.5118108	total: 722ms	remaining: 4.47s
    139:	learn: 0.5108899	total: 730ms	remaining: 4.48s
    140:	learn: 0.5098505	total: 734ms	remaining: 4.47s
    141:	learn: 0.5086875	total: 739ms	remaining: 4.47s
    142:	learn: 0.5077278	total: 744ms	remaining: 4.46s
    143:	learn: 0.5066673	total: 746ms	remaining: 4.43s
    144:	learn: 0.5058345	total: 747ms	remaining: 4.41s
    145:	learn: 0.5047942	total: 748ms	remaining: 4.38s
    146:	learn: 0.5038261	total: 750ms	remaining: 4.35s
    147:	learn: 0.5025643	total: 752ms	remaining: 4.33s
    148:	learn: 0.5017820	total: 753ms	remaining: 4.3s
    149:	learn: 0.5007986	total: 754ms	remaining: 4.28s
    150:	learn: 0.4999881	total: 756ms	remaining: 4.25s
    151:	learn: 0.4991027	total: 757ms	remaining: 4.22s
    152:	learn: 0.4981344	total: 758ms	remaining: 4.2s
    153:	learn: 0.4973197	total: 759ms	remaining: 4.17s
    154:	learn: 0.4963139	total: 760ms	remaining: 4.14s
    155:	learn: 0.4954750	total: 762ms	remaining: 4.12s
    156:	learn: 0.4945960	total: 763ms	remaining: 4.09s
    157:	learn: 0.4935981	total: 765ms	remaining: 4.07s
    158:	learn: 0.4924508	total: 766ms	remaining: 4.05s
    159:	learn: 0.4916743	total: 767ms	remaining: 4.03s
    160:	learn: 0.4908338	total: 768ms	remaining: 4s
    161:	learn: 0.4898612	total: 770ms	remaining: 3.98s
    162:	learn: 0.4889968	total: 771ms	remaining: 3.96s
    163:	learn: 0.4879872	total: 772ms	remaining: 3.94s
    164:	learn: 0.4869989	total: 773ms	remaining: 3.91s
    165:	learn: 0.4861378	total: 775ms	remaining: 3.89s
    166:	learn: 0.4855161	total: 776ms	remaining: 3.87s
    167:	learn: 0.4846660	total: 778ms	remaining: 3.85s
    168:	learn: 0.4838519	total: 780ms	remaining: 3.83s
    169:	learn: 0.4829264	total: 781ms	remaining: 3.81s
    170:	learn: 0.4819107	total: 782ms	remaining: 3.79s
    171:	learn: 0.4811528	total: 783ms	remaining: 3.77s
    172:	learn: 0.4802587	total: 785ms	remaining: 3.75s
    173:	learn: 0.4794931	total: 786ms	remaining: 3.73s
    174:	learn: 0.4785975	total: 787ms	remaining: 3.71s
    175:	learn: 0.4777623	total: 788ms	remaining: 3.69s
    176:	learn: 0.4769111	total: 790ms	remaining: 3.67s
    177:	learn: 0.4760671	total: 792ms	remaining: 3.66s
    178:	learn: 0.4753627	total: 793ms	remaining: 3.64s
    179:	learn: 0.4745071	total: 794ms	remaining: 3.62s
    180:	learn: 0.4736382	total: 810ms	remaining: 3.67s
    181:	learn: 0.4728406	total: 812ms	remaining: 3.65s
    182:	learn: 0.4717846	total: 813ms	remaining: 3.63s
    183:	learn: 0.4709508	total: 815ms	remaining: 3.61s
    184:	learn: 0.4703161	total: 831ms	remaining: 3.66s
    185:	learn: 0.4695485	total: 840ms	remaining: 3.68s
    186:	learn: 0.4688131	total: 843ms	remaining: 3.66s
    187:	learn: 0.4678114	total: 847ms	remaining: 3.66s
    188:	learn: 0.4669940	total: 851ms	remaining: 3.65s
    189:	learn: 0.4663793	total: 857ms	remaining: 3.65s
    190:	learn: 0.4658176	total: 864ms	remaining: 3.66s
    191:	learn: 0.4649109	total: 871ms	remaining: 3.66s
    192:	learn: 0.4641905	total: 881ms	remaining: 3.69s
    193:	learn: 0.4634401	total: 885ms	remaining: 3.68s
    194:	learn: 0.4624081	total: 896ms	remaining: 3.7s
    195:	learn: 0.4615176	total: 898ms	remaining: 3.68s
    196:	learn: 0.4608337	total: 902ms	remaining: 3.68s
    197:	learn: 0.4599976	total: 907ms	remaining: 3.67s
    198:	learn: 0.4591047	total: 909ms	remaining: 3.66s
    199:	learn: 0.4584569	total: 911ms	remaining: 3.64s
    200:	learn: 0.4578395	total: 913ms	remaining: 3.63s
    201:	learn: 0.4571071	total: 914ms	remaining: 3.61s
    202:	learn: 0.4562983	total: 919ms	remaining: 3.61s
    203:	learn: 0.4556219	total: 926ms	remaining: 3.61s
    204:	learn: 0.4547996	total: 927ms	remaining: 3.59s
    205:	learn: 0.4540717	total: 928ms	remaining: 3.58s
    206:	learn: 0.4532668	total: 930ms	remaining: 3.56s
    207:	learn: 0.4525894	total: 938ms	remaining: 3.57s
    208:	learn: 0.4516051	total: 939ms	remaining: 3.55s
    209:	learn: 0.4508773	total: 940ms	remaining: 3.54s
    210:	learn: 0.4501685	total: 958ms	remaining: 3.58s
    211:	learn: 0.4493660	total: 959ms	remaining: 3.57s
    212:	learn: 0.4487884	total: 961ms	remaining: 3.55s
    213:	learn: 0.4477990	total: 962ms	remaining: 3.53s
    214:	learn: 0.4470603	total: 970ms	remaining: 3.54s
    215:	learn: 0.4463550	total: 979ms	remaining: 3.55s
    216:	learn: 0.4455783	total: 995ms	remaining: 3.59s
    217:	learn: 0.4449971	total: 1s	remaining: 3.59s
    218:	learn: 0.4443250	total: 1s	remaining: 3.58s
    219:	learn: 0.4434473	total: 1.01s	remaining: 3.57s
    220:	learn: 0.4429290	total: 1.01s	remaining: 3.57s
    221:	learn: 0.4422166	total: 1.01s	remaining: 3.56s
    222:	learn: 0.4417268	total: 1.02s	remaining: 3.55s
    223:	learn: 0.4410274	total: 1.02s	remaining: 3.54s
    224:	learn: 0.4402457	total: 1.03s	remaining: 3.54s
    225:	learn: 0.4395826	total: 1.03s	remaining: 3.53s
    226:	learn: 0.4390065	total: 1.03s	remaining: 3.52s
    227:	learn: 0.4384387	total: 1.04s	remaining: 3.52s
    228:	learn: 0.4375899	total: 1.04s	remaining: 3.51s
    229:	learn: 0.4368810	total: 1.05s	remaining: 3.51s
    230:	learn: 0.4361623	total: 1.05s	remaining: 3.5s
    231:	learn: 0.4353733	total: 1.06s	remaining: 3.5s
    232:	learn: 0.4346762	total: 1.06s	remaining: 3.49s
    233:	learn: 0.4342142	total: 1.06s	remaining: 3.48s
    234:	learn: 0.4336200	total: 1.07s	remaining: 3.48s
    235:	learn: 0.4327963	total: 1.07s	remaining: 3.48s
    236:	learn: 0.4320818	total: 1.08s	remaining: 3.47s
    237:	learn: 0.4313628	total: 1.08s	remaining: 3.46s
    238:	learn: 0.4306946	total: 1.08s	remaining: 3.45s
    239:	learn: 0.4300466	total: 1.09s	remaining: 3.44s
    240:	learn: 0.4293394	total: 1.09s	remaining: 3.44s
    241:	learn: 0.4285010	total: 1.09s	remaining: 3.43s
    242:	learn: 0.4279960	total: 1.1s	remaining: 3.43s
    243:	learn: 0.4273805	total: 1.1s	remaining: 3.42s
    244:	learn: 0.4265464	total: 1.11s	remaining: 3.42s
    245:	learn: 0.4259914	total: 1.11s	remaining: 3.41s
    246:	learn: 0.4251848	total: 1.11s	remaining: 3.4s
    247:	learn: 0.4245760	total: 1.12s	remaining: 3.4s
    248:	learn: 0.4238835	total: 1.12s	remaining: 3.39s
    249:	learn: 0.4232601	total: 1.13s	remaining: 3.38s
    250:	learn: 0.4226900	total: 1.13s	remaining: 3.38s
    251:	learn: 0.4219728	total: 1.14s	remaining: 3.37s
    252:	learn: 0.4213673	total: 1.14s	remaining: 3.36s
    253:	learn: 0.4206479	total: 1.14s	remaining: 3.36s
    254:	learn: 0.4200714	total: 1.15s	remaining: 3.35s
    255:	learn: 0.4191414	total: 1.17s	remaining: 3.4s
    256:	learn: 0.4185875	total: 1.17s	remaining: 3.39s
    257:	learn: 0.4177696	total: 1.18s	remaining: 3.38s
    258:	learn: 0.4172747	total: 1.18s	remaining: 3.38s
    259:	learn: 0.4167096	total: 1.18s	remaining: 3.37s
    260:	learn: 0.4160081	total: 1.19s	remaining: 3.37s
    261:	learn: 0.4153119	total: 1.19s	remaining: 3.36s
    262:	learn: 0.4148252	total: 1.2s	remaining: 3.35s
    263:	learn: 0.4142162	total: 1.2s	remaining: 3.35s
    264:	learn: 0.4136207	total: 1.2s	remaining: 3.34s
    265:	learn: 0.4128804	total: 1.21s	remaining: 3.33s
    266:	learn: 0.4121797	total: 1.22s	remaining: 3.34s
    267:	learn: 0.4116524	total: 1.23s	remaining: 3.35s
    268:	learn: 0.4109866	total: 1.23s	remaining: 3.33s
    269:	learn: 0.4103362	total: 1.23s	remaining: 3.34s
    270:	learn: 0.4097170	total: 1.24s	remaining: 3.33s
    271:	learn: 0.4090302	total: 1.24s	remaining: 3.33s
    272:	learn: 0.4085681	total: 1.25s	remaining: 3.32s
    273:	learn: 0.4080054	total: 1.25s	remaining: 3.31s
    274:	learn: 0.4073489	total: 1.25s	remaining: 3.31s
    275:	learn: 0.4068724	total: 1.26s	remaining: 3.3s
    276:	learn: 0.4063424	total: 1.26s	remaining: 3.29s
    277:	learn: 0.4056728	total: 1.26s	remaining: 3.29s
    278:	learn: 0.4049894	total: 1.27s	remaining: 3.28s
    279:	learn: 0.4044938	total: 1.27s	remaining: 3.27s
    280:	learn: 0.4038993	total: 1.28s	remaining: 3.27s
    281:	learn: 0.4031474	total: 1.28s	remaining: 3.26s
    282:	learn: 0.4025591	total: 1.28s	remaining: 3.26s
    283:	learn: 0.4019257	total: 1.29s	remaining: 3.25s
    284:	learn: 0.4015937	total: 1.29s	remaining: 3.24s
    285:	learn: 0.4011566	total: 1.29s	remaining: 3.23s
    286:	learn: 0.4005165	total: 1.3s	remaining: 3.23s
    287:	learn: 0.3999882	total: 1.3s	remaining: 3.22s
    288:	learn: 0.3992051	total: 1.31s	remaining: 3.22s
    289:	learn: 0.3984955	total: 1.31s	remaining: 3.21s
    290:	learn: 0.3979636	total: 1.31s	remaining: 3.21s
    291:	learn: 0.3972740	total: 1.33s	remaining: 3.23s
    292:	learn: 0.3967546	total: 1.33s	remaining: 3.22s
    293:	learn: 0.3961513	total: 1.34s	remaining: 3.22s
    294:	learn: 0.3954874	total: 1.35s	remaining: 3.22s
    295:	learn: 0.3948472	total: 1.35s	remaining: 3.21s
    296:	learn: 0.3943297	total: 1.35s	remaining: 3.2s
    297:	learn: 0.3938217	total: 1.36s	remaining: 3.2s
    298:	learn: 0.3931685	total: 1.36s	remaining: 3.19s
    299:	learn: 0.3925345	total: 1.36s	remaining: 3.18s
    300:	learn: 0.3920940	total: 1.36s	remaining: 3.17s
    301:	learn: 0.3915255	total: 1.37s	remaining: 3.17s
    302:	learn: 0.3908212	total: 1.37s	remaining: 3.16s
    303:	learn: 0.3902699	total: 1.38s	remaining: 3.15s
    304:	learn: 0.3897762	total: 1.38s	remaining: 3.14s
    305:	learn: 0.3892569	total: 1.38s	remaining: 3.13s
    306:	learn: 0.3887556	total: 1.39s	remaining: 3.14s
    307:	learn: 0.3882112	total: 1.39s	remaining: 3.13s
    308:	learn: 0.3875177	total: 1.39s	remaining: 3.11s
    309:	learn: 0.3868626	total: 1.39s	remaining: 3.1s
    310:	learn: 0.3863909	total: 1.4s	remaining: 3.1s
    311:	learn: 0.3857040	total: 1.4s	remaining: 3.09s
    312:	learn: 0.3850242	total: 1.41s	remaining: 3.08s
    313:	learn: 0.3843214	total: 1.41s	remaining: 3.07s
    314:	learn: 0.3838761	total: 1.41s	remaining: 3.06s
    315:	learn: 0.3833891	total: 1.42s	remaining: 3.06s
    316:	learn: 0.3827665	total: 1.42s	remaining: 3.06s
    317:	learn: 0.3823744	total: 1.42s	remaining: 3.04s
    318:	learn: 0.3818037	total: 1.42s	remaining: 3.04s
    319:	learn: 0.3813530	total: 1.43s	remaining: 3.03s
    320:	learn: 0.3808996	total: 1.43s	remaining: 3.03s
    321:	learn: 0.3803609	total: 1.44s	remaining: 3.02s
    322:	learn: 0.3796278	total: 1.44s	remaining: 3.02s
    323:	learn: 0.3791932	total: 1.45s	remaining: 3.02s
    324:	learn: 0.3784559	total: 1.45s	remaining: 3.02s
    325:	learn: 0.3778514	total: 1.47s	remaining: 3.03s
    326:	learn: 0.3774335	total: 1.47s	remaining: 3.02s
    327:	learn: 0.3768639	total: 1.48s	remaining: 3.02s
    328:	learn: 0.3764818	total: 1.48s	remaining: 3.01s
    329:	learn: 0.3759251	total: 1.48s	remaining: 3s
    330:	learn: 0.3753949	total: 1.48s	remaining: 3s
    331:	learn: 0.3750183	total: 1.49s	remaining: 3s
    332:	learn: 0.3745771	total: 1.5s	remaining: 2.99s
    333:	learn: 0.3740987	total: 1.5s	remaining: 2.99s
    334:	learn: 0.3736224	total: 1.5s	remaining: 2.98s
    335:	learn: 0.3731486	total: 1.5s	remaining: 2.96s
    336:	learn: 0.3726451	total: 1.5s	remaining: 2.95s
    337:	learn: 0.3721202	total: 1.5s	remaining: 2.94s
    338:	learn: 0.3717141	total: 1.5s	remaining: 2.93s
    339:	learn: 0.3712915	total: 1.5s	remaining: 2.92s
    340:	learn: 0.3707742	total: 1.52s	remaining: 2.93s
    341:	learn: 0.3703292	total: 1.52s	remaining: 2.92s
    342:	learn: 0.3698128	total: 1.53s	remaining: 2.92s
    343:	learn: 0.3691610	total: 1.53s	remaining: 2.92s
    344:	learn: 0.3685646	total: 1.53s	remaining: 2.91s
    345:	learn: 0.3680540	total: 1.54s	remaining: 2.91s
    346:	learn: 0.3675064	total: 1.55s	remaining: 2.92s
    347:	learn: 0.3668875	total: 1.55s	remaining: 2.91s
    348:	learn: 0.3665062	total: 1.55s	remaining: 2.9s
    349:	learn: 0.3661266	total: 1.56s	remaining: 2.91s
    350:	learn: 0.3656196	total: 1.57s	remaining: 2.9s
    351:	learn: 0.3650219	total: 1.57s	remaining: 2.89s
    352:	learn: 0.3645383	total: 1.57s	remaining: 2.89s
    353:	learn: 0.3641725	total: 1.58s	remaining: 2.88s
    354:	learn: 0.3635423	total: 1.59s	remaining: 2.9s
    355:	learn: 0.3630410	total: 1.6s	remaining: 2.9s
    356:	learn: 0.3624437	total: 1.61s	remaining: 2.9s
    357:	learn: 0.3620870	total: 1.63s	remaining: 2.92s
    358:	learn: 0.3615205	total: 1.64s	remaining: 2.92s
    359:	learn: 0.3611226	total: 1.65s	remaining: 2.93s
    360:	learn: 0.3605546	total: 1.65s	remaining: 2.92s
    361:	learn: 0.3600744	total: 1.66s	remaining: 2.92s
    362:	learn: 0.3595042	total: 1.66s	remaining: 2.92s
    363:	learn: 0.3590617	total: 1.67s	remaining: 2.91s
    364:	learn: 0.3586016	total: 1.67s	remaining: 2.9s
    365:	learn: 0.3580611	total: 1.68s	remaining: 2.91s
    366:	learn: 0.3574802	total: 1.68s	remaining: 2.9s
    367:	learn: 0.3570567	total: 1.69s	remaining: 2.9s
    368:	learn: 0.3565087	total: 1.7s	remaining: 2.9s
    369:	learn: 0.3559342	total: 1.71s	remaining: 2.91s
    370:	learn: 0.3554306	total: 1.71s	remaining: 2.9s
    371:	learn: 0.3548665	total: 1.72s	remaining: 2.9s
    372:	learn: 0.3543744	total: 1.72s	remaining: 2.9s
    373:	learn: 0.3539497	total: 1.73s	remaining: 2.89s
    374:	learn: 0.3535974	total: 1.73s	remaining: 2.88s
    375:	learn: 0.3530800	total: 1.73s	remaining: 2.87s
    376:	learn: 0.3527574	total: 1.73s	remaining: 2.86s
    377:	learn: 0.3521453	total: 1.73s	remaining: 2.85s
    378:	learn: 0.3518922	total: 1.73s	remaining: 2.84s
    379:	learn: 0.3515252	total: 1.73s	remaining: 2.83s
    380:	learn: 0.3511527	total: 1.74s	remaining: 2.82s
    381:	learn: 0.3506439	total: 1.74s	remaining: 2.81s
    382:	learn: 0.3501740	total: 1.74s	remaining: 2.8s
    383:	learn: 0.3498193	total: 1.74s	remaining: 2.79s
    384:	learn: 0.3494429	total: 1.74s	remaining: 2.78s
    385:	learn: 0.3489123	total: 1.74s	remaining: 2.77s
    386:	learn: 0.3484813	total: 1.74s	remaining: 2.76s
    387:	learn: 0.3481483	total: 1.74s	remaining: 2.75s
    388:	learn: 0.3476793	total: 1.75s	remaining: 2.74s
    389:	learn: 0.3471910	total: 1.75s	remaining: 2.73s
    390:	learn: 0.3467060	total: 1.75s	remaining: 2.72s
    391:	learn: 0.3462851	total: 1.75s	remaining: 2.72s
    392:	learn: 0.3455595	total: 1.75s	remaining: 2.71s
    393:	learn: 0.3450566	total: 1.76s	remaining: 2.7s
    394:	learn: 0.3447797	total: 1.76s	remaining: 2.69s
    395:	learn: 0.3443234	total: 1.76s	remaining: 2.68s
    396:	learn: 0.3440525	total: 1.76s	remaining: 2.67s
    397:	learn: 0.3436570	total: 1.76s	remaining: 2.66s
    398:	learn: 0.3432502	total: 1.76s	remaining: 2.65s
    399:	learn: 0.3426712	total: 1.76s	remaining: 2.65s
    400:	learn: 0.3421830	total: 1.76s	remaining: 2.64s
    401:	learn: 0.3416369	total: 1.77s	remaining: 2.63s
    402:	learn: 0.3412002	total: 1.77s	remaining: 2.62s
    403:	learn: 0.3408398	total: 1.77s	remaining: 2.61s
    404:	learn: 0.3403203	total: 1.77s	remaining: 2.6s
    405:	learn: 0.3398938	total: 1.78s	remaining: 2.6s
    406:	learn: 0.3393590	total: 1.78s	remaining: 2.6s
    407:	learn: 0.3388617	total: 1.78s	remaining: 2.59s
    408:	learn: 0.3384443	total: 1.79s	remaining: 2.58s
    409:	learn: 0.3379450	total: 1.8s	remaining: 2.59s
    410:	learn: 0.3375037	total: 1.8s	remaining: 2.58s
    411:	learn: 0.3371684	total: 1.8s	remaining: 2.57s
    412:	learn: 0.3367914	total: 1.8s	remaining: 2.56s
    413:	learn: 0.3363510	total: 1.8s	remaining: 2.55s
    414:	learn: 0.3357567	total: 1.8s	remaining: 2.54s
    415:	learn: 0.3352301	total: 1.81s	remaining: 2.54s
    416:	learn: 0.3348729	total: 1.81s	remaining: 2.53s
    417:	learn: 0.3345361	total: 1.81s	remaining: 2.52s
    418:	learn: 0.3342598	total: 1.81s	remaining: 2.51s
    419:	learn: 0.3336054	total: 1.81s	remaining: 2.5s
    420:	learn: 0.3331483	total: 1.82s	remaining: 2.51s
    421:	learn: 0.3327198	total: 1.82s	remaining: 2.5s
    422:	learn: 0.3322434	total: 1.83s	remaining: 2.5s
    423:	learn: 0.3319115	total: 1.83s	remaining: 2.49s
    424:	learn: 0.3315882	total: 1.83s	remaining: 2.48s
    425:	learn: 0.3310203	total: 1.83s	remaining: 2.47s
    426:	learn: 0.3306779	total: 1.84s	remaining: 2.46s
    427:	learn: 0.3303787	total: 1.84s	remaining: 2.46s
    428:	learn: 0.3300041	total: 1.84s	remaining: 2.45s
    429:	learn: 0.3297152	total: 1.84s	remaining: 2.44s
    430:	learn: 0.3292932	total: 1.84s	remaining: 2.43s
    431:	learn: 0.3289150	total: 1.84s	remaining: 2.42s
    432:	learn: 0.3284996	total: 1.84s	remaining: 2.42s
    433:	learn: 0.3278760	total: 1.85s	remaining: 2.41s
    434:	learn: 0.3274604	total: 1.85s	remaining: 2.4s
    435:	learn: 0.3270393	total: 1.85s	remaining: 2.39s
    436:	learn: 0.3265952	total: 1.85s	remaining: 2.38s
    437:	learn: 0.3262615	total: 1.85s	remaining: 2.38s
    438:	learn: 0.3258411	total: 1.85s	remaining: 2.37s
    439:	learn: 0.3254365	total: 1.85s	remaining: 2.36s
    440:	learn: 0.3250712	total: 1.86s	remaining: 2.35s
    441:	learn: 0.3246904	total: 1.86s	remaining: 2.35s
    442:	learn: 0.3242511	total: 1.86s	remaining: 2.34s
    443:	learn: 0.3237705	total: 1.86s	remaining: 2.33s
    444:	learn: 0.3234198	total: 1.86s	remaining: 2.32s
    445:	learn: 0.3229582	total: 1.86s	remaining: 2.31s
    446:	learn: 0.3225868	total: 1.86s	remaining: 2.31s
    447:	learn: 0.3222503	total: 1.87s	remaining: 2.3s
    448:	learn: 0.3219284	total: 1.87s	remaining: 2.3s
    449:	learn: 0.3216397	total: 1.88s	remaining: 2.29s
    450:	learn: 0.3212378	total: 1.88s	remaining: 2.28s
    451:	learn: 0.3208276	total: 1.88s	remaining: 2.28s
    452:	learn: 0.3204117	total: 1.89s	remaining: 2.28s
    453:	learn: 0.3200407	total: 1.89s	remaining: 2.27s
    454:	learn: 0.3196164	total: 1.89s	remaining: 2.26s
    455:	learn: 0.3192852	total: 1.89s	remaining: 2.25s
    456:	learn: 0.3190313	total: 1.89s	remaining: 2.25s
    457:	learn: 0.3186540	total: 1.9s	remaining: 2.25s
    458:	learn: 0.3182957	total: 1.9s	remaining: 2.24s
    459:	learn: 0.3179742	total: 1.9s	remaining: 2.23s
    460:	learn: 0.3176397	total: 1.9s	remaining: 2.23s
    461:	learn: 0.3172729	total: 1.91s	remaining: 2.22s
    462:	learn: 0.3169007	total: 1.91s	remaining: 2.21s
    463:	learn: 0.3166042	total: 1.91s	remaining: 2.21s
    464:	learn: 0.3160966	total: 1.91s	remaining: 2.2s
    465:	learn: 0.3158178	total: 1.91s	remaining: 2.19s
    466:	learn: 0.3152804	total: 1.92s	remaining: 2.19s
    467:	learn: 0.3149591	total: 1.92s	remaining: 2.18s
    468:	learn: 0.3143903	total: 1.92s	remaining: 2.17s
    469:	learn: 0.3140874	total: 1.92s	remaining: 2.16s
    470:	learn: 0.3135009	total: 1.92s	remaining: 2.16s
    471:	learn: 0.3131499	total: 1.94s	remaining: 2.17s
    472:	learn: 0.3127816	total: 1.94s	remaining: 2.16s
    473:	learn: 0.3124635	total: 1.94s	remaining: 2.15s
    474:	learn: 0.3120747	total: 1.94s	remaining: 2.15s
    475:	learn: 0.3117447	total: 1.94s	remaining: 2.14s
    476:	learn: 0.3114109	total: 1.94s	remaining: 2.13s
    477:	learn: 0.3110358	total: 1.95s	remaining: 2.12s
    478:	learn: 0.3107500	total: 1.95s	remaining: 2.12s
    479:	learn: 0.3104440	total: 1.95s	remaining: 2.11s
    480:	learn: 0.3101247	total: 1.95s	remaining: 2.1s
    481:	learn: 0.3098487	total: 1.95s	remaining: 2.1s
    482:	learn: 0.3094179	total: 1.95s	remaining: 2.09s
    483:	learn: 0.3089158	total: 1.96s	remaining: 2.09s
    484:	learn: 0.3085657	total: 1.96s	remaining: 2.08s
    485:	learn: 0.3081013	total: 1.97s	remaining: 2.08s
    486:	learn: 0.3076811	total: 1.97s	remaining: 2.07s
    487:	learn: 0.3073807	total: 1.97s	remaining: 2.07s
    488:	learn: 0.3070515	total: 1.97s	remaining: 2.06s
    489:	learn: 0.3066399	total: 1.97s	remaining: 2.05s
    490:	learn: 0.3064076	total: 1.97s	remaining: 2.05s
    491:	learn: 0.3060208	total: 1.98s	remaining: 2.04s
    492:	learn: 0.3057685	total: 1.98s	remaining: 2.03s
    493:	learn: 0.3053461	total: 1.98s	remaining: 2.03s
    494:	learn: 0.3050652	total: 1.98s	remaining: 2.02s
    495:	learn: 0.3045929	total: 1.98s	remaining: 2.01s
    496:	learn: 0.3043601	total: 1.98s	remaining: 2.01s
    497:	learn: 0.3039781	total: 1.98s	remaining: 2s
    498:	learn: 0.3037515	total: 1.99s	remaining: 1.99s
    499:	learn: 0.3033919	total: 1.99s	remaining: 1.99s
    500:	learn: 0.3031056	total: 2s	remaining: 1.99s
    501:	learn: 0.3027835	total: 2s	remaining: 1.98s
    502:	learn: 0.3025286	total: 2.01s	remaining: 1.99s
    503:	learn: 0.3022478	total: 2.01s	remaining: 1.98s
    504:	learn: 0.3019370	total: 2.02s	remaining: 1.98s
    505:	learn: 0.3016408	total: 2.02s	remaining: 1.97s
    506:	learn: 0.3012854	total: 2.02s	remaining: 1.96s
    507:	learn: 0.3008095	total: 2.02s	remaining: 1.96s
    508:	learn: 0.3005145	total: 2.03s	remaining: 1.96s
    509:	learn: 0.3000170	total: 2.03s	remaining: 1.95s
    510:	learn: 0.2997095	total: 2.03s	remaining: 1.95s
    511:	learn: 0.2993748	total: 2.04s	remaining: 1.94s
    512:	learn: 0.2988376	total: 2.04s	remaining: 1.94s
    513:	learn: 0.2985508	total: 2.04s	remaining: 1.93s
    514:	learn: 0.2981663	total: 2.04s	remaining: 1.93s
    515:	learn: 0.2978169	total: 2.06s	remaining: 1.93s
    516:	learn: 0.2973984	total: 2.06s	remaining: 1.92s
    517:	learn: 0.2970141	total: 2.06s	remaining: 1.92s
    518:	learn: 0.2966780	total: 2.07s	remaining: 1.92s
    519:	learn: 0.2964423	total: 2.07s	remaining: 1.91s
    520:	learn: 0.2961084	total: 2.08s	remaining: 1.91s
    521:	learn: 0.2958728	total: 2.08s	remaining: 1.91s
    522:	learn: 0.2954674	total: 2.09s	remaining: 1.91s
    523:	learn: 0.2951104	total: 2.1s	remaining: 1.9s
    524:	learn: 0.2946468	total: 2.1s	remaining: 1.9s
    525:	learn: 0.2943505	total: 2.1s	remaining: 1.89s
    526:	learn: 0.2939055	total: 2.1s	remaining: 1.89s
    527:	learn: 0.2936309	total: 2.1s	remaining: 1.88s
    528:	learn: 0.2934518	total: 2.1s	remaining: 1.87s
    529:	learn: 0.2931494	total: 2.1s	remaining: 1.87s
    530:	learn: 0.2929032	total: 2.11s	remaining: 1.86s
    531:	learn: 0.2926552	total: 2.11s	remaining: 1.85s
    532:	learn: 0.2924043	total: 2.11s	remaining: 1.85s
    533:	learn: 0.2920858	total: 2.11s	remaining: 1.84s
    534:	learn: 0.2918330	total: 2.11s	remaining: 1.84s
    535:	learn: 0.2914877	total: 2.11s	remaining: 1.83s
    536:	learn: 0.2912009	total: 2.12s	remaining: 1.82s
    537:	learn: 0.2910066	total: 2.12s	remaining: 1.82s
    538:	learn: 0.2907739	total: 2.12s	remaining: 1.81s
    539:	learn: 0.2903020	total: 2.12s	remaining: 1.8s
    540:	learn: 0.2900493	total: 2.12s	remaining: 1.8s
    541:	learn: 0.2897481	total: 2.12s	remaining: 1.79s
    542:	learn: 0.2893427	total: 2.13s	remaining: 1.79s
    543:	learn: 0.2890919	total: 2.13s	remaining: 1.78s
    544:	learn: 0.2888003	total: 2.13s	remaining: 1.78s
    545:	learn: 0.2884345	total: 2.13s	remaining: 1.77s
    546:	learn: 0.2882733	total: 2.13s	remaining: 1.77s
    547:	learn: 0.2879764	total: 2.13s	remaining: 1.76s
    548:	learn: 0.2877696	total: 2.14s	remaining: 1.75s
    549:	learn: 0.2873989	total: 2.14s	remaining: 1.75s
    550:	learn: 0.2871171	total: 2.14s	remaining: 1.74s
    551:	learn: 0.2866705	total: 2.14s	remaining: 1.74s
    552:	learn: 0.2863974	total: 2.14s	remaining: 1.73s
    553:	learn: 0.2861067	total: 2.14s	remaining: 1.73s
    554:	learn: 0.2857859	total: 2.14s	remaining: 1.72s
    555:	learn: 0.2854859	total: 2.15s	remaining: 1.71s
    556:	learn: 0.2852679	total: 2.15s	remaining: 1.71s
    557:	learn: 0.2849359	total: 2.15s	remaining: 1.7s
    558:	learn: 0.2847324	total: 2.15s	remaining: 1.7s
    559:	learn: 0.2845192	total: 2.15s	remaining: 1.69s
    560:	learn: 0.2842733	total: 2.15s	remaining: 1.68s
    561:	learn: 0.2839620	total: 2.15s	remaining: 1.68s
    562:	learn: 0.2836300	total: 2.15s	remaining: 1.67s
    563:	learn: 0.2834520	total: 2.16s	remaining: 1.67s
    564:	learn: 0.2831288	total: 2.16s	remaining: 1.66s
    565:	learn: 0.2827007	total: 2.16s	remaining: 1.66s
    566:	learn: 0.2823925	total: 2.17s	remaining: 1.66s
    567:	learn: 0.2821461	total: 2.17s	remaining: 1.65s
    568:	learn: 0.2818215	total: 2.17s	remaining: 1.65s
    569:	learn: 0.2814931	total: 2.18s	remaining: 1.64s
    570:	learn: 0.2812420	total: 2.18s	remaining: 1.64s
    571:	learn: 0.2809906	total: 2.18s	remaining: 1.63s
    572:	learn: 0.2807537	total: 2.18s	remaining: 1.63s
    573:	learn: 0.2805616	total: 2.18s	remaining: 1.62s
    574:	learn: 0.2801565	total: 2.19s	remaining: 1.62s
    575:	learn: 0.2799680	total: 2.19s	remaining: 1.61s
    576:	learn: 0.2797373	total: 2.19s	remaining: 1.6s
    577:	learn: 0.2793498	total: 2.19s	remaining: 1.6s
    578:	learn: 0.2789605	total: 2.19s	remaining: 1.59s
    579:	learn: 0.2787969	total: 2.19s	remaining: 1.59s
    580:	learn: 0.2785321	total: 2.19s	remaining: 1.58s
    581:	learn: 0.2780051	total: 2.2s	remaining: 1.58s
    582:	learn: 0.2776566	total: 2.2s	remaining: 1.57s
    583:	learn: 0.2774483	total: 2.2s	remaining: 1.57s
    584:	learn: 0.2770369	total: 2.2s	remaining: 1.56s
    585:	learn: 0.2766620	total: 2.2s	remaining: 1.56s
    586:	learn: 0.2764250	total: 2.2s	remaining: 1.55s
    587:	learn: 0.2760114	total: 2.21s	remaining: 1.55s
    588:	learn: 0.2756307	total: 2.21s	remaining: 1.54s
    589:	learn: 0.2753289	total: 2.21s	remaining: 1.54s
    590:	learn: 0.2751285	total: 2.21s	remaining: 1.53s
    591:	learn: 0.2748705	total: 2.21s	remaining: 1.53s
    592:	learn: 0.2745051	total: 2.22s	remaining: 1.52s
    593:	learn: 0.2741735	total: 2.22s	remaining: 1.51s
    594:	learn: 0.2739014	total: 2.22s	remaining: 1.51s
    595:	learn: 0.2736385	total: 2.22s	remaining: 1.5s
    596:	learn: 0.2734127	total: 2.22s	remaining: 1.5s
    597:	learn: 0.2731807	total: 2.22s	remaining: 1.49s
    598:	learn: 0.2728641	total: 2.22s	remaining: 1.49s
    599:	learn: 0.2725570	total: 2.23s	remaining: 1.48s
    600:	learn: 0.2723018	total: 2.23s	remaining: 1.48s
    601:	learn: 0.2720885	total: 2.23s	remaining: 1.47s
    602:	learn: 0.2718826	total: 2.23s	remaining: 1.47s
    603:	learn: 0.2714747	total: 2.23s	remaining: 1.46s
    604:	learn: 0.2711734	total: 2.23s	remaining: 1.46s
    605:	learn: 0.2709040	total: 2.23s	remaining: 1.45s
    606:	learn: 0.2705919	total: 2.23s	remaining: 1.45s
    607:	learn: 0.2704178	total: 2.24s	remaining: 1.44s
    608:	learn: 0.2701672	total: 2.24s	remaining: 1.44s
    609:	learn: 0.2700293	total: 2.24s	remaining: 1.43s
    610:	learn: 0.2698435	total: 2.24s	remaining: 1.43s
    611:	learn: 0.2696007	total: 2.24s	remaining: 1.42s
    612:	learn: 0.2693789	total: 2.25s	remaining: 1.42s
    613:	learn: 0.2690995	total: 2.25s	remaining: 1.42s
    614:	learn: 0.2686902	total: 2.26s	remaining: 1.41s
    615:	learn: 0.2684898	total: 2.26s	remaining: 1.41s
    616:	learn: 0.2682029	total: 2.26s	remaining: 1.41s
    617:	learn: 0.2679550	total: 2.27s	remaining: 1.4s
    618:	learn: 0.2676066	total: 2.27s	remaining: 1.4s
    619:	learn: 0.2672769	total: 2.27s	remaining: 1.39s
    620:	learn: 0.2670360	total: 2.27s	remaining: 1.39s
    621:	learn: 0.2668332	total: 2.28s	remaining: 1.38s
    622:	learn: 0.2666559	total: 2.28s	remaining: 1.38s
    623:	learn: 0.2663714	total: 2.29s	remaining: 1.38s
    624:	learn: 0.2660770	total: 2.29s	remaining: 1.37s
    625:	learn: 0.2658893	total: 2.29s	remaining: 1.37s
    626:	learn: 0.2656841	total: 2.29s	remaining: 1.36s
    627:	learn: 0.2652961	total: 2.29s	remaining: 1.36s
    628:	learn: 0.2651700	total: 2.29s	remaining: 1.35s
    629:	learn: 0.2649374	total: 2.29s	remaining: 1.35s
    630:	learn: 0.2647148	total: 2.29s	remaining: 1.34s
    631:	learn: 0.2643154	total: 2.29s	remaining: 1.34s
    632:	learn: 0.2640574	total: 2.3s	remaining: 1.33s
    633:	learn: 0.2638080	total: 2.3s	remaining: 1.33s
    634:	learn: 0.2636168	total: 2.3s	remaining: 1.32s
    635:	learn: 0.2633490	total: 2.3s	remaining: 1.32s
    636:	learn: 0.2631305	total: 2.3s	remaining: 1.31s
    637:	learn: 0.2630712	total: 2.3s	remaining: 1.31s
    638:	learn: 0.2627850	total: 2.3s	remaining: 1.3s
    639:	learn: 0.2624751	total: 2.31s	remaining: 1.3s
    640:	learn: 0.2622236	total: 2.31s	remaining: 1.29s
    641:	learn: 0.2619213	total: 2.31s	remaining: 1.29s
    642:	learn: 0.2616492	total: 2.31s	remaining: 1.28s
    643:	learn: 0.2614568	total: 2.31s	remaining: 1.28s
    644:	learn: 0.2613670	total: 2.31s	remaining: 1.27s
    645:	learn: 0.2611115	total: 2.31s	remaining: 1.27s
    646:	learn: 0.2609564	total: 2.31s	remaining: 1.26s
    647:	learn: 0.2607256	total: 2.31s	remaining: 1.26s
    648:	learn: 0.2605420	total: 2.32s	remaining: 1.25s
    649:	learn: 0.2602660	total: 2.32s	remaining: 1.25s
    650:	learn: 0.2600614	total: 2.32s	remaining: 1.24s
    651:	learn: 0.2598172	total: 2.32s	remaining: 1.24s
    652:	learn: 0.2596006	total: 2.32s	remaining: 1.23s
    653:	learn: 0.2593483	total: 2.32s	remaining: 1.23s
    654:	learn: 0.2592057	total: 2.32s	remaining: 1.22s
    655:	learn: 0.2589099	total: 2.33s	remaining: 1.22s
    656:	learn: 0.2585904	total: 2.33s	remaining: 1.21s
    657:	learn: 0.2583868	total: 2.33s	remaining: 1.21s
    658:	learn: 0.2581116	total: 2.33s	remaining: 1.21s
    659:	learn: 0.2579014	total: 2.33s	remaining: 1.2s
    660:	learn: 0.2576405	total: 2.33s	remaining: 1.2s
    661:	learn: 0.2573270	total: 2.33s	remaining: 1.19s
    662:	learn: 0.2570407	total: 2.33s	remaining: 1.19s
    663:	learn: 0.2567864	total: 2.34s	remaining: 1.18s
    664:	learn: 0.2565548	total: 2.34s	remaining: 1.18s
    665:	learn: 0.2562714	total: 2.34s	remaining: 1.17s
    666:	learn: 0.2560116	total: 2.34s	remaining: 1.17s
    667:	learn: 0.2558739	total: 2.34s	remaining: 1.16s
    668:	learn: 0.2557269	total: 2.34s	remaining: 1.16s
    669:	learn: 0.2554629	total: 2.34s	remaining: 1.15s
    670:	learn: 0.2552458	total: 2.35s	remaining: 1.15s
    671:	learn: 0.2550157	total: 2.35s	remaining: 1.15s
    672:	learn: 0.2548639	total: 2.35s	remaining: 1.14s
    673:	learn: 0.2546102	total: 2.35s	remaining: 1.14s
    674:	learn: 0.2543343	total: 2.35s	remaining: 1.13s
    675:	learn: 0.2540210	total: 2.35s	remaining: 1.13s
    676:	learn: 0.2537275	total: 2.35s	remaining: 1.12s
    677:	learn: 0.2534741	total: 2.36s	remaining: 1.12s
    678:	learn: 0.2532918	total: 2.36s	remaining: 1.11s
    679:	learn: 0.2530151	total: 2.36s	remaining: 1.11s
    680:	learn: 0.2527868	total: 2.36s	remaining: 1.11s
    681:	learn: 0.2525205	total: 2.36s	remaining: 1.1s
    682:	learn: 0.2523683	total: 2.36s	remaining: 1.1s
    683:	learn: 0.2522223	total: 2.37s	remaining: 1.09s
    684:	learn: 0.2519187	total: 2.37s	remaining: 1.09s
    685:	learn: 0.2516991	total: 2.37s	remaining: 1.08s
    686:	learn: 0.2515226	total: 2.37s	remaining: 1.08s
    687:	learn: 0.2512730	total: 2.37s	remaining: 1.08s
    688:	learn: 0.2509817	total: 2.38s	remaining: 1.07s
    689:	learn: 0.2507159	total: 2.38s	remaining: 1.07s
    690:	learn: 0.2504417	total: 2.38s	remaining: 1.07s
    691:	learn: 0.2501636	total: 2.39s	remaining: 1.06s
    692:	learn: 0.2499261	total: 2.39s	remaining: 1.06s
    693:	learn: 0.2496931	total: 2.4s	remaining: 1.06s
    694:	learn: 0.2494738	total: 2.4s	remaining: 1.05s
    695:	learn: 0.2492935	total: 2.4s	remaining: 1.05s
    696:	learn: 0.2491214	total: 2.4s	remaining: 1.04s
    697:	learn: 0.2489561	total: 2.4s	remaining: 1.04s
    698:	learn: 0.2487961	total: 2.41s	remaining: 1.04s
    699:	learn: 0.2486266	total: 2.42s	remaining: 1.03s
    700:	learn: 0.2484346	total: 2.42s	remaining: 1.03s
    701:	learn: 0.2482791	total: 2.42s	remaining: 1.03s
    702:	learn: 0.2481713	total: 2.42s	remaining: 1.02s
    703:	learn: 0.2479744	total: 2.42s	remaining: 1.02s
    704:	learn: 0.2477075	total: 2.43s	remaining: 1.02s
    705:	learn: 0.2475398	total: 2.43s	remaining: 1.01s
    706:	learn: 0.2471808	total: 2.43s	remaining: 1.01s
    707:	learn: 0.2469103	total: 2.44s	remaining: 1s
    708:	learn: 0.2467567	total: 2.44s	remaining: 1s
    709:	learn: 0.2465977	total: 2.44s	remaining: 996ms
    710:	learn: 0.2463584	total: 2.44s	remaining: 994ms
    711:	learn: 0.2460927	total: 2.45s	remaining: 990ms
    712:	learn: 0.2458425	total: 2.45s	remaining: 985ms
    713:	learn: 0.2457384	total: 2.45s	remaining: 983ms
    714:	learn: 0.2456240	total: 2.46s	remaining: 980ms
    715:	learn: 0.2455112	total: 2.46s	remaining: 977ms
    716:	learn: 0.2452829	total: 2.47s	remaining: 975ms
    717:	learn: 0.2451011	total: 2.47s	remaining: 971ms
    718:	learn: 0.2449370	total: 2.47s	remaining: 966ms
    719:	learn: 0.2446350	total: 2.48s	remaining: 965ms
    720:	learn: 0.2444253	total: 2.48s	remaining: 961ms
    721:	learn: 0.2441967	total: 2.48s	remaining: 956ms
    722:	learn: 0.2439462	total: 2.48s	remaining: 952ms
    723:	learn: 0.2437795	total: 2.49s	remaining: 950ms
    724:	learn: 0.2435928	total: 2.49s	remaining: 946ms
    725:	learn: 0.2433732	total: 2.5s	remaining: 942ms
    726:	learn: 0.2432261	total: 2.5s	remaining: 938ms
    727:	learn: 0.2429580	total: 2.5s	remaining: 935ms
    728:	learn: 0.2427509	total: 2.5s	remaining: 930ms
    729:	learn: 0.2425872	total: 2.51s	remaining: 927ms
    730:	learn: 0.2423368	total: 2.51s	remaining: 923ms
    731:	learn: 0.2422223	total: 2.51s	remaining: 919ms
    732:	learn: 0.2420200	total: 2.51s	remaining: 915ms
    733:	learn: 0.2418382	total: 2.51s	remaining: 911ms
    734:	learn: 0.2416306	total: 2.51s	remaining: 907ms
    735:	learn: 0.2414405	total: 2.52s	remaining: 902ms
    736:	learn: 0.2411571	total: 2.52s	remaining: 899ms
    737:	learn: 0.2410485	total: 2.52s	remaining: 894ms
    738:	learn: 0.2408496	total: 2.52s	remaining: 890ms
    739:	learn: 0.2406600	total: 2.52s	remaining: 886ms
    740:	learn: 0.2405116	total: 2.52s	remaining: 883ms
    741:	learn: 0.2402926	total: 2.53s	remaining: 879ms
    742:	learn: 0.2401016	total: 2.53s	remaining: 875ms
    743:	learn: 0.2399393	total: 2.53s	remaining: 870ms
    744:	learn: 0.2397054	total: 2.53s	remaining: 866ms
    745:	learn: 0.2394991	total: 2.53s	remaining: 862ms
    746:	learn: 0.2392989	total: 2.53s	remaining: 858ms
    747:	learn: 0.2389871	total: 2.53s	remaining: 854ms
    748:	learn: 0.2387986	total: 2.54s	remaining: 850ms
    749:	learn: 0.2386173	total: 2.54s	remaining: 846ms
    750:	learn: 0.2383704	total: 2.54s	remaining: 844ms
    751:	learn: 0.2381506	total: 2.55s	remaining: 840ms
    752:	learn: 0.2378906	total: 2.55s	remaining: 836ms
    753:	learn: 0.2376180	total: 2.55s	remaining: 832ms
    754:	learn: 0.2374491	total: 2.55s	remaining: 827ms
    755:	learn: 0.2373265	total: 2.55s	remaining: 823ms
    756:	learn: 0.2370363	total: 2.55s	remaining: 819ms
    757:	learn: 0.2368880	total: 2.55s	remaining: 815ms
    758:	learn: 0.2366981	total: 2.56s	remaining: 811ms
    759:	learn: 0.2364798	total: 2.56s	remaining: 808ms
    760:	learn: 0.2362517	total: 2.56s	remaining: 804ms
    761:	learn: 0.2359464	total: 2.56s	remaining: 800ms
    762:	learn: 0.2357191	total: 2.56s	remaining: 796ms
    763:	learn: 0.2354328	total: 2.56s	remaining: 793ms
    764:	learn: 0.2352757	total: 2.57s	remaining: 790ms
    765:	learn: 0.2349964	total: 2.57s	remaining: 786ms
    766:	learn: 0.2347631	total: 2.58s	remaining: 783ms
    767:	learn: 0.2346358	total: 2.58s	remaining: 779ms
    768:	learn: 0.2344428	total: 2.58s	remaining: 775ms
    769:	learn: 0.2342801	total: 2.59s	remaining: 775ms
    770:	learn: 0.2340303	total: 2.6s	remaining: 771ms
    771:	learn: 0.2338215	total: 2.6s	remaining: 768ms
    772:	learn: 0.2336825	total: 2.6s	remaining: 764ms
    773:	learn: 0.2334978	total: 2.6s	remaining: 760ms
    774:	learn: 0.2332568	total: 2.6s	remaining: 756ms
    775:	learn: 0.2330695	total: 2.61s	remaining: 753ms
    776:	learn: 0.2328420	total: 2.61s	remaining: 749ms
    777:	learn: 0.2326463	total: 2.61s	remaining: 745ms
    778:	learn: 0.2324285	total: 2.61s	remaining: 741ms
    779:	learn: 0.2322408	total: 2.61s	remaining: 737ms
    780:	learn: 0.2320827	total: 2.61s	remaining: 733ms
    781:	learn: 0.2318909	total: 2.62s	remaining: 729ms
    782:	learn: 0.2317213	total: 2.62s	remaining: 725ms
    783:	learn: 0.2315256	total: 2.62s	remaining: 721ms
    784:	learn: 0.2312418	total: 2.63s	remaining: 721ms
    785:	learn: 0.2310051	total: 2.64s	remaining: 718ms
    786:	learn: 0.2308032	total: 2.65s	remaining: 717ms
    787:	learn: 0.2305683	total: 2.65s	remaining: 713ms
    788:	learn: 0.2302944	total: 2.66s	remaining: 712ms
    789:	learn: 0.2301519	total: 2.67s	remaining: 711ms
    790:	learn: 0.2299530	total: 2.68s	remaining: 708ms
    791:	learn: 0.2296603	total: 2.69s	remaining: 706ms
    792:	learn: 0.2295069	total: 2.69s	remaining: 703ms
    793:	learn: 0.2293821	total: 2.7s	remaining: 700ms
    794:	learn: 0.2290975	total: 2.7s	remaining: 697ms
    795:	learn: 0.2288538	total: 2.71s	remaining: 693ms
    796:	learn: 0.2285628	total: 2.71s	remaining: 690ms
    797:	learn: 0.2283870	total: 2.71s	remaining: 686ms
    798:	learn: 0.2281763	total: 2.71s	remaining: 682ms
    799:	learn: 0.2279953	total: 2.71s	remaining: 678ms
    800:	learn: 0.2278824	total: 2.71s	remaining: 674ms
    801:	learn: 0.2277192	total: 2.71s	remaining: 670ms
    802:	learn: 0.2275222	total: 2.71s	remaining: 666ms
    803:	learn: 0.2273585	total: 2.72s	remaining: 662ms
    804:	learn: 0.2271877	total: 2.72s	remaining: 658ms
    805:	learn: 0.2268745	total: 2.72s	remaining: 655ms
    806:	learn: 0.2267009	total: 2.72s	remaining: 651ms
    807:	learn: 0.2265323	total: 2.72s	remaining: 647ms
    808:	learn: 0.2264282	total: 2.73s	remaining: 644ms
    809:	learn: 0.2262549	total: 2.73s	remaining: 640ms
    810:	learn: 0.2259639	total: 2.73s	remaining: 636ms
    811:	learn: 0.2256980	total: 2.73s	remaining: 632ms
    812:	learn: 0.2254202	total: 2.73s	remaining: 629ms
    813:	learn: 0.2252304	total: 2.73s	remaining: 625ms
    814:	learn: 0.2251090	total: 2.73s	remaining: 621ms
    815:	learn: 0.2249010	total: 2.74s	remaining: 617ms
    816:	learn: 0.2248207	total: 2.74s	remaining: 613ms
    817:	learn: 0.2246411	total: 2.74s	remaining: 610ms
    818:	learn: 0.2243708	total: 2.74s	remaining: 606ms
    819:	learn: 0.2241768	total: 2.74s	remaining: 602ms
    820:	learn: 0.2239457	total: 2.75s	remaining: 600ms
    821:	learn: 0.2238689	total: 2.75s	remaining: 596ms
    822:	learn: 0.2237225	total: 2.75s	remaining: 592ms
    823:	learn: 0.2234584	total: 2.75s	remaining: 588ms
    824:	learn: 0.2233017	total: 2.76s	remaining: 585ms
    825:	learn: 0.2230684	total: 2.76s	remaining: 581ms
    826:	learn: 0.2228262	total: 2.76s	remaining: 577ms
    827:	learn: 0.2226481	total: 2.76s	remaining: 573ms
    828:	learn: 0.2223835	total: 2.76s	remaining: 570ms
    829:	learn: 0.2222006	total: 2.76s	remaining: 566ms
    830:	learn: 0.2220782	total: 2.77s	remaining: 562ms
    831:	learn: 0.2218848	total: 2.77s	remaining: 559ms
    832:	learn: 0.2216839	total: 2.77s	remaining: 555ms
    833:	learn: 0.2216123	total: 2.77s	remaining: 551ms
    834:	learn: 0.2214461	total: 2.77s	remaining: 547ms
    835:	learn: 0.2211622	total: 2.78s	remaining: 545ms
    836:	learn: 0.2209913	total: 2.78s	remaining: 541ms
    837:	learn: 0.2207848	total: 2.78s	remaining: 537ms
    838:	learn: 0.2206158	total: 2.78s	remaining: 534ms
    839:	learn: 0.2204463	total: 2.79s	remaining: 531ms
    840:	learn: 0.2203566	total: 2.79s	remaining: 528ms
    841:	learn: 0.2201947	total: 2.79s	remaining: 524ms
    842:	learn: 0.2199975	total: 2.79s	remaining: 520ms
    843:	learn: 0.2198509	total: 2.79s	remaining: 516ms
    844:	learn: 0.2196786	total: 2.79s	remaining: 513ms
    845:	learn: 0.2194827	total: 2.8s	remaining: 509ms
    846:	learn: 0.2193084	total: 2.8s	remaining: 505ms
    847:	learn: 0.2191912	total: 2.8s	remaining: 502ms
    848:	learn: 0.2189905	total: 2.8s	remaining: 498ms
    849:	learn: 0.2188308	total: 2.8s	remaining: 495ms
    850:	learn: 0.2185952	total: 2.8s	remaining: 491ms
    851:	learn: 0.2183468	total: 2.81s	remaining: 488ms
    852:	learn: 0.2181409	total: 2.81s	remaining: 485ms
    853:	learn: 0.2179586	total: 2.81s	remaining: 481ms
    854:	learn: 0.2178303	total: 2.81s	remaining: 477ms
    855:	learn: 0.2176006	total: 2.81s	remaining: 474ms
    856:	learn: 0.2173826	total: 2.82s	remaining: 470ms
    857:	learn: 0.2172477	total: 2.82s	remaining: 466ms
    858:	learn: 0.2170174	total: 2.82s	remaining: 463ms
    859:	learn: 0.2168186	total: 2.82s	remaining: 459ms
    860:	learn: 0.2166445	total: 2.82s	remaining: 456ms
    861:	learn: 0.2164725	total: 2.83s	remaining: 452ms
    862:	learn: 0.2162903	total: 2.83s	remaining: 449ms
    863:	learn: 0.2160317	total: 2.83s	remaining: 445ms
    864:	learn: 0.2158823	total: 2.83s	remaining: 442ms
    865:	learn: 0.2157189	total: 2.83s	remaining: 438ms
    866:	learn: 0.2155965	total: 2.83s	remaining: 434ms
    867:	learn: 0.2154727	total: 2.83s	remaining: 431ms
    868:	learn: 0.2154003	total: 2.83s	remaining: 427ms
    869:	learn: 0.2152012	total: 2.83s	remaining: 424ms
    870:	learn: 0.2149657	total: 2.84s	remaining: 420ms
    871:	learn: 0.2148844	total: 2.84s	remaining: 417ms
    872:	learn: 0.2147410	total: 2.84s	remaining: 413ms
    873:	learn: 0.2146393	total: 2.84s	remaining: 410ms
    874:	learn: 0.2145433	total: 2.84s	remaining: 406ms
    875:	learn: 0.2144104	total: 2.84s	remaining: 402ms
    876:	learn: 0.2142978	total: 2.84s	remaining: 399ms
    877:	learn: 0.2141552	total: 2.85s	remaining: 395ms
    878:	learn: 0.2140396	total: 2.85s	remaining: 392ms
    879:	learn: 0.2139503	total: 2.85s	remaining: 389ms
    880:	learn: 0.2137147	total: 2.85s	remaining: 385ms
    881:	learn: 0.2135135	total: 2.85s	remaining: 382ms
    882:	learn: 0.2133592	total: 2.85s	remaining: 378ms
    883:	learn: 0.2132602	total: 2.85s	remaining: 374ms
    884:	learn: 0.2130422	total: 2.85s	remaining: 371ms
    885:	learn: 0.2128580	total: 2.86s	remaining: 368ms
    886:	learn: 0.2126837	total: 2.86s	remaining: 364ms
    887:	learn: 0.2124325	total: 2.86s	remaining: 361ms
    888:	learn: 0.2124126	total: 2.86s	remaining: 357ms
    889:	learn: 0.2122193	total: 2.87s	remaining: 354ms
    890:	learn: 0.2119709	total: 2.87s	remaining: 351ms
    891:	learn: 0.2118794	total: 2.87s	remaining: 347ms
    892:	learn: 0.2117443	total: 2.87s	remaining: 344ms
    893:	learn: 0.2115188	total: 2.87s	remaining: 340ms
    894:	learn: 0.2113960	total: 2.87s	remaining: 337ms
    895:	learn: 0.2112128	total: 2.88s	remaining: 334ms
    896:	learn: 0.2110242	total: 2.88s	remaining: 331ms
    897:	learn: 0.2107495	total: 2.88s	remaining: 327ms
    898:	learn: 0.2105580	total: 2.88s	remaining: 324ms
    899:	learn: 0.2103457	total: 2.88s	remaining: 320ms
    900:	learn: 0.2101332	total: 2.88s	remaining: 317ms
    901:	learn: 0.2100149	total: 2.89s	remaining: 314ms
    902:	learn: 0.2099564	total: 2.89s	remaining: 310ms
    903:	learn: 0.2097801	total: 2.89s	remaining: 307ms
    904:	learn: 0.2096299	total: 2.89s	remaining: 303ms
    905:	learn: 0.2094628	total: 2.89s	remaining: 300ms
    906:	learn: 0.2093701	total: 2.89s	remaining: 297ms
    907:	learn: 0.2091394	total: 2.89s	remaining: 293ms
    908:	learn: 0.2090093	total: 2.9s	remaining: 290ms
    909:	learn: 0.2088169	total: 2.9s	remaining: 287ms
    910:	learn: 0.2086476	total: 2.9s	remaining: 283ms
    911:	learn: 0.2084487	total: 2.9s	remaining: 280ms
    912:	learn: 0.2083323	total: 2.9s	remaining: 276ms
    913:	learn: 0.2082169	total: 2.9s	remaining: 273ms
    914:	learn: 0.2080506	total: 2.9s	remaining: 270ms
    915:	learn: 0.2078259	total: 2.9s	remaining: 266ms
    916:	learn: 0.2076538	total: 2.9s	remaining: 263ms
    917:	learn: 0.2074735	total: 2.91s	remaining: 260ms
    918:	learn: 0.2073427	total: 2.91s	remaining: 256ms
    919:	learn: 0.2072668	total: 2.91s	remaining: 253ms
    920:	learn: 0.2071781	total: 2.91s	remaining: 250ms
    921:	learn: 0.2070540	total: 2.91s	remaining: 246ms
    922:	learn: 0.2069171	total: 2.91s	remaining: 243ms
    923:	learn: 0.2067301	total: 2.92s	remaining: 240ms
    924:	learn: 0.2065461	total: 2.92s	remaining: 236ms
    925:	learn: 0.2064036	total: 2.93s	remaining: 234ms
    926:	learn: 0.2062113	total: 2.93s	remaining: 231ms
    927:	learn: 0.2060842	total: 2.94s	remaining: 228ms
    928:	learn: 0.2058966	total: 2.94s	remaining: 224ms
    929:	learn: 0.2057157	total: 2.94s	remaining: 221ms
    930:	learn: 0.2056182	total: 2.94s	remaining: 218ms
    931:	learn: 0.2054450	total: 2.94s	remaining: 215ms
    932:	learn: 0.2052037	total: 2.94s	remaining: 211ms
    933:	learn: 0.2050580	total: 2.94s	remaining: 208ms
    934:	learn: 0.2048257	total: 2.94s	remaining: 205ms
    935:	learn: 0.2047219	total: 2.95s	remaining: 201ms
    936:	learn: 0.2046374	total: 2.95s	remaining: 198ms
    937:	learn: 0.2044376	total: 2.95s	remaining: 195ms
    938:	learn: 0.2043307	total: 2.95s	remaining: 192ms
    939:	learn: 0.2041968	total: 2.95s	remaining: 188ms
    940:	learn: 0.2039970	total: 2.95s	remaining: 185ms
    941:	learn: 0.2038299	total: 2.96s	remaining: 182ms
    942:	learn: 0.2037169	total: 2.96s	remaining: 179ms
    943:	learn: 0.2034858	total: 2.96s	remaining: 175ms
    944:	learn: 0.2034146	total: 2.96s	remaining: 172ms
    945:	learn: 0.2033033	total: 2.96s	remaining: 169ms
    946:	learn: 0.2030568	total: 2.96s	remaining: 166ms
    947:	learn: 0.2028564	total: 2.96s	remaining: 163ms
    948:	learn: 0.2027175	total: 2.97s	remaining: 159ms
    949:	learn: 0.2026020	total: 2.97s	remaining: 156ms
    950:	learn: 0.2024828	total: 2.97s	remaining: 153ms
    951:	learn: 0.2022904	total: 2.97s	remaining: 150ms
    952:	learn: 0.2021212	total: 2.98s	remaining: 147ms
    953:	learn: 0.2020026	total: 2.98s	remaining: 144ms
    954:	learn: 0.2018566	total: 2.99s	remaining: 141ms
    955:	learn: 0.2017614	total: 2.99s	remaining: 137ms
    956:	learn: 0.2016419	total: 2.99s	remaining: 134ms
    957:	learn: 0.2015527	total: 2.99s	remaining: 131ms
    958:	learn: 0.2014243	total: 2.99s	remaining: 128ms
    959:	learn: 0.2012074	total: 2.99s	remaining: 125ms
    960:	learn: 0.2011212	total: 2.99s	remaining: 121ms
    961:	learn: 0.2009440	total: 3s	remaining: 118ms
    962:	learn: 0.2008064	total: 3s	remaining: 115ms
    963:	learn: 0.2006254	total: 3s	remaining: 112ms
    964:	learn: 0.2004066	total: 3s	remaining: 109ms
    965:	learn: 0.2002880	total: 3s	remaining: 106ms
    966:	learn: 0.2001414	total: 3.01s	remaining: 103ms
    967:	learn: 0.2000786	total: 3.01s	remaining: 99.5ms
    968:	learn: 0.1998965	total: 3.01s	remaining: 96.3ms
    969:	learn: 0.1997146	total: 3.01s	remaining: 93.2ms
    970:	learn: 0.1995524	total: 3.02s	remaining: 90.2ms
    971:	learn: 0.1994197	total: 3.02s	remaining: 87ms
    972:	learn: 0.1992826	total: 3.02s	remaining: 83.9ms
    973:	learn: 0.1991081	total: 3.02s	remaining: 80.7ms
    974:	learn: 0.1989842	total: 3.02s	remaining: 77.6ms
    975:	learn: 0.1987844	total: 3.03s	remaining: 74.4ms
    976:	learn: 0.1986616	total: 3.03s	remaining: 71.3ms
    977:	learn: 0.1985143	total: 3.04s	remaining: 68.5ms
    978:	learn: 0.1983525	total: 3.05s	remaining: 65.5ms
    979:	learn: 0.1982284	total: 3.06s	remaining: 62.4ms
    980:	learn: 0.1980550	total: 3.07s	remaining: 59.4ms
    981:	learn: 0.1979119	total: 3.07s	remaining: 56.3ms
    982:	learn: 0.1978305	total: 3.08s	remaining: 53.3ms
    983:	learn: 0.1976788	total: 3.09s	remaining: 50.2ms
    984:	learn: 0.1974894	total: 3.09s	remaining: 47.1ms
    985:	learn: 0.1973861	total: 3.12s	remaining: 44.3ms
    986:	learn: 0.1972701	total: 3.13s	remaining: 41.2ms
    987:	learn: 0.1971236	total: 3.13s	remaining: 38.1ms
    988:	learn: 0.1970436	total: 3.14s	remaining: 34.9ms
    989:	learn: 0.1968167	total: 3.15s	remaining: 31.8ms
    990:	learn: 0.1966295	total: 3.15s	remaining: 28.7ms
    991:	learn: 0.1964678	total: 3.17s	remaining: 25.5ms
    992:	learn: 0.1963600	total: 3.17s	remaining: 22.3ms
    993:	learn: 0.1961994	total: 3.18s	remaining: 19.2ms
    994:	learn: 0.1960094	total: 3.19s	remaining: 16ms
    995:	learn: 0.1959085	total: 3.19s	remaining: 12.8ms
    996:	learn: 0.1957488	total: 3.19s	remaining: 9.61ms
    997:	learn: 0.1956462	total: 3.19s	remaining: 6.4ms
    998:	learn: 0.1955390	total: 3.2s	remaining: 3.2ms
    999:	learn: 0.1953857	total: 3.2s	remaining: 0us
    0:	learn: 0.6911383	total: 2.19ms	remaining: 2.19s
    1:	learn: 0.6885918	total: 5.66ms	remaining: 2.83s
    2:	learn: 0.6866163	total: 9.63ms	remaining: 3.2s
    3:	learn: 0.6843547	total: 11ms	remaining: 2.73s
    4:	learn: 0.6817839	total: 12.3ms	remaining: 2.45s
    5:	learn: 0.6797349	total: 13.6ms	remaining: 2.26s
    6:	learn: 0.6775883	total: 14.9ms	remaining: 2.12s
    7:	learn: 0.6757058	total: 16.2ms	remaining: 2s
    8:	learn: 0.6735491	total: 17.5ms	remaining: 1.93s
    9:	learn: 0.6719165	total: 18.8ms	remaining: 1.86s
    10:	learn: 0.6697978	total: 20.2ms	remaining: 1.81s
    11:	learn: 0.6679611	total: 22.5ms	remaining: 1.85s
    12:	learn: 0.6661096	total: 24.2ms	remaining: 1.84s
    13:	learn: 0.6638339	total: 25.5ms	remaining: 1.79s
    14:	learn: 0.6618357	total: 26.7ms	remaining: 1.75s
    15:	learn: 0.6597410	total: 30.3ms	remaining: 1.86s
    16:	learn: 0.6579417	total: 34.4ms	remaining: 1.99s
    17:	learn: 0.6561764	total: 37.4ms	remaining: 2.04s
    18:	learn: 0.6541464	total: 38.8ms	remaining: 2s
    19:	learn: 0.6527596	total: 40ms	remaining: 1.96s
    20:	learn: 0.6509464	total: 47.1ms	remaining: 2.2s
    21:	learn: 0.6491590	total: 48.4ms	remaining: 2.15s
    22:	learn: 0.6475575	total: 49.6ms	remaining: 2.11s
    23:	learn: 0.6458615	total: 50.9ms	remaining: 2.07s
    24:	learn: 0.6438979	total: 55ms	remaining: 2.15s
    25:	learn: 0.6419502	total: 61.2ms	remaining: 2.29s
    26:	learn: 0.6398815	total: 62.4ms	remaining: 2.25s
    27:	learn: 0.6381166	total: 63.6ms	remaining: 2.21s
    28:	learn: 0.6364274	total: 64.8ms	remaining: 2.17s
    29:	learn: 0.6344558	total: 72.1ms	remaining: 2.33s
    30:	learn: 0.6326463	total: 73.6ms	remaining: 2.3s
    31:	learn: 0.6306376	total: 74.9ms	remaining: 2.27s
    32:	learn: 0.6290271	total: 82.8ms	remaining: 2.42s
    33:	learn: 0.6274379	total: 84.1ms	remaining: 2.39s
    34:	learn: 0.6257987	total: 85.5ms	remaining: 2.36s
    35:	learn: 0.6240313	total: 92ms	remaining: 2.46s
    36:	learn: 0.6221788	total: 93.3ms	remaining: 2.43s
    37:	learn: 0.6203680	total: 94.5ms	remaining: 2.39s
    38:	learn: 0.6188713	total: 100ms	remaining: 2.47s
    39:	learn: 0.6171431	total: 101ms	remaining: 2.43s
    40:	learn: 0.6154516	total: 103ms	remaining: 2.4s
    41:	learn: 0.6139617	total: 109ms	remaining: 2.48s
    42:	learn: 0.6122283	total: 110ms	remaining: 2.45s
    43:	learn: 0.6106340	total: 111ms	remaining: 2.42s
    44:	learn: 0.6088500	total: 113ms	remaining: 2.39s
    45:	learn: 0.6074363	total: 114ms	remaining: 2.36s
    46:	learn: 0.6058891	total: 115ms	remaining: 2.33s
    47:	learn: 0.6043252	total: 116ms	remaining: 2.31s
    48:	learn: 0.6021648	total: 117ms	remaining: 2.28s
    49:	learn: 0.6007347	total: 127ms	remaining: 2.42s
    50:	learn: 0.5992056	total: 134ms	remaining: 2.5s
    51:	learn: 0.5978049	total: 136ms	remaining: 2.47s
    52:	learn: 0.5963823	total: 137ms	remaining: 2.45s
    53:	learn: 0.5949289	total: 138ms	remaining: 2.42s
    54:	learn: 0.5933712	total: 140ms	remaining: 2.4s
    55:	learn: 0.5915033	total: 141ms	remaining: 2.37s
    56:	learn: 0.5898480	total: 142ms	remaining: 2.35s
    57:	learn: 0.5886154	total: 143ms	remaining: 2.33s
    58:	learn: 0.5869565	total: 144ms	remaining: 2.3s
    59:	learn: 0.5851382	total: 152ms	remaining: 2.38s
    60:	learn: 0.5836114	total: 153ms	remaining: 2.36s
    61:	learn: 0.5822439	total: 154ms	remaining: 2.34s
    62:	learn: 0.5804612	total: 156ms	remaining: 2.31s
    63:	learn: 0.5784955	total: 157ms	remaining: 2.3s
    64:	learn: 0.5771415	total: 158ms	remaining: 2.28s
    65:	learn: 0.5753148	total: 160ms	remaining: 2.26s
    66:	learn: 0.5738886	total: 161ms	remaining: 2.24s
    67:	learn: 0.5725165	total: 167ms	remaining: 2.29s
    68:	learn: 0.5712178	total: 168ms	remaining: 2.27s
    69:	learn: 0.5699471	total: 174ms	remaining: 2.32s
    70:	learn: 0.5682378	total: 176ms	remaining: 2.3s
    71:	learn: 0.5667725	total: 177ms	remaining: 2.28s
    72:	learn: 0.5653399	total: 183ms	remaining: 2.32s
    73:	learn: 0.5637511	total: 184ms	remaining: 2.3s
    74:	learn: 0.5622062	total: 185ms	remaining: 2.29s
    75:	learn: 0.5608091	total: 187ms	remaining: 2.27s
    76:	learn: 0.5592533	total: 188ms	remaining: 2.26s
    77:	learn: 0.5578391	total: 190ms	remaining: 2.25s
    78:	learn: 0.5561123	total: 192ms	remaining: 2.23s
    79:	learn: 0.5546528	total: 193ms	remaining: 2.22s
    80:	learn: 0.5531912	total: 194ms	remaining: 2.2s
    81:	learn: 0.5515071	total: 195ms	remaining: 2.19s
    82:	learn: 0.5502822	total: 197ms	remaining: 2.17s
    83:	learn: 0.5488514	total: 198ms	remaining: 2.16s
    84:	learn: 0.5476948	total: 199ms	remaining: 2.15s
    85:	learn: 0.5462958	total: 201ms	remaining: 2.14s
    86:	learn: 0.5448845	total: 204ms	remaining: 2.14s
    87:	learn: 0.5434695	total: 228ms	remaining: 2.36s
    88:	learn: 0.5422303	total: 231ms	remaining: 2.36s
    89:	learn: 0.5408043	total: 242ms	remaining: 2.45s
    90:	learn: 0.5393050	total: 248ms	remaining: 2.48s
    91:	learn: 0.5379353	total: 252ms	remaining: 2.49s
    92:	learn: 0.5362748	total: 263ms	remaining: 2.57s
    93:	learn: 0.5347720	total: 278ms	remaining: 2.68s
    94:	learn: 0.5330830	total: 282ms	remaining: 2.69s
    95:	learn: 0.5317144	total: 284ms	remaining: 2.67s
    96:	learn: 0.5304413	total: 285ms	remaining: 2.65s
    97:	learn: 0.5291161	total: 286ms	remaining: 2.64s
    98:	learn: 0.5276380	total: 289ms	remaining: 2.63s
    99:	learn: 0.5261708	total: 290ms	remaining: 2.61s
    100:	learn: 0.5247355	total: 292ms	remaining: 2.6s
    101:	learn: 0.5237122	total: 321ms	remaining: 2.83s
    102:	learn: 0.5223774	total: 324ms	remaining: 2.82s
    103:	learn: 0.5210040	total: 326ms	remaining: 2.81s
    104:	learn: 0.5195823	total: 332ms	remaining: 2.83s
    105:	learn: 0.5182128	total: 348ms	remaining: 2.93s
    106:	learn: 0.5171010	total: 376ms	remaining: 3.13s
    107:	learn: 0.5154453	total: 377ms	remaining: 3.11s
    108:	learn: 0.5140762	total: 378ms	remaining: 3.09s
    109:	learn: 0.5128809	total: 379ms	remaining: 3.07s
    110:	learn: 0.5113136	total: 381ms	remaining: 3.05s
    111:	learn: 0.5099528	total: 392ms	remaining: 3.11s
    112:	learn: 0.5087099	total: 393ms	remaining: 3.09s
    113:	learn: 0.5075710	total: 407ms	remaining: 3.17s
    114:	learn: 0.5063746	total: 409ms	remaining: 3.14s
    115:	learn: 0.5052392	total: 410ms	remaining: 3.12s
    116:	learn: 0.5039981	total: 418ms	remaining: 3.16s
    117:	learn: 0.5027485	total: 428ms	remaining: 3.2s
    118:	learn: 0.5017238	total: 434ms	remaining: 3.21s
    119:	learn: 0.5005180	total: 443ms	remaining: 3.25s
    120:	learn: 0.4995321	total: 451ms	remaining: 3.28s
    121:	learn: 0.4982390	total: 453ms	remaining: 3.26s
    122:	learn: 0.4965852	total: 461ms	remaining: 3.28s
    123:	learn: 0.4955757	total: 462ms	remaining: 3.26s
    124:	learn: 0.4945564	total: 463ms	remaining: 3.24s
    125:	learn: 0.4934651	total: 464ms	remaining: 3.22s
    126:	learn: 0.4922702	total: 466ms	remaining: 3.2s
    127:	learn: 0.4911709	total: 467ms	remaining: 3.18s
    128:	learn: 0.4898813	total: 477ms	remaining: 3.22s
    129:	learn: 0.4889741	total: 478ms	remaining: 3.2s
    130:	learn: 0.4878748	total: 479ms	remaining: 3.18s
    131:	learn: 0.4868282	total: 490ms	remaining: 3.22s
    132:	learn: 0.4857749	total: 494ms	remaining: 3.22s
    133:	learn: 0.4849637	total: 499ms	remaining: 3.22s
    134:	learn: 0.4837374	total: 500ms	remaining: 3.2s
    135:	learn: 0.4827893	total: 501ms	remaining: 3.18s
    136:	learn: 0.4816063	total: 507ms	remaining: 3.19s
    137:	learn: 0.4805441	total: 516ms	remaining: 3.22s
    138:	learn: 0.4791551	total: 517ms	remaining: 3.2s
    139:	learn: 0.4780436	total: 523ms	remaining: 3.21s
    140:	learn: 0.4771532	total: 529ms	remaining: 3.22s
    141:	learn: 0.4758364	total: 535ms	remaining: 3.23s
    142:	learn: 0.4749786	total: 537ms	remaining: 3.22s
    143:	learn: 0.4737938	total: 538ms	remaining: 3.2s
    144:	learn: 0.4726401	total: 539ms	remaining: 3.18s
    145:	learn: 0.4716186	total: 546ms	remaining: 3.19s
    146:	learn: 0.4704294	total: 548ms	remaining: 3.18s
    147:	learn: 0.4692975	total: 549ms	remaining: 3.16s
    148:	learn: 0.4682360	total: 550ms	remaining: 3.14s
    149:	learn: 0.4670508	total: 551ms	remaining: 3.12s
    150:	learn: 0.4661076	total: 553ms	remaining: 3.11s
    151:	learn: 0.4652287	total: 554ms	remaining: 3.09s
    152:	learn: 0.4640738	total: 560ms	remaining: 3.1s
    153:	learn: 0.4630563	total: 566ms	remaining: 3.11s
    154:	learn: 0.4620782	total: 570ms	remaining: 3.1s
    155:	learn: 0.4608829	total: 571ms	remaining: 3.09s
    156:	learn: 0.4599296	total: 577ms	remaining: 3.1s
    157:	learn: 0.4591482	total: 579ms	remaining: 3.08s
    158:	learn: 0.4579964	total: 580ms	remaining: 3.07s
    159:	learn: 0.4570654	total: 581ms	remaining: 3.05s
    160:	learn: 0.4560071	total: 583ms	remaining: 3.04s
    161:	learn: 0.4551913	total: 584ms	remaining: 3.02s
    162:	learn: 0.4540082	total: 586ms	remaining: 3.01s
    163:	learn: 0.4528849	total: 588ms	remaining: 3s
    164:	learn: 0.4518167	total: 589ms	remaining: 2.98s
    165:	learn: 0.4507635	total: 591ms	remaining: 2.97s
    166:	learn: 0.4499163	total: 599ms	remaining: 2.99s
    167:	learn: 0.4483632	total: 600ms	remaining: 2.97s
    168:	learn: 0.4473301	total: 602ms	remaining: 2.96s
    169:	learn: 0.4463483	total: 603ms	remaining: 2.95s
    170:	learn: 0.4452950	total: 605ms	remaining: 2.93s
    171:	learn: 0.4444347	total: 606ms	remaining: 2.92s
    172:	learn: 0.4433858	total: 608ms	remaining: 2.9s
    173:	learn: 0.4424606	total: 609ms	remaining: 2.89s
    174:	learn: 0.4414629	total: 619ms	remaining: 2.92s
    175:	learn: 0.4406022	total: 620ms	remaining: 2.9s
    176:	learn: 0.4396689	total: 629ms	remaining: 2.92s
    177:	learn: 0.4387505	total: 630ms	remaining: 2.91s
    178:	learn: 0.4377719	total: 631ms	remaining: 2.89s
    179:	learn: 0.4369275	total: 632ms	remaining: 2.88s
    180:	learn: 0.4358432	total: 634ms	remaining: 2.87s
    181:	learn: 0.4348113	total: 640ms	remaining: 2.88s
    182:	learn: 0.4339665	total: 641ms	remaining: 2.86s
    183:	learn: 0.4330720	total: 643ms	remaining: 2.85s
    184:	learn: 0.4321389	total: 644ms	remaining: 2.84s
    185:	learn: 0.4311181	total: 653ms	remaining: 2.86s
    186:	learn: 0.4301325	total: 654ms	remaining: 2.85s
    187:	learn: 0.4291398	total: 656ms	remaining: 2.83s
    188:	learn: 0.4280697	total: 664ms	remaining: 2.85s
    189:	learn: 0.4270324	total: 665ms	remaining: 2.83s
    190:	learn: 0.4262547	total: 666ms	remaining: 2.82s
    191:	learn: 0.4253000	total: 668ms	remaining: 2.81s
    192:	learn: 0.4244767	total: 669ms	remaining: 2.8s
    193:	learn: 0.4234863	total: 670ms	remaining: 2.78s
    194:	learn: 0.4222637	total: 672ms	remaining: 2.77s
    195:	learn: 0.4212016	total: 676ms	remaining: 2.77s
    196:	learn: 0.4202589	total: 678ms	remaining: 2.76s
    197:	learn: 0.4191853	total: 679ms	remaining: 2.75s
    198:	learn: 0.4182935	total: 688ms	remaining: 2.77s
    199:	learn: 0.4172772	total: 690ms	remaining: 2.76s
    200:	learn: 0.4163435	total: 700ms	remaining: 2.78s
    201:	learn: 0.4152868	total: 701ms	remaining: 2.77s
    202:	learn: 0.4142657	total: 702ms	remaining: 2.76s
    203:	learn: 0.4134885	total: 703ms	remaining: 2.74s
    204:	learn: 0.4127018	total: 705ms	remaining: 2.73s
    205:	learn: 0.4116318	total: 712ms	remaining: 2.74s
    206:	learn: 0.4107109	total: 713ms	remaining: 2.73s
    207:	learn: 0.4096986	total: 714ms	remaining: 2.72s
    208:	learn: 0.4089435	total: 715ms	remaining: 2.71s
    209:	learn: 0.4079575	total: 717ms	remaining: 2.7s
    210:	learn: 0.4068538	total: 718ms	remaining: 2.69s
    211:	learn: 0.4059947	total: 724ms	remaining: 2.69s
    212:	learn: 0.4051675	total: 735ms	remaining: 2.71s
    213:	learn: 0.4044280	total: 736ms	remaining: 2.7s
    214:	learn: 0.4035601	total: 737ms	remaining: 2.69s
    215:	learn: 0.4027667	total: 738ms	remaining: 2.68s
    216:	learn: 0.4018758	total: 740ms	remaining: 2.67s
    217:	learn: 0.4008227	total: 752ms	remaining: 2.7s
    218:	learn: 0.3998099	total: 754ms	remaining: 2.69s
    219:	learn: 0.3988106	total: 763ms	remaining: 2.7s
    220:	learn: 0.3977956	total: 764ms	remaining: 2.69s
    221:	learn: 0.3970227	total: 770ms	remaining: 2.7s
    222:	learn: 0.3962992	total: 775ms	remaining: 2.7s
    223:	learn: 0.3955867	total: 781ms	remaining: 2.71s
    224:	learn: 0.3948700	total: 782ms	remaining: 2.69s
    225:	learn: 0.3939560	total: 791ms	remaining: 2.71s
    226:	learn: 0.3932772	total: 794ms	remaining: 2.7s
    227:	learn: 0.3925000	total: 796ms	remaining: 2.69s
    228:	learn: 0.3917208	total: 802ms	remaining: 2.7s
    229:	learn: 0.3910261	total: 811ms	remaining: 2.71s
    230:	learn: 0.3901264	total: 812ms	remaining: 2.7s
    231:	learn: 0.3893815	total: 813ms	remaining: 2.69s
    232:	learn: 0.3886140	total: 814ms	remaining: 2.68s
    233:	learn: 0.3878760	total: 816ms	remaining: 2.67s
    234:	learn: 0.3869758	total: 821ms	remaining: 2.67s
    235:	learn: 0.3861070	total: 826ms	remaining: 2.67s
    236:	learn: 0.3852621	total: 838ms	remaining: 2.7s
    237:	learn: 0.3845646	total: 839ms	remaining: 2.69s
    238:	learn: 0.3838317	total: 840ms	remaining: 2.67s
    239:	learn: 0.3829835	total: 841ms	remaining: 2.66s
    240:	learn: 0.3820073	total: 843ms	remaining: 2.65s
    241:	learn: 0.3814470	total: 844ms	remaining: 2.64s
    242:	learn: 0.3807217	total: 854ms	remaining: 2.66s
    243:	learn: 0.3799907	total: 856ms	remaining: 2.65s
    244:	learn: 0.3791525	total: 862ms	remaining: 2.66s
    245:	learn: 0.3782599	total: 864ms	remaining: 2.65s
    246:	learn: 0.3776954	total: 873ms	remaining: 2.66s
    247:	learn: 0.3771596	total: 874ms	remaining: 2.65s
    248:	learn: 0.3764676	total: 876ms	remaining: 2.64s
    249:	learn: 0.3756033	total: 877ms	remaining: 2.63s
    250:	learn: 0.3748574	total: 879ms	remaining: 2.62s
    251:	learn: 0.3741521	total: 880ms	remaining: 2.61s
    252:	learn: 0.3735373	total: 886ms	remaining: 2.62s
    253:	learn: 0.3726707	total: 887ms	remaining: 2.6s
    254:	learn: 0.3719370	total: 893ms	remaining: 2.61s
    255:	learn: 0.3712878	total: 894ms	remaining: 2.6s
    256:	learn: 0.3706344	total: 901ms	remaining: 2.6s
    257:	learn: 0.3698126	total: 902ms	remaining: 2.6s
    258:	learn: 0.3692689	total: 904ms	remaining: 2.58s
    259:	learn: 0.3684632	total: 910ms	remaining: 2.59s
    260:	learn: 0.3677466	total: 911ms	remaining: 2.58s
    261:	learn: 0.3671327	total: 912ms	remaining: 2.57s
    262:	learn: 0.3665564	total: 932ms	remaining: 2.61s
    263:	learn: 0.3659407	total: 937ms	remaining: 2.61s
    264:	learn: 0.3651113	total: 939ms	remaining: 2.6s
    265:	learn: 0.3642668	total: 944ms	remaining: 2.6s
    266:	learn: 0.3635892	total: 950ms	remaining: 2.61s
    267:	learn: 0.3630675	total: 955ms	remaining: 2.61s
    268:	learn: 0.3624013	total: 957ms	remaining: 2.6s
    269:	learn: 0.3616951	total: 964ms	remaining: 2.6s
    270:	learn: 0.3609394	total: 966ms	remaining: 2.6s
    271:	learn: 0.3601518	total: 973ms	remaining: 2.6s
    272:	learn: 0.3595363	total: 975ms	remaining: 2.6s
    273:	learn: 0.3587891	total: 986ms	remaining: 2.61s
    274:	learn: 0.3581906	total: 991ms	remaining: 2.61s
    275:	learn: 0.3575159	total: 998ms	remaining: 2.62s
    276:	learn: 0.3570034	total: 1s	remaining: 2.62s
    277:	learn: 0.3564511	total: 1.02s	remaining: 2.65s
    278:	learn: 0.3557321	total: 1.02s	remaining: 2.65s
    279:	learn: 0.3550267	total: 1.03s	remaining: 2.65s
    280:	learn: 0.3540649	total: 1.04s	remaining: 2.66s
    281:	learn: 0.3534089	total: 1.04s	remaining: 2.66s
    282:	learn: 0.3524684	total: 1.05s	remaining: 2.66s
    283:	learn: 0.3517167	total: 1.05s	remaining: 2.66s
    284:	learn: 0.3511599	total: 1.06s	remaining: 2.65s
    285:	learn: 0.3505288	total: 1.06s	remaining: 2.66s
    286:	learn: 0.3497674	total: 1.07s	remaining: 2.65s
    287:	learn: 0.3490211	total: 1.07s	remaining: 2.65s
    288:	learn: 0.3486808	total: 1.07s	remaining: 2.65s
    289:	learn: 0.3482409	total: 1.08s	remaining: 2.64s
    290:	learn: 0.3476104	total: 1.08s	remaining: 2.64s
    291:	learn: 0.3471115	total: 1.09s	remaining: 2.64s
    292:	learn: 0.3464207	total: 1.09s	remaining: 2.63s
    293:	learn: 0.3457073	total: 1.09s	remaining: 2.62s
    294:	learn: 0.3448493	total: 1.1s	remaining: 2.62s
    295:	learn: 0.3442764	total: 1.1s	remaining: 2.62s
    296:	learn: 0.3437679	total: 1.12s	remaining: 2.65s
    297:	learn: 0.3431912	total: 1.13s	remaining: 2.66s
    298:	learn: 0.3425625	total: 1.13s	remaining: 2.65s
    299:	learn: 0.3418909	total: 1.14s	remaining: 2.67s
    300:	learn: 0.3411826	total: 1.15s	remaining: 2.67s
    301:	learn: 0.3404468	total: 1.16s	remaining: 2.67s
    302:	learn: 0.3397911	total: 1.17s	remaining: 2.68s
    303:	learn: 0.3392819	total: 1.17s	remaining: 2.68s
    304:	learn: 0.3386785	total: 1.18s	remaining: 2.68s
    305:	learn: 0.3381723	total: 1.18s	remaining: 2.68s
    306:	learn: 0.3374953	total: 1.19s	remaining: 2.69s
    307:	learn: 0.3368311	total: 1.2s	remaining: 2.69s
    308:	learn: 0.3361975	total: 1.2s	remaining: 2.69s
    309:	learn: 0.3355929	total: 1.21s	remaining: 2.68s
    310:	learn: 0.3351610	total: 1.21s	remaining: 2.69s
    311:	learn: 0.3344132	total: 1.22s	remaining: 2.69s
    312:	learn: 0.3338982	total: 1.22s	remaining: 2.69s
    313:	learn: 0.3331771	total: 1.22s	remaining: 2.67s
    314:	learn: 0.3325166	total: 1.23s	remaining: 2.67s
    315:	learn: 0.3319689	total: 1.23s	remaining: 2.67s
    316:	learn: 0.3314436	total: 1.24s	remaining: 2.67s
    317:	learn: 0.3308998	total: 1.24s	remaining: 2.67s
    318:	learn: 0.3301422	total: 1.25s	remaining: 2.66s
    319:	learn: 0.3295908	total: 1.25s	remaining: 2.66s
    320:	learn: 0.3289123	total: 1.25s	remaining: 2.66s
    321:	learn: 0.3281617	total: 1.26s	remaining: 2.65s
    322:	learn: 0.3276499	total: 1.26s	remaining: 2.65s
    323:	learn: 0.3271874	total: 1.27s	remaining: 2.65s
    324:	learn: 0.3264112	total: 1.27s	remaining: 2.64s
    325:	learn: 0.3258449	total: 1.28s	remaining: 2.64s
    326:	learn: 0.3250780	total: 1.29s	remaining: 2.66s
    327:	learn: 0.3241434	total: 1.29s	remaining: 2.65s
    328:	learn: 0.3233908	total: 1.3s	remaining: 2.65s
    329:	learn: 0.3229982	total: 1.3s	remaining: 2.65s
    330:	learn: 0.3225123	total: 1.31s	remaining: 2.64s
    331:	learn: 0.3220280	total: 1.31s	remaining: 2.64s
    332:	learn: 0.3214247	total: 1.32s	remaining: 2.63s
    333:	learn: 0.3208399	total: 1.32s	remaining: 2.63s
    334:	learn: 0.3203381	total: 1.32s	remaining: 2.63s
    335:	learn: 0.3198142	total: 1.33s	remaining: 2.62s
    336:	learn: 0.3192629	total: 1.33s	remaining: 2.62s
    337:	learn: 0.3186935	total: 1.33s	remaining: 2.62s
    338:	learn: 0.3179686	total: 1.34s	remaining: 2.61s
    339:	learn: 0.3174602	total: 1.34s	remaining: 2.61s
    340:	learn: 0.3168891	total: 1.35s	remaining: 2.6s
    341:	learn: 0.3163692	total: 1.35s	remaining: 2.6s
    342:	learn: 0.3159716	total: 1.35s	remaining: 2.59s
    343:	learn: 0.3154536	total: 1.36s	remaining: 2.59s
    344:	learn: 0.3149685	total: 1.36s	remaining: 2.59s
    345:	learn: 0.3144024	total: 1.37s	remaining: 2.58s
    346:	learn: 0.3140353	total: 1.37s	remaining: 2.58s
    347:	learn: 0.3135035	total: 1.37s	remaining: 2.57s
    348:	learn: 0.3130855	total: 1.38s	remaining: 2.57s
    349:	learn: 0.3126477	total: 1.38s	remaining: 2.56s
    350:	learn: 0.3121701	total: 1.38s	remaining: 2.55s
    351:	learn: 0.3115854	total: 1.38s	remaining: 2.55s
    352:	learn: 0.3111464	total: 1.39s	remaining: 2.54s
    353:	learn: 0.3104357	total: 1.39s	remaining: 2.53s
    354:	learn: 0.3098581	total: 1.39s	remaining: 2.52s
    355:	learn: 0.3093025	total: 1.39s	remaining: 2.51s
    356:	learn: 0.3085438	total: 1.39s	remaining: 2.51s
    357:	learn: 0.3081124	total: 1.39s	remaining: 2.5s
    358:	learn: 0.3074302	total: 1.39s	remaining: 2.49s
    359:	learn: 0.3069290	total: 1.4s	remaining: 2.48s
    360:	learn: 0.3063572	total: 1.4s	remaining: 2.47s
    361:	learn: 0.3058487	total: 1.4s	remaining: 2.46s
    362:	learn: 0.3053731	total: 1.4s	remaining: 2.46s
    363:	learn: 0.3049950	total: 1.4s	remaining: 2.45s
    364:	learn: 0.3044817	total: 1.4s	remaining: 2.44s
    365:	learn: 0.3041031	total: 1.4s	remaining: 2.43s
    366:	learn: 0.3036329	total: 1.41s	remaining: 2.42s
    367:	learn: 0.3029988	total: 1.41s	remaining: 2.42s
    368:	learn: 0.3025647	total: 1.41s	remaining: 2.41s
    369:	learn: 0.3022169	total: 1.41s	remaining: 2.4s
    370:	learn: 0.3016009	total: 1.41s	remaining: 2.39s
    371:	learn: 0.3011917	total: 1.41s	remaining: 2.38s
    372:	learn: 0.3007721	total: 1.41s	remaining: 2.38s
    373:	learn: 0.3002796	total: 1.42s	remaining: 2.37s
    374:	learn: 0.2998885	total: 1.42s	remaining: 2.36s
    375:	learn: 0.2993784	total: 1.42s	remaining: 2.35s
    376:	learn: 0.2989046	total: 1.42s	remaining: 2.35s
    377:	learn: 0.2984268	total: 1.42s	remaining: 2.34s
    378:	learn: 0.2980376	total: 1.43s	remaining: 2.34s
    379:	learn: 0.2976386	total: 1.43s	remaining: 2.34s
    380:	learn: 0.2971803	total: 1.44s	remaining: 2.33s
    381:	learn: 0.2966664	total: 1.44s	remaining: 2.33s
    382:	learn: 0.2961564	total: 1.45s	remaining: 2.33s
    383:	learn: 0.2956202	total: 1.45s	remaining: 2.33s
    384:	learn: 0.2951527	total: 1.46s	remaining: 2.32s
    385:	learn: 0.2947671	total: 1.46s	remaining: 2.33s
    386:	learn: 0.2942167	total: 1.47s	remaining: 2.33s
    387:	learn: 0.2938094	total: 1.47s	remaining: 2.32s
    388:	learn: 0.2934343	total: 1.48s	remaining: 2.32s
    389:	learn: 0.2929567	total: 1.48s	remaining: 2.32s
    390:	learn: 0.2925197	total: 1.49s	remaining: 2.31s
    391:	learn: 0.2921517	total: 1.49s	remaining: 2.31s
    392:	learn: 0.2915990	total: 1.49s	remaining: 2.31s
    393:	learn: 0.2911530	total: 1.5s	remaining: 2.3s
    394:	learn: 0.2907136	total: 1.5s	remaining: 2.3s
    395:	learn: 0.2901805	total: 1.51s	remaining: 2.3s
    396:	learn: 0.2897985	total: 1.51s	remaining: 2.29s
    397:	learn: 0.2893307	total: 1.51s	remaining: 2.29s
    398:	learn: 0.2887063	total: 1.52s	remaining: 2.29s
    399:	learn: 0.2883666	total: 1.52s	remaining: 2.28s
    400:	learn: 0.2880451	total: 1.53s	remaining: 2.28s
    401:	learn: 0.2875984	total: 1.53s	remaining: 2.28s
    402:	learn: 0.2871877	total: 1.53s	remaining: 2.27s
    403:	learn: 0.2868942	total: 1.54s	remaining: 2.27s
    404:	learn: 0.2864472	total: 1.54s	remaining: 2.27s
    405:	learn: 0.2858871	total: 1.55s	remaining: 2.27s
    406:	learn: 0.2854912	total: 1.55s	remaining: 2.26s
    407:	learn: 0.2848975	total: 1.56s	remaining: 2.26s
    408:	learn: 0.2843898	total: 1.56s	remaining: 2.25s
    409:	learn: 0.2840452	total: 1.56s	remaining: 2.25s
    410:	learn: 0.2834345	total: 1.57s	remaining: 2.25s
    411:	learn: 0.2830059	total: 1.57s	remaining: 2.24s
    412:	learn: 0.2826489	total: 1.58s	remaining: 2.24s
    413:	learn: 0.2821333	total: 1.58s	remaining: 2.24s
    414:	learn: 0.2816383	total: 1.58s	remaining: 2.23s
    415:	learn: 0.2810573	total: 1.59s	remaining: 2.23s
    416:	learn: 0.2806727	total: 1.59s	remaining: 2.23s
    417:	learn: 0.2802031	total: 1.59s	remaining: 2.22s
    418:	learn: 0.2796201	total: 1.6s	remaining: 2.21s
    419:	learn: 0.2791867	total: 1.6s	remaining: 2.21s
    420:	learn: 0.2786717	total: 1.6s	remaining: 2.2s
    421:	learn: 0.2782879	total: 1.6s	remaining: 2.19s
    422:	learn: 0.2779091	total: 1.6s	remaining: 2.19s
    423:	learn: 0.2773204	total: 1.6s	remaining: 2.18s
    424:	learn: 0.2767898	total: 1.61s	remaining: 2.17s
    425:	learn: 0.2763485	total: 1.61s	remaining: 2.17s
    426:	learn: 0.2760821	total: 1.61s	remaining: 2.16s
    427:	learn: 0.2756501	total: 1.61s	remaining: 2.16s
    428:	learn: 0.2753336	total: 1.61s	remaining: 2.15s
    429:	learn: 0.2748098	total: 1.62s	remaining: 2.14s
    430:	learn: 0.2742634	total: 1.62s	remaining: 2.14s
    431:	learn: 0.2738034	total: 1.62s	remaining: 2.13s
    432:	learn: 0.2733852	total: 1.62s	remaining: 2.12s
    433:	learn: 0.2730193	total: 1.62s	remaining: 2.12s
    434:	learn: 0.2723791	total: 1.62s	remaining: 2.11s
    435:	learn: 0.2720132	total: 1.62s	remaining: 2.1s
    436:	learn: 0.2717234	total: 1.63s	remaining: 2.09s
    437:	learn: 0.2713446	total: 1.63s	remaining: 2.09s
    438:	learn: 0.2709566	total: 1.63s	remaining: 2.08s
    439:	learn: 0.2706907	total: 1.63s	remaining: 2.07s
    440:	learn: 0.2704036	total: 1.63s	remaining: 2.07s
    441:	learn: 0.2701125	total: 1.63s	remaining: 2.06s
    442:	learn: 0.2696652	total: 1.63s	remaining: 2.05s
    443:	learn: 0.2692436	total: 1.63s	remaining: 2.05s
    444:	learn: 0.2688672	total: 1.64s	remaining: 2.04s
    445:	learn: 0.2686117	total: 1.64s	remaining: 2.03s
    446:	learn: 0.2682218	total: 1.64s	remaining: 2.03s
    447:	learn: 0.2678798	total: 1.64s	remaining: 2.02s
    448:	learn: 0.2676169	total: 1.64s	remaining: 2.01s
    449:	learn: 0.2670965	total: 1.64s	remaining: 2.01s
    450:	learn: 0.2667749	total: 1.64s	remaining: 2s
    451:	learn: 0.2663350	total: 1.64s	remaining: 1.99s
    452:	learn: 0.2657786	total: 1.65s	remaining: 1.99s
    453:	learn: 0.2654621	total: 1.65s	remaining: 1.98s
    454:	learn: 0.2651185	total: 1.65s	remaining: 1.97s
    455:	learn: 0.2646181	total: 1.65s	remaining: 1.97s
    456:	learn: 0.2640489	total: 1.65s	remaining: 1.96s
    457:	learn: 0.2637333	total: 1.65s	remaining: 1.96s
    458:	learn: 0.2633727	total: 1.65s	remaining: 1.95s
    459:	learn: 0.2629595	total: 1.66s	remaining: 1.94s
    460:	learn: 0.2625130	total: 1.66s	remaining: 1.94s
    461:	learn: 0.2620180	total: 1.66s	remaining: 1.93s
    462:	learn: 0.2616593	total: 1.66s	remaining: 1.92s
    463:	learn: 0.2610619	total: 1.66s	remaining: 1.92s
    464:	learn: 0.2608050	total: 1.66s	remaining: 1.91s
    465:	learn: 0.2604130	total: 1.66s	remaining: 1.9s
    466:	learn: 0.2600092	total: 1.67s	remaining: 1.9s
    467:	learn: 0.2594932	total: 1.67s	remaining: 1.9s
    468:	learn: 0.2590833	total: 1.67s	remaining: 1.89s
    469:	learn: 0.2586633	total: 1.67s	remaining: 1.89s
    470:	learn: 0.2582646	total: 1.67s	remaining: 1.88s
    471:	learn: 0.2578490	total: 1.68s	remaining: 1.88s
    472:	learn: 0.2575844	total: 1.68s	remaining: 1.87s
    473:	learn: 0.2570344	total: 1.68s	remaining: 1.86s
    474:	learn: 0.2567278	total: 1.68s	remaining: 1.86s
    475:	learn: 0.2564029	total: 1.68s	remaining: 1.85s
    476:	learn: 0.2560195	total: 1.69s	remaining: 1.85s
    477:	learn: 0.2557816	total: 1.69s	remaining: 1.84s
    478:	learn: 0.2554329	total: 1.69s	remaining: 1.84s
    479:	learn: 0.2550979	total: 1.69s	remaining: 1.83s
    480:	learn: 0.2548465	total: 1.69s	remaining: 1.82s
    481:	learn: 0.2545279	total: 1.69s	remaining: 1.82s
    482:	learn: 0.2540403	total: 1.69s	remaining: 1.81s
    483:	learn: 0.2536986	total: 1.69s	remaining: 1.8s
    484:	learn: 0.2533229	total: 1.7s	remaining: 1.8s
    485:	learn: 0.2529596	total: 1.7s	remaining: 1.79s
    486:	learn: 0.2526620	total: 1.7s	remaining: 1.79s
    487:	learn: 0.2522225	total: 1.7s	remaining: 1.78s
    488:	learn: 0.2518720	total: 1.7s	remaining: 1.78s
    489:	learn: 0.2516374	total: 1.7s	remaining: 1.77s
    490:	learn: 0.2512580	total: 1.7s	remaining: 1.77s
    491:	learn: 0.2509041	total: 1.7s	remaining: 1.76s
    492:	learn: 0.2505165	total: 1.71s	remaining: 1.75s
    493:	learn: 0.2501407	total: 1.71s	remaining: 1.75s
    494:	learn: 0.2497492	total: 1.71s	remaining: 1.74s
    495:	learn: 0.2494590	total: 1.71s	remaining: 1.74s
    496:	learn: 0.2491722	total: 1.71s	remaining: 1.73s
    497:	learn: 0.2488336	total: 1.71s	remaining: 1.73s
    498:	learn: 0.2484837	total: 1.71s	remaining: 1.72s
    499:	learn: 0.2481679	total: 1.72s	remaining: 1.72s
    500:	learn: 0.2478490	total: 1.72s	remaining: 1.71s
    501:	learn: 0.2473930	total: 1.72s	remaining: 1.7s
    502:	learn: 0.2469644	total: 1.72s	remaining: 1.7s
    503:	learn: 0.2465281	total: 1.72s	remaining: 1.69s
    504:	learn: 0.2461485	total: 1.72s	remaining: 1.69s
    505:	learn: 0.2459035	total: 1.72s	remaining: 1.68s
    506:	learn: 0.2456081	total: 1.72s	remaining: 1.68s
    507:	learn: 0.2453251	total: 1.73s	remaining: 1.67s
    508:	learn: 0.2451622	total: 1.73s	remaining: 1.67s
    509:	learn: 0.2448403	total: 1.73s	remaining: 1.66s
    510:	learn: 0.2444679	total: 1.73s	remaining: 1.65s
    511:	learn: 0.2441152	total: 1.73s	remaining: 1.65s
    512:	learn: 0.2437654	total: 1.73s	remaining: 1.64s
    513:	learn: 0.2434364	total: 1.73s	remaining: 1.64s
    514:	learn: 0.2431713	total: 1.75s	remaining: 1.64s
    515:	learn: 0.2428516	total: 1.75s	remaining: 1.64s
    516:	learn: 0.2423800	total: 1.75s	remaining: 1.63s
    517:	learn: 0.2420030	total: 1.75s	remaining: 1.63s
    518:	learn: 0.2416189	total: 1.75s	remaining: 1.62s
    519:	learn: 0.2413112	total: 1.75s	remaining: 1.62s
    520:	learn: 0.2409968	total: 1.75s	remaining: 1.61s
    521:	learn: 0.2407144	total: 1.75s	remaining: 1.61s
    522:	learn: 0.2405005	total: 1.76s	remaining: 1.6s
    523:	learn: 0.2400975	total: 1.76s	remaining: 1.6s
    524:	learn: 0.2396519	total: 1.76s	remaining: 1.59s
    525:	learn: 0.2393105	total: 1.76s	remaining: 1.59s
    526:	learn: 0.2390107	total: 1.76s	remaining: 1.58s
    527:	learn: 0.2387796	total: 1.76s	remaining: 1.58s
    528:	learn: 0.2383853	total: 1.76s	remaining: 1.57s
    529:	learn: 0.2380187	total: 1.76s	remaining: 1.57s
    530:	learn: 0.2377028	total: 1.77s	remaining: 1.56s
    531:	learn: 0.2375156	total: 1.77s	remaining: 1.55s
    532:	learn: 0.2372164	total: 1.77s	remaining: 1.55s
    533:	learn: 0.2369146	total: 1.77s	remaining: 1.55s
    534:	learn: 0.2366065	total: 1.77s	remaining: 1.54s
    535:	learn: 0.2363448	total: 1.77s	remaining: 1.53s
    536:	learn: 0.2360492	total: 1.77s	remaining: 1.53s
    537:	learn: 0.2358844	total: 1.78s	remaining: 1.52s
    538:	learn: 0.2356659	total: 1.78s	remaining: 1.52s
    539:	learn: 0.2352691	total: 1.78s	remaining: 1.52s
    540:	learn: 0.2350419	total: 1.78s	remaining: 1.51s
    541:	learn: 0.2347482	total: 1.78s	remaining: 1.51s
    542:	learn: 0.2344424	total: 1.78s	remaining: 1.5s
    543:	learn: 0.2342469	total: 1.79s	remaining: 1.5s
    544:	learn: 0.2339228	total: 1.79s	remaining: 1.5s
    545:	learn: 0.2336282	total: 1.8s	remaining: 1.49s
    546:	learn: 0.2333830	total: 1.8s	remaining: 1.49s
    547:	learn: 0.2329885	total: 1.8s	remaining: 1.48s
    548:	learn: 0.2327492	total: 1.8s	remaining: 1.48s
    549:	learn: 0.2324078	total: 1.8s	remaining: 1.47s
    550:	learn: 0.2321474	total: 1.8s	remaining: 1.47s
    551:	learn: 0.2318735	total: 1.8s	remaining: 1.46s
    552:	learn: 0.2314612	total: 1.8s	remaining: 1.46s
    553:	learn: 0.2310767	total: 1.81s	remaining: 1.45s
    554:	learn: 0.2309167	total: 1.81s	remaining: 1.45s
    555:	learn: 0.2305761	total: 1.81s	remaining: 1.45s
    556:	learn: 0.2302469	total: 1.81s	remaining: 1.44s
    557:	learn: 0.2299750	total: 1.81s	remaining: 1.44s
    558:	learn: 0.2297495	total: 1.81s	remaining: 1.43s
    559:	learn: 0.2294821	total: 1.81s	remaining: 1.43s
    560:	learn: 0.2292364	total: 1.81s	remaining: 1.42s
    561:	learn: 0.2289496	total: 1.82s	remaining: 1.42s
    562:	learn: 0.2286585	total: 1.82s	remaining: 1.41s
    563:	learn: 0.2284182	total: 1.82s	remaining: 1.41s
    564:	learn: 0.2280876	total: 1.82s	remaining: 1.4s
    565:	learn: 0.2278016	total: 1.82s	remaining: 1.4s
    566:	learn: 0.2275360	total: 1.82s	remaining: 1.39s
    567:	learn: 0.2273335	total: 1.82s	remaining: 1.39s
    568:	learn: 0.2270234	total: 1.83s	remaining: 1.38s
    569:	learn: 0.2267649	total: 1.83s	remaining: 1.38s
    570:	learn: 0.2265213	total: 1.83s	remaining: 1.37s
    571:	learn: 0.2262158	total: 1.83s	remaining: 1.37s
    572:	learn: 0.2260402	total: 1.83s	remaining: 1.36s
    573:	learn: 0.2256939	total: 1.83s	remaining: 1.36s
    574:	learn: 0.2255059	total: 1.83s	remaining: 1.36s
    575:	learn: 0.2252901	total: 1.84s	remaining: 1.35s
    576:	learn: 0.2250568	total: 1.84s	remaining: 1.35s
    577:	learn: 0.2247808	total: 1.84s	remaining: 1.34s
    578:	learn: 0.2243982	total: 1.84s	remaining: 1.34s
    579:	learn: 0.2240717	total: 1.84s	remaining: 1.33s
    580:	learn: 0.2237615	total: 1.84s	remaining: 1.33s
    581:	learn: 0.2235118	total: 1.84s	remaining: 1.32s
    582:	learn: 0.2233229	total: 1.84s	remaining: 1.32s
    583:	learn: 0.2230402	total: 1.85s	remaining: 1.31s
    584:	learn: 0.2226428	total: 1.85s	remaining: 1.31s
    585:	learn: 0.2222478	total: 1.85s	remaining: 1.31s
    586:	learn: 0.2219533	total: 1.85s	remaining: 1.3s
    587:	learn: 0.2217893	total: 1.85s	remaining: 1.3s
    588:	learn: 0.2215629	total: 1.85s	remaining: 1.29s
    589:	learn: 0.2213844	total: 1.85s	remaining: 1.29s
    590:	learn: 0.2211217	total: 1.85s	remaining: 1.28s
    591:	learn: 0.2208070	total: 1.86s	remaining: 1.28s
    592:	learn: 0.2205984	total: 1.86s	remaining: 1.27s
    593:	learn: 0.2204360	total: 1.86s	remaining: 1.27s
    594:	learn: 0.2202355	total: 1.86s	remaining: 1.27s
    595:	learn: 0.2199388	total: 1.86s	remaining: 1.26s
    596:	learn: 0.2196028	total: 1.86s	remaining: 1.26s
    597:	learn: 0.2194720	total: 1.86s	remaining: 1.25s
    598:	learn: 0.2191861	total: 1.86s	remaining: 1.25s
    599:	learn: 0.2189812	total: 1.87s	remaining: 1.25s
    600:	learn: 0.2187491	total: 1.87s	remaining: 1.24s
    601:	learn: 0.2184993	total: 1.87s	remaining: 1.24s
    602:	learn: 0.2182312	total: 1.88s	remaining: 1.23s
    603:	learn: 0.2179048	total: 1.88s	remaining: 1.23s
    604:	learn: 0.2176619	total: 1.88s	remaining: 1.23s
    605:	learn: 0.2174789	total: 1.88s	remaining: 1.22s
    606:	learn: 0.2173124	total: 1.88s	remaining: 1.22s
    607:	learn: 0.2170755	total: 1.89s	remaining: 1.22s
    608:	learn: 0.2167933	total: 1.89s	remaining: 1.21s
    609:	learn: 0.2165464	total: 1.89s	remaining: 1.21s
    610:	learn: 0.2163295	total: 1.89s	remaining: 1.2s
    611:	learn: 0.2161077	total: 1.89s	remaining: 1.2s
    612:	learn: 0.2159526	total: 1.89s	remaining: 1.19s
    613:	learn: 0.2157916	total: 1.89s	remaining: 1.19s
    614:	learn: 0.2155877	total: 1.89s	remaining: 1.19s
    615:	learn: 0.2153547	total: 1.9s	remaining: 1.18s
    616:	learn: 0.2151374	total: 1.9s	remaining: 1.18s
    617:	learn: 0.2148505	total: 1.9s	remaining: 1.17s
    618:	learn: 0.2146456	total: 1.9s	remaining: 1.17s
    619:	learn: 0.2143099	total: 1.9s	remaining: 1.16s
    620:	learn: 0.2141113	total: 1.9s	remaining: 1.16s
    621:	learn: 0.2138241	total: 1.9s	remaining: 1.16s
    622:	learn: 0.2135714	total: 1.9s	remaining: 1.15s
    623:	learn: 0.2133305	total: 1.91s	remaining: 1.15s
    624:	learn: 0.2130557	total: 1.91s	remaining: 1.14s
    625:	learn: 0.2127836	total: 1.91s	remaining: 1.14s
    626:	learn: 0.2125129	total: 1.91s	remaining: 1.14s
    627:	learn: 0.2122744	total: 1.91s	remaining: 1.13s
    628:	learn: 0.2120097	total: 1.92s	remaining: 1.13s
    629:	learn: 0.2117803	total: 1.92s	remaining: 1.13s
    630:	learn: 0.2114843	total: 1.92s	remaining: 1.12s
    631:	learn: 0.2112653	total: 1.92s	remaining: 1.12s
    632:	learn: 0.2110786	total: 1.92s	remaining: 1.11s
    633:	learn: 0.2108491	total: 1.92s	remaining: 1.11s
    634:	learn: 0.2106548	total: 1.92s	remaining: 1.11s
    635:	learn: 0.2103541	total: 1.93s	remaining: 1.1s
    636:	learn: 0.2100442	total: 1.93s	remaining: 1.1s
    637:	learn: 0.2098573	total: 1.93s	remaining: 1.09s
    638:	learn: 0.2095055	total: 1.93s	remaining: 1.09s
    639:	learn: 0.2093106	total: 1.93s	remaining: 1.09s
    640:	learn: 0.2090526	total: 1.93s	remaining: 1.08s
    641:	learn: 0.2088128	total: 1.93s	remaining: 1.08s
    642:	learn: 0.2085560	total: 1.93s	remaining: 1.07s
    643:	learn: 0.2083085	total: 1.94s	remaining: 1.07s
    644:	learn: 0.2080493	total: 1.94s	remaining: 1.07s
    645:	learn: 0.2078654	total: 1.94s	remaining: 1.06s
    646:	learn: 0.2076429	total: 1.94s	remaining: 1.06s
    647:	learn: 0.2074211	total: 1.94s	remaining: 1.05s
    648:	learn: 0.2070036	total: 1.94s	remaining: 1.05s
    649:	learn: 0.2066961	total: 1.94s	remaining: 1.05s
    650:	learn: 0.2064640	total: 1.95s	remaining: 1.04s
    651:	learn: 0.2062261	total: 1.95s	remaining: 1.04s
    652:	learn: 0.2060180	total: 1.95s	remaining: 1.03s
    653:	learn: 0.2058004	total: 1.95s	remaining: 1.03s
    654:	learn: 0.2055314	total: 1.95s	remaining: 1.03s
    655:	learn: 0.2052985	total: 1.95s	remaining: 1.02s
    656:	learn: 0.2050233	total: 1.95s	remaining: 1.02s
    657:	learn: 0.2046901	total: 1.95s	remaining: 1.02s
    658:	learn: 0.2045525	total: 1.96s	remaining: 1.01s
    659:	learn: 0.2042892	total: 1.96s	remaining: 1.01s
    660:	learn: 0.2040670	total: 1.96s	remaining: 1s
    661:	learn: 0.2038644	total: 1.96s	remaining: 1s
    662:	learn: 0.2037219	total: 1.96s	remaining: 997ms
    663:	learn: 0.2035028	total: 1.96s	remaining: 993ms
    664:	learn: 0.2032665	total: 1.96s	remaining: 990ms
    665:	learn: 0.2030467	total: 1.97s	remaining: 986ms
    666:	learn: 0.2028411	total: 1.97s	remaining: 982ms
    667:	learn: 0.2024784	total: 1.97s	remaining: 978ms
    668:	learn: 0.2023498	total: 1.97s	remaining: 975ms
    669:	learn: 0.2020165	total: 1.97s	remaining: 971ms
    670:	learn: 0.2018284	total: 1.97s	remaining: 967ms
    671:	learn: 0.2016231	total: 1.97s	remaining: 963ms
    672:	learn: 0.2013732	total: 1.98s	remaining: 960ms
    673:	learn: 0.2010755	total: 1.98s	remaining: 956ms
    674:	learn: 0.2008374	total: 1.98s	remaining: 954ms
    675:	learn: 0.2005553	total: 1.98s	remaining: 951ms
    676:	learn: 0.2003587	total: 1.99s	remaining: 948ms
    677:	learn: 0.2000526	total: 1.99s	remaining: 944ms
    678:	learn: 0.1998831	total: 1.99s	remaining: 940ms
    679:	learn: 0.1997291	total: 1.99s	remaining: 936ms
    680:	learn: 0.1994216	total: 1.99s	remaining: 933ms
    681:	learn: 0.1992324	total: 1.99s	remaining: 929ms
    682:	learn: 0.1989363	total: 1.99s	remaining: 925ms
    683:	learn: 0.1986430	total: 2s	remaining: 922ms
    684:	learn: 0.1984366	total: 2s	remaining: 918ms
    685:	learn: 0.1981642	total: 2s	remaining: 914ms
    686:	learn: 0.1979691	total: 2s	remaining: 911ms
    687:	learn: 0.1978234	total: 2s	remaining: 907ms
    688:	learn: 0.1975995	total: 2.01s	remaining: 908ms
    689:	learn: 0.1974457	total: 2.02s	remaining: 907ms
    690:	learn: 0.1971816	total: 2.02s	remaining: 905ms
    691:	learn: 0.1969985	total: 2.02s	remaining: 901ms
    692:	learn: 0.1967947	total: 2.03s	remaining: 900ms
    693:	learn: 0.1964801	total: 2.04s	remaining: 898ms
    694:	learn: 0.1962314	total: 2.04s	remaining: 895ms
    695:	learn: 0.1960316	total: 2.04s	remaining: 893ms
    696:	learn: 0.1958359	total: 2.05s	remaining: 890ms
    697:	learn: 0.1956162	total: 2.05s	remaining: 888ms
    698:	learn: 0.1954382	total: 2.05s	remaining: 884ms
    699:	learn: 0.1951703	total: 2.06s	remaining: 881ms
    700:	learn: 0.1949062	total: 2.06s	remaining: 878ms
    701:	learn: 0.1947062	total: 2.06s	remaining: 874ms
    702:	learn: 0.1944979	total: 2.06s	remaining: 871ms
    703:	learn: 0.1942648	total: 2.06s	remaining: 867ms
    704:	learn: 0.1939977	total: 2.06s	remaining: 864ms
    705:	learn: 0.1937824	total: 2.06s	remaining: 860ms
    706:	learn: 0.1935741	total: 2.07s	remaining: 856ms
    707:	learn: 0.1933340	total: 2.07s	remaining: 853ms
    708:	learn: 0.1930791	total: 2.07s	remaining: 849ms
    709:	learn: 0.1929426	total: 2.07s	remaining: 845ms
    710:	learn: 0.1927686	total: 2.07s	remaining: 842ms
    711:	learn: 0.1925062	total: 2.07s	remaining: 838ms
    712:	learn: 0.1923013	total: 2.07s	remaining: 835ms
    713:	learn: 0.1921343	total: 2.08s	remaining: 831ms
    714:	learn: 0.1919629	total: 2.08s	remaining: 828ms
    715:	learn: 0.1917238	total: 2.08s	remaining: 824ms
    716:	learn: 0.1914298	total: 2.08s	remaining: 821ms
    717:	learn: 0.1911985	total: 2.08s	remaining: 817ms
    718:	learn: 0.1910248	total: 2.08s	remaining: 814ms
    719:	learn: 0.1908187	total: 2.08s	remaining: 810ms
    720:	learn: 0.1905606	total: 2.08s	remaining: 807ms
    721:	learn: 0.1903669	total: 2.08s	remaining: 803ms
    722:	learn: 0.1902308	total: 2.09s	remaining: 800ms
    723:	learn: 0.1899287	total: 2.09s	remaining: 796ms
    724:	learn: 0.1898109	total: 2.09s	remaining: 793ms
    725:	learn: 0.1896464	total: 2.09s	remaining: 789ms
    726:	learn: 0.1893627	total: 2.09s	remaining: 786ms
    727:	learn: 0.1892554	total: 2.09s	remaining: 782ms
    728:	learn: 0.1890625	total: 2.1s	remaining: 779ms
    729:	learn: 0.1888089	total: 2.1s	remaining: 775ms
    730:	learn: 0.1886550	total: 2.1s	remaining: 772ms
    731:	learn: 0.1884326	total: 2.1s	remaining: 769ms
    732:	learn: 0.1883110	total: 2.1s	remaining: 765ms
    733:	learn: 0.1881577	total: 2.1s	remaining: 762ms
    734:	learn: 0.1879141	total: 2.1s	remaining: 759ms
    735:	learn: 0.1877437	total: 2.1s	remaining: 755ms
    736:	learn: 0.1875953	total: 2.11s	remaining: 752ms
    737:	learn: 0.1874794	total: 2.11s	remaining: 748ms
    738:	learn: 0.1873668	total: 2.11s	remaining: 745ms
    739:	learn: 0.1872035	total: 2.11s	remaining: 741ms
    740:	learn: 0.1870745	total: 2.11s	remaining: 738ms
    741:	learn: 0.1869023	total: 2.11s	remaining: 734ms
    742:	learn: 0.1866802	total: 2.11s	remaining: 731ms
    743:	learn: 0.1864442	total: 2.12s	remaining: 728ms
    744:	learn: 0.1863371	total: 2.12s	remaining: 725ms
    745:	learn: 0.1860808	total: 2.12s	remaining: 723ms
    746:	learn: 0.1858950	total: 2.12s	remaining: 720ms
    747:	learn: 0.1857619	total: 2.13s	remaining: 716ms
    748:	learn: 0.1855279	total: 2.13s	remaining: 713ms
    749:	learn: 0.1853801	total: 2.13s	remaining: 710ms
    750:	learn: 0.1852299	total: 2.13s	remaining: 706ms
    751:	learn: 0.1849581	total: 2.13s	remaining: 703ms
    752:	learn: 0.1847370	total: 2.13s	remaining: 699ms
    753:	learn: 0.1845795	total: 2.13s	remaining: 696ms
    754:	learn: 0.1843570	total: 2.13s	remaining: 693ms
    755:	learn: 0.1841727	total: 2.14s	remaining: 689ms
    756:	learn: 0.1839405	total: 2.14s	remaining: 686ms
    757:	learn: 0.1837444	total: 2.14s	remaining: 683ms
    758:	learn: 0.1835577	total: 2.14s	remaining: 680ms
    759:	learn: 0.1834072	total: 2.14s	remaining: 677ms
    760:	learn: 0.1832458	total: 2.15s	remaining: 676ms
    761:	learn: 0.1830310	total: 2.15s	remaining: 673ms
    762:	learn: 0.1828628	total: 2.16s	remaining: 670ms
    763:	learn: 0.1827431	total: 2.16s	remaining: 667ms
    764:	learn: 0.1825209	total: 2.16s	remaining: 663ms
    765:	learn: 0.1823271	total: 2.16s	remaining: 660ms
    766:	learn: 0.1821145	total: 2.16s	remaining: 657ms
    767:	learn: 0.1818968	total: 2.16s	remaining: 653ms
    768:	learn: 0.1816805	total: 2.16s	remaining: 650ms
    769:	learn: 0.1815560	total: 2.17s	remaining: 647ms
    770:	learn: 0.1813870	total: 2.17s	remaining: 644ms
    771:	learn: 0.1811777	total: 2.17s	remaining: 640ms
    772:	learn: 0.1809440	total: 2.17s	remaining: 637ms
    773:	learn: 0.1807740	total: 2.17s	remaining: 634ms
    774:	learn: 0.1805270	total: 2.17s	remaining: 631ms
    775:	learn: 0.1803343	total: 2.17s	remaining: 627ms
    776:	learn: 0.1800902	total: 2.17s	remaining: 624ms
    777:	learn: 0.1798802	total: 2.17s	remaining: 621ms
    778:	learn: 0.1796791	total: 2.18s	remaining: 618ms
    779:	learn: 0.1795071	total: 2.18s	remaining: 615ms
    780:	learn: 0.1792813	total: 2.18s	remaining: 611ms
    781:	learn: 0.1790725	total: 2.18s	remaining: 608ms
    782:	learn: 0.1789756	total: 2.18s	remaining: 605ms
    783:	learn: 0.1787815	total: 2.18s	remaining: 602ms
    784:	learn: 0.1786093	total: 2.19s	remaining: 599ms
    785:	learn: 0.1784486	total: 2.19s	remaining: 595ms
    786:	learn: 0.1782153	total: 2.19s	remaining: 592ms
    787:	learn: 0.1780564	total: 2.19s	remaining: 589ms
    788:	learn: 0.1779013	total: 2.19s	remaining: 586ms
    789:	learn: 0.1776759	total: 2.19s	remaining: 583ms
    790:	learn: 0.1774475	total: 2.19s	remaining: 580ms
    791:	learn: 0.1772716	total: 2.19s	remaining: 576ms
    792:	learn: 0.1770801	total: 2.2s	remaining: 573ms
    793:	learn: 0.1769370	total: 2.2s	remaining: 570ms
    794:	learn: 0.1767282	total: 2.2s	remaining: 567ms
    795:	learn: 0.1765836	total: 2.2s	remaining: 564ms
    796:	learn: 0.1763405	total: 2.2s	remaining: 561ms
    797:	learn: 0.1762129	total: 2.2s	remaining: 557ms
    798:	learn: 0.1760235	total: 2.2s	remaining: 554ms
    799:	learn: 0.1758189	total: 2.21s	remaining: 551ms
    800:	learn: 0.1756702	total: 2.21s	remaining: 548ms
    801:	learn: 0.1755450	total: 2.21s	remaining: 545ms
    802:	learn: 0.1753332	total: 2.21s	remaining: 542ms
    803:	learn: 0.1752189	total: 2.21s	remaining: 539ms
    804:	learn: 0.1750728	total: 2.21s	remaining: 536ms
    805:	learn: 0.1749159	total: 2.21s	remaining: 533ms
    806:	learn: 0.1747684	total: 2.21s	remaining: 529ms
    807:	learn: 0.1746228	total: 2.21s	remaining: 526ms
    808:	learn: 0.1744255	total: 2.22s	remaining: 523ms
    809:	learn: 0.1742790	total: 2.22s	remaining: 520ms
    810:	learn: 0.1742306	total: 2.22s	remaining: 517ms
    811:	learn: 0.1741252	total: 2.22s	remaining: 514ms
    812:	learn: 0.1739433	total: 2.22s	remaining: 511ms
    813:	learn: 0.1737325	total: 2.22s	remaining: 508ms
    814:	learn: 0.1735887	total: 2.22s	remaining: 505ms
    815:	learn: 0.1734158	total: 2.23s	remaining: 502ms
    816:	learn: 0.1732640	total: 2.23s	remaining: 499ms
    817:	learn: 0.1730771	total: 2.23s	remaining: 496ms
    818:	learn: 0.1728427	total: 2.23s	remaining: 493ms
    819:	learn: 0.1727148	total: 2.23s	remaining: 490ms
    820:	learn: 0.1726151	total: 2.23s	remaining: 487ms
    821:	learn: 0.1724531	total: 2.23s	remaining: 484ms
    822:	learn: 0.1722405	total: 2.23s	remaining: 481ms
    823:	learn: 0.1720868	total: 2.24s	remaining: 478ms
    824:	learn: 0.1719532	total: 2.24s	remaining: 475ms
    825:	learn: 0.1717664	total: 2.24s	remaining: 472ms
    826:	learn: 0.1715672	total: 2.24s	remaining: 469ms
    827:	learn: 0.1714585	total: 2.24s	remaining: 466ms
    828:	learn: 0.1713378	total: 2.24s	remaining: 463ms
    829:	learn: 0.1712344	total: 2.25s	remaining: 460ms
    830:	learn: 0.1711275	total: 2.25s	remaining: 458ms
    831:	learn: 0.1710643	total: 2.26s	remaining: 456ms
    832:	learn: 0.1709130	total: 2.26s	remaining: 453ms
    833:	learn: 0.1707184	total: 2.26s	remaining: 451ms
    834:	learn: 0.1705323	total: 2.27s	remaining: 448ms
    835:	learn: 0.1703361	total: 2.27s	remaining: 446ms
    836:	learn: 0.1701466	total: 2.28s	remaining: 443ms
    837:	learn: 0.1699468	total: 2.28s	remaining: 441ms
    838:	learn: 0.1697696	total: 2.29s	remaining: 439ms
    839:	learn: 0.1695825	total: 2.29s	remaining: 436ms
    840:	learn: 0.1694047	total: 2.3s	remaining: 435ms
    841:	learn: 0.1692851	total: 2.3s	remaining: 432ms
    842:	learn: 0.1691157	total: 2.31s	remaining: 429ms
    843:	learn: 0.1688711	total: 2.31s	remaining: 427ms
    844:	learn: 0.1687820	total: 2.31s	remaining: 424ms
    845:	learn: 0.1686828	total: 2.32s	remaining: 422ms
    846:	learn: 0.1684238	total: 2.32s	remaining: 419ms
    847:	learn: 0.1682248	total: 2.33s	remaining: 417ms
    848:	learn: 0.1679504	total: 2.33s	remaining: 414ms
    849:	learn: 0.1677554	total: 2.33s	remaining: 412ms
    850:	learn: 0.1676341	total: 2.33s	remaining: 409ms
    851:	learn: 0.1675346	total: 2.34s	remaining: 406ms
    852:	learn: 0.1673314	total: 2.34s	remaining: 403ms
    853:	learn: 0.1671557	total: 2.34s	remaining: 400ms
    854:	learn: 0.1669157	total: 2.34s	remaining: 397ms
    855:	learn: 0.1668059	total: 2.34s	remaining: 394ms
    856:	learn: 0.1666470	total: 2.34s	remaining: 391ms
    857:	learn: 0.1665119	total: 2.34s	remaining: 388ms
    858:	learn: 0.1663130	total: 2.35s	remaining: 385ms
    859:	learn: 0.1661849	total: 2.35s	remaining: 382ms
    860:	learn: 0.1660905	total: 2.35s	remaining: 379ms
    861:	learn: 0.1659814	total: 2.35s	remaining: 376ms
    862:	learn: 0.1658224	total: 2.35s	remaining: 373ms
    863:	learn: 0.1656633	total: 2.35s	remaining: 370ms
    864:	learn: 0.1654616	total: 2.35s	remaining: 367ms
    865:	learn: 0.1653035	total: 2.35s	remaining: 364ms
    866:	learn: 0.1651659	total: 2.35s	remaining: 361ms
    867:	learn: 0.1650027	total: 2.36s	remaining: 358ms
    868:	learn: 0.1648196	total: 2.36s	remaining: 356ms
    869:	learn: 0.1646475	total: 2.36s	remaining: 353ms
    870:	learn: 0.1644930	total: 2.36s	remaining: 350ms
    871:	learn: 0.1643962	total: 2.36s	remaining: 347ms
    872:	learn: 0.1642436	total: 2.36s	remaining: 344ms
    873:	learn: 0.1640270	total: 2.37s	remaining: 341ms
    874:	learn: 0.1638385	total: 2.37s	remaining: 338ms
    875:	learn: 0.1637115	total: 2.37s	remaining: 335ms
    876:	learn: 0.1636267	total: 2.37s	remaining: 332ms
    877:	learn: 0.1635386	total: 2.37s	remaining: 329ms
    878:	learn: 0.1634232	total: 2.37s	remaining: 326ms
    879:	learn: 0.1632174	total: 2.37s	remaining: 324ms
    880:	learn: 0.1630249	total: 2.37s	remaining: 321ms
    881:	learn: 0.1628712	total: 2.38s	remaining: 318ms
    882:	learn: 0.1627648	total: 2.38s	remaining: 315ms
    883:	learn: 0.1625785	total: 2.38s	remaining: 312ms
    884:	learn: 0.1624283	total: 2.38s	remaining: 309ms
    885:	learn: 0.1623230	total: 2.38s	remaining: 306ms
    886:	learn: 0.1621857	total: 2.38s	remaining: 303ms
    887:	learn: 0.1620284	total: 2.38s	remaining: 301ms
    888:	learn: 0.1619104	total: 2.38s	remaining: 298ms
    889:	learn: 0.1618253	total: 2.39s	remaining: 295ms
    890:	learn: 0.1616158	total: 2.39s	remaining: 292ms
    891:	learn: 0.1614334	total: 2.39s	remaining: 289ms
    892:	learn: 0.1613278	total: 2.39s	remaining: 286ms
    893:	learn: 0.1611948	total: 2.39s	remaining: 283ms
    894:	learn: 0.1610125	total: 2.39s	remaining: 281ms
    895:	learn: 0.1608685	total: 2.39s	remaining: 278ms
    896:	learn: 0.1606578	total: 2.4s	remaining: 275ms
    897:	learn: 0.1604568	total: 2.4s	remaining: 272ms
    898:	learn: 0.1603552	total: 2.4s	remaining: 269ms
    899:	learn: 0.1602050	total: 2.4s	remaining: 267ms
    900:	learn: 0.1600276	total: 2.4s	remaining: 264ms
    901:	learn: 0.1598679	total: 2.4s	remaining: 261ms
    902:	learn: 0.1597268	total: 2.4s	remaining: 258ms
    903:	learn: 0.1595353	total: 2.4s	remaining: 255ms
    904:	learn: 0.1594047	total: 2.4s	remaining: 252ms
    905:	learn: 0.1591321	total: 2.41s	remaining: 250ms
    906:	learn: 0.1590200	total: 2.41s	remaining: 247ms
    907:	learn: 0.1588619	total: 2.41s	remaining: 244ms
    908:	learn: 0.1586496	total: 2.41s	remaining: 241ms
    909:	learn: 0.1585360	total: 2.41s	remaining: 239ms
    910:	learn: 0.1584123	total: 2.41s	remaining: 236ms
    911:	learn: 0.1581315	total: 2.41s	remaining: 233ms
    912:	learn: 0.1580284	total: 2.42s	remaining: 230ms
    913:	learn: 0.1579134	total: 2.42s	remaining: 228ms
    914:	learn: 0.1576965	total: 2.42s	remaining: 225ms
    915:	learn: 0.1575631	total: 2.42s	remaining: 222ms
    916:	learn: 0.1573195	total: 2.43s	remaining: 220ms
    917:	learn: 0.1571985	total: 2.43s	remaining: 217ms
    918:	learn: 0.1570316	total: 2.43s	remaining: 214ms
    919:	learn: 0.1569559	total: 2.43s	remaining: 211ms
    920:	learn: 0.1567983	total: 2.43s	remaining: 209ms
    921:	learn: 0.1566163	total: 2.43s	remaining: 206ms
    922:	learn: 0.1564047	total: 2.43s	remaining: 203ms
    923:	learn: 0.1562604	total: 2.43s	remaining: 200ms
    924:	learn: 0.1560988	total: 2.44s	remaining: 198ms
    925:	learn: 0.1559335	total: 2.44s	remaining: 195ms
    926:	learn: 0.1557727	total: 2.44s	remaining: 192ms
    927:	learn: 0.1556045	total: 2.44s	remaining: 189ms
    928:	learn: 0.1554279	total: 2.44s	remaining: 187ms
    929:	learn: 0.1552939	total: 2.44s	remaining: 184ms
    930:	learn: 0.1551815	total: 2.44s	remaining: 181ms
    931:	learn: 0.1550513	total: 2.44s	remaining: 178ms
    932:	learn: 0.1548965	total: 2.45s	remaining: 176ms
    933:	learn: 0.1548350	total: 2.45s	remaining: 173ms
    934:	learn: 0.1546728	total: 2.45s	remaining: 170ms
    935:	learn: 0.1545464	total: 2.45s	remaining: 168ms
    936:	learn: 0.1543975	total: 2.45s	remaining: 165ms
    937:	learn: 0.1541951	total: 2.45s	remaining: 162ms
    938:	learn: 0.1540613	total: 2.45s	remaining: 159ms
    939:	learn: 0.1538955	total: 2.46s	remaining: 157ms
    940:	learn: 0.1537964	total: 2.46s	remaining: 154ms
    941:	learn: 0.1536613	total: 2.46s	remaining: 151ms
    942:	learn: 0.1534730	total: 2.46s	remaining: 149ms
    943:	learn: 0.1533342	total: 2.46s	remaining: 146ms
    944:	learn: 0.1532246	total: 2.46s	remaining: 143ms
    945:	learn: 0.1530955	total: 2.46s	remaining: 141ms
    946:	learn: 0.1529400	total: 2.46s	remaining: 138ms
    947:	learn: 0.1528365	total: 2.47s	remaining: 135ms
    948:	learn: 0.1526804	total: 2.47s	remaining: 133ms
    949:	learn: 0.1524719	total: 2.47s	remaining: 130ms
    950:	learn: 0.1523470	total: 2.48s	remaining: 128ms
    951:	learn: 0.1522206	total: 2.48s	remaining: 125ms
    952:	learn: 0.1520748	total: 2.48s	remaining: 122ms
    953:	learn: 0.1518997	total: 2.48s	remaining: 120ms
    954:	learn: 0.1518284	total: 2.49s	remaining: 117ms
    955:	learn: 0.1517171	total: 2.49s	remaining: 114ms
    956:	learn: 0.1515919	total: 2.49s	remaining: 112ms
    957:	learn: 0.1514092	total: 2.49s	remaining: 109ms
    958:	learn: 0.1512453	total: 2.49s	remaining: 107ms
    959:	learn: 0.1511709	total: 2.49s	remaining: 104ms
    960:	learn: 0.1510302	total: 2.49s	remaining: 101ms
    961:	learn: 0.1509030	total: 2.49s	remaining: 98.5ms
    962:	learn: 0.1508020	total: 2.5s	remaining: 95.9ms
    963:	learn: 0.1507202	total: 2.5s	remaining: 93.3ms
    964:	learn: 0.1505643	total: 2.5s	remaining: 90.7ms
    965:	learn: 0.1504692	total: 2.5s	remaining: 88ms
    966:	learn: 0.1503167	total: 2.5s	remaining: 85.4ms
    967:	learn: 0.1500881	total: 2.5s	remaining: 82.7ms
    968:	learn: 0.1499522	total: 2.5s	remaining: 80.1ms
    969:	learn: 0.1497710	total: 2.5s	remaining: 77.5ms
    970:	learn: 0.1496962	total: 2.51s	remaining: 74.9ms
    971:	learn: 0.1494771	total: 2.51s	remaining: 72.3ms
    972:	learn: 0.1492930	total: 2.51s	remaining: 69.6ms
    973:	learn: 0.1491693	total: 2.51s	remaining: 67ms
    974:	learn: 0.1490576	total: 2.51s	remaining: 64.4ms
    975:	learn: 0.1489654	total: 2.51s	remaining: 61.8ms
    976:	learn: 0.1488193	total: 2.51s	remaining: 59.2ms
    977:	learn: 0.1487737	total: 2.52s	remaining: 56.6ms
    978:	learn: 0.1486333	total: 2.52s	remaining: 54ms
    979:	learn: 0.1484506	total: 2.52s	remaining: 51.4ms
    980:	learn: 0.1482593	total: 2.52s	remaining: 48.8ms
    981:	learn: 0.1481609	total: 2.52s	remaining: 46.2ms
    982:	learn: 0.1480046	total: 2.52s	remaining: 43.6ms
    983:	learn: 0.1479486	total: 2.52s	remaining: 41ms
    984:	learn: 0.1478514	total: 2.52s	remaining: 38.4ms
    985:	learn: 0.1477449	total: 2.53s	remaining: 35.9ms
    986:	learn: 0.1475843	total: 2.53s	remaining: 33.3ms
    987:	learn: 0.1474700	total: 2.53s	remaining: 30.7ms
    988:	learn: 0.1473128	total: 2.53s	remaining: 28.1ms
    989:	learn: 0.1472493	total: 2.53s	remaining: 25.6ms
    990:	learn: 0.1471369	total: 2.53s	remaining: 23ms
    991:	learn: 0.1470064	total: 2.53s	remaining: 20.4ms
    992:	learn: 0.1468605	total: 2.54s	remaining: 17.9ms
    993:	learn: 0.1467172	total: 2.54s	remaining: 15.3ms
    994:	learn: 0.1465716	total: 2.54s	remaining: 12.8ms
    995:	learn: 0.1464307	total: 2.54s	remaining: 10.2ms
    996:	learn: 0.1463201	total: 2.54s	remaining: 7.64ms
    997:	learn: 0.1461544	total: 2.54s	remaining: 5.09ms
    998:	learn: 0.1460391	total: 2.54s	remaining: 2.54ms
    999:	learn: 0.1459447	total: 2.54s	remaining: 0us
    0:	learn: 0.6911383	total: 4.95ms	remaining: 4.94s
    1:	learn: 0.6885918	total: 11.1ms	remaining: 5.53s
    2:	learn: 0.6866163	total: 14.1ms	remaining: 4.68s
    3:	learn: 0.6843547	total: 17.3ms	remaining: 4.3s
    4:	learn: 0.6817839	total: 21.4ms	remaining: 4.26s
    5:	learn: 0.6797349	total: 25.6ms	remaining: 4.24s
    6:	learn: 0.6775883	total: 31.7ms	remaining: 4.5s
    7:	learn: 0.6757058	total: 35.1ms	remaining: 4.35s
    8:	learn: 0.6735491	total: 37.2ms	remaining: 4.09s
    9:	learn: 0.6719165	total: 41.2ms	remaining: 4.08s
    10:	learn: 0.6697978	total: 45.3ms	remaining: 4.08s
    11:	learn: 0.6679611	total: 49.5ms	remaining: 4.08s
    12:	learn: 0.6661096	total: 53.5ms	remaining: 4.06s
    13:	learn: 0.6638339	total: 57.5ms	remaining: 4.05s
    14:	learn: 0.6618357	total: 61.6ms	remaining: 4.05s
    15:	learn: 0.6597410	total: 65.6ms	remaining: 4.04s
    16:	learn: 0.6579417	total: 69.6ms	remaining: 4.03s
    17:	learn: 0.6561764	total: 73.7ms	remaining: 4.02s
    18:	learn: 0.6541464	total: 77.9ms	remaining: 4.02s
    19:	learn: 0.6527596	total: 79.1ms	remaining: 3.88s
    20:	learn: 0.6509464	total: 81ms	remaining: 3.78s
    21:	learn: 0.6491590	total: 83.3ms	remaining: 3.7s
    22:	learn: 0.6475575	total: 85.1ms	remaining: 3.62s
    23:	learn: 0.6458615	total: 87ms	remaining: 3.54s
    24:	learn: 0.6438979	total: 88.6ms	remaining: 3.45s
    25:	learn: 0.6419502	total: 89.7ms	remaining: 3.36s
    26:	learn: 0.6398815	total: 90.9ms	remaining: 3.27s
    27:	learn: 0.6381166	total: 92ms	remaining: 3.19s
    28:	learn: 0.6364274	total: 93.2ms	remaining: 3.12s
    29:	learn: 0.6344558	total: 94.4ms	remaining: 3.05s
    30:	learn: 0.6326463	total: 96ms	remaining: 3s
    31:	learn: 0.6306376	total: 97.4ms	remaining: 2.95s
    32:	learn: 0.6290271	total: 98.6ms	remaining: 2.89s
    33:	learn: 0.6274379	total: 101ms	remaining: 2.86s
    34:	learn: 0.6257987	total: 102ms	remaining: 2.81s
    35:	learn: 0.6240313	total: 103ms	remaining: 2.76s
    36:	learn: 0.6221788	total: 104ms	remaining: 2.72s
    37:	learn: 0.6203680	total: 106ms	remaining: 2.67s
    38:	learn: 0.6188713	total: 106ms	remaining: 2.62s
    39:	learn: 0.6171431	total: 108ms	remaining: 2.58s
    40:	learn: 0.6154516	total: 109ms	remaining: 2.54s
    41:	learn: 0.6139617	total: 110ms	remaining: 2.51s
    42:	learn: 0.6122283	total: 112ms	remaining: 2.49s
    43:	learn: 0.6106340	total: 113ms	remaining: 2.46s
    44:	learn: 0.6088500	total: 115ms	remaining: 2.44s
    45:	learn: 0.6074363	total: 116ms	remaining: 2.41s
    46:	learn: 0.6058891	total: 118ms	remaining: 2.38s
    47:	learn: 0.6043252	total: 118ms	remaining: 2.35s
    48:	learn: 0.6021648	total: 120ms	remaining: 2.32s
    49:	learn: 0.6007347	total: 121ms	remaining: 2.3s
    50:	learn: 0.5992056	total: 122ms	remaining: 2.27s
    51:	learn: 0.5978049	total: 124ms	remaining: 2.26s
    52:	learn: 0.5963823	total: 126ms	remaining: 2.24s
    53:	learn: 0.5949289	total: 127ms	remaining: 2.22s
    54:	learn: 0.5933712	total: 128ms	remaining: 2.2s
    55:	learn: 0.5915033	total: 129ms	remaining: 2.18s
    56:	learn: 0.5898480	total: 130ms	remaining: 2.15s
    57:	learn: 0.5886154	total: 131ms	remaining: 2.13s
    58:	learn: 0.5869565	total: 133ms	remaining: 2.11s
    59:	learn: 0.5851382	total: 134ms	remaining: 2.09s
    60:	learn: 0.5836114	total: 135ms	remaining: 2.08s
    61:	learn: 0.5822439	total: 137ms	remaining: 2.07s
    62:	learn: 0.5804612	total: 139ms	remaining: 2.06s
    63:	learn: 0.5784955	total: 140ms	remaining: 2.05s
    64:	learn: 0.5771415	total: 141ms	remaining: 2.03s
    65:	learn: 0.5753148	total: 142ms	remaining: 2.01s
    66:	learn: 0.5738886	total: 144ms	remaining: 2.01s
    67:	learn: 0.5725165	total: 146ms	remaining: 2s
    68:	learn: 0.5712178	total: 147ms	remaining: 1.98s
    69:	learn: 0.5699471	total: 148ms	remaining: 1.97s
    70:	learn: 0.5682378	total: 151ms	remaining: 1.97s
    71:	learn: 0.5667725	total: 153ms	remaining: 1.97s
    72:	learn: 0.5653399	total: 155ms	remaining: 1.97s
    73:	learn: 0.5637511	total: 157ms	remaining: 1.96s
    74:	learn: 0.5622062	total: 158ms	remaining: 1.95s
    75:	learn: 0.5608091	total: 160ms	remaining: 1.95s
    76:	learn: 0.5592533	total: 162ms	remaining: 1.94s
    77:	learn: 0.5578391	total: 164ms	remaining: 1.94s
    78:	learn: 0.5561123	total: 166ms	remaining: 1.93s
    79:	learn: 0.5546528	total: 168ms	remaining: 1.93s
    80:	learn: 0.5531912	total: 169ms	remaining: 1.92s
    81:	learn: 0.5515071	total: 171ms	remaining: 1.92s
    82:	learn: 0.5502822	total: 173ms	remaining: 1.91s
    83:	learn: 0.5488514	total: 174ms	remaining: 1.9s
    84:	learn: 0.5476948	total: 175ms	remaining: 1.88s
    85:	learn: 0.5462958	total: 176ms	remaining: 1.87s
    86:	learn: 0.5448845	total: 177ms	remaining: 1.86s
    87:	learn: 0.5434695	total: 179ms	remaining: 1.85s
    88:	learn: 0.5422303	total: 180ms	remaining: 1.84s
    89:	learn: 0.5408043	total: 181ms	remaining: 1.83s
    90:	learn: 0.5393050	total: 183ms	remaining: 1.82s
    91:	learn: 0.5379353	total: 184ms	remaining: 1.82s
    92:	learn: 0.5362748	total: 186ms	remaining: 1.81s
    93:	learn: 0.5347720	total: 187ms	remaining: 1.8s
    94:	learn: 0.5330830	total: 188ms	remaining: 1.79s
    95:	learn: 0.5317144	total: 189ms	remaining: 1.78s
    96:	learn: 0.5304413	total: 190ms	remaining: 1.77s
    97:	learn: 0.5291161	total: 192ms	remaining: 1.76s
    98:	learn: 0.5276380	total: 193ms	remaining: 1.75s
    99:	learn: 0.5261708	total: 195ms	remaining: 1.75s
    100:	learn: 0.5247355	total: 196ms	remaining: 1.75s
    101:	learn: 0.5237122	total: 197ms	remaining: 1.74s
    102:	learn: 0.5223774	total: 199ms	remaining: 1.73s
    103:	learn: 0.5210040	total: 200ms	remaining: 1.72s
    104:	learn: 0.5195823	total: 201ms	remaining: 1.71s
    105:	learn: 0.5182128	total: 202ms	remaining: 1.7s
    106:	learn: 0.5171010	total: 203ms	remaining: 1.7s
    107:	learn: 0.5154453	total: 205ms	remaining: 1.69s
    108:	learn: 0.5140762	total: 206ms	remaining: 1.68s
    109:	learn: 0.5128809	total: 207ms	remaining: 1.68s
    110:	learn: 0.5113136	total: 208ms	remaining: 1.67s
    111:	learn: 0.5099528	total: 210ms	remaining: 1.66s
    112:	learn: 0.5087099	total: 211ms	remaining: 1.65s
    113:	learn: 0.5075710	total: 214ms	remaining: 1.66s
    114:	learn: 0.5063746	total: 216ms	remaining: 1.66s
    115:	learn: 0.5052392	total: 218ms	remaining: 1.66s
    116:	learn: 0.5039981	total: 219ms	remaining: 1.66s
    117:	learn: 0.5027485	total: 221ms	remaining: 1.65s
    118:	learn: 0.5017238	total: 222ms	remaining: 1.65s
    119:	learn: 0.5005180	total: 224ms	remaining: 1.64s
    120:	learn: 0.4995321	total: 225ms	remaining: 1.63s
    121:	learn: 0.4982390	total: 226ms	remaining: 1.63s
    122:	learn: 0.4965852	total: 227ms	remaining: 1.62s
    123:	learn: 0.4955757	total: 229ms	remaining: 1.61s
    124:	learn: 0.4945564	total: 230ms	remaining: 1.61s
    125:	learn: 0.4934651	total: 232ms	remaining: 1.61s
    126:	learn: 0.4922702	total: 233ms	remaining: 1.6s
    127:	learn: 0.4911709	total: 236ms	remaining: 1.6s
    128:	learn: 0.4898813	total: 237ms	remaining: 1.6s
    129:	learn: 0.4889741	total: 239ms	remaining: 1.6s
    130:	learn: 0.4878748	total: 240ms	remaining: 1.59s
    131:	learn: 0.4868282	total: 241ms	remaining: 1.58s
    132:	learn: 0.4857749	total: 248ms	remaining: 1.61s
    133:	learn: 0.4849637	total: 249ms	remaining: 1.61s
    134:	learn: 0.4837374	total: 250ms	remaining: 1.6s
    135:	learn: 0.4827893	total: 251ms	remaining: 1.6s
    136:	learn: 0.4816063	total: 253ms	remaining: 1.59s
    137:	learn: 0.4805441	total: 254ms	remaining: 1.59s
    138:	learn: 0.4791551	total: 255ms	remaining: 1.58s
    139:	learn: 0.4780436	total: 257ms	remaining: 1.58s
    140:	learn: 0.4771532	total: 258ms	remaining: 1.57s
    141:	learn: 0.4758364	total: 259ms	remaining: 1.57s
    142:	learn: 0.4749786	total: 261ms	remaining: 1.56s
    143:	learn: 0.4737938	total: 263ms	remaining: 1.56s
    144:	learn: 0.4726401	total: 264ms	remaining: 1.56s
    145:	learn: 0.4716186	total: 266ms	remaining: 1.55s
    146:	learn: 0.4704294	total: 267ms	remaining: 1.55s
    147:	learn: 0.4692975	total: 268ms	remaining: 1.54s
    148:	learn: 0.4682360	total: 269ms	remaining: 1.53s
    149:	learn: 0.4670508	total: 270ms	remaining: 1.53s
    150:	learn: 0.4661076	total: 271ms	remaining: 1.52s
    151:	learn: 0.4652287	total: 272ms	remaining: 1.52s
    152:	learn: 0.4640738	total: 274ms	remaining: 1.51s
    153:	learn: 0.4630563	total: 276ms	remaining: 1.51s
    154:	learn: 0.4620782	total: 278ms	remaining: 1.51s
    155:	learn: 0.4608829	total: 280ms	remaining: 1.51s
    156:	learn: 0.4599296	total: 281ms	remaining: 1.51s
    157:	learn: 0.4591482	total: 284ms	remaining: 1.52s
    158:	learn: 0.4579964	total: 287ms	remaining: 1.52s
    159:	learn: 0.4570654	total: 288ms	remaining: 1.51s
    160:	learn: 0.4560071	total: 289ms	remaining: 1.51s
    161:	learn: 0.4551913	total: 290ms	remaining: 1.5s
    162:	learn: 0.4540082	total: 292ms	remaining: 1.5s
    163:	learn: 0.4528849	total: 293ms	remaining: 1.49s
    164:	learn: 0.4518167	total: 294ms	remaining: 1.49s
    165:	learn: 0.4507635	total: 295ms	remaining: 1.48s
    166:	learn: 0.4499163	total: 296ms	remaining: 1.48s
    167:	learn: 0.4483632	total: 299ms	remaining: 1.48s
    168:	learn: 0.4473301	total: 300ms	remaining: 1.47s
    169:	learn: 0.4463483	total: 301ms	remaining: 1.47s
    170:	learn: 0.4452950	total: 302ms	remaining: 1.47s
    171:	learn: 0.4444347	total: 303ms	remaining: 1.46s
    172:	learn: 0.4433858	total: 304ms	remaining: 1.46s
    173:	learn: 0.4424606	total: 306ms	remaining: 1.45s
    174:	learn: 0.4414629	total: 307ms	remaining: 1.45s
    175:	learn: 0.4406022	total: 308ms	remaining: 1.44s
    176:	learn: 0.4396689	total: 309ms	remaining: 1.44s
    177:	learn: 0.4387505	total: 312ms	remaining: 1.44s
    178:	learn: 0.4377719	total: 313ms	remaining: 1.44s
    179:	learn: 0.4369275	total: 314ms	remaining: 1.43s
    180:	learn: 0.4358432	total: 316ms	remaining: 1.43s
    181:	learn: 0.4348113	total: 317ms	remaining: 1.42s
    182:	learn: 0.4339665	total: 318ms	remaining: 1.42s
    183:	learn: 0.4330720	total: 319ms	remaining: 1.42s
    184:	learn: 0.4321389	total: 320ms	remaining: 1.41s
    185:	learn: 0.4311181	total: 322ms	remaining: 1.41s
    186:	learn: 0.4301325	total: 323ms	remaining: 1.41s
    187:	learn: 0.4291398	total: 325ms	remaining: 1.4s
    188:	learn: 0.4280697	total: 326ms	remaining: 1.4s
    189:	learn: 0.4270324	total: 328ms	remaining: 1.4s
    190:	learn: 0.4262547	total: 329ms	remaining: 1.39s
    191:	learn: 0.4253000	total: 332ms	remaining: 1.4s
    192:	learn: 0.4244767	total: 333ms	remaining: 1.39s
    193:	learn: 0.4234863	total: 335ms	remaining: 1.39s
    194:	learn: 0.4222637	total: 337ms	remaining: 1.39s
    195:	learn: 0.4212016	total: 338ms	remaining: 1.39s
    196:	learn: 0.4202589	total: 339ms	remaining: 1.38s
    197:	learn: 0.4191853	total: 344ms	remaining: 1.39s
    198:	learn: 0.4182935	total: 347ms	remaining: 1.39s
    199:	learn: 0.4172772	total: 349ms	remaining: 1.39s
    200:	learn: 0.4163435	total: 350ms	remaining: 1.39s
    201:	learn: 0.4152868	total: 351ms	remaining: 1.39s
    202:	learn: 0.4142657	total: 352ms	remaining: 1.38s
    203:	learn: 0.4134885	total: 354ms	remaining: 1.38s
    204:	learn: 0.4127018	total: 355ms	remaining: 1.38s
    205:	learn: 0.4116318	total: 356ms	remaining: 1.37s
    206:	learn: 0.4107109	total: 357ms	remaining: 1.37s
    207:	learn: 0.4096986	total: 359ms	remaining: 1.37s
    208:	learn: 0.4089435	total: 361ms	remaining: 1.37s
    209:	learn: 0.4079575	total: 363ms	remaining: 1.36s
    210:	learn: 0.4068538	total: 364ms	remaining: 1.36s
    211:	learn: 0.4059947	total: 365ms	remaining: 1.36s
    212:	learn: 0.4051675	total: 366ms	remaining: 1.35s
    213:	learn: 0.4044280	total: 368ms	remaining: 1.35s
    214:	learn: 0.4035601	total: 369ms	remaining: 1.35s
    215:	learn: 0.4027667	total: 370ms	remaining: 1.34s
    216:	learn: 0.4018758	total: 372ms	remaining: 1.34s
    217:	learn: 0.4008227	total: 374ms	remaining: 1.34s
    218:	learn: 0.3998099	total: 392ms	remaining: 1.4s
    219:	learn: 0.3988106	total: 401ms	remaining: 1.42s
    220:	learn: 0.3977956	total: 412ms	remaining: 1.45s
    221:	learn: 0.3970227	total: 415ms	remaining: 1.45s
    222:	learn: 0.3962992	total: 434ms	remaining: 1.51s
    223:	learn: 0.3955867	total: 439ms	remaining: 1.52s
    224:	learn: 0.3948700	total: 441ms	remaining: 1.52s
    225:	learn: 0.3939560	total: 445ms	remaining: 1.52s
    226:	learn: 0.3932772	total: 449ms	remaining: 1.53s
    227:	learn: 0.3925000	total: 454ms	remaining: 1.53s
    228:	learn: 0.3917208	total: 458ms	remaining: 1.54s
    229:	learn: 0.3910261	total: 462ms	remaining: 1.55s
    230:	learn: 0.3901264	total: 466ms	remaining: 1.55s
    231:	learn: 0.3893815	total: 470ms	remaining: 1.55s
    232:	learn: 0.3886140	total: 474ms	remaining: 1.56s
    233:	learn: 0.3878760	total: 478ms	remaining: 1.56s
    234:	learn: 0.3869758	total: 482ms	remaining: 1.57s
    235:	learn: 0.3861070	total: 486ms	remaining: 1.57s
    236:	learn: 0.3852621	total: 490ms	remaining: 1.58s
    237:	learn: 0.3845646	total: 494ms	remaining: 1.58s
    238:	learn: 0.3838317	total: 498ms	remaining: 1.59s
    239:	learn: 0.3829835	total: 501ms	remaining: 1.59s
    240:	learn: 0.3820073	total: 506ms	remaining: 1.59s
    241:	learn: 0.3814470	total: 510ms	remaining: 1.6s
    242:	learn: 0.3807217	total: 514ms	remaining: 1.6s
    243:	learn: 0.3799907	total: 525ms	remaining: 1.63s
    244:	learn: 0.3791525	total: 528ms	remaining: 1.63s
    245:	learn: 0.3782599	total: 532ms	remaining: 1.63s
    246:	learn: 0.3776954	total: 536ms	remaining: 1.63s
    247:	learn: 0.3771596	total: 539ms	remaining: 1.63s
    248:	learn: 0.3764676	total: 542ms	remaining: 1.64s
    249:	learn: 0.3756033	total: 547ms	remaining: 1.64s
    250:	learn: 0.3748574	total: 551ms	remaining: 1.64s
    251:	learn: 0.3741521	total: 555ms	remaining: 1.65s
    252:	learn: 0.3735373	total: 559ms	remaining: 1.65s
    253:	learn: 0.3726707	total: 563ms	remaining: 1.65s
    254:	learn: 0.3719370	total: 567ms	remaining: 1.66s
    255:	learn: 0.3712878	total: 572ms	remaining: 1.66s
    256:	learn: 0.3706344	total: 576ms	remaining: 1.66s
    257:	learn: 0.3698126	total: 580ms	remaining: 1.67s
    258:	learn: 0.3692689	total: 611ms	remaining: 1.75s
    259:	learn: 0.3684632	total: 614ms	remaining: 1.75s
    260:	learn: 0.3677466	total: 619ms	remaining: 1.75s
    261:	learn: 0.3671327	total: 624ms	remaining: 1.76s
    262:	learn: 0.3665564	total: 628ms	remaining: 1.76s
    263:	learn: 0.3659407	total: 640ms	remaining: 1.78s
    264:	learn: 0.3651113	total: 642ms	remaining: 1.78s
    265:	learn: 0.3642668	total: 646ms	remaining: 1.78s
    266:	learn: 0.3635892	total: 650ms	remaining: 1.78s
    267:	learn: 0.3630675	total: 655ms	remaining: 1.79s
    268:	learn: 0.3624013	total: 659ms	remaining: 1.79s
    269:	learn: 0.3616951	total: 665ms	remaining: 1.8s
    270:	learn: 0.3609394	total: 672ms	remaining: 1.81s
    271:	learn: 0.3601518	total: 674ms	remaining: 1.8s
    272:	learn: 0.3595363	total: 680ms	remaining: 1.81s
    273:	learn: 0.3587891	total: 685ms	remaining: 1.81s
    274:	learn: 0.3581906	total: 689ms	remaining: 1.81s
    275:	learn: 0.3575159	total: 692ms	remaining: 1.82s
    276:	learn: 0.3570034	total: 697ms	remaining: 1.82s
    277:	learn: 0.3564511	total: 699ms	remaining: 1.81s
    278:	learn: 0.3557321	total: 703ms	remaining: 1.82s
    279:	learn: 0.3550267	total: 707ms	remaining: 1.82s
    280:	learn: 0.3540649	total: 712ms	remaining: 1.82s
    281:	learn: 0.3534089	total: 717ms	remaining: 1.83s
    282:	learn: 0.3524684	total: 722ms	remaining: 1.83s
    283:	learn: 0.3517167	total: 727ms	remaining: 1.83s
    284:	learn: 0.3511599	total: 730ms	remaining: 1.83s
    285:	learn: 0.3505288	total: 744ms	remaining: 1.86s
    286:	learn: 0.3497674	total: 747ms	remaining: 1.85s
    287:	learn: 0.3490211	total: 758ms	remaining: 1.87s
    288:	learn: 0.3486808	total: 762ms	remaining: 1.87s
    289:	learn: 0.3482409	total: 765ms	remaining: 1.87s
    290:	learn: 0.3476104	total: 769ms	remaining: 1.87s
    291:	learn: 0.3471115	total: 773ms	remaining: 1.88s
    292:	learn: 0.3464207	total: 778ms	remaining: 1.88s
    293:	learn: 0.3457073	total: 783ms	remaining: 1.88s
    294:	learn: 0.3448493	total: 786ms	remaining: 1.88s
    295:	learn: 0.3442764	total: 790ms	remaining: 1.88s
    296:	learn: 0.3437679	total: 794ms	remaining: 1.88s
    297:	learn: 0.3431912	total: 798ms	remaining: 1.88s
    298:	learn: 0.3425625	total: 803ms	remaining: 1.88s
    299:	learn: 0.3418909	total: 811ms	remaining: 1.89s
    300:	learn: 0.3411826	total: 813ms	remaining: 1.89s
    301:	learn: 0.3404468	total: 817ms	remaining: 1.89s
    302:	learn: 0.3397911	total: 822ms	remaining: 1.89s
    303:	learn: 0.3392819	total: 826ms	remaining: 1.89s
    304:	learn: 0.3386785	total: 831ms	remaining: 1.89s
    305:	learn: 0.3381723	total: 835ms	remaining: 1.89s
    306:	learn: 0.3374953	total: 839ms	remaining: 1.89s
    307:	learn: 0.3368311	total: 843ms	remaining: 1.89s
    308:	learn: 0.3361975	total: 847ms	remaining: 1.89s
    309:	learn: 0.3355929	total: 848ms	remaining: 1.89s
    310:	learn: 0.3351610	total: 856ms	remaining: 1.9s
    311:	learn: 0.3344132	total: 858ms	remaining: 1.89s
    312:	learn: 0.3338982	total: 860ms	remaining: 1.89s
    313:	learn: 0.3331771	total: 866ms	remaining: 1.89s
    314:	learn: 0.3325166	total: 868ms	remaining: 1.89s
    315:	learn: 0.3319689	total: 870ms	remaining: 1.88s
    316:	learn: 0.3314436	total: 888ms	remaining: 1.91s
    317:	learn: 0.3308998	total: 905ms	remaining: 1.94s
    318:	learn: 0.3301422	total: 908ms	remaining: 1.94s
    319:	learn: 0.3295908	total: 915ms	remaining: 1.94s
    320:	learn: 0.3289123	total: 916ms	remaining: 1.94s
    321:	learn: 0.3281617	total: 918ms	remaining: 1.93s
    322:	learn: 0.3276499	total: 920ms	remaining: 1.93s
    323:	learn: 0.3271874	total: 926ms	remaining: 1.93s
    324:	learn: 0.3264112	total: 931ms	remaining: 1.93s
    325:	learn: 0.3258449	total: 937ms	remaining: 1.94s
    326:	learn: 0.3250780	total: 941ms	remaining: 1.94s
    327:	learn: 0.3241434	total: 973ms	remaining: 1.99s
    328:	learn: 0.3233908	total: 988ms	remaining: 2.01s
    329:	learn: 0.3229982	total: 990ms	remaining: 2.01s
    330:	learn: 0.3225123	total: 994ms	remaining: 2.01s
    331:	learn: 0.3220280	total: 998ms	remaining: 2.01s
    332:	learn: 0.3214247	total: 1.01s	remaining: 2.02s
    333:	learn: 0.3208399	total: 1.01s	remaining: 2.01s
    334:	learn: 0.3203381	total: 1.02s	remaining: 2.02s
    335:	learn: 0.3198142	total: 1.02s	remaining: 2.02s
    336:	learn: 0.3192629	total: 1.02s	remaining: 2.02s
    337:	learn: 0.3186935	total: 1.03s	remaining: 2.02s
    338:	learn: 0.3179686	total: 1.03s	remaining: 2.01s
    339:	learn: 0.3174602	total: 1.04s	remaining: 2.02s
    340:	learn: 0.3168891	total: 1.04s	remaining: 2.02s
    341:	learn: 0.3163692	total: 1.05s	remaining: 2.02s
    342:	learn: 0.3159716	total: 1.05s	remaining: 2.01s
    343:	learn: 0.3154536	total: 1.06s	remaining: 2.02s
    344:	learn: 0.3149685	total: 1.07s	remaining: 2.03s
    345:	learn: 0.3144024	total: 1.07s	remaining: 2.02s
    346:	learn: 0.3140353	total: 1.08s	remaining: 2.02s
    347:	learn: 0.3135035	total: 1.08s	remaining: 2.02s
    348:	learn: 0.3130855	total: 1.08s	remaining: 2.02s
    349:	learn: 0.3126477	total: 1.09s	remaining: 2.02s
    350:	learn: 0.3121701	total: 1.09s	remaining: 2.03s
    351:	learn: 0.3115854	total: 1.1s	remaining: 2.03s
    352:	learn: 0.3111464	total: 1.11s	remaining: 2.04s
    353:	learn: 0.3104357	total: 1.12s	remaining: 2.05s
    354:	learn: 0.3098581	total: 1.14s	remaining: 2.07s
    355:	learn: 0.3093025	total: 1.14s	remaining: 2.06s
    356:	learn: 0.3085438	total: 1.14s	remaining: 2.06s
    357:	learn: 0.3081124	total: 1.16s	remaining: 2.07s
    358:	learn: 0.3074302	total: 1.16s	remaining: 2.07s
    359:	learn: 0.3069290	total: 1.17s	remaining: 2.07s
    360:	learn: 0.3063572	total: 1.17s	remaining: 2.07s
    361:	learn: 0.3058487	total: 1.17s	remaining: 2.07s
    362:	learn: 0.3053731	total: 1.18s	remaining: 2.07s
    363:	learn: 0.3049950	total: 1.18s	remaining: 2.06s
    364:	learn: 0.3044817	total: 1.19s	remaining: 2.06s
    365:	learn: 0.3041031	total: 1.19s	remaining: 2.06s
    366:	learn: 0.3036329	total: 1.19s	remaining: 2.06s
    367:	learn: 0.3029988	total: 1.2s	remaining: 2.06s
    368:	learn: 0.3025647	total: 1.2s	remaining: 2.06s
    369:	learn: 0.3022169	total: 1.21s	remaining: 2.06s
    370:	learn: 0.3016009	total: 1.21s	remaining: 2.05s
    371:	learn: 0.3011917	total: 1.22s	remaining: 2.05s
    372:	learn: 0.3007721	total: 1.22s	remaining: 2.05s
    373:	learn: 0.3002796	total: 1.22s	remaining: 2.05s
    374:	learn: 0.2998885	total: 1.23s	remaining: 2.05s
    375:	learn: 0.2993784	total: 1.23s	remaining: 2.05s
    376:	learn: 0.2989046	total: 1.24s	remaining: 2.05s
    377:	learn: 0.2984268	total: 1.24s	remaining: 2.05s
    378:	learn: 0.2980376	total: 1.25s	remaining: 2.05s
    379:	learn: 0.2976386	total: 1.25s	remaining: 2.05s
    380:	learn: 0.2971803	total: 1.26s	remaining: 2.05s
    381:	learn: 0.2966664	total: 1.28s	remaining: 2.06s
    382:	learn: 0.2961564	total: 1.28s	remaining: 2.07s
    383:	learn: 0.2956202	total: 1.29s	remaining: 2.06s
    384:	learn: 0.2951527	total: 1.29s	remaining: 2.06s
    385:	learn: 0.2947671	total: 1.3s	remaining: 2.07s
    386:	learn: 0.2942167	total: 1.3s	remaining: 2.06s
    387:	learn: 0.2938094	total: 1.31s	remaining: 2.06s
    388:	learn: 0.2934343	total: 1.31s	remaining: 2.06s
    389:	learn: 0.2929567	total: 1.32s	remaining: 2.06s
    390:	learn: 0.2925197	total: 1.33s	remaining: 2.08s
    391:	learn: 0.2921517	total: 1.34s	remaining: 2.07s
    392:	learn: 0.2915990	total: 1.34s	remaining: 2.07s
    393:	learn: 0.2911530	total: 1.35s	remaining: 2.08s
    394:	learn: 0.2907136	total: 1.36s	remaining: 2.08s
    395:	learn: 0.2901805	total: 1.36s	remaining: 2.08s
    396:	learn: 0.2897985	total: 1.37s	remaining: 2.08s
    397:	learn: 0.2893307	total: 1.38s	remaining: 2.09s
    398:	learn: 0.2887063	total: 1.39s	remaining: 2.09s
    399:	learn: 0.2883666	total: 1.39s	remaining: 2.09s
    400:	learn: 0.2880451	total: 1.4s	remaining: 2.08s
    401:	learn: 0.2875984	total: 1.4s	remaining: 2.08s
    402:	learn: 0.2871877	total: 1.4s	remaining: 2.08s
    403:	learn: 0.2868942	total: 1.41s	remaining: 2.07s
    404:	learn: 0.2864472	total: 1.41s	remaining: 2.07s
    405:	learn: 0.2858871	total: 1.41s	remaining: 2.07s
    406:	learn: 0.2854912	total: 1.42s	remaining: 2.06s
    407:	learn: 0.2848975	total: 1.42s	remaining: 2.06s
    408:	learn: 0.2843898	total: 1.42s	remaining: 2.06s
    409:	learn: 0.2840452	total: 1.43s	remaining: 2.05s
    410:	learn: 0.2834345	total: 1.43s	remaining: 2.04s
    411:	learn: 0.2830059	total: 1.43s	remaining: 2.04s
    412:	learn: 0.2826489	total: 1.43s	remaining: 2.04s
    413:	learn: 0.2821333	total: 1.44s	remaining: 2.03s
    414:	learn: 0.2816383	total: 1.44s	remaining: 2.03s
    415:	learn: 0.2810573	total: 1.44s	remaining: 2.03s
    416:	learn: 0.2806727	total: 1.45s	remaining: 2.03s
    417:	learn: 0.2802031	total: 1.46s	remaining: 2.03s
    418:	learn: 0.2796201	total: 1.46s	remaining: 2.02s
    419:	learn: 0.2791867	total: 1.46s	remaining: 2.02s
    420:	learn: 0.2786717	total: 1.47s	remaining: 2.02s
    421:	learn: 0.2782879	total: 1.47s	remaining: 2.02s
    422:	learn: 0.2779091	total: 1.5s	remaining: 2.04s
    423:	learn: 0.2773204	total: 1.5s	remaining: 2.04s
    424:	learn: 0.2767898	total: 1.5s	remaining: 2.03s
    425:	learn: 0.2763485	total: 1.51s	remaining: 2.03s
    426:	learn: 0.2760821	total: 1.51s	remaining: 2.03s
    427:	learn: 0.2756501	total: 1.51s	remaining: 2.02s
    428:	learn: 0.2753336	total: 1.52s	remaining: 2.02s
    429:	learn: 0.2748098	total: 1.52s	remaining: 2.02s
    430:	learn: 0.2742634	total: 1.53s	remaining: 2.02s
    431:	learn: 0.2738034	total: 1.53s	remaining: 2.01s
    432:	learn: 0.2733852	total: 1.54s	remaining: 2.01s
    433:	learn: 0.2730193	total: 1.54s	remaining: 2.01s
    434:	learn: 0.2723791	total: 1.54s	remaining: 2s
    435:	learn: 0.2720132	total: 1.55s	remaining: 2s
    436:	learn: 0.2717234	total: 1.55s	remaining: 2s
    437:	learn: 0.2713446	total: 1.56s	remaining: 2s
    438:	learn: 0.2709566	total: 1.56s	remaining: 1.99s
    439:	learn: 0.2706907	total: 1.56s	remaining: 1.99s
    440:	learn: 0.2704036	total: 1.56s	remaining: 1.98s
    441:	learn: 0.2701125	total: 1.56s	remaining: 1.98s
    442:	learn: 0.2696652	total: 1.57s	remaining: 1.97s
    443:	learn: 0.2692436	total: 1.57s	remaining: 1.96s
    444:	learn: 0.2688672	total: 1.57s	remaining: 1.96s
    445:	learn: 0.2686117	total: 1.57s	remaining: 1.95s
    446:	learn: 0.2682218	total: 1.57s	remaining: 1.95s
    447:	learn: 0.2678798	total: 1.57s	remaining: 1.94s
    448:	learn: 0.2676169	total: 1.58s	remaining: 1.94s
    449:	learn: 0.2670965	total: 1.59s	remaining: 1.94s
    450:	learn: 0.2667749	total: 1.59s	remaining: 1.94s
    451:	learn: 0.2663350	total: 1.59s	remaining: 1.93s
    452:	learn: 0.2657786	total: 1.59s	remaining: 1.92s
    453:	learn: 0.2654621	total: 1.6s	remaining: 1.93s
    454:	learn: 0.2651185	total: 1.6s	remaining: 1.92s
    455:	learn: 0.2646181	total: 1.6s	remaining: 1.91s
    456:	learn: 0.2640489	total: 1.61s	remaining: 1.91s
    457:	learn: 0.2637333	total: 1.61s	remaining: 1.91s
    458:	learn: 0.2633727	total: 1.61s	remaining: 1.9s
    459:	learn: 0.2629595	total: 1.62s	remaining: 1.9s
    460:	learn: 0.2625130	total: 1.62s	remaining: 1.9s
    461:	learn: 0.2620180	total: 1.62s	remaining: 1.89s
    462:	learn: 0.2616593	total: 1.63s	remaining: 1.89s
    463:	learn: 0.2610619	total: 1.63s	remaining: 1.88s
    464:	learn: 0.2608050	total: 1.63s	remaining: 1.88s
    465:	learn: 0.2604130	total: 1.64s	remaining: 1.88s
    466:	learn: 0.2600092	total: 1.64s	remaining: 1.87s
    467:	learn: 0.2594932	total: 1.64s	remaining: 1.87s
    468:	learn: 0.2590833	total: 1.65s	remaining: 1.87s
    469:	learn: 0.2586633	total: 1.66s	remaining: 1.87s
    470:	learn: 0.2582646	total: 1.66s	remaining: 1.86s
    471:	learn: 0.2578490	total: 1.66s	remaining: 1.85s
    472:	learn: 0.2575844	total: 1.66s	remaining: 1.85s
    473:	learn: 0.2570344	total: 1.67s	remaining: 1.85s
    474:	learn: 0.2567278	total: 1.67s	remaining: 1.84s
    475:	learn: 0.2564029	total: 1.67s	remaining: 1.84s
    476:	learn: 0.2560195	total: 1.68s	remaining: 1.84s
    477:	learn: 0.2557816	total: 1.68s	remaining: 1.83s
    478:	learn: 0.2554329	total: 1.68s	remaining: 1.83s
    479:	learn: 0.2550979	total: 1.69s	remaining: 1.83s
    480:	learn: 0.2548465	total: 1.69s	remaining: 1.82s
    481:	learn: 0.2545279	total: 1.69s	remaining: 1.82s
    482:	learn: 0.2540403	total: 1.7s	remaining: 1.81s
    483:	learn: 0.2536986	total: 1.7s	remaining: 1.81s
    484:	learn: 0.2533229	total: 1.7s	remaining: 1.8s
    485:	learn: 0.2529596	total: 1.7s	remaining: 1.8s
    486:	learn: 0.2526620	total: 1.7s	remaining: 1.79s
    487:	learn: 0.2522225	total: 1.7s	remaining: 1.78s
    488:	learn: 0.2518720	total: 1.71s	remaining: 1.79s
    489:	learn: 0.2516374	total: 1.71s	remaining: 1.78s
    490:	learn: 0.2512580	total: 1.71s	remaining: 1.77s
    491:	learn: 0.2509041	total: 1.71s	remaining: 1.77s
    492:	learn: 0.2505165	total: 1.72s	remaining: 1.77s
    493:	learn: 0.2501407	total: 1.72s	remaining: 1.76s
    494:	learn: 0.2497492	total: 1.72s	remaining: 1.76s
    495:	learn: 0.2494590	total: 1.72s	remaining: 1.75s
    496:	learn: 0.2491722	total: 1.73s	remaining: 1.75s
    497:	learn: 0.2488336	total: 1.73s	remaining: 1.74s
    498:	learn: 0.2484837	total: 1.74s	remaining: 1.74s
    499:	learn: 0.2481679	total: 1.74s	remaining: 1.74s
    500:	learn: 0.2478490	total: 1.74s	remaining: 1.74s
    501:	learn: 0.2473930	total: 1.75s	remaining: 1.73s
    502:	learn: 0.2469644	total: 1.76s	remaining: 1.74s
    503:	learn: 0.2465281	total: 1.77s	remaining: 1.74s
    504:	learn: 0.2461485	total: 1.77s	remaining: 1.73s
    505:	learn: 0.2459035	total: 1.77s	remaining: 1.73s
    506:	learn: 0.2456081	total: 1.78s	remaining: 1.73s
    507:	learn: 0.2453251	total: 1.78s	remaining: 1.72s
    508:	learn: 0.2451622	total: 1.78s	remaining: 1.72s
    509:	learn: 0.2448403	total: 1.78s	remaining: 1.71s
    510:	learn: 0.2444679	total: 1.79s	remaining: 1.72s
    511:	learn: 0.2441152	total: 1.8s	remaining: 1.72s
    512:	learn: 0.2437654	total: 1.82s	remaining: 1.73s
    513:	learn: 0.2434364	total: 1.82s	remaining: 1.72s
    514:	learn: 0.2431713	total: 1.83s	remaining: 1.72s
    515:	learn: 0.2428516	total: 1.83s	remaining: 1.72s
    516:	learn: 0.2423800	total: 1.83s	remaining: 1.71s
    517:	learn: 0.2420030	total: 1.84s	remaining: 1.71s
    518:	learn: 0.2416189	total: 1.84s	remaining: 1.71s
    519:	learn: 0.2413112	total: 1.84s	remaining: 1.7s
    520:	learn: 0.2409968	total: 1.84s	remaining: 1.7s
    521:	learn: 0.2407144	total: 1.85s	remaining: 1.69s
    522:	learn: 0.2405005	total: 1.85s	remaining: 1.69s
    523:	learn: 0.2400975	total: 1.85s	remaining: 1.69s
    524:	learn: 0.2396519	total: 1.86s	remaining: 1.68s
    525:	learn: 0.2393105	total: 1.86s	remaining: 1.68s
    526:	learn: 0.2390107	total: 1.87s	remaining: 1.68s
    527:	learn: 0.2387796	total: 1.87s	remaining: 1.67s
    528:	learn: 0.2383853	total: 1.88s	remaining: 1.67s
    529:	learn: 0.2380187	total: 1.89s	remaining: 1.67s
    530:	learn: 0.2377028	total: 1.89s	remaining: 1.67s
    531:	learn: 0.2375156	total: 1.89s	remaining: 1.66s
    532:	learn: 0.2372164	total: 1.9s	remaining: 1.66s
    533:	learn: 0.2369146	total: 1.91s	remaining: 1.66s
    534:	learn: 0.2366065	total: 1.91s	remaining: 1.66s
    535:	learn: 0.2363448	total: 1.91s	remaining: 1.65s
    536:	learn: 0.2360492	total: 1.92s	remaining: 1.65s
    537:	learn: 0.2358844	total: 1.92s	remaining: 1.65s
    538:	learn: 0.2356659	total: 1.93s	remaining: 1.65s
    539:	learn: 0.2352691	total: 1.94s	remaining: 1.65s
    540:	learn: 0.2350419	total: 1.94s	remaining: 1.65s
    541:	learn: 0.2347482	total: 1.95s	remaining: 1.65s
    542:	learn: 0.2344424	total: 1.95s	remaining: 1.64s
    543:	learn: 0.2342469	total: 1.96s	remaining: 1.65s
    544:	learn: 0.2339228	total: 1.97s	remaining: 1.64s
    545:	learn: 0.2336282	total: 1.97s	remaining: 1.64s
    546:	learn: 0.2333830	total: 1.98s	remaining: 1.64s
    547:	learn: 0.2329885	total: 1.98s	remaining: 1.63s
    548:	learn: 0.2327492	total: 1.99s	remaining: 1.64s
    549:	learn: 0.2324078	total: 2s	remaining: 1.64s
    550:	learn: 0.2321474	total: 2.01s	remaining: 1.64s
    551:	learn: 0.2318735	total: 2.02s	remaining: 1.64s
    552:	learn: 0.2314612	total: 2.03s	remaining: 1.64s
    553:	learn: 0.2310767	total: 2.04s	remaining: 1.64s
    554:	learn: 0.2309167	total: 2.04s	remaining: 1.64s
    555:	learn: 0.2305761	total: 2.05s	remaining: 1.64s
    556:	learn: 0.2302469	total: 2.06s	remaining: 1.63s
    557:	learn: 0.2299750	total: 2.06s	remaining: 1.63s
    558:	learn: 0.2297495	total: 2.08s	remaining: 1.64s
    559:	learn: 0.2294821	total: 2.08s	remaining: 1.63s
    560:	learn: 0.2292364	total: 2.08s	remaining: 1.63s
    561:	learn: 0.2289496	total: 2.08s	remaining: 1.62s
    562:	learn: 0.2286585	total: 2.09s	remaining: 1.62s
    563:	learn: 0.2284182	total: 2.1s	remaining: 1.62s
    564:	learn: 0.2280876	total: 2.1s	remaining: 1.62s
    565:	learn: 0.2278016	total: 2.1s	remaining: 1.61s
    566:	learn: 0.2275360	total: 2.1s	remaining: 1.61s
    567:	learn: 0.2273335	total: 2.11s	remaining: 1.61s
    568:	learn: 0.2270234	total: 2.12s	remaining: 1.6s
    569:	learn: 0.2267649	total: 2.12s	remaining: 1.6s
    570:	learn: 0.2265213	total: 2.12s	remaining: 1.59s
    571:	learn: 0.2262158	total: 2.13s	remaining: 1.59s
    572:	learn: 0.2260402	total: 2.13s	remaining: 1.59s
    573:	learn: 0.2256939	total: 2.14s	remaining: 1.59s
    574:	learn: 0.2255059	total: 2.15s	remaining: 1.59s
    575:	learn: 0.2252901	total: 2.15s	remaining: 1.59s
    576:	learn: 0.2250568	total: 2.17s	remaining: 1.59s
    577:	learn: 0.2247808	total: 2.17s	remaining: 1.58s
    578:	learn: 0.2243982	total: 2.18s	remaining: 1.58s
    579:	learn: 0.2240717	total: 2.18s	remaining: 1.58s
    580:	learn: 0.2237615	total: 2.19s	remaining: 1.58s
    581:	learn: 0.2235118	total: 2.19s	remaining: 1.57s
    582:	learn: 0.2233229	total: 2.21s	remaining: 1.58s
    583:	learn: 0.2230402	total: 2.22s	remaining: 1.58s
    584:	learn: 0.2226428	total: 2.23s	remaining: 1.58s
    585:	learn: 0.2222478	total: 2.23s	remaining: 1.58s
    586:	learn: 0.2219533	total: 2.24s	remaining: 1.58s
    587:	learn: 0.2217893	total: 2.24s	remaining: 1.57s
    588:	learn: 0.2215629	total: 2.25s	remaining: 1.57s
    589:	learn: 0.2213844	total: 2.25s	remaining: 1.57s
    590:	learn: 0.2211217	total: 2.26s	remaining: 1.56s
    591:	learn: 0.2208070	total: 2.26s	remaining: 1.56s
    592:	learn: 0.2205984	total: 2.27s	remaining: 1.56s
    593:	learn: 0.2204360	total: 2.27s	remaining: 1.55s
    594:	learn: 0.2202355	total: 2.28s	remaining: 1.55s
    595:	learn: 0.2199388	total: 2.28s	remaining: 1.55s
    596:	learn: 0.2196028	total: 2.28s	remaining: 1.54s
    597:	learn: 0.2194720	total: 2.29s	remaining: 1.54s
    598:	learn: 0.2191861	total: 2.29s	remaining: 1.53s
    599:	learn: 0.2189812	total: 2.29s	remaining: 1.53s
    600:	learn: 0.2187491	total: 2.3s	remaining: 1.52s
    601:	learn: 0.2184993	total: 2.31s	remaining: 1.52s
    602:	learn: 0.2182312	total: 2.31s	remaining: 1.52s
    603:	learn: 0.2179048	total: 2.32s	remaining: 1.52s
    604:	learn: 0.2176619	total: 2.32s	remaining: 1.51s
    605:	learn: 0.2174789	total: 2.33s	remaining: 1.51s
    606:	learn: 0.2173124	total: 2.34s	remaining: 1.51s
    607:	learn: 0.2170755	total: 2.34s	remaining: 1.51s
    608:	learn: 0.2167933	total: 2.35s	remaining: 1.51s
    609:	learn: 0.2165464	total: 2.35s	remaining: 1.5s
    610:	learn: 0.2163295	total: 2.36s	remaining: 1.5s
    611:	learn: 0.2161077	total: 2.36s	remaining: 1.5s
    612:	learn: 0.2159526	total: 2.37s	remaining: 1.5s
    613:	learn: 0.2157916	total: 2.37s	remaining: 1.49s
    614:	learn: 0.2155877	total: 2.39s	remaining: 1.49s
    615:	learn: 0.2153547	total: 2.39s	remaining: 1.49s
    616:	learn: 0.2151374	total: 2.39s	remaining: 1.48s
    617:	learn: 0.2148505	total: 2.4s	remaining: 1.48s
    618:	learn: 0.2146456	total: 2.4s	remaining: 1.48s
    619:	learn: 0.2143099	total: 2.41s	remaining: 1.48s
    620:	learn: 0.2141113	total: 2.41s	remaining: 1.47s
    621:	learn: 0.2138241	total: 2.42s	remaining: 1.47s
    622:	learn: 0.2135714	total: 2.42s	remaining: 1.46s
    623:	learn: 0.2133305	total: 2.42s	remaining: 1.46s
    624:	learn: 0.2130557	total: 2.43s	remaining: 1.46s
    625:	learn: 0.2127836	total: 2.43s	remaining: 1.45s
    626:	learn: 0.2125129	total: 2.43s	remaining: 1.45s
    627:	learn: 0.2122744	total: 2.43s	remaining: 1.44s
    628:	learn: 0.2120097	total: 2.44s	remaining: 1.44s
    629:	learn: 0.2117803	total: 2.44s	remaining: 1.43s
    630:	learn: 0.2114843	total: 2.44s	remaining: 1.43s
    631:	learn: 0.2112653	total: 2.45s	remaining: 1.42s
    632:	learn: 0.2110786	total: 2.45s	remaining: 1.42s
    633:	learn: 0.2108491	total: 2.46s	remaining: 1.42s
    634:	learn: 0.2106548	total: 2.46s	remaining: 1.41s
    635:	learn: 0.2103541	total: 2.46s	remaining: 1.41s
    636:	learn: 0.2100442	total: 2.46s	remaining: 1.4s
    637:	learn: 0.2098573	total: 2.47s	remaining: 1.4s
    638:	learn: 0.2095055	total: 2.47s	remaining: 1.39s
    639:	learn: 0.2093106	total: 2.48s	remaining: 1.39s
    640:	learn: 0.2090526	total: 2.48s	remaining: 1.39s
    641:	learn: 0.2088128	total: 2.48s	remaining: 1.38s
    642:	learn: 0.2085560	total: 2.48s	remaining: 1.38s
    643:	learn: 0.2083085	total: 2.48s	remaining: 1.37s
    644:	learn: 0.2080493	total: 2.49s	remaining: 1.37s
    645:	learn: 0.2078654	total: 2.5s	remaining: 1.37s
    646:	learn: 0.2076429	total: 2.5s	remaining: 1.36s
    647:	learn: 0.2074211	total: 2.5s	remaining: 1.36s
    648:	learn: 0.2070036	total: 2.5s	remaining: 1.35s
    649:	learn: 0.2066961	total: 2.51s	remaining: 1.35s
    650:	learn: 0.2064640	total: 2.51s	remaining: 1.34s
    651:	learn: 0.2062261	total: 2.51s	remaining: 1.34s
    652:	learn: 0.2060180	total: 2.52s	remaining: 1.34s
    653:	learn: 0.2058004	total: 2.52s	remaining: 1.33s
    654:	learn: 0.2055314	total: 2.52s	remaining: 1.33s
    655:	learn: 0.2052985	total: 2.53s	remaining: 1.33s
    656:	learn: 0.2050233	total: 2.53s	remaining: 1.32s
    657:	learn: 0.2046901	total: 2.54s	remaining: 1.32s
    658:	learn: 0.2045525	total: 2.54s	remaining: 1.31s
    659:	learn: 0.2042892	total: 2.54s	remaining: 1.31s
    660:	learn: 0.2040670	total: 2.54s	remaining: 1.3s
    661:	learn: 0.2038644	total: 2.55s	remaining: 1.3s
    662:	learn: 0.2037219	total: 2.55s	remaining: 1.3s
    663:	learn: 0.2035028	total: 2.55s	remaining: 1.29s
    664:	learn: 0.2032665	total: 2.56s	remaining: 1.29s
    665:	learn: 0.2030467	total: 2.56s	remaining: 1.28s
    666:	learn: 0.2028411	total: 2.56s	remaining: 1.28s
    667:	learn: 0.2024784	total: 2.57s	remaining: 1.28s
    668:	learn: 0.2023498	total: 2.57s	remaining: 1.27s
    669:	learn: 0.2020165	total: 2.58s	remaining: 1.27s
    670:	learn: 0.2018284	total: 2.59s	remaining: 1.27s
    671:	learn: 0.2016231	total: 2.59s	remaining: 1.26s
    672:	learn: 0.2013732	total: 2.59s	remaining: 1.26s
    673:	learn: 0.2010755	total: 2.59s	remaining: 1.25s
    674:	learn: 0.2008374	total: 2.6s	remaining: 1.25s
    675:	learn: 0.2005553	total: 2.61s	remaining: 1.25s
    676:	learn: 0.2003587	total: 2.61s	remaining: 1.25s
    677:	learn: 0.2000526	total: 2.62s	remaining: 1.24s
    678:	learn: 0.1998831	total: 2.62s	remaining: 1.24s
    679:	learn: 0.1997291	total: 2.63s	remaining: 1.24s
    680:	learn: 0.1994216	total: 2.63s	remaining: 1.23s
    681:	learn: 0.1992324	total: 2.64s	remaining: 1.23s
    682:	learn: 0.1989363	total: 2.65s	remaining: 1.23s
    683:	learn: 0.1986430	total: 2.65s	remaining: 1.22s
    684:	learn: 0.1984366	total: 2.66s	remaining: 1.22s
    685:	learn: 0.1981642	total: 2.66s	remaining: 1.22s
    686:	learn: 0.1979691	total: 2.66s	remaining: 1.21s
    687:	learn: 0.1978234	total: 2.67s	remaining: 1.21s
    688:	learn: 0.1975995	total: 2.67s	remaining: 1.21s
    689:	learn: 0.1974457	total: 2.67s	remaining: 1.2s
    690:	learn: 0.1971816	total: 2.68s	remaining: 1.2s
    691:	learn: 0.1969985	total: 2.68s	remaining: 1.19s
    692:	learn: 0.1967947	total: 2.69s	remaining: 1.19s
    693:	learn: 0.1964801	total: 2.69s	remaining: 1.19s
    694:	learn: 0.1962314	total: 2.69s	remaining: 1.18s
    695:	learn: 0.1960316	total: 2.69s	remaining: 1.18s
    696:	learn: 0.1958359	total: 2.69s	remaining: 1.17s
    697:	learn: 0.1956162	total: 2.7s	remaining: 1.17s
    698:	learn: 0.1954382	total: 2.7s	remaining: 1.16s
    699:	learn: 0.1951703	total: 2.7s	remaining: 1.16s
    700:	learn: 0.1949062	total: 2.71s	remaining: 1.16s
    701:	learn: 0.1947062	total: 2.71s	remaining: 1.15s
    702:	learn: 0.1944979	total: 2.71s	remaining: 1.15s
    703:	learn: 0.1942648	total: 2.72s	remaining: 1.14s
    704:	learn: 0.1939977	total: 2.72s	remaining: 1.14s
    705:	learn: 0.1937824	total: 2.73s	remaining: 1.14s
    706:	learn: 0.1935741	total: 2.73s	remaining: 1.13s
    707:	learn: 0.1933340	total: 2.73s	remaining: 1.13s
    708:	learn: 0.1930791	total: 2.74s	remaining: 1.12s
    709:	learn: 0.1929426	total: 2.74s	remaining: 1.12s
    710:	learn: 0.1927686	total: 2.74s	remaining: 1.11s
    711:	learn: 0.1925062	total: 2.75s	remaining: 1.11s
    712:	learn: 0.1923013	total: 2.75s	remaining: 1.11s
    713:	learn: 0.1921343	total: 2.75s	remaining: 1.1s
    714:	learn: 0.1919629	total: 2.76s	remaining: 1.1s
    715:	learn: 0.1917238	total: 2.76s	remaining: 1.09s
    716:	learn: 0.1914298	total: 2.76s	remaining: 1.09s
    717:	learn: 0.1911985	total: 2.76s	remaining: 1.08s
    718:	learn: 0.1910248	total: 2.76s	remaining: 1.08s
    719:	learn: 0.1908187	total: 2.77s	remaining: 1.08s
    720:	learn: 0.1905606	total: 2.77s	remaining: 1.07s
    721:	learn: 0.1903669	total: 2.77s	remaining: 1.07s
    722:	learn: 0.1902308	total: 2.79s	remaining: 1.07s
    723:	learn: 0.1899287	total: 2.79s	remaining: 1.06s
    724:	learn: 0.1898109	total: 2.8s	remaining: 1.06s
    725:	learn: 0.1896464	total: 2.81s	remaining: 1.06s
    726:	learn: 0.1893627	total: 2.81s	remaining: 1.06s
    727:	learn: 0.1892554	total: 2.82s	remaining: 1.05s
    728:	learn: 0.1890625	total: 2.82s	remaining: 1.05s
    729:	learn: 0.1888089	total: 2.83s	remaining: 1.04s
    730:	learn: 0.1886550	total: 2.83s	remaining: 1.04s
    731:	learn: 0.1884326	total: 2.83s	remaining: 1.04s
    732:	learn: 0.1883110	total: 2.83s	remaining: 1.03s
    733:	learn: 0.1881577	total: 2.84s	remaining: 1.03s
    734:	learn: 0.1879141	total: 2.84s	remaining: 1.02s
    735:	learn: 0.1877437	total: 2.85s	remaining: 1.02s
    736:	learn: 0.1875953	total: 2.85s	remaining: 1.02s
    737:	learn: 0.1874794	total: 2.85s	remaining: 1.01s
    738:	learn: 0.1873668	total: 2.85s	remaining: 1.01s
    739:	learn: 0.1872035	total: 2.86s	remaining: 1s
    740:	learn: 0.1870745	total: 2.86s	remaining: 1s
    741:	learn: 0.1869023	total: 2.87s	remaining: 998ms
    742:	learn: 0.1866802	total: 2.87s	remaining: 994ms
    743:	learn: 0.1864442	total: 2.88s	remaining: 992ms
    744:	learn: 0.1863371	total: 2.88s	remaining: 987ms
    745:	learn: 0.1860808	total: 2.89s	remaining: 985ms
    746:	learn: 0.1858950	total: 2.89s	remaining: 980ms
    747:	learn: 0.1857619	total: 2.9s	remaining: 978ms
    748:	learn: 0.1855279	total: 2.9s	remaining: 973ms
    749:	learn: 0.1853801	total: 2.91s	remaining: 971ms
    750:	learn: 0.1852299	total: 2.91s	remaining: 966ms
    751:	learn: 0.1849581	total: 2.92s	remaining: 962ms
    752:	learn: 0.1847370	total: 2.92s	remaining: 959ms
    753:	learn: 0.1845795	total: 2.92s	remaining: 954ms
    754:	learn: 0.1843570	total: 2.93s	remaining: 950ms
    755:	learn: 0.1841727	total: 2.93s	remaining: 945ms
    756:	learn: 0.1839405	total: 2.93s	remaining: 942ms
    757:	learn: 0.1837444	total: 2.94s	remaining: 937ms
    758:	learn: 0.1835577	total: 2.94s	remaining: 933ms
    759:	learn: 0.1834072	total: 2.94s	remaining: 930ms
    760:	learn: 0.1832458	total: 2.95s	remaining: 926ms
    761:	learn: 0.1830310	total: 2.95s	remaining: 921ms
    762:	learn: 0.1828628	total: 2.96s	remaining: 918ms
    763:	learn: 0.1827431	total: 2.96s	remaining: 914ms
    764:	learn: 0.1825209	total: 2.96s	remaining: 909ms
    765:	learn: 0.1823271	total: 2.97s	remaining: 907ms
    766:	learn: 0.1821145	total: 2.97s	remaining: 902ms
    767:	learn: 0.1818968	total: 2.97s	remaining: 898ms
    768:	learn: 0.1816805	total: 2.97s	remaining: 893ms
    769:	learn: 0.1815560	total: 2.98s	remaining: 889ms
    770:	learn: 0.1813870	total: 2.98s	remaining: 887ms
    771:	learn: 0.1811777	total: 3s	remaining: 886ms
    772:	learn: 0.1809440	total: 3s	remaining: 883ms
    773:	learn: 0.1807740	total: 3.01s	remaining: 880ms
    774:	learn: 0.1805270	total: 3.02s	remaining: 876ms
    775:	learn: 0.1803343	total: 3.02s	remaining: 871ms
    776:	learn: 0.1800902	total: 3.02s	remaining: 867ms
    777:	learn: 0.1798802	total: 3.02s	remaining: 862ms
    778:	learn: 0.1796791	total: 3.03s	remaining: 859ms
    779:	learn: 0.1795071	total: 3.03s	remaining: 855ms
    780:	learn: 0.1792813	total: 3.05s	remaining: 855ms
    781:	learn: 0.1790725	total: 3.05s	remaining: 850ms
    782:	learn: 0.1789756	total: 3.05s	remaining: 847ms
    783:	learn: 0.1787815	total: 3.06s	remaining: 843ms
    784:	learn: 0.1786093	total: 3.07s	remaining: 840ms
    785:	learn: 0.1784486	total: 3.08s	remaining: 837ms
    786:	learn: 0.1782153	total: 3.08s	remaining: 833ms
    787:	learn: 0.1780564	total: 3.09s	remaining: 830ms
    788:	learn: 0.1779013	total: 3.09s	remaining: 826ms
    789:	learn: 0.1776759	total: 3.11s	remaining: 827ms
    790:	learn: 0.1774475	total: 3.11s	remaining: 822ms
    791:	learn: 0.1772716	total: 3.12s	remaining: 819ms
    792:	learn: 0.1770801	total: 3.12s	remaining: 815ms
    793:	learn: 0.1769370	total: 3.12s	remaining: 810ms
    794:	learn: 0.1767282	total: 3.12s	remaining: 806ms
    795:	learn: 0.1765836	total: 3.13s	remaining: 801ms
    796:	learn: 0.1763405	total: 3.13s	remaining: 798ms
    797:	learn: 0.1762129	total: 3.14s	remaining: 794ms
    798:	learn: 0.1760235	total: 3.14s	remaining: 789ms
    799:	learn: 0.1758189	total: 3.14s	remaining: 785ms
    800:	learn: 0.1756702	total: 3.14s	remaining: 781ms
    801:	learn: 0.1755450	total: 3.14s	remaining: 776ms
    802:	learn: 0.1753332	total: 3.15s	remaining: 772ms
    803:	learn: 0.1752189	total: 3.15s	remaining: 767ms
    804:	learn: 0.1750728	total: 3.15s	remaining: 763ms
    805:	learn: 0.1749159	total: 3.15s	remaining: 758ms
    806:	learn: 0.1747684	total: 3.16s	remaining: 755ms
    807:	learn: 0.1746228	total: 3.16s	remaining: 751ms
    808:	learn: 0.1744255	total: 3.16s	remaining: 747ms
    809:	learn: 0.1742790	total: 3.17s	remaining: 744ms
    810:	learn: 0.1742306	total: 3.17s	remaining: 740ms
    811:	learn: 0.1741252	total: 3.17s	remaining: 735ms
    812:	learn: 0.1739433	total: 3.18s	remaining: 731ms
    813:	learn: 0.1737325	total: 3.18s	remaining: 727ms
    814:	learn: 0.1735887	total: 3.19s	remaining: 724ms
    815:	learn: 0.1734158	total: 3.19s	remaining: 719ms
    816:	learn: 0.1732640	total: 3.19s	remaining: 715ms
    817:	learn: 0.1730771	total: 3.2s	remaining: 712ms
    818:	learn: 0.1728427	total: 3.2s	remaining: 708ms
    819:	learn: 0.1727148	total: 3.2s	remaining: 703ms
    820:	learn: 0.1726151	total: 3.21s	remaining: 699ms
    821:	learn: 0.1724531	total: 3.21s	remaining: 695ms
    822:	learn: 0.1722405	total: 3.21s	remaining: 691ms
    823:	learn: 0.1720868	total: 3.22s	remaining: 688ms
    824:	learn: 0.1719532	total: 3.22s	remaining: 684ms
    825:	learn: 0.1717664	total: 3.23s	remaining: 680ms
    826:	learn: 0.1715672	total: 3.23s	remaining: 675ms
    827:	learn: 0.1714585	total: 3.23s	remaining: 671ms
    828:	learn: 0.1713378	total: 3.24s	remaining: 668ms
    829:	learn: 0.1712344	total: 3.24s	remaining: 663ms
    830:	learn: 0.1711275	total: 3.24s	remaining: 659ms
    831:	learn: 0.1710643	total: 3.25s	remaining: 657ms
    832:	learn: 0.1709130	total: 3.27s	remaining: 656ms
    833:	learn: 0.1707184	total: 3.28s	remaining: 653ms
    834:	learn: 0.1705323	total: 3.28s	remaining: 649ms
    835:	learn: 0.1703361	total: 3.29s	remaining: 645ms
    836:	learn: 0.1701466	total: 3.29s	remaining: 642ms
    837:	learn: 0.1699468	total: 3.3s	remaining: 638ms
    838:	learn: 0.1697696	total: 3.3s	remaining: 634ms
    839:	learn: 0.1695825	total: 3.31s	remaining: 630ms
    840:	learn: 0.1694047	total: 3.32s	remaining: 628ms
    841:	learn: 0.1692851	total: 3.32s	remaining: 623ms
    842:	learn: 0.1691157	total: 3.33s	remaining: 620ms
    843:	learn: 0.1688711	total: 3.34s	remaining: 617ms
    844:	learn: 0.1687820	total: 3.34s	remaining: 613ms
    845:	learn: 0.1686828	total: 3.35s	remaining: 610ms
    846:	learn: 0.1684238	total: 3.35s	remaining: 605ms
    847:	learn: 0.1682248	total: 3.35s	remaining: 601ms
    848:	learn: 0.1679504	total: 3.37s	remaining: 599ms
    849:	learn: 0.1677554	total: 3.38s	remaining: 598ms
    850:	learn: 0.1676341	total: 3.39s	remaining: 594ms
    851:	learn: 0.1675346	total: 3.4s	remaining: 590ms
    852:	learn: 0.1673314	total: 3.4s	remaining: 586ms
    853:	learn: 0.1671557	total: 3.4s	remaining: 582ms
    854:	learn: 0.1669157	total: 3.41s	remaining: 578ms
    855:	learn: 0.1668059	total: 3.41s	remaining: 574ms
    856:	learn: 0.1666470	total: 3.42s	remaining: 570ms
    857:	learn: 0.1665119	total: 3.42s	remaining: 566ms
    858:	learn: 0.1663130	total: 3.42s	remaining: 561ms
    859:	learn: 0.1661849	total: 3.42s	remaining: 557ms
    860:	learn: 0.1660905	total: 3.43s	remaining: 554ms
    861:	learn: 0.1659814	total: 3.43s	remaining: 549ms
    862:	learn: 0.1658224	total: 3.43s	remaining: 545ms
    863:	learn: 0.1656633	total: 3.44s	remaining: 541ms
    864:	learn: 0.1654616	total: 3.44s	remaining: 537ms
    865:	learn: 0.1653035	total: 3.44s	remaining: 533ms
    866:	learn: 0.1651659	total: 3.45s	remaining: 530ms
    867:	learn: 0.1650027	total: 3.45s	remaining: 525ms
    868:	learn: 0.1648196	total: 3.46s	remaining: 521ms
    869:	learn: 0.1646475	total: 3.46s	remaining: 517ms
    870:	learn: 0.1644930	total: 3.46s	remaining: 513ms
    871:	learn: 0.1643962	total: 3.47s	remaining: 510ms
    872:	learn: 0.1642436	total: 3.47s	remaining: 505ms
    873:	learn: 0.1640270	total: 3.48s	remaining: 501ms
    874:	learn: 0.1638385	total: 3.48s	remaining: 497ms
    875:	learn: 0.1637115	total: 3.48s	remaining: 493ms
    876:	learn: 0.1636267	total: 3.49s	remaining: 489ms
    877:	learn: 0.1635386	total: 3.49s	remaining: 485ms
    878:	learn: 0.1634232	total: 3.49s	remaining: 481ms
    879:	learn: 0.1632174	total: 3.5s	remaining: 477ms
    880:	learn: 0.1630249	total: 3.5s	remaining: 473ms
    881:	learn: 0.1628712	total: 3.5s	remaining: 469ms
    882:	learn: 0.1627648	total: 3.51s	remaining: 465ms
    883:	learn: 0.1625785	total: 3.51s	remaining: 461ms
    884:	learn: 0.1624283	total: 3.51s	remaining: 457ms
    885:	learn: 0.1623230	total: 3.52s	remaining: 453ms
    886:	learn: 0.1621857	total: 3.53s	remaining: 450ms
    887:	learn: 0.1620284	total: 3.53s	remaining: 445ms
    888:	learn: 0.1619104	total: 3.53s	remaining: 441ms
    889:	learn: 0.1618253	total: 3.53s	remaining: 437ms
    890:	learn: 0.1616158	total: 3.54s	remaining: 433ms
    891:	learn: 0.1614334	total: 3.54s	remaining: 429ms
    892:	learn: 0.1613278	total: 3.54s	remaining: 425ms
    893:	learn: 0.1611948	total: 3.56s	remaining: 422ms
    894:	learn: 0.1610125	total: 3.56s	remaining: 418ms
    895:	learn: 0.1608685	total: 3.57s	remaining: 415ms
    896:	learn: 0.1606578	total: 3.57s	remaining: 410ms
    897:	learn: 0.1604568	total: 3.57s	remaining: 406ms
    898:	learn: 0.1603552	total: 3.58s	remaining: 402ms
    899:	learn: 0.1602050	total: 3.58s	remaining: 398ms
    900:	learn: 0.1600276	total: 3.58s	remaining: 394ms
    901:	learn: 0.1598679	total: 3.59s	remaining: 390ms
    902:	learn: 0.1597268	total: 3.59s	remaining: 386ms
    903:	learn: 0.1595353	total: 3.59s	remaining: 382ms
    904:	learn: 0.1594047	total: 3.6s	remaining: 378ms
    905:	learn: 0.1591321	total: 3.6s	remaining: 374ms
    906:	learn: 0.1590200	total: 3.61s	remaining: 370ms
    907:	learn: 0.1588619	total: 3.62s	remaining: 367ms
    908:	learn: 0.1586496	total: 3.63s	remaining: 364ms
    909:	learn: 0.1585360	total: 3.64s	remaining: 360ms
    910:	learn: 0.1584123	total: 3.65s	remaining: 356ms
    911:	learn: 0.1581315	total: 3.65s	remaining: 352ms
    912:	learn: 0.1580284	total: 3.66s	remaining: 349ms
    913:	learn: 0.1579134	total: 3.66s	remaining: 345ms
    914:	learn: 0.1576965	total: 3.67s	remaining: 341ms
    915:	learn: 0.1575631	total: 3.67s	remaining: 337ms
    916:	learn: 0.1573195	total: 3.68s	remaining: 333ms
    917:	learn: 0.1571985	total: 3.7s	remaining: 330ms
    918:	learn: 0.1570316	total: 3.7s	remaining: 326ms
    919:	learn: 0.1569559	total: 3.71s	remaining: 322ms
    920:	learn: 0.1567983	total: 3.72s	remaining: 319ms
    921:	learn: 0.1566163	total: 3.75s	remaining: 317ms
    922:	learn: 0.1564047	total: 3.76s	remaining: 313ms
    923:	learn: 0.1562604	total: 3.76s	remaining: 309ms
    924:	learn: 0.1560988	total: 3.77s	remaining: 306ms
    925:	learn: 0.1559335	total: 3.79s	remaining: 303ms
    926:	learn: 0.1557727	total: 3.8s	remaining: 299ms
    927:	learn: 0.1556045	total: 3.81s	remaining: 296ms
    928:	learn: 0.1554279	total: 3.82s	remaining: 292ms
    929:	learn: 0.1552939	total: 3.82s	remaining: 288ms
    930:	learn: 0.1551815	total: 3.83s	remaining: 284ms
    931:	learn: 0.1550513	total: 3.83s	remaining: 280ms
    932:	learn: 0.1548965	total: 3.84s	remaining: 276ms
    933:	learn: 0.1548350	total: 3.85s	remaining: 272ms
    934:	learn: 0.1546728	total: 3.85s	remaining: 268ms
    935:	learn: 0.1545464	total: 3.85s	remaining: 264ms
    936:	learn: 0.1543975	total: 3.86s	remaining: 260ms
    937:	learn: 0.1541951	total: 3.87s	remaining: 256ms
    938:	learn: 0.1540613	total: 3.87s	remaining: 251ms
    939:	learn: 0.1538955	total: 3.88s	remaining: 248ms
    940:	learn: 0.1537964	total: 3.88s	remaining: 244ms
    941:	learn: 0.1536613	total: 3.9s	remaining: 240ms
    942:	learn: 0.1534730	total: 3.9s	remaining: 236ms
    943:	learn: 0.1533342	total: 3.9s	remaining: 232ms
    944:	learn: 0.1532246	total: 3.91s	remaining: 228ms
    945:	learn: 0.1530955	total: 3.92s	remaining: 224ms
    946:	learn: 0.1529400	total: 3.93s	remaining: 220ms
    947:	learn: 0.1528365	total: 3.93s	remaining: 216ms
    948:	learn: 0.1526804	total: 3.94s	remaining: 212ms
    949:	learn: 0.1524719	total: 3.95s	remaining: 208ms
    950:	learn: 0.1523470	total: 3.96s	remaining: 204ms
    951:	learn: 0.1522206	total: 3.97s	remaining: 200ms
    952:	learn: 0.1520748	total: 3.97s	remaining: 196ms
    953:	learn: 0.1518997	total: 3.98s	remaining: 192ms
    954:	learn: 0.1518284	total: 3.98s	remaining: 187ms
    955:	learn: 0.1517171	total: 3.98s	remaining: 183ms
    956:	learn: 0.1515919	total: 3.99s	remaining: 179ms
    957:	learn: 0.1514092	total: 3.99s	remaining: 175ms
    958:	learn: 0.1512453	total: 3.99s	remaining: 171ms
    959:	learn: 0.1511709	total: 4s	remaining: 167ms
    960:	learn: 0.1510302	total: 4.01s	remaining: 163ms
    961:	learn: 0.1509030	total: 4.02s	remaining: 159ms
    962:	learn: 0.1508020	total: 4.02s	remaining: 155ms
    963:	learn: 0.1507202	total: 4.03s	remaining: 150ms
    964:	learn: 0.1505643	total: 4.03s	remaining: 146ms
    965:	learn: 0.1504692	total: 4.04s	remaining: 142ms
    966:	learn: 0.1503167	total: 4.04s	remaining: 138ms
    967:	learn: 0.1500881	total: 4.05s	remaining: 134ms
    968:	learn: 0.1499522	total: 4.06s	remaining: 130ms
    969:	learn: 0.1497710	total: 4.06s	remaining: 126ms
    970:	learn: 0.1496962	total: 4.07s	remaining: 121ms
    971:	learn: 0.1494771	total: 4.07s	remaining: 117ms
    972:	learn: 0.1492930	total: 4.07s	remaining: 113ms
    973:	learn: 0.1491693	total: 4.08s	remaining: 109ms
    974:	learn: 0.1490576	total: 4.08s	remaining: 105ms
    975:	learn: 0.1489654	total: 4.09s	remaining: 101ms
    976:	learn: 0.1488193	total: 4.1s	remaining: 96.5ms
    977:	learn: 0.1487737	total: 4.1s	remaining: 92.2ms
    978:	learn: 0.1486333	total: 4.1s	remaining: 88ms
    979:	learn: 0.1484506	total: 4.11s	remaining: 83.8ms
    980:	learn: 0.1482593	total: 4.11s	remaining: 79.6ms
    981:	learn: 0.1481609	total: 4.12s	remaining: 75.6ms
    982:	learn: 0.1480046	total: 4.13s	remaining: 71.3ms
    983:	learn: 0.1479486	total: 4.13s	remaining: 67.1ms
    984:	learn: 0.1478514	total: 4.13s	remaining: 62.9ms
    985:	learn: 0.1477449	total: 4.14s	remaining: 58.8ms
    986:	learn: 0.1475843	total: 4.15s	remaining: 54.7ms
    987:	learn: 0.1474700	total: 4.16s	remaining: 50.5ms
    988:	learn: 0.1473128	total: 4.16s	remaining: 46.3ms
    989:	learn: 0.1472493	total: 4.17s	remaining: 42.1ms
    990:	learn: 0.1471369	total: 4.17s	remaining: 37.9ms
    991:	learn: 0.1470064	total: 4.17s	remaining: 33.7ms
    992:	learn: 0.1468605	total: 4.18s	remaining: 29.4ms
    993:	learn: 0.1467172	total: 4.18s	remaining: 25.2ms
    994:	learn: 0.1465716	total: 4.18s	remaining: 21ms
    995:	learn: 0.1464307	total: 4.19s	remaining: 16.8ms
    996:	learn: 0.1463201	total: 4.19s	remaining: 12.6ms
    997:	learn: 0.1461544	total: 4.2s	remaining: 8.41ms
    998:	learn: 0.1460391	total: 4.2s	remaining: 4.21ms
    999:	learn: 0.1459447	total: 4.21s	remaining: 0us
    0:	learn: 0.6918869	total: 3.88ms	remaining: 3.88s
    1:	learn: 0.6907095	total: 6.79ms	remaining: 3.39s
    2:	learn: 0.6896000	total: 10.9ms	remaining: 3.63s
    3:	learn: 0.6884974	total: 15.1ms	remaining: 3.75s
    4:	learn: 0.6874596	total: 19.2ms	remaining: 3.81s
    5:	learn: 0.6864332	total: 24.1ms	remaining: 3.99s
    6:	learn: 0.6849787	total: 26.7ms	remaining: 3.79s
    7:	learn: 0.6840458	total: 32.6ms	remaining: 4.05s
    8:	learn: 0.6830164	total: 35.3ms	remaining: 3.89s
    9:	learn: 0.6817751	total: 40.1ms	remaining: 3.97s
    10:	learn: 0.6807793	total: 43.8ms	remaining: 3.94s
    11:	learn: 0.6797654	total: 48.1ms	remaining: 3.96s
    12:	learn: 0.6786874	total: 52ms	remaining: 3.95s
    13:	learn: 0.6774234	total: 56.2ms	remaining: 3.96s
    14:	learn: 0.6762335	total: 60.5ms	remaining: 3.97s
    15:	learn: 0.6750528	total: 65ms	remaining: 4s
    16:	learn: 0.6739153	total: 69.8ms	remaining: 4.03s
    17:	learn: 0.6726632	total: 74.7ms	remaining: 4.08s
    18:	learn: 0.6717189	total: 79.8ms	remaining: 4.12s
    19:	learn: 0.6704536	total: 86.5ms	remaining: 4.24s
    20:	learn: 0.6694389	total: 89.1ms	remaining: 4.15s
    21:	learn: 0.6683602	total: 92.6ms	remaining: 4.11s
    22:	learn: 0.6673338	total: 96ms	remaining: 4.08s
    23:	learn: 0.6664527	total: 99.9ms	remaining: 4.06s
    24:	learn: 0.6653395	total: 104ms	remaining: 4.05s
    25:	learn: 0.6642815	total: 108ms	remaining: 4.04s
    26:	learn: 0.6632941	total: 112ms	remaining: 4.03s
    27:	learn: 0.6622495	total: 116ms	remaining: 4.01s
    28:	learn: 0.6611268	total: 120ms	remaining: 4.02s
    29:	learn: 0.6602187	total: 124ms	remaining: 4s
    30:	learn: 0.6587994	total: 128ms	remaining: 3.99s
    31:	learn: 0.6577835	total: 132ms	remaining: 3.99s
    32:	learn: 0.6567842	total: 137ms	remaining: 4.03s
    33:	learn: 0.6558671	total: 141ms	remaining: 4.01s
    34:	learn: 0.6548285	total: 144ms	remaining: 3.96s
    35:	learn: 0.6538094	total: 148ms	remaining: 3.95s
    36:	learn: 0.6527777	total: 151ms	remaining: 3.94s
    37:	learn: 0.6516232	total: 155ms	remaining: 3.93s
    38:	learn: 0.6504498	total: 159ms	remaining: 3.92s
    39:	learn: 0.6492084	total: 173ms	remaining: 4.15s
    40:	learn: 0.6482530	total: 176ms	remaining: 4.13s
    41:	learn: 0.6472314	total: 180ms	remaining: 4.12s
    42:	learn: 0.6462297	total: 185ms	remaining: 4.11s
    43:	learn: 0.6452469	total: 189ms	remaining: 4.11s
    44:	learn: 0.6442824	total: 193ms	remaining: 4.09s
    45:	learn: 0.6432856	total: 197ms	remaining: 4.09s
    46:	learn: 0.6424260	total: 201ms	remaining: 4.07s
    47:	learn: 0.6414519	total: 205ms	remaining: 4.06s
    48:	learn: 0.6404024	total: 209ms	remaining: 4.05s
    49:	learn: 0.6394951	total: 213ms	remaining: 4.04s
    50:	learn: 0.6384562	total: 217ms	remaining: 4.04s
    51:	learn: 0.6377340	total: 230ms	remaining: 4.19s
    52:	learn: 0.6370376	total: 232ms	remaining: 4.15s
    53:	learn: 0.6361699	total: 236ms	remaining: 4.14s
    54:	learn: 0.6349325	total: 240ms	remaining: 4.13s
    55:	learn: 0.6340741	total: 244ms	remaining: 4.12s
    56:	learn: 0.6330330	total: 248ms	remaining: 4.11s
    57:	learn: 0.6319279	total: 252ms	remaining: 4.1s
    58:	learn: 0.6308126	total: 256ms	remaining: 4.09s
    59:	learn: 0.6296362	total: 261ms	remaining: 4.09s
    60:	learn: 0.6288846	total: 264ms	remaining: 4.06s
    61:	learn: 0.6280550	total: 268ms	remaining: 4.05s
    62:	learn: 0.6270512	total: 272ms	remaining: 4.04s
    63:	learn: 0.6261965	total: 275ms	remaining: 4.02s
    64:	learn: 0.6253990	total: 279ms	remaining: 4.02s
    65:	learn: 0.6241247	total: 286ms	remaining: 4.04s
    66:	learn: 0.6233855	total: 289ms	remaining: 4.02s
    67:	learn: 0.6225575	total: 293ms	remaining: 4.01s
    68:	learn: 0.6217561	total: 297ms	remaining: 4.01s
    69:	learn: 0.6210245	total: 302ms	remaining: 4.01s
    70:	learn: 0.6200431	total: 306ms	remaining: 4s
    71:	learn: 0.6191761	total: 310ms	remaining: 4s
    72:	learn: 0.6182466	total: 314ms	remaining: 3.99s
    73:	learn: 0.6172773	total: 318ms	remaining: 3.98s
    74:	learn: 0.6164477	total: 322ms	remaining: 3.98s
    75:	learn: 0.6155428	total: 327ms	remaining: 3.97s
    76:	learn: 0.6147587	total: 329ms	remaining: 3.94s
    77:	learn: 0.6138804	total: 342ms	remaining: 4.04s
    78:	learn: 0.6129919	total: 349ms	remaining: 4.06s
    79:	learn: 0.6116394	total: 350ms	remaining: 4.03s
    80:	learn: 0.6108288	total: 351ms	remaining: 3.98s
    81:	learn: 0.6097555	total: 353ms	remaining: 3.95s
    82:	learn: 0.6086159	total: 354ms	remaining: 3.91s
    83:	learn: 0.6079374	total: 355ms	remaining: 3.87s
    84:	learn: 0.6068348	total: 356ms	remaining: 3.83s
    85:	learn: 0.6058541	total: 358ms	remaining: 3.81s
    86:	learn: 0.6048878	total: 360ms	remaining: 3.78s
    87:	learn: 0.6041164	total: 362ms	remaining: 3.75s
    88:	learn: 0.6030150	total: 363ms	remaining: 3.72s
    89:	learn: 0.6023052	total: 365ms	remaining: 3.69s
    90:	learn: 0.6014823	total: 366ms	remaining: 3.66s
    91:	learn: 0.6008691	total: 368ms	remaining: 3.63s
    92:	learn: 0.6000707	total: 369ms	remaining: 3.6s
    93:	learn: 0.5992743	total: 371ms	remaining: 3.57s
    94:	learn: 0.5981709	total: 372ms	remaining: 3.54s
    95:	learn: 0.5973049	total: 376ms	remaining: 3.54s
    96:	learn: 0.5965799	total: 381ms	remaining: 3.54s
    97:	learn: 0.5958428	total: 382ms	remaining: 3.51s
    98:	learn: 0.5951556	total: 383ms	remaining: 3.48s
    99:	learn: 0.5943759	total: 384ms	remaining: 3.46s
    100:	learn: 0.5933803	total: 386ms	remaining: 3.44s
    101:	learn: 0.5924483	total: 387ms	remaining: 3.41s
    102:	learn: 0.5914717	total: 389ms	remaining: 3.39s
    103:	learn: 0.5905939	total: 390ms	remaining: 3.36s
    104:	learn: 0.5898563	total: 391ms	remaining: 3.33s
    105:	learn: 0.5888931	total: 392ms	remaining: 3.31s
    106:	learn: 0.5880638	total: 394ms	remaining: 3.29s
    107:	learn: 0.5870656	total: 395ms	remaining: 3.26s
    108:	learn: 0.5862185	total: 397ms	remaining: 3.25s
    109:	learn: 0.5854891	total: 398ms	remaining: 3.22s
    110:	learn: 0.5848289	total: 399ms	remaining: 3.2s
    111:	learn: 0.5839230	total: 401ms	remaining: 3.18s
    112:	learn: 0.5831818	total: 402ms	remaining: 3.16s
    113:	learn: 0.5826976	total: 404ms	remaining: 3.14s
    114:	learn: 0.5819756	total: 405ms	remaining: 3.12s
    115:	learn: 0.5809390	total: 407ms	remaining: 3.1s
    116:	learn: 0.5799830	total: 408ms	remaining: 3.08s
    117:	learn: 0.5794922	total: 409ms	remaining: 3.06s
    118:	learn: 0.5786623	total: 410ms	remaining: 3.04s
    119:	learn: 0.5779462	total: 412ms	remaining: 3.02s
    120:	learn: 0.5773177	total: 413ms	remaining: 3s
    121:	learn: 0.5764507	total: 414ms	remaining: 2.98s
    122:	learn: 0.5756193	total: 415ms	remaining: 2.96s
    123:	learn: 0.5746836	total: 417ms	remaining: 2.94s
    124:	learn: 0.5738920	total: 418ms	remaining: 2.92s
    125:	learn: 0.5733100	total: 419ms	remaining: 2.9s
    126:	learn: 0.5725773	total: 420ms	remaining: 2.89s
    127:	learn: 0.5720072	total: 421ms	remaining: 2.87s
    128:	learn: 0.5711971	total: 423ms	remaining: 2.86s
    129:	learn: 0.5704806	total: 425ms	remaining: 2.84s
    130:	learn: 0.5697909	total: 427ms	remaining: 2.83s
    131:	learn: 0.5690912	total: 428ms	remaining: 2.81s
    132:	learn: 0.5682001	total: 429ms	remaining: 2.8s
    133:	learn: 0.5675095	total: 430ms	remaining: 2.78s
    134:	learn: 0.5666746	total: 432ms	remaining: 2.77s
    135:	learn: 0.5656949	total: 433ms	remaining: 2.75s
    136:	learn: 0.5647606	total: 435ms	remaining: 2.74s
    137:	learn: 0.5639987	total: 437ms	remaining: 2.73s
    138:	learn: 0.5630932	total: 438ms	remaining: 2.71s
    139:	learn: 0.5624942	total: 439ms	remaining: 2.7s
    140:	learn: 0.5617045	total: 440ms	remaining: 2.68s
    141:	learn: 0.5608551	total: 442ms	remaining: 2.67s
    142:	learn: 0.5601727	total: 443ms	remaining: 2.65s
    143:	learn: 0.5593583	total: 445ms	remaining: 2.64s
    144:	learn: 0.5583766	total: 446ms	remaining: 2.63s
    145:	learn: 0.5577476	total: 447ms	remaining: 2.62s
    146:	learn: 0.5566969	total: 449ms	remaining: 2.6s
    147:	learn: 0.5558364	total: 450ms	remaining: 2.59s
    148:	learn: 0.5553401	total: 452ms	remaining: 2.58s
    149:	learn: 0.5544860	total: 453ms	remaining: 2.57s
    150:	learn: 0.5540291	total: 454ms	remaining: 2.55s
    151:	learn: 0.5533045	total: 456ms	remaining: 2.54s
    152:	learn: 0.5527557	total: 458ms	remaining: 2.54s
    153:	learn: 0.5521565	total: 464ms	remaining: 2.55s
    154:	learn: 0.5516302	total: 468ms	remaining: 2.55s
    155:	learn: 0.5506567	total: 494ms	remaining: 2.67s
    156:	learn: 0.5499638	total: 495ms	remaining: 2.66s
    157:	learn: 0.5492059	total: 497ms	remaining: 2.65s
    158:	learn: 0.5486160	total: 498ms	remaining: 2.63s
    159:	learn: 0.5478743	total: 499ms	remaining: 2.62s
    160:	learn: 0.5471911	total: 500ms	remaining: 2.61s
    161:	learn: 0.5462748	total: 501ms	remaining: 2.59s
    162:	learn: 0.5457225	total: 503ms	remaining: 2.58s
    163:	learn: 0.5450170	total: 505ms	remaining: 2.57s
    164:	learn: 0.5445286	total: 506ms	remaining: 2.56s
    165:	learn: 0.5438266	total: 508ms	remaining: 2.55s
    166:	learn: 0.5431915	total: 510ms	remaining: 2.54s
    167:	learn: 0.5424464	total: 511ms	remaining: 2.53s
    168:	learn: 0.5416509	total: 513ms	remaining: 2.52s
    169:	learn: 0.5409259	total: 514ms	remaining: 2.51s
    170:	learn: 0.5403612	total: 516ms	remaining: 2.5s
    171:	learn: 0.5395569	total: 517ms	remaining: 2.49s
    172:	learn: 0.5390692	total: 519ms	remaining: 2.48s
    173:	learn: 0.5380616	total: 520ms	remaining: 2.47s
    174:	learn: 0.5370532	total: 521ms	remaining: 2.46s
    175:	learn: 0.5362207	total: 523ms	remaining: 2.45s
    176:	learn: 0.5354971	total: 524ms	remaining: 2.44s
    177:	learn: 0.5346360	total: 526ms	remaining: 2.43s
    178:	learn: 0.5341902	total: 527ms	remaining: 2.42s
    179:	learn: 0.5333694	total: 528ms	remaining: 2.41s
    180:	learn: 0.5327304	total: 530ms	remaining: 2.4s
    181:	learn: 0.5323432	total: 531ms	remaining: 2.39s
    182:	learn: 0.5315507	total: 533ms	remaining: 2.38s
    183:	learn: 0.5310209	total: 534ms	remaining: 2.37s
    184:	learn: 0.5304177	total: 536ms	remaining: 2.36s
    185:	learn: 0.5297874	total: 537ms	remaining: 2.35s
    186:	learn: 0.5293058	total: 538ms	remaining: 2.34s
    187:	learn: 0.5286236	total: 540ms	remaining: 2.33s
    188:	learn: 0.5277143	total: 553ms	remaining: 2.37s
    189:	learn: 0.5266913	total: 554ms	remaining: 2.36s
    190:	learn: 0.5261298	total: 562ms	remaining: 2.38s
    191:	learn: 0.5255553	total: 563ms	remaining: 2.37s
    192:	learn: 0.5249771	total: 564ms	remaining: 2.36s
    193:	learn: 0.5242177	total: 571ms	remaining: 2.37s
    194:	learn: 0.5235802	total: 572ms	remaining: 2.36s
    195:	learn: 0.5225486	total: 573ms	remaining: 2.35s
    196:	learn: 0.5217112	total: 574ms	remaining: 2.34s
    197:	learn: 0.5211934	total: 576ms	remaining: 2.33s
    198:	learn: 0.5207752	total: 577ms	remaining: 2.32s
    199:	learn: 0.5200123	total: 579ms	remaining: 2.31s
    200:	learn: 0.5193172	total: 585ms	remaining: 2.33s
    201:	learn: 0.5185237	total: 587ms	remaining: 2.32s
    202:	learn: 0.5180801	total: 588ms	remaining: 2.31s
    203:	learn: 0.5176541	total: 596ms	remaining: 2.32s
    204:	learn: 0.5170277	total: 598ms	remaining: 2.32s
    205:	learn: 0.5161101	total: 599ms	remaining: 2.31s
    206:	learn: 0.5154903	total: 600ms	remaining: 2.3s
    207:	learn: 0.5147008	total: 601ms	remaining: 2.29s
    208:	learn: 0.5139488	total: 607ms	remaining: 2.3s
    209:	learn: 0.5132651	total: 636ms	remaining: 2.39s
    210:	learn: 0.5127195	total: 639ms	remaining: 2.39s
    211:	learn: 0.5121326	total: 643ms	remaining: 2.39s
    212:	learn: 0.5116904	total: 648ms	remaining: 2.39s
    213:	learn: 0.5107378	total: 662ms	remaining: 2.43s
    214:	learn: 0.5100782	total: 666ms	remaining: 2.43s
    215:	learn: 0.5096489	total: 677ms	remaining: 2.46s
    216:	learn: 0.5089351	total: 686ms	remaining: 2.48s
    217:	learn: 0.5083718	total: 690ms	remaining: 2.48s
    218:	learn: 0.5078532	total: 695ms	remaining: 2.48s
    219:	learn: 0.5070823	total: 697ms	remaining: 2.47s
    220:	learn: 0.5067222	total: 699ms	remaining: 2.46s
    221:	learn: 0.5060558	total: 701ms	remaining: 2.46s
    222:	learn: 0.5054522	total: 702ms	remaining: 2.44s
    223:	learn: 0.5047812	total: 703ms	remaining: 2.44s
    224:	learn: 0.5043270	total: 704ms	remaining: 2.42s
    225:	learn: 0.5038082	total: 705ms	remaining: 2.42s
    226:	learn: 0.5032403	total: 712ms	remaining: 2.42s
    227:	learn: 0.5026089	total: 714ms	remaining: 2.42s
    228:	learn: 0.5019078	total: 724ms	remaining: 2.44s
    229:	learn: 0.5010677	total: 732ms	remaining: 2.45s
    230:	learn: 0.5004797	total: 733ms	remaining: 2.44s
    231:	learn: 0.5000801	total: 735ms	remaining: 2.43s
    232:	learn: 0.4995281	total: 736ms	remaining: 2.42s
    233:	learn: 0.4989349	total: 737ms	remaining: 2.41s
    234:	learn: 0.4983239	total: 739ms	remaining: 2.4s
    235:	learn: 0.4977345	total: 740ms	remaining: 2.4s
    236:	learn: 0.4972209	total: 742ms	remaining: 2.39s
    237:	learn: 0.4967576	total: 743ms	remaining: 2.38s
    238:	learn: 0.4963517	total: 744ms	remaining: 2.37s
    239:	learn: 0.4958271	total: 746ms	remaining: 2.36s
    240:	learn: 0.4951212	total: 748ms	remaining: 2.35s
    241:	learn: 0.4947071	total: 749ms	remaining: 2.35s
    242:	learn: 0.4942853	total: 750ms	remaining: 2.34s
    243:	learn: 0.4936040	total: 752ms	remaining: 2.33s
    244:	learn: 0.4930561	total: 754ms	remaining: 2.32s
    245:	learn: 0.4924403	total: 755ms	remaining: 2.31s
    246:	learn: 0.4917661	total: 757ms	remaining: 2.31s
    247:	learn: 0.4912270	total: 763ms	remaining: 2.31s
    248:	learn: 0.4905116	total: 770ms	remaining: 2.32s
    249:	learn: 0.4899101	total: 772ms	remaining: 2.31s
    250:	learn: 0.4893575	total: 773ms	remaining: 2.31s
    251:	learn: 0.4887786	total: 775ms	remaining: 2.3s
    252:	learn: 0.4881231	total: 776ms	remaining: 2.29s
    253:	learn: 0.4873455	total: 777ms	remaining: 2.28s
    254:	learn: 0.4867256	total: 778ms	remaining: 2.27s
    255:	learn: 0.4859819	total: 780ms	remaining: 2.27s
    256:	learn: 0.4856628	total: 781ms	remaining: 2.26s
    257:	learn: 0.4851707	total: 782ms	remaining: 2.25s
    258:	learn: 0.4845777	total: 783ms	remaining: 2.24s
    259:	learn: 0.4839485	total: 785ms	remaining: 2.23s
    260:	learn: 0.4833956	total: 787ms	remaining: 2.23s
    261:	learn: 0.4826634	total: 789ms	remaining: 2.22s
    262:	learn: 0.4820420	total: 791ms	remaining: 2.21s
    263:	learn: 0.4814126	total: 792ms	remaining: 2.21s
    264:	learn: 0.4807546	total: 793ms	remaining: 2.2s
    265:	learn: 0.4801425	total: 794ms	remaining: 2.19s
    266:	learn: 0.4797000	total: 796ms	remaining: 2.18s
    267:	learn: 0.4792489	total: 809ms	remaining: 2.21s
    268:	learn: 0.4784225	total: 811ms	remaining: 2.2s
    269:	learn: 0.4779274	total: 812ms	remaining: 2.2s
    270:	learn: 0.4774020	total: 814ms	remaining: 2.19s
    271:	learn: 0.4769307	total: 815ms	remaining: 2.18s
    272:	learn: 0.4763279	total: 818ms	remaining: 2.18s
    273:	learn: 0.4760015	total: 820ms	remaining: 2.17s
    274:	learn: 0.4755537	total: 822ms	remaining: 2.17s
    275:	learn: 0.4751463	total: 825ms	remaining: 2.16s
    276:	learn: 0.4747130	total: 829ms	remaining: 2.16s
    277:	learn: 0.4742695	total: 839ms	remaining: 2.18s
    278:	learn: 0.4735550	total: 843ms	remaining: 2.18s
    279:	learn: 0.4729069	total: 847ms	remaining: 2.18s
    280:	learn: 0.4724714	total: 851ms	remaining: 2.18s
    281:	learn: 0.4721044	total: 855ms	remaining: 2.17s
    282:	learn: 0.4717955	total: 859ms	remaining: 2.18s
    283:	learn: 0.4711951	total: 865ms	remaining: 2.18s
    284:	learn: 0.4706111	total: 871ms	remaining: 2.18s
    285:	learn: 0.4701546	total: 875ms	remaining: 2.18s
    286:	learn: 0.4697646	total: 877ms	remaining: 2.18s
    287:	learn: 0.4689982	total: 881ms	remaining: 2.18s
    288:	learn: 0.4685527	total: 884ms	remaining: 2.17s
    289:	learn: 0.4680757	total: 887ms	remaining: 2.17s
    290:	learn: 0.4675406	total: 891ms	remaining: 2.17s
    291:	learn: 0.4669605	total: 895ms	remaining: 2.17s
    292:	learn: 0.4663477	total: 899ms	remaining: 2.17s
    293:	learn: 0.4658416	total: 903ms	remaining: 2.17s
    294:	learn: 0.4652179	total: 907ms	remaining: 2.17s
    295:	learn: 0.4646821	total: 911ms	remaining: 2.17s
    296:	learn: 0.4642878	total: 915ms	remaining: 2.17s
    297:	learn: 0.4637833	total: 919ms	remaining: 2.16s
    298:	learn: 0.4632003	total: 922ms	remaining: 2.16s
    299:	learn: 0.4626257	total: 926ms	remaining: 2.16s
    300:	learn: 0.4622435	total: 932ms	remaining: 2.16s
    301:	learn: 0.4617246	total: 938ms	remaining: 2.17s
    302:	learn: 0.4612985	total: 940ms	remaining: 2.16s
    303:	learn: 0.4609645	total: 943ms	remaining: 2.16s
    304:	learn: 0.4603634	total: 955ms	remaining: 2.17s
    305:	learn: 0.4596479	total: 957ms	remaining: 2.17s
    306:	learn: 0.4591460	total: 961ms	remaining: 2.17s
    307:	learn: 0.4586914	total: 964ms	remaining: 2.17s
    308:	learn: 0.4583886	total: 967ms	remaining: 2.16s
    309:	learn: 0.4579716	total: 970ms	remaining: 2.16s
    310:	learn: 0.4576709	total: 975ms	remaining: 2.16s
    311:	learn: 0.4571110	total: 978ms	remaining: 2.16s
    312:	learn: 0.4567054	total: 981ms	remaining: 2.15s
    313:	learn: 0.4562964	total: 985ms	remaining: 2.15s
    314:	learn: 0.4557273	total: 990ms	remaining: 2.15s
    315:	learn: 0.4553919	total: 993ms	remaining: 2.15s
    316:	learn: 0.4549232	total: 997ms	remaining: 2.15s
    317:	learn: 0.4544991	total: 1s	remaining: 2.15s
    318:	learn: 0.4540360	total: 1s	remaining: 2.15s
    319:	learn: 0.4534247	total: 1.01s	remaining: 2.14s
    320:	learn: 0.4529385	total: 1.01s	remaining: 2.14s
    321:	learn: 0.4525199	total: 1.02s	remaining: 2.14s
    322:	learn: 0.4521400	total: 1.02s	remaining: 2.14s
    323:	learn: 0.4516770	total: 1.02s	remaining: 2.14s
    324:	learn: 0.4511209	total: 1.03s	remaining: 2.14s
    325:	learn: 0.4506782	total: 1.03s	remaining: 2.14s
    326:	learn: 0.4502335	total: 1.04s	remaining: 2.14s
    327:	learn: 0.4498270	total: 1.05s	remaining: 2.15s
    328:	learn: 0.4492761	total: 1.05s	remaining: 2.15s
    329:	learn: 0.4486022	total: 1.06s	remaining: 2.15s
    330:	learn: 0.4481993	total: 1.06s	remaining: 2.15s
    331:	learn: 0.4475916	total: 1.06s	remaining: 2.14s
    332:	learn: 0.4469475	total: 1.07s	remaining: 2.14s
    333:	learn: 0.4465096	total: 1.07s	remaining: 2.13s
    334:	learn: 0.4458964	total: 1.07s	remaining: 2.13s
    335:	learn: 0.4453308	total: 1.08s	remaining: 2.13s
    336:	learn: 0.4448189	total: 1.08s	remaining: 2.13s
    337:	learn: 0.4441843	total: 1.08s	remaining: 2.12s
    338:	learn: 0.4435656	total: 1.08s	remaining: 2.12s
    339:	learn: 0.4431130	total: 1.09s	remaining: 2.12s
    340:	learn: 0.4424023	total: 1.1s	remaining: 2.12s
    341:	learn: 0.4420524	total: 1.1s	remaining: 2.12s
    342:	learn: 0.4414892	total: 1.1s	remaining: 2.11s
    343:	learn: 0.4409313	total: 1.11s	remaining: 2.11s
    344:	learn: 0.4404261	total: 1.11s	remaining: 2.11s
    345:	learn: 0.4400485	total: 1.11s	remaining: 2.11s
    346:	learn: 0.4396016	total: 1.12s	remaining: 2.1s
    347:	learn: 0.4390671	total: 1.12s	remaining: 2.1s
    348:	learn: 0.4387800	total: 1.12s	remaining: 2.1s
    349:	learn: 0.4384767	total: 1.13s	remaining: 2.1s
    350:	learn: 0.4381728	total: 1.13s	remaining: 2.1s
    351:	learn: 0.4379399	total: 1.14s	remaining: 2.09s
    352:	learn: 0.4376068	total: 1.14s	remaining: 2.09s
    353:	learn: 0.4370865	total: 1.14s	remaining: 2.09s
    354:	learn: 0.4366078	total: 1.15s	remaining: 2.09s
    355:	learn: 0.4361617	total: 1.16s	remaining: 2.09s
    356:	learn: 0.4358538	total: 1.16s	remaining: 2.09s
    357:	learn: 0.4355243	total: 1.17s	remaining: 2.09s
    358:	learn: 0.4350539	total: 1.17s	remaining: 2.09s
    359:	learn: 0.4345864	total: 1.17s	remaining: 2.08s
    360:	learn: 0.4340925	total: 1.18s	remaining: 2.08s
    361:	learn: 0.4337524	total: 1.18s	remaining: 2.08s
    362:	learn: 0.4332822	total: 1.18s	remaining: 2.08s
    363:	learn: 0.4329155	total: 1.19s	remaining: 2.07s
    364:	learn: 0.4324423	total: 1.19s	remaining: 2.07s
    365:	learn: 0.4320703	total: 1.19s	remaining: 2.06s
    366:	learn: 0.4315321	total: 1.19s	remaining: 2.06s
    367:	learn: 0.4311592	total: 1.19s	remaining: 2.05s
    368:	learn: 0.4305721	total: 1.19s	remaining: 2.04s
    369:	learn: 0.4302706	total: 1.2s	remaining: 2.04s
    370:	learn: 0.4298455	total: 1.2s	remaining: 2.03s
    371:	learn: 0.4295163	total: 1.2s	remaining: 2.02s
    372:	learn: 0.4289848	total: 1.2s	remaining: 2.02s
    373:	learn: 0.4282655	total: 1.2s	remaining: 2.01s
    374:	learn: 0.4277896	total: 1.2s	remaining: 2s
    375:	learn: 0.4273579	total: 1.2s	remaining: 2s
    376:	learn: 0.4267384	total: 1.2s	remaining: 1.99s
    377:	learn: 0.4261097	total: 1.21s	remaining: 1.98s
    378:	learn: 0.4256745	total: 1.21s	remaining: 1.98s
    379:	learn: 0.4252608	total: 1.21s	remaining: 1.97s
    380:	learn: 0.4248515	total: 1.21s	remaining: 1.97s
    381:	learn: 0.4244838	total: 1.21s	remaining: 1.96s
    382:	learn: 0.4243126	total: 1.21s	remaining: 1.95s
    383:	learn: 0.4239668	total: 1.21s	remaining: 1.95s
    384:	learn: 0.4233470	total: 1.21s	remaining: 1.94s
    385:	learn: 0.4229491	total: 1.22s	remaining: 1.93s
    386:	learn: 0.4225574	total: 1.22s	remaining: 1.93s
    387:	learn: 0.4221927	total: 1.22s	remaining: 1.92s
    388:	learn: 0.4217105	total: 1.22s	remaining: 1.92s
    389:	learn: 0.4212659	total: 1.22s	remaining: 1.91s
    390:	learn: 0.4206719	total: 1.22s	remaining: 1.9s
    391:	learn: 0.4202688	total: 1.22s	remaining: 1.9s
    392:	learn: 0.4199662	total: 1.23s	remaining: 1.89s
    393:	learn: 0.4196230	total: 1.23s	remaining: 1.89s
    394:	learn: 0.4192292	total: 1.23s	remaining: 1.88s
    395:	learn: 0.4188960	total: 1.23s	remaining: 1.87s
    396:	learn: 0.4183157	total: 1.23s	remaining: 1.87s
    397:	learn: 0.4177423	total: 1.23s	remaining: 1.86s
    398:	learn: 0.4173232	total: 1.23s	remaining: 1.86s
    399:	learn: 0.4169872	total: 1.23s	remaining: 1.85s
    400:	learn: 0.4165300	total: 1.23s	remaining: 1.84s
    401:	learn: 0.4161933	total: 1.24s	remaining: 1.84s
    402:	learn: 0.4158733	total: 1.24s	remaining: 1.83s
    403:	learn: 0.4155185	total: 1.24s	remaining: 1.83s
    404:	learn: 0.4150895	total: 1.24s	remaining: 1.82s
    405:	learn: 0.4146444	total: 1.24s	remaining: 1.82s
    406:	learn: 0.4143439	total: 1.24s	remaining: 1.81s
    407:	learn: 0.4139090	total: 1.24s	remaining: 1.8s
    408:	learn: 0.4135823	total: 1.25s	remaining: 1.8s
    409:	learn: 0.4131873	total: 1.25s	remaining: 1.79s
    410:	learn: 0.4128178	total: 1.25s	remaining: 1.79s
    411:	learn: 0.4123896	total: 1.25s	remaining: 1.78s
    412:	learn: 0.4119503	total: 1.25s	remaining: 1.78s
    413:	learn: 0.4116167	total: 1.25s	remaining: 1.77s
    414:	learn: 0.4113682	total: 1.25s	remaining: 1.77s
    415:	learn: 0.4110871	total: 1.26s	remaining: 1.77s
    416:	learn: 0.4108131	total: 1.26s	remaining: 1.76s
    417:	learn: 0.4103981	total: 1.26s	remaining: 1.75s
    418:	learn: 0.4099065	total: 1.26s	remaining: 1.75s
    419:	learn: 0.4095592	total: 1.26s	remaining: 1.74s
    420:	learn: 0.4093085	total: 1.26s	remaining: 1.74s
    421:	learn: 0.4087403	total: 1.26s	remaining: 1.73s
    422:	learn: 0.4085358	total: 1.27s	remaining: 1.73s
    423:	learn: 0.4082140	total: 1.27s	remaining: 1.72s
    424:	learn: 0.4076625	total: 1.27s	remaining: 1.72s
    425:	learn: 0.4073838	total: 1.27s	remaining: 1.71s
    426:	learn: 0.4069956	total: 1.27s	remaining: 1.71s
    427:	learn: 0.4064779	total: 1.27s	remaining: 1.7s
    428:	learn: 0.4062312	total: 1.27s	remaining: 1.7s
    429:	learn: 0.4059226	total: 1.28s	remaining: 1.69s
    430:	learn: 0.4056839	total: 1.28s	remaining: 1.69s
    431:	learn: 0.4052678	total: 1.28s	remaining: 1.68s
    432:	learn: 0.4048708	total: 1.28s	remaining: 1.68s
    433:	learn: 0.4043585	total: 1.28s	remaining: 1.67s
    434:	learn: 0.4040660	total: 1.28s	remaining: 1.67s
    435:	learn: 0.4038167	total: 1.28s	remaining: 1.66s
    436:	learn: 0.4033260	total: 1.29s	remaining: 1.66s
    437:	learn: 0.4028786	total: 1.29s	remaining: 1.65s
    438:	learn: 0.4025291	total: 1.29s	remaining: 1.65s
    439:	learn: 0.4022020	total: 1.29s	remaining: 1.64s
    440:	learn: 0.4016643	total: 1.29s	remaining: 1.64s
    441:	learn: 0.4013553	total: 1.29s	remaining: 1.63s
    442:	learn: 0.4008144	total: 1.29s	remaining: 1.63s
    443:	learn: 0.4005761	total: 1.29s	remaining: 1.62s
    444:	learn: 0.4002045	total: 1.3s	remaining: 1.62s
    445:	learn: 0.3997710	total: 1.3s	remaining: 1.61s
    446:	learn: 0.3993118	total: 1.3s	remaining: 1.61s
    447:	learn: 0.3990391	total: 1.3s	remaining: 1.6s
    448:	learn: 0.3986102	total: 1.3s	remaining: 1.6s
    449:	learn: 0.3982317	total: 1.3s	remaining: 1.59s
    450:	learn: 0.3977189	total: 1.3s	remaining: 1.59s
    451:	learn: 0.3973824	total: 1.3s	remaining: 1.58s
    452:	learn: 0.3969249	total: 1.31s	remaining: 1.58s
    453:	learn: 0.3965304	total: 1.31s	remaining: 1.57s
    454:	learn: 0.3961990	total: 1.31s	remaining: 1.57s
    455:	learn: 0.3958629	total: 1.31s	remaining: 1.56s
    456:	learn: 0.3954519	total: 1.31s	remaining: 1.56s
    457:	learn: 0.3952265	total: 1.31s	remaining: 1.55s
    458:	learn: 0.3948552	total: 1.31s	remaining: 1.55s
    459:	learn: 0.3945055	total: 1.32s	remaining: 1.54s
    460:	learn: 0.3942388	total: 1.32s	remaining: 1.54s
    461:	learn: 0.3938845	total: 1.32s	remaining: 1.53s
    462:	learn: 0.3934466	total: 1.32s	remaining: 1.53s
    463:	learn: 0.3931510	total: 1.32s	remaining: 1.53s
    464:	learn: 0.3928267	total: 1.33s	remaining: 1.53s
    465:	learn: 0.3926062	total: 1.33s	remaining: 1.52s
    466:	learn: 0.3921506	total: 1.33s	remaining: 1.52s
    467:	learn: 0.3917620	total: 1.33s	remaining: 1.51s
    468:	learn: 0.3914566	total: 1.33s	remaining: 1.51s
    469:	learn: 0.3912414	total: 1.33s	remaining: 1.5s
    470:	learn: 0.3908352	total: 1.33s	remaining: 1.5s
    471:	learn: 0.3905011	total: 1.34s	remaining: 1.5s
    472:	learn: 0.3901716	total: 1.34s	remaining: 1.49s
    473:	learn: 0.3899627	total: 1.34s	remaining: 1.49s
    474:	learn: 0.3894250	total: 1.34s	remaining: 1.48s
    475:	learn: 0.3890944	total: 1.34s	remaining: 1.48s
    476:	learn: 0.3886896	total: 1.34s	remaining: 1.47s
    477:	learn: 0.3883534	total: 1.34s	remaining: 1.47s
    478:	learn: 0.3880367	total: 1.35s	remaining: 1.47s
    479:	learn: 0.3878126	total: 1.35s	remaining: 1.46s
    480:	learn: 0.3874376	total: 1.35s	remaining: 1.46s
    481:	learn: 0.3871304	total: 1.35s	remaining: 1.45s
    482:	learn: 0.3868231	total: 1.35s	remaining: 1.45s
    483:	learn: 0.3864982	total: 1.36s	remaining: 1.45s
    484:	learn: 0.3861254	total: 1.36s	remaining: 1.44s
    485:	learn: 0.3858998	total: 1.36s	remaining: 1.44s
    486:	learn: 0.3854810	total: 1.36s	remaining: 1.43s
    487:	learn: 0.3851311	total: 1.36s	remaining: 1.43s
    488:	learn: 0.3848123	total: 1.36s	remaining: 1.43s
    489:	learn: 0.3844635	total: 1.36s	remaining: 1.42s
    490:	learn: 0.3840809	total: 1.37s	remaining: 1.42s
    491:	learn: 0.3836944	total: 1.37s	remaining: 1.41s
    492:	learn: 0.3833551	total: 1.37s	remaining: 1.41s
    493:	learn: 0.3830792	total: 1.37s	remaining: 1.4s
    494:	learn: 0.3825047	total: 1.37s	remaining: 1.4s
    495:	learn: 0.3821551	total: 1.37s	remaining: 1.4s
    496:	learn: 0.3819040	total: 1.38s	remaining: 1.39s
    497:	learn: 0.3815504	total: 1.38s	remaining: 1.39s
    498:	learn: 0.3812619	total: 1.38s	remaining: 1.38s
    499:	learn: 0.3809595	total: 1.38s	remaining: 1.38s
    500:	learn: 0.3805517	total: 1.38s	remaining: 1.37s
    501:	learn: 0.3802126	total: 1.38s	remaining: 1.37s
    502:	learn: 0.3800408	total: 1.38s	remaining: 1.37s
    503:	learn: 0.3795083	total: 1.39s	remaining: 1.37s
    504:	learn: 0.3791381	total: 1.4s	remaining: 1.37s
    505:	learn: 0.3786674	total: 1.4s	remaining: 1.36s
    506:	learn: 0.3782959	total: 1.4s	remaining: 1.36s
    507:	learn: 0.3779692	total: 1.4s	remaining: 1.36s
    508:	learn: 0.3776246	total: 1.4s	remaining: 1.35s
    509:	learn: 0.3772367	total: 1.4s	remaining: 1.35s
    510:	learn: 0.3768605	total: 1.4s	remaining: 1.34s
    511:	learn: 0.3763271	total: 1.41s	remaining: 1.34s
    512:	learn: 0.3761111	total: 1.41s	remaining: 1.34s
    513:	learn: 0.3758758	total: 1.41s	remaining: 1.33s
    514:	learn: 0.3754354	total: 1.41s	remaining: 1.33s
    515:	learn: 0.3749483	total: 1.41s	remaining: 1.32s
    516:	learn: 0.3745156	total: 1.41s	remaining: 1.32s
    517:	learn: 0.3741987	total: 1.42s	remaining: 1.32s
    518:	learn: 0.3739278	total: 1.42s	remaining: 1.32s
    519:	learn: 0.3736857	total: 1.43s	remaining: 1.32s
    520:	learn: 0.3732556	total: 1.43s	remaining: 1.31s
    521:	learn: 0.3730407	total: 1.43s	remaining: 1.31s
    522:	learn: 0.3728372	total: 1.43s	remaining: 1.3s
    523:	learn: 0.3725236	total: 1.43s	remaining: 1.3s
    524:	learn: 0.3721006	total: 1.43s	remaining: 1.3s
    525:	learn: 0.3717057	total: 1.44s	remaining: 1.29s
    526:	learn: 0.3713627	total: 1.44s	remaining: 1.29s
    527:	learn: 0.3710899	total: 1.44s	remaining: 1.28s
    528:	learn: 0.3707694	total: 1.44s	remaining: 1.28s
    529:	learn: 0.3705163	total: 1.44s	remaining: 1.28s
    530:	learn: 0.3702163	total: 1.44s	remaining: 1.27s
    531:	learn: 0.3699680	total: 1.44s	remaining: 1.27s
    532:	learn: 0.3695804	total: 1.44s	remaining: 1.26s
    533:	learn: 0.3692362	total: 1.45s	remaining: 1.26s
    534:	learn: 0.3690209	total: 1.45s	remaining: 1.26s
    535:	learn: 0.3685981	total: 1.45s	remaining: 1.25s
    536:	learn: 0.3682060	total: 1.45s	remaining: 1.25s
    537:	learn: 0.3679561	total: 1.45s	remaining: 1.25s
    538:	learn: 0.3676233	total: 1.45s	remaining: 1.24s
    539:	learn: 0.3673112	total: 1.45s	remaining: 1.24s
    540:	learn: 0.3669195	total: 1.45s	remaining: 1.23s
    541:	learn: 0.3665953	total: 1.46s	remaining: 1.23s
    542:	learn: 0.3662761	total: 1.46s	remaining: 1.23s
    543:	learn: 0.3658649	total: 1.46s	remaining: 1.22s
    544:	learn: 0.3653294	total: 1.46s	remaining: 1.22s
    545:	learn: 0.3650385	total: 1.46s	remaining: 1.22s
    546:	learn: 0.3646401	total: 1.46s	remaining: 1.21s
    547:	learn: 0.3645019	total: 1.46s	remaining: 1.21s
    548:	learn: 0.3641810	total: 1.46s	remaining: 1.2s
    549:	learn: 0.3639342	total: 1.47s	remaining: 1.2s
    550:	learn: 0.3637084	total: 1.47s	remaining: 1.2s
    551:	learn: 0.3634693	total: 1.47s	remaining: 1.19s
    552:	learn: 0.3630533	total: 1.47s	remaining: 1.19s
    553:	learn: 0.3626798	total: 1.48s	remaining: 1.19s
    554:	learn: 0.3625060	total: 1.48s	remaining: 1.18s
    555:	learn: 0.3620805	total: 1.48s	remaining: 1.18s
    556:	learn: 0.3618519	total: 1.48s	remaining: 1.18s
    557:	learn: 0.3615626	total: 1.48s	remaining: 1.17s
    558:	learn: 0.3613213	total: 1.48s	remaining: 1.17s
    559:	learn: 0.3610301	total: 1.48s	remaining: 1.17s
    560:	learn: 0.3608510	total: 1.49s	remaining: 1.16s
    561:	learn: 0.3606721	total: 1.49s	remaining: 1.16s
    562:	learn: 0.3604439	total: 1.49s	remaining: 1.15s
    563:	learn: 0.3600980	total: 1.49s	remaining: 1.15s
    564:	learn: 0.3596534	total: 1.49s	remaining: 1.15s
    565:	learn: 0.3594761	total: 1.49s	remaining: 1.14s
    566:	learn: 0.3593242	total: 1.49s	remaining: 1.14s
    567:	learn: 0.3591736	total: 1.49s	remaining: 1.14s
    568:	learn: 0.3588585	total: 1.5s	remaining: 1.13s
    569:	learn: 0.3586155	total: 1.5s	remaining: 1.13s
    570:	learn: 0.3583969	total: 1.5s	remaining: 1.13s
    571:	learn: 0.3580985	total: 1.5s	remaining: 1.12s
    572:	learn: 0.3578687	total: 1.5s	remaining: 1.12s
    573:	learn: 0.3575707	total: 1.5s	remaining: 1.11s
    574:	learn: 0.3571789	total: 1.5s	remaining: 1.11s
    575:	learn: 0.3568428	total: 1.5s	remaining: 1.11s
    576:	learn: 0.3565674	total: 1.51s	remaining: 1.1s
    577:	learn: 0.3562385	total: 1.51s	remaining: 1.1s
    578:	learn: 0.3560349	total: 1.52s	remaining: 1.11s
    579:	learn: 0.3556953	total: 1.53s	remaining: 1.11s
    580:	learn: 0.3553258	total: 1.53s	remaining: 1.1s
    581:	learn: 0.3548626	total: 1.54s	remaining: 1.1s
    582:	learn: 0.3545285	total: 1.54s	remaining: 1.1s
    583:	learn: 0.3542536	total: 1.54s	remaining: 1.1s
    584:	learn: 0.3539073	total: 1.54s	remaining: 1.09s
    585:	learn: 0.3534859	total: 1.54s	remaining: 1.09s
    586:	learn: 0.3532509	total: 1.54s	remaining: 1.09s
    587:	learn: 0.3530436	total: 1.55s	remaining: 1.09s
    588:	learn: 0.3527570	total: 1.55s	remaining: 1.08s
    589:	learn: 0.3525101	total: 1.56s	remaining: 1.08s
    590:	learn: 0.3522113	total: 1.56s	remaining: 1.08s
    591:	learn: 0.3520489	total: 1.56s	remaining: 1.08s
    592:	learn: 0.3516889	total: 1.57s	remaining: 1.08s
    593:	learn: 0.3514473	total: 1.57s	remaining: 1.07s
    594:	learn: 0.3511787	total: 1.57s	remaining: 1.07s
    595:	learn: 0.3509142	total: 1.57s	remaining: 1.07s
    596:	learn: 0.3507387	total: 1.57s	remaining: 1.06s
    597:	learn: 0.3504683	total: 1.58s	remaining: 1.06s
    598:	learn: 0.3501645	total: 1.58s	remaining: 1.06s
    599:	learn: 0.3498414	total: 1.58s	remaining: 1.05s
    600:	learn: 0.3493886	total: 1.58s	remaining: 1.05s
    601:	learn: 0.3490806	total: 1.58s	remaining: 1.05s
    602:	learn: 0.3488135	total: 1.58s	remaining: 1.04s
    603:	learn: 0.3485939	total: 1.58s	remaining: 1.04s
    604:	learn: 0.3482250	total: 1.58s	remaining: 1.03s
    605:	learn: 0.3479442	total: 1.59s	remaining: 1.03s
    606:	learn: 0.3477725	total: 1.59s	remaining: 1.03s
    607:	learn: 0.3474272	total: 1.59s	remaining: 1.02s
    608:	learn: 0.3470944	total: 1.59s	remaining: 1.02s
    609:	learn: 0.3466551	total: 1.59s	remaining: 1.02s
    610:	learn: 0.3463616	total: 1.59s	remaining: 1.01s
    611:	learn: 0.3461570	total: 1.59s	remaining: 1.01s
    612:	learn: 0.3458491	total: 1.6s	remaining: 1.01s
    613:	learn: 0.3455455	total: 1.6s	remaining: 1s
    614:	learn: 0.3453008	total: 1.6s	remaining: 1s
    615:	learn: 0.3451190	total: 1.6s	remaining: 998ms
    616:	learn: 0.3449249	total: 1.6s	remaining: 994ms
    617:	learn: 0.3446516	total: 1.6s	remaining: 991ms
    618:	learn: 0.3444380	total: 1.61s	remaining: 989ms
    619:	learn: 0.3442321	total: 1.61s	remaining: 986ms
    620:	learn: 0.3440420	total: 1.61s	remaining: 983ms
    621:	learn: 0.3437195	total: 1.61s	remaining: 980ms
    622:	learn: 0.3432327	total: 1.61s	remaining: 977ms
    623:	learn: 0.3430127	total: 1.61s	remaining: 973ms
    624:	learn: 0.3427330	total: 1.62s	remaining: 970ms
    625:	learn: 0.3422502	total: 1.62s	remaining: 966ms
    626:	learn: 0.3420584	total: 1.62s	remaining: 963ms
    627:	learn: 0.3417045	total: 1.62s	remaining: 959ms
    628:	learn: 0.3416501	total: 1.62s	remaining: 956ms
    629:	learn: 0.3413845	total: 1.62s	remaining: 952ms
    630:	learn: 0.3410835	total: 1.62s	remaining: 949ms
    631:	learn: 0.3408639	total: 1.62s	remaining: 946ms
    632:	learn: 0.3403330	total: 1.63s	remaining: 943ms
    633:	learn: 0.3400616	total: 1.63s	remaining: 939ms
    634:	learn: 0.3396417	total: 1.63s	remaining: 936ms
    635:	learn: 0.3393275	total: 1.63s	remaining: 933ms
    636:	learn: 0.3389483	total: 1.63s	remaining: 929ms
    637:	learn: 0.3387498	total: 1.63s	remaining: 926ms
    638:	learn: 0.3384303	total: 1.63s	remaining: 923ms
    639:	learn: 0.3381935	total: 1.65s	remaining: 926ms
    640:	learn: 0.3380160	total: 1.65s	remaining: 923ms
    641:	learn: 0.3378595	total: 1.65s	remaining: 920ms
    642:	learn: 0.3377456	total: 1.65s	remaining: 916ms
    643:	learn: 0.3374696	total: 1.65s	remaining: 913ms
    644:	learn: 0.3370792	total: 1.65s	remaining: 909ms
    645:	learn: 0.3368629	total: 1.67s	remaining: 913ms
    646:	learn: 0.3365676	total: 1.67s	remaining: 911ms
    647:	learn: 0.3363391	total: 1.67s	remaining: 908ms
    648:	learn: 0.3361647	total: 1.67s	remaining: 904ms
    649:	learn: 0.3358873	total: 1.67s	remaining: 901ms
    650:	learn: 0.3355135	total: 1.67s	remaining: 898ms
    651:	learn: 0.3351650	total: 1.68s	remaining: 895ms
    652:	learn: 0.3348108	total: 1.68s	remaining: 891ms
    653:	learn: 0.3345348	total: 1.68s	remaining: 888ms
    654:	learn: 0.3343680	total: 1.68s	remaining: 885ms
    655:	learn: 0.3342893	total: 1.68s	remaining: 881ms
    656:	learn: 0.3341190	total: 1.68s	remaining: 878ms
    657:	learn: 0.3339014	total: 1.68s	remaining: 875ms
    658:	learn: 0.3336664	total: 1.68s	remaining: 872ms
    659:	learn: 0.3335302	total: 1.69s	remaining: 868ms
    660:	learn: 0.3331241	total: 1.69s	remaining: 865ms
    661:	learn: 0.3328024	total: 1.69s	remaining: 862ms
    662:	learn: 0.3325316	total: 1.69s	remaining: 858ms
    663:	learn: 0.3324100	total: 1.69s	remaining: 855ms
    664:	learn: 0.3321541	total: 1.69s	remaining: 852ms
    665:	learn: 0.3319974	total: 1.69s	remaining: 849ms
    666:	learn: 0.3317975	total: 1.69s	remaining: 846ms
    667:	learn: 0.3315524	total: 1.7s	remaining: 843ms
    668:	learn: 0.3313412	total: 1.7s	remaining: 840ms
    669:	learn: 0.3311172	total: 1.7s	remaining: 836ms
    670:	learn: 0.3309319	total: 1.7s	remaining: 833ms
    671:	learn: 0.3307170	total: 1.7s	remaining: 830ms
    672:	learn: 0.3304421	total: 1.7s	remaining: 827ms
    673:	learn: 0.3301300	total: 1.7s	remaining: 824ms
    674:	learn: 0.3298557	total: 1.7s	remaining: 821ms
    675:	learn: 0.3295640	total: 1.71s	remaining: 818ms
    676:	learn: 0.3293214	total: 1.71s	remaining: 815ms
    677:	learn: 0.3289244	total: 1.71s	remaining: 812ms
    678:	learn: 0.3287224	total: 1.71s	remaining: 809ms
    679:	learn: 0.3284095	total: 1.71s	remaining: 805ms
    680:	learn: 0.3282182	total: 1.71s	remaining: 802ms
    681:	learn: 0.3279353	total: 1.71s	remaining: 799ms
    682:	learn: 0.3277441	total: 1.72s	remaining: 796ms
    683:	learn: 0.3274582	total: 1.72s	remaining: 793ms
    684:	learn: 0.3271778	total: 1.72s	remaining: 790ms
    685:	learn: 0.3267508	total: 1.72s	remaining: 787ms
    686:	learn: 0.3264459	total: 1.72s	remaining: 784ms
    687:	learn: 0.3260522	total: 1.72s	remaining: 781ms
    688:	learn: 0.3257209	total: 1.72s	remaining: 778ms
    689:	learn: 0.3253320	total: 1.73s	remaining: 775ms
    690:	learn: 0.3249599	total: 1.73s	remaining: 772ms
    691:	learn: 0.3246715	total: 1.73s	remaining: 769ms
    692:	learn: 0.3246043	total: 1.73s	remaining: 766ms
    693:	learn: 0.3243228	total: 1.73s	remaining: 763ms
    694:	learn: 0.3240322	total: 1.73s	remaining: 760ms
    695:	learn: 0.3238588	total: 1.73s	remaining: 757ms
    696:	learn: 0.3237242	total: 1.74s	remaining: 755ms
    697:	learn: 0.3235334	total: 1.74s	remaining: 752ms
    698:	learn: 0.3234583	total: 1.74s	remaining: 749ms
    699:	learn: 0.3233351	total: 1.74s	remaining: 746ms
    700:	learn: 0.3230621	total: 1.74s	remaining: 743ms
    701:	learn: 0.3228487	total: 1.74s	remaining: 740ms
    702:	learn: 0.3225795	total: 1.75s	remaining: 737ms
    703:	learn: 0.3222725	total: 1.75s	remaining: 734ms
    704:	learn: 0.3220679	total: 1.75s	remaining: 731ms
    705:	learn: 0.3218231	total: 1.75s	remaining: 728ms
    706:	learn: 0.3215532	total: 1.75s	remaining: 725ms
    707:	learn: 0.3213057	total: 1.75s	remaining: 722ms
    708:	learn: 0.3211120	total: 1.75s	remaining: 719ms
    709:	learn: 0.3209608	total: 1.75s	remaining: 716ms
    710:	learn: 0.3206958	total: 1.75s	remaining: 713ms
    711:	learn: 0.3205042	total: 1.76s	remaining: 710ms
    712:	learn: 0.3203639	total: 1.76s	remaining: 708ms
    713:	learn: 0.3202346	total: 1.76s	remaining: 705ms
    714:	learn: 0.3200862	total: 1.76s	remaining: 702ms
    715:	learn: 0.3199268	total: 1.76s	remaining: 699ms
    716:	learn: 0.3196081	total: 1.76s	remaining: 696ms
    717:	learn: 0.3194574	total: 1.76s	remaining: 693ms
    718:	learn: 0.3191280	total: 1.76s	remaining: 690ms
    719:	learn: 0.3188974	total: 1.77s	remaining: 687ms
    720:	learn: 0.3186039	total: 1.77s	remaining: 684ms
    721:	learn: 0.3185274	total: 1.77s	remaining: 681ms
    722:	learn: 0.3184672	total: 1.77s	remaining: 678ms
    723:	learn: 0.3182843	total: 1.77s	remaining: 675ms
    724:	learn: 0.3181445	total: 1.77s	remaining: 672ms
    725:	learn: 0.3180535	total: 1.77s	remaining: 669ms
    726:	learn: 0.3176699	total: 1.77s	remaining: 666ms
    727:	learn: 0.3174515	total: 1.77s	remaining: 664ms
    728:	learn: 0.3172517	total: 1.78s	remaining: 661ms
    729:	learn: 0.3169398	total: 1.78s	remaining: 658ms
    730:	learn: 0.3166433	total: 1.78s	remaining: 655ms
    731:	learn: 0.3162918	total: 1.78s	remaining: 652ms
    732:	learn: 0.3160154	total: 1.78s	remaining: 649ms
    733:	learn: 0.3158345	total: 1.78s	remaining: 646ms
    734:	learn: 0.3156884	total: 1.78s	remaining: 643ms
    735:	learn: 0.3154700	total: 1.78s	remaining: 640ms
    736:	learn: 0.3151725	total: 1.79s	remaining: 638ms
    737:	learn: 0.3149833	total: 1.79s	remaining: 635ms
    738:	learn: 0.3148945	total: 1.79s	remaining: 632ms
    739:	learn: 0.3146413	total: 1.79s	remaining: 629ms
    740:	learn: 0.3145453	total: 1.79s	remaining: 626ms
    741:	learn: 0.3142532	total: 1.8s	remaining: 627ms
    742:	learn: 0.3140808	total: 1.81s	remaining: 625ms
    743:	learn: 0.3138641	total: 1.81s	remaining: 622ms
    744:	learn: 0.3137304	total: 1.81s	remaining: 620ms
    745:	learn: 0.3135155	total: 1.81s	remaining: 617ms
    746:	learn: 0.3133424	total: 1.81s	remaining: 614ms
    747:	learn: 0.3130712	total: 1.81s	remaining: 611ms
    748:	learn: 0.3127004	total: 1.81s	remaining: 608ms
    749:	learn: 0.3124777	total: 1.82s	remaining: 605ms
    750:	learn: 0.3121394	total: 1.82s	remaining: 603ms
    751:	learn: 0.3119436	total: 1.82s	remaining: 600ms
    752:	learn: 0.3116986	total: 1.82s	remaining: 597ms
    753:	learn: 0.3115656	total: 1.82s	remaining: 594ms
    754:	learn: 0.3114501	total: 1.82s	remaining: 592ms
    755:	learn: 0.3111048	total: 1.82s	remaining: 589ms
    756:	learn: 0.3109193	total: 1.82s	remaining: 586ms
    757:	learn: 0.3107448	total: 1.83s	remaining: 583ms
    758:	learn: 0.3105257	total: 1.83s	remaining: 580ms
    759:	learn: 0.3104061	total: 1.83s	remaining: 578ms
    760:	learn: 0.3101747	total: 1.83s	remaining: 575ms
    761:	learn: 0.3099412	total: 1.83s	remaining: 572ms
    762:	learn: 0.3097221	total: 1.83s	remaining: 570ms
    763:	learn: 0.3094487	total: 1.83s	remaining: 567ms
    764:	learn: 0.3091716	total: 1.84s	remaining: 564ms
    765:	learn: 0.3090421	total: 1.84s	remaining: 562ms
    766:	learn: 0.3088640	total: 1.84s	remaining: 559ms
    767:	learn: 0.3085965	total: 1.84s	remaining: 556ms
    768:	learn: 0.3083616	total: 1.84s	remaining: 553ms
    769:	learn: 0.3080985	total: 1.84s	remaining: 551ms
    770:	learn: 0.3079715	total: 1.84s	remaining: 548ms
    771:	learn: 0.3076964	total: 1.85s	remaining: 545ms
    772:	learn: 0.3075994	total: 1.85s	remaining: 543ms
    773:	learn: 0.3074512	total: 1.85s	remaining: 540ms
    774:	learn: 0.3072880	total: 1.85s	remaining: 537ms
    775:	learn: 0.3070059	total: 1.85s	remaining: 535ms
    776:	learn: 0.3065409	total: 1.85s	remaining: 532ms
    777:	learn: 0.3062357	total: 1.85s	remaining: 529ms
    778:	learn: 0.3060494	total: 1.86s	remaining: 527ms
    779:	learn: 0.3057171	total: 1.86s	remaining: 524ms
    780:	learn: 0.3054242	total: 1.86s	remaining: 521ms
    781:	learn: 0.3052925	total: 1.86s	remaining: 519ms
    782:	learn: 0.3051321	total: 1.86s	remaining: 516ms
    783:	learn: 0.3050003	total: 1.86s	remaining: 513ms
    784:	learn: 0.3048260	total: 1.86s	remaining: 511ms
    785:	learn: 0.3046389	total: 1.87s	remaining: 508ms
    786:	learn: 0.3044302	total: 1.87s	remaining: 506ms
    787:	learn: 0.3042453	total: 1.87s	remaining: 503ms
    788:	learn: 0.3040766	total: 1.87s	remaining: 500ms
    789:	learn: 0.3039220	total: 1.87s	remaining: 498ms
    790:	learn: 0.3037328	total: 1.87s	remaining: 495ms
    791:	learn: 0.3035034	total: 1.88s	remaining: 493ms
    792:	learn: 0.3034325	total: 1.88s	remaining: 490ms
    793:	learn: 0.3032128	total: 1.88s	remaining: 487ms
    794:	learn: 0.3030553	total: 1.88s	remaining: 485ms
    795:	learn: 0.3029265	total: 1.88s	remaining: 482ms
    796:	learn: 0.3025759	total: 1.88s	remaining: 480ms
    797:	learn: 0.3025014	total: 1.89s	remaining: 477ms
    798:	learn: 0.3021780	total: 1.89s	remaining: 475ms
    799:	learn: 0.3018760	total: 1.89s	remaining: 472ms
    800:	learn: 0.3017365	total: 1.89s	remaining: 470ms
    801:	learn: 0.3015434	total: 1.9s	remaining: 468ms
    802:	learn: 0.3013035	total: 1.9s	remaining: 466ms
    803:	learn: 0.3011298	total: 1.9s	remaining: 463ms
    804:	learn: 0.3009583	total: 1.9s	remaining: 461ms
    805:	learn: 0.3007937	total: 1.9s	remaining: 458ms
    806:	learn: 0.3005020	total: 1.9s	remaining: 455ms
    807:	learn: 0.3002857	total: 1.91s	remaining: 453ms
    808:	learn: 0.2999542	total: 1.91s	remaining: 450ms
    809:	learn: 0.2998034	total: 1.91s	remaining: 447ms
    810:	learn: 0.2994679	total: 1.91s	remaining: 445ms
    811:	learn: 0.2992922	total: 1.91s	remaining: 442ms
    812:	learn: 0.2989525	total: 1.91s	remaining: 440ms
    813:	learn: 0.2988843	total: 1.91s	remaining: 437ms
    814:	learn: 0.2986333	total: 1.92s	remaining: 435ms
    815:	learn: 0.2984650	total: 1.92s	remaining: 432ms
    816:	learn: 0.2983660	total: 1.92s	remaining: 430ms
    817:	learn: 0.2981047	total: 1.92s	remaining: 427ms
    818:	learn: 0.2978962	total: 1.92s	remaining: 424ms
    819:	learn: 0.2976227	total: 1.92s	remaining: 422ms
    820:	learn: 0.2973651	total: 1.92s	remaining: 419ms
    821:	learn: 0.2970952	total: 1.93s	remaining: 417ms
    822:	learn: 0.2969549	total: 1.93s	remaining: 415ms
    823:	learn: 0.2968257	total: 1.93s	remaining: 412ms
    824:	learn: 0.2965174	total: 1.93s	remaining: 410ms
    825:	learn: 0.2962890	total: 1.93s	remaining: 407ms
    826:	learn: 0.2959075	total: 1.93s	remaining: 404ms
    827:	learn: 0.2956116	total: 1.93s	remaining: 402ms
    828:	learn: 0.2952377	total: 1.94s	remaining: 399ms
    829:	learn: 0.2949474	total: 1.94s	remaining: 397ms
    830:	learn: 0.2948073	total: 1.94s	remaining: 394ms
    831:	learn: 0.2945125	total: 1.94s	remaining: 392ms
    832:	learn: 0.2943166	total: 1.94s	remaining: 389ms
    833:	learn: 0.2940166	total: 1.94s	remaining: 387ms
    834:	learn: 0.2938380	total: 1.94s	remaining: 384ms
    835:	learn: 0.2936978	total: 1.95s	remaining: 382ms
    836:	learn: 0.2935170	total: 1.95s	remaining: 379ms
    837:	learn: 0.2932565	total: 1.95s	remaining: 376ms
    838:	learn: 0.2929052	total: 1.95s	remaining: 374ms
    839:	learn: 0.2927079	total: 1.95s	remaining: 371ms
    840:	learn: 0.2924398	total: 1.95s	remaining: 369ms
    841:	learn: 0.2923313	total: 1.95s	remaining: 366ms
    842:	learn: 0.2921684	total: 1.95s	remaining: 364ms
    843:	learn: 0.2919597	total: 1.96s	remaining: 361ms
    844:	learn: 0.2918318	total: 1.96s	remaining: 359ms
    845:	learn: 0.2916668	total: 1.96s	remaining: 356ms
    846:	learn: 0.2914693	total: 1.96s	remaining: 354ms
    847:	learn: 0.2912804	total: 1.96s	remaining: 351ms
    848:	learn: 0.2911508	total: 1.96s	remaining: 349ms
    849:	learn: 0.2910642	total: 1.96s	remaining: 346ms
    850:	learn: 0.2908617	total: 1.96s	remaining: 344ms
    851:	learn: 0.2906814	total: 1.97s	remaining: 341ms
    852:	learn: 0.2904690	total: 1.97s	remaining: 339ms
    853:	learn: 0.2902362	total: 1.97s	remaining: 337ms
    854:	learn: 0.2900135	total: 1.97s	remaining: 334ms
    855:	learn: 0.2897123	total: 1.97s	remaining: 332ms
    856:	learn: 0.2894698	total: 1.97s	remaining: 329ms
    857:	learn: 0.2893142	total: 1.97s	remaining: 327ms
    858:	learn: 0.2890136	total: 1.97s	remaining: 324ms
    859:	learn: 0.2888601	total: 1.98s	remaining: 322ms
    860:	learn: 0.2886630	total: 1.98s	remaining: 319ms
    861:	learn: 0.2885175	total: 1.98s	remaining: 317ms
    862:	learn: 0.2883170	total: 1.98s	remaining: 315ms
    863:	learn: 0.2880664	total: 1.98s	remaining: 312ms
    864:	learn: 0.2878671	total: 1.99s	remaining: 310ms
    865:	learn: 0.2875972	total: 1.99s	remaining: 308ms
    866:	learn: 0.2874414	total: 1.99s	remaining: 305ms
    867:	learn: 0.2872545	total: 1.99s	remaining: 303ms
    868:	learn: 0.2870708	total: 1.99s	remaining: 300ms
    869:	learn: 0.2868316	total: 1.99s	remaining: 298ms
    870:	learn: 0.2866339	total: 1.99s	remaining: 295ms
    871:	learn: 0.2865923	total: 1.99s	remaining: 293ms
    872:	learn: 0.2864664	total: 2s	remaining: 290ms
    873:	learn: 0.2863478	total: 2.02s	remaining: 291ms
    874:	learn: 0.2861626	total: 2.02s	remaining: 289ms
    875:	learn: 0.2859715	total: 2.03s	remaining: 287ms
    876:	learn: 0.2858514	total: 2.03s	remaining: 285ms
    877:	learn: 0.2856724	total: 2.03s	remaining: 283ms
    878:	learn: 0.2855089	total: 2.03s	remaining: 280ms
    879:	learn: 0.2854204	total: 2.04s	remaining: 278ms
    880:	learn: 0.2851937	total: 2.04s	remaining: 275ms
    881:	learn: 0.2850213	total: 2.04s	remaining: 273ms
    882:	learn: 0.2848920	total: 2.04s	remaining: 270ms
    883:	learn: 0.2847252	total: 2.04s	remaining: 268ms
    884:	learn: 0.2845051	total: 2.04s	remaining: 265ms
    885:	learn: 0.2843960	total: 2.04s	remaining: 263ms
    886:	learn: 0.2841953	total: 2.04s	remaining: 261ms
    887:	learn: 0.2839355	total: 2.05s	remaining: 258ms
    888:	learn: 0.2836438	total: 2.05s	remaining: 256ms
    889:	learn: 0.2834348	total: 2.05s	remaining: 253ms
    890:	learn: 0.2832590	total: 2.05s	remaining: 251ms
    891:	learn: 0.2829858	total: 2.05s	remaining: 248ms
    892:	learn: 0.2828478	total: 2.05s	remaining: 246ms
    893:	learn: 0.2825301	total: 2.05s	remaining: 244ms
    894:	learn: 0.2824010	total: 2.06s	remaining: 241ms
    895:	learn: 0.2822710	total: 2.06s	remaining: 239ms
    896:	learn: 0.2819156	total: 2.06s	remaining: 236ms
    897:	learn: 0.2817344	total: 2.06s	remaining: 234ms
    898:	learn: 0.2815895	total: 2.06s	remaining: 231ms
    899:	learn: 0.2814930	total: 2.06s	remaining: 229ms
    900:	learn: 0.2812095	total: 2.06s	remaining: 227ms
    901:	learn: 0.2809294	total: 2.07s	remaining: 225ms
    902:	learn: 0.2807974	total: 2.07s	remaining: 223ms
    903:	learn: 0.2806573	total: 2.08s	remaining: 221ms
    904:	learn: 0.2803645	total: 2.08s	remaining: 218ms
    905:	learn: 0.2802090	total: 2.08s	remaining: 216ms
    906:	learn: 0.2800461	total: 2.08s	remaining: 214ms
    907:	learn: 0.2798569	total: 2.08s	remaining: 211ms
    908:	learn: 0.2797643	total: 2.09s	remaining: 209ms
    909:	learn: 0.2795885	total: 2.09s	remaining: 207ms
    910:	learn: 0.2793944	total: 2.09s	remaining: 204ms
    911:	learn: 0.2791919	total: 2.09s	remaining: 202ms
    912:	learn: 0.2790906	total: 2.09s	remaining: 199ms
    913:	learn: 0.2789060	total: 2.09s	remaining: 197ms
    914:	learn: 0.2786720	total: 2.1s	remaining: 195ms
    915:	learn: 0.2784611	total: 2.1s	remaining: 192ms
    916:	learn: 0.2782113	total: 2.1s	remaining: 190ms
    917:	learn: 0.2780414	total: 2.1s	remaining: 188ms
    918:	learn: 0.2778083	total: 2.1s	remaining: 185ms
    919:	learn: 0.2775713	total: 2.1s	remaining: 183ms
    920:	learn: 0.2774738	total: 2.1s	remaining: 181ms
    921:	learn: 0.2773348	total: 2.1s	remaining: 178ms
    922:	learn: 0.2772183	total: 2.11s	remaining: 176ms
    923:	learn: 0.2770912	total: 2.11s	remaining: 173ms
    924:	learn: 0.2769969	total: 2.11s	remaining: 171ms
    925:	learn: 0.2768031	total: 2.11s	remaining: 169ms
    926:	learn: 0.2766616	total: 2.11s	remaining: 166ms
    927:	learn: 0.2765076	total: 2.11s	remaining: 164ms
    928:	learn: 0.2761540	total: 2.12s	remaining: 162ms
    929:	learn: 0.2759572	total: 2.12s	remaining: 159ms
    930:	learn: 0.2757937	total: 2.12s	remaining: 157ms
    931:	learn: 0.2755387	total: 2.12s	remaining: 155ms
    932:	learn: 0.2753424	total: 2.12s	remaining: 152ms
    933:	learn: 0.2750983	total: 2.12s	remaining: 150ms
    934:	learn: 0.2749394	total: 2.12s	remaining: 148ms
    935:	learn: 0.2748684	total: 2.12s	remaining: 145ms
    936:	learn: 0.2747025	total: 2.12s	remaining: 143ms
    937:	learn: 0.2744501	total: 2.13s	remaining: 141ms
    938:	learn: 0.2741766	total: 2.13s	remaining: 138ms
    939:	learn: 0.2740751	total: 2.13s	remaining: 136ms
    940:	learn: 0.2738601	total: 2.13s	remaining: 134ms
    941:	learn: 0.2736221	total: 2.13s	remaining: 131ms
    942:	learn: 0.2734254	total: 2.13s	remaining: 129ms
    943:	learn: 0.2731311	total: 2.13s	remaining: 127ms
    944:	learn: 0.2730030	total: 2.13s	remaining: 124ms
    945:	learn: 0.2728339	total: 2.14s	remaining: 122ms
    946:	learn: 0.2726247	total: 2.14s	remaining: 120ms
    947:	learn: 0.2725411	total: 2.14s	remaining: 117ms
    948:	learn: 0.2723206	total: 2.14s	remaining: 115ms
    949:	learn: 0.2721476	total: 2.14s	remaining: 113ms
    950:	learn: 0.2718759	total: 2.14s	remaining: 110ms
    951:	learn: 0.2717793	total: 2.15s	remaining: 108ms
    952:	learn: 0.2716197	total: 2.15s	remaining: 106ms
    953:	learn: 0.2715248	total: 2.15s	remaining: 104ms
    954:	learn: 0.2714280	total: 2.15s	remaining: 101ms
    955:	learn: 0.2713150	total: 2.15s	remaining: 99ms
    956:	learn: 0.2711096	total: 2.15s	remaining: 96.7ms
    957:	learn: 0.2709656	total: 2.15s	remaining: 94.4ms
    958:	learn: 0.2708148	total: 2.15s	remaining: 92.1ms
    959:	learn: 0.2706601	total: 2.15s	remaining: 89.8ms
    960:	learn: 0.2703661	total: 2.16s	remaining: 87.5ms
    961:	learn: 0.2701886	total: 2.16s	remaining: 85.3ms
    962:	learn: 0.2701206	total: 2.16s	remaining: 83ms
    963:	learn: 0.2699416	total: 2.16s	remaining: 80.7ms
    964:	learn: 0.2698577	total: 2.16s	remaining: 78.4ms
    965:	learn: 0.2697207	total: 2.16s	remaining: 76.1ms
    966:	learn: 0.2694234	total: 2.16s	remaining: 73.9ms
    967:	learn: 0.2691837	total: 2.17s	remaining: 71.6ms
    968:	learn: 0.2689275	total: 2.17s	remaining: 69.3ms
    969:	learn: 0.2687682	total: 2.17s	remaining: 67ms
    970:	learn: 0.2687316	total: 2.17s	remaining: 64.8ms
    971:	learn: 0.2685794	total: 2.17s	remaining: 62.5ms
    972:	learn: 0.2683534	total: 2.17s	remaining: 60.3ms
    973:	learn: 0.2681200	total: 2.17s	remaining: 58ms
    974:	learn: 0.2679638	total: 2.17s	remaining: 55.8ms
    975:	learn: 0.2678516	total: 2.17s	remaining: 53.5ms
    976:	learn: 0.2676302	total: 2.18s	remaining: 51.2ms
    977:	learn: 0.2675006	total: 2.18s	remaining: 49ms
    978:	learn: 0.2673920	total: 2.18s	remaining: 46.8ms
    979:	learn: 0.2673105	total: 2.18s	remaining: 44.5ms
    980:	learn: 0.2671645	total: 2.18s	remaining: 42.3ms
    981:	learn: 0.2670783	total: 2.18s	remaining: 40ms
    982:	learn: 0.2669992	total: 2.18s	remaining: 37.8ms
    983:	learn: 0.2668481	total: 2.19s	remaining: 35.5ms
    984:	learn: 0.2667583	total: 2.19s	remaining: 33.4ms
    985:	learn: 0.2665053	total: 2.19s	remaining: 31.2ms
    986:	learn: 0.2663741	total: 2.2s	remaining: 28.9ms
    987:	learn: 0.2662453	total: 2.2s	remaining: 26.7ms
    988:	learn: 0.2661092	total: 2.2s	remaining: 24.5ms
    989:	learn: 0.2659775	total: 2.2s	remaining: 22.2ms
    990:	learn: 0.2657491	total: 2.2s	remaining: 20ms
    991:	learn: 0.2655746	total: 2.2s	remaining: 17.8ms
    992:	learn: 0.2654548	total: 2.21s	remaining: 15.5ms
    993:	learn: 0.2651612	total: 2.21s	remaining: 13.3ms
    994:	learn: 0.2648038	total: 2.21s	remaining: 11.1ms
    995:	learn: 0.2646414	total: 2.21s	remaining: 8.87ms
    996:	learn: 0.2645150	total: 2.21s	remaining: 6.65ms
    997:	learn: 0.2643620	total: 2.21s	remaining: 4.43ms
    998:	learn: 0.2642650	total: 2.21s	remaining: 2.22ms
    999:	learn: 0.2640218	total: 2.21s	remaining: 0us
    0:	learn: 0.6918869	total: 2.02ms	remaining: 2.02s
    1:	learn: 0.6907095	total: 3.26ms	remaining: 1.63s
    2:	learn: 0.6896000	total: 4.48ms	remaining: 1.49s
    3:	learn: 0.6884974	total: 5.68ms	remaining: 1.41s
    4:	learn: 0.6874596	total: 6.86ms	remaining: 1.36s
    5:	learn: 0.6864332	total: 7.76ms	remaining: 1.29s
    6:	learn: 0.6849787	total: 9.65ms	remaining: 1.37s
    7:	learn: 0.6840458	total: 11.6ms	remaining: 1.44s
    8:	learn: 0.6830164	total: 12.8ms	remaining: 1.41s
    9:	learn: 0.6817751	total: 14ms	remaining: 1.39s
    10:	learn: 0.6807793	total: 15.8ms	remaining: 1.42s
    11:	learn: 0.6797654	total: 17.6ms	remaining: 1.45s
    12:	learn: 0.6786874	total: 18.9ms	remaining: 1.44s
    13:	learn: 0.6774234	total: 20.1ms	remaining: 1.41s
    14:	learn: 0.6762335	total: 21.2ms	remaining: 1.4s
    15:	learn: 0.6750528	total: 22.5ms	remaining: 1.38s
    16:	learn: 0.6739153	total: 23.6ms	remaining: 1.36s
    17:	learn: 0.6726632	total: 24.9ms	remaining: 1.36s
    18:	learn: 0.6717189	total: 26.3ms	remaining: 1.36s
    19:	learn: 0.6704536	total: 27.5ms	remaining: 1.35s
    20:	learn: 0.6694389	total: 29.3ms	remaining: 1.36s
    21:	learn: 0.6683602	total: 30.6ms	remaining: 1.36s
    22:	learn: 0.6673338	total: 31.7ms	remaining: 1.35s
    23:	learn: 0.6664527	total: 32.8ms	remaining: 1.33s
    24:	learn: 0.6653395	total: 34.2ms	remaining: 1.33s
    25:	learn: 0.6642815	total: 35.8ms	remaining: 1.34s
    26:	learn: 0.6632941	total: 37.1ms	remaining: 1.33s
    27:	learn: 0.6622495	total: 38.3ms	remaining: 1.33s
    28:	learn: 0.6611268	total: 39.5ms	remaining: 1.32s
    29:	learn: 0.6602187	total: 41.4ms	remaining: 1.34s
    30:	learn: 0.6587994	total: 42.8ms	remaining: 1.34s
    31:	learn: 0.6577835	total: 44ms	remaining: 1.33s
    32:	learn: 0.6567842	total: 45.1ms	remaining: 1.32s
    33:	learn: 0.6558671	total: 46.3ms	remaining: 1.31s
    34:	learn: 0.6548285	total: 47.5ms	remaining: 1.31s
    35:	learn: 0.6538094	total: 48.6ms	remaining: 1.3s
    36:	learn: 0.6527777	total: 49.7ms	remaining: 1.29s
    37:	learn: 0.6516232	total: 50.8ms	remaining: 1.29s
    38:	learn: 0.6504498	total: 51.9ms	remaining: 1.28s
    39:	learn: 0.6492084	total: 53.7ms	remaining: 1.29s
    40:	learn: 0.6482530	total: 56.4ms	remaining: 1.32s
    41:	learn: 0.6472314	total: 65.8ms	remaining: 1.5s
    42:	learn: 0.6462297	total: 67.1ms	remaining: 1.49s
    43:	learn: 0.6452469	total: 68.4ms	remaining: 1.49s
    44:	learn: 0.6442824	total: 69.6ms	remaining: 1.48s
    45:	learn: 0.6432856	total: 74.2ms	remaining: 1.54s
    46:	learn: 0.6424260	total: 77ms	remaining: 1.56s
    47:	learn: 0.6414519	total: 79.1ms	remaining: 1.57s
    48:	learn: 0.6404024	total: 81.6ms	remaining: 1.58s
    49:	learn: 0.6394951	total: 83.1ms	remaining: 1.58s
    50:	learn: 0.6384562	total: 84.4ms	remaining: 1.57s
    51:	learn: 0.6377340	total: 86.3ms	remaining: 1.57s
    52:	learn: 0.6370376	total: 88.1ms	remaining: 1.57s
    53:	learn: 0.6361699	total: 89.4ms	remaining: 1.57s
    54:	learn: 0.6349325	total: 90.6ms	remaining: 1.56s
    55:	learn: 0.6340741	total: 92.1ms	remaining: 1.55s
    56:	learn: 0.6330330	total: 93.7ms	remaining: 1.55s
    57:	learn: 0.6319279	total: 95.3ms	remaining: 1.55s
    58:	learn: 0.6308126	total: 96.5ms	remaining: 1.54s
    59:	learn: 0.6296362	total: 97.7ms	remaining: 1.53s
    60:	learn: 0.6288846	total: 98.8ms	remaining: 1.52s
    61:	learn: 0.6280550	total: 100ms	remaining: 1.52s
    62:	learn: 0.6270512	total: 102ms	remaining: 1.51s
    63:	learn: 0.6261965	total: 103ms	remaining: 1.51s
    64:	learn: 0.6253990	total: 105ms	remaining: 1.5s
    65:	learn: 0.6241247	total: 106ms	remaining: 1.5s
    66:	learn: 0.6233855	total: 107ms	remaining: 1.49s
    67:	learn: 0.6225575	total: 109ms	remaining: 1.49s
    68:	learn: 0.6217561	total: 110ms	remaining: 1.49s
    69:	learn: 0.6210245	total: 111ms	remaining: 1.48s
    70:	learn: 0.6200431	total: 113ms	remaining: 1.47s
    71:	learn: 0.6191761	total: 114ms	remaining: 1.47s
    72:	learn: 0.6182466	total: 116ms	remaining: 1.47s
    73:	learn: 0.6172773	total: 117ms	remaining: 1.47s
    74:	learn: 0.6164477	total: 118ms	remaining: 1.46s
    75:	learn: 0.6155428	total: 120ms	remaining: 1.46s
    76:	learn: 0.6147587	total: 121ms	remaining: 1.45s
    77:	learn: 0.6138804	total: 123ms	remaining: 1.45s
    78:	learn: 0.6129919	total: 125ms	remaining: 1.45s
    79:	learn: 0.6116394	total: 126ms	remaining: 1.44s
    80:	learn: 0.6108288	total: 127ms	remaining: 1.44s
    81:	learn: 0.6097555	total: 128ms	remaining: 1.43s
    82:	learn: 0.6086159	total: 130ms	remaining: 1.43s
    83:	learn: 0.6079374	total: 131ms	remaining: 1.42s
    84:	learn: 0.6068348	total: 132ms	remaining: 1.42s
    85:	learn: 0.6058541	total: 133ms	remaining: 1.42s
    86:	learn: 0.6048878	total: 135ms	remaining: 1.41s
    87:	learn: 0.6041164	total: 136ms	remaining: 1.41s
    88:	learn: 0.6030150	total: 138ms	remaining: 1.41s
    89:	learn: 0.6023052	total: 139ms	remaining: 1.4s
    90:	learn: 0.6014823	total: 140ms	remaining: 1.4s
    91:	learn: 0.6008691	total: 141ms	remaining: 1.39s
    92:	learn: 0.6000707	total: 148ms	remaining: 1.45s
    93:	learn: 0.5992743	total: 149ms	remaining: 1.44s
    94:	learn: 0.5981709	total: 151ms	remaining: 1.43s
    95:	learn: 0.5973049	total: 152ms	remaining: 1.43s
    96:	learn: 0.5965799	total: 153ms	remaining: 1.42s
    97:	learn: 0.5958428	total: 154ms	remaining: 1.42s
    98:	learn: 0.5951556	total: 155ms	remaining: 1.41s
    99:	learn: 0.5943759	total: 156ms	remaining: 1.41s
    100:	learn: 0.5933803	total: 158ms	remaining: 1.4s
    101:	learn: 0.5924483	total: 159ms	remaining: 1.4s
    102:	learn: 0.5914717	total: 160ms	remaining: 1.4s
    103:	learn: 0.5905939	total: 162ms	remaining: 1.39s
    104:	learn: 0.5898563	total: 163ms	remaining: 1.39s
    105:	learn: 0.5888931	total: 164ms	remaining: 1.38s
    106:	learn: 0.5880638	total: 165ms	remaining: 1.38s
    107:	learn: 0.5870656	total: 166ms	remaining: 1.37s
    108:	learn: 0.5862185	total: 167ms	remaining: 1.37s
    109:	learn: 0.5854891	total: 169ms	remaining: 1.37s
    110:	learn: 0.5848289	total: 170ms	remaining: 1.36s
    111:	learn: 0.5839230	total: 172ms	remaining: 1.36s
    112:	learn: 0.5831818	total: 173ms	remaining: 1.36s
    113:	learn: 0.5826976	total: 175ms	remaining: 1.36s
    114:	learn: 0.5819756	total: 176ms	remaining: 1.36s
    115:	learn: 0.5809390	total: 178ms	remaining: 1.35s
    116:	learn: 0.5799830	total: 179ms	remaining: 1.35s
    117:	learn: 0.5794922	total: 181ms	remaining: 1.35s
    118:	learn: 0.5786623	total: 183ms	remaining: 1.35s
    119:	learn: 0.5779462	total: 184ms	remaining: 1.35s
    120:	learn: 0.5773177	total: 186ms	remaining: 1.35s
    121:	learn: 0.5764507	total: 187ms	remaining: 1.35s
    122:	learn: 0.5756193	total: 190ms	remaining: 1.35s
    123:	learn: 0.5746836	total: 193ms	remaining: 1.36s
    124:	learn: 0.5738920	total: 202ms	remaining: 1.41s
    125:	learn: 0.5733100	total: 204ms	remaining: 1.41s
    126:	learn: 0.5725773	total: 205ms	remaining: 1.41s
    127:	learn: 0.5720072	total: 207ms	remaining: 1.41s
    128:	learn: 0.5711971	total: 209ms	remaining: 1.41s
    129:	learn: 0.5704806	total: 210ms	remaining: 1.41s
    130:	learn: 0.5697909	total: 212ms	remaining: 1.41s
    131:	learn: 0.5690912	total: 213ms	remaining: 1.4s
    132:	learn: 0.5682001	total: 215ms	remaining: 1.4s
    133:	learn: 0.5675095	total: 217ms	remaining: 1.4s
    134:	learn: 0.5666746	total: 218ms	remaining: 1.4s
    135:	learn: 0.5656949	total: 220ms	remaining: 1.4s
    136:	learn: 0.5647606	total: 221ms	remaining: 1.39s
    137:	learn: 0.5639987	total: 223ms	remaining: 1.39s
    138:	learn: 0.5630932	total: 224ms	remaining: 1.39s
    139:	learn: 0.5624942	total: 226ms	remaining: 1.39s
    140:	learn: 0.5617045	total: 228ms	remaining: 1.39s
    141:	learn: 0.5608551	total: 229ms	remaining: 1.38s
    142:	learn: 0.5601727	total: 231ms	remaining: 1.38s
    143:	learn: 0.5593583	total: 232ms	remaining: 1.38s
    144:	learn: 0.5583766	total: 234ms	remaining: 1.38s
    145:	learn: 0.5577476	total: 235ms	remaining: 1.38s
    146:	learn: 0.5566969	total: 237ms	remaining: 1.37s
    147:	learn: 0.5558364	total: 238ms	remaining: 1.37s
    148:	learn: 0.5553401	total: 240ms	remaining: 1.37s
    149:	learn: 0.5544860	total: 241ms	remaining: 1.37s
    150:	learn: 0.5540291	total: 243ms	remaining: 1.36s
    151:	learn: 0.5533045	total: 244ms	remaining: 1.36s
    152:	learn: 0.5527557	total: 246ms	remaining: 1.36s
    153:	learn: 0.5521565	total: 250ms	remaining: 1.37s
    154:	learn: 0.5516302	total: 251ms	remaining: 1.37s
    155:	learn: 0.5506567	total: 254ms	remaining: 1.37s
    156:	learn: 0.5499638	total: 255ms	remaining: 1.37s
    157:	learn: 0.5492059	total: 257ms	remaining: 1.37s
    158:	learn: 0.5486160	total: 258ms	remaining: 1.37s
    159:	learn: 0.5478743	total: 260ms	remaining: 1.36s
    160:	learn: 0.5471911	total: 261ms	remaining: 1.36s
    161:	learn: 0.5462748	total: 263ms	remaining: 1.36s
    162:	learn: 0.5457225	total: 265ms	remaining: 1.36s
    163:	learn: 0.5450170	total: 266ms	remaining: 1.36s
    164:	learn: 0.5445286	total: 267ms	remaining: 1.35s
    165:	learn: 0.5438266	total: 269ms	remaining: 1.35s
    166:	learn: 0.5431915	total: 270ms	remaining: 1.35s
    167:	learn: 0.5424464	total: 272ms	remaining: 1.35s
    168:	learn: 0.5416509	total: 274ms	remaining: 1.35s
    169:	learn: 0.5409259	total: 276ms	remaining: 1.34s
    170:	learn: 0.5403612	total: 277ms	remaining: 1.34s
    171:	learn: 0.5395569	total: 279ms	remaining: 1.34s
    172:	learn: 0.5390692	total: 281ms	remaining: 1.34s
    173:	learn: 0.5380616	total: 283ms	remaining: 1.34s
    174:	learn: 0.5370532	total: 285ms	remaining: 1.34s
    175:	learn: 0.5362207	total: 286ms	remaining: 1.34s
    176:	learn: 0.5354971	total: 288ms	remaining: 1.34s
    177:	learn: 0.5346360	total: 290ms	remaining: 1.34s
    178:	learn: 0.5341902	total: 291ms	remaining: 1.34s
    179:	learn: 0.5333694	total: 293ms	remaining: 1.33s
    180:	learn: 0.5327304	total: 295ms	remaining: 1.33s
    181:	learn: 0.5323432	total: 296ms	remaining: 1.33s
    182:	learn: 0.5315507	total: 297ms	remaining: 1.33s
    183:	learn: 0.5310209	total: 299ms	remaining: 1.32s
    184:	learn: 0.5304177	total: 301ms	remaining: 1.32s
    185:	learn: 0.5297874	total: 302ms	remaining: 1.32s
    186:	learn: 0.5293058	total: 303ms	remaining: 1.32s
    187:	learn: 0.5286236	total: 305ms	remaining: 1.32s
    188:	learn: 0.5277143	total: 306ms	remaining: 1.31s
    189:	learn: 0.5266913	total: 308ms	remaining: 1.31s
    190:	learn: 0.5261298	total: 310ms	remaining: 1.31s
    191:	learn: 0.5255553	total: 311ms	remaining: 1.31s
    192:	learn: 0.5249771	total: 313ms	remaining: 1.31s
    193:	learn: 0.5242177	total: 315ms	remaining: 1.31s
    194:	learn: 0.5235802	total: 318ms	remaining: 1.31s
    195:	learn: 0.5225486	total: 323ms	remaining: 1.32s
    196:	learn: 0.5217112	total: 326ms	remaining: 1.33s
    197:	learn: 0.5211934	total: 330ms	remaining: 1.34s
    198:	learn: 0.5207752	total: 334ms	remaining: 1.34s
    199:	learn: 0.5200123	total: 337ms	remaining: 1.35s
    200:	learn: 0.5193172	total: 341ms	remaining: 1.36s
    201:	learn: 0.5185237	total: 345ms	remaining: 1.36s
    202:	learn: 0.5180801	total: 349ms	remaining: 1.37s
    203:	learn: 0.5176541	total: 353ms	remaining: 1.38s
    204:	learn: 0.5170277	total: 357ms	remaining: 1.38s
    205:	learn: 0.5161101	total: 361ms	remaining: 1.39s
    206:	learn: 0.5154903	total: 364ms	remaining: 1.4s
    207:	learn: 0.5147008	total: 375ms	remaining: 1.43s
    208:	learn: 0.5139488	total: 386ms	remaining: 1.46s
    209:	learn: 0.5132651	total: 389ms	remaining: 1.46s
    210:	learn: 0.5127195	total: 392ms	remaining: 1.47s
    211:	learn: 0.5121326	total: 396ms	remaining: 1.47s
    212:	learn: 0.5116904	total: 400ms	remaining: 1.48s
    213:	learn: 0.5107378	total: 403ms	remaining: 1.48s
    214:	learn: 0.5100782	total: 407ms	remaining: 1.49s
    215:	learn: 0.5096489	total: 411ms	remaining: 1.49s
    216:	learn: 0.5089351	total: 416ms	remaining: 1.5s
    217:	learn: 0.5083718	total: 419ms	remaining: 1.5s
    218:	learn: 0.5078532	total: 423ms	remaining: 1.51s
    219:	learn: 0.5070823	total: 428ms	remaining: 1.52s
    220:	learn: 0.5067222	total: 431ms	remaining: 1.52s
    221:	learn: 0.5060558	total: 436ms	remaining: 1.53s
    222:	learn: 0.5054522	total: 439ms	remaining: 1.53s
    223:	learn: 0.5047812	total: 443ms	remaining: 1.53s
    224:	learn: 0.5043270	total: 447ms	remaining: 1.54s
    225:	learn: 0.5038082	total: 451ms	remaining: 1.54s
    226:	learn: 0.5032403	total: 455ms	remaining: 1.55s
    227:	learn: 0.5026089	total: 459ms	remaining: 1.55s
    228:	learn: 0.5019078	total: 463ms	remaining: 1.56s
    229:	learn: 0.5010677	total: 467ms	remaining: 1.56s
    230:	learn: 0.5004797	total: 472ms	remaining: 1.57s
    231:	learn: 0.5000801	total: 475ms	remaining: 1.57s
    232:	learn: 0.4995281	total: 479ms	remaining: 1.58s
    233:	learn: 0.4989349	total: 483ms	remaining: 1.58s
    234:	learn: 0.4983239	total: 487ms	remaining: 1.58s
    235:	learn: 0.4977345	total: 490ms	remaining: 1.59s
    236:	learn: 0.4972209	total: 494ms	remaining: 1.59s
    237:	learn: 0.4967576	total: 498ms	remaining: 1.59s
    238:	learn: 0.4963517	total: 502ms	remaining: 1.6s
    239:	learn: 0.4958271	total: 506ms	remaining: 1.6s
    240:	learn: 0.4951212	total: 510ms	remaining: 1.6s
    241:	learn: 0.4947071	total: 514ms	remaining: 1.61s
    242:	learn: 0.4942853	total: 518ms	remaining: 1.61s
    243:	learn: 0.4936040	total: 522ms	remaining: 1.62s
    244:	learn: 0.4930561	total: 526ms	remaining: 1.62s
    245:	learn: 0.4924403	total: 530ms	remaining: 1.62s
    246:	learn: 0.4917661	total: 546ms	remaining: 1.67s
    247:	learn: 0.4912270	total: 555ms	remaining: 1.68s
    248:	learn: 0.4905116	total: 559ms	remaining: 1.69s
    249:	learn: 0.4899101	total: 564ms	remaining: 1.69s
    250:	learn: 0.4893575	total: 568ms	remaining: 1.7s
    251:	learn: 0.4887786	total: 573ms	remaining: 1.7s
    252:	learn: 0.4881231	total: 576ms	remaining: 1.7s
    253:	learn: 0.4873455	total: 580ms	remaining: 1.7s
    254:	learn: 0.4867256	total: 584ms	remaining: 1.71s
    255:	learn: 0.4859819	total: 588ms	remaining: 1.71s
    256:	learn: 0.4856628	total: 592ms	remaining: 1.71s
    257:	learn: 0.4851707	total: 596ms	remaining: 1.71s
    258:	learn: 0.4845777	total: 600ms	remaining: 1.72s
    259:	learn: 0.4839485	total: 604ms	remaining: 1.72s
    260:	learn: 0.4833956	total: 608ms	remaining: 1.72s
    261:	learn: 0.4826634	total: 612ms	remaining: 1.72s
    262:	learn: 0.4820420	total: 616ms	remaining: 1.73s
    263:	learn: 0.4814126	total: 620ms	remaining: 1.73s
    264:	learn: 0.4807546	total: 624ms	remaining: 1.73s
    265:	learn: 0.4801425	total: 628ms	remaining: 1.73s
    266:	learn: 0.4797000	total: 632ms	remaining: 1.73s
    267:	learn: 0.4792489	total: 636ms	remaining: 1.74s
    268:	learn: 0.4784225	total: 640ms	remaining: 1.74s
    269:	learn: 0.4779274	total: 644ms	remaining: 1.74s
    270:	learn: 0.4774020	total: 648ms	remaining: 1.74s
    271:	learn: 0.4769307	total: 653ms	remaining: 1.75s
    272:	learn: 0.4763279	total: 656ms	remaining: 1.75s
    273:	learn: 0.4760015	total: 659ms	remaining: 1.75s
    274:	learn: 0.4755537	total: 665ms	remaining: 1.75s
    275:	learn: 0.4751463	total: 667ms	remaining: 1.75s
    276:	learn: 0.4747130	total: 671ms	remaining: 1.75s
    277:	learn: 0.4742695	total: 675ms	remaining: 1.75s
    278:	learn: 0.4735550	total: 679ms	remaining: 1.75s
    279:	learn: 0.4729069	total: 683ms	remaining: 1.76s
    280:	learn: 0.4724714	total: 684ms	remaining: 1.75s
    281:	learn: 0.4721044	total: 686ms	remaining: 1.75s
    282:	learn: 0.4717955	total: 687ms	remaining: 1.74s
    283:	learn: 0.4711951	total: 688ms	remaining: 1.74s
    284:	learn: 0.4706111	total: 690ms	remaining: 1.73s
    285:	learn: 0.4701546	total: 691ms	remaining: 1.73s
    286:	learn: 0.4697646	total: 693ms	remaining: 1.72s
    287:	learn: 0.4689982	total: 694ms	remaining: 1.72s
    288:	learn: 0.4685527	total: 695ms	remaining: 1.71s
    289:	learn: 0.4680757	total: 696ms	remaining: 1.7s
    290:	learn: 0.4675406	total: 697ms	remaining: 1.7s
    291:	learn: 0.4669605	total: 698ms	remaining: 1.69s
    292:	learn: 0.4663477	total: 700ms	remaining: 1.69s
    293:	learn: 0.4658416	total: 702ms	remaining: 1.69s
    294:	learn: 0.4652179	total: 704ms	remaining: 1.68s
    295:	learn: 0.4646821	total: 705ms	remaining: 1.68s
    296:	learn: 0.4642878	total: 706ms	remaining: 1.67s
    297:	learn: 0.4637833	total: 708ms	remaining: 1.67s
    298:	learn: 0.4632003	total: 709ms	remaining: 1.66s
    299:	learn: 0.4626257	total: 710ms	remaining: 1.66s
    300:	learn: 0.4622435	total: 712ms	remaining: 1.65s
    301:	learn: 0.4617246	total: 713ms	remaining: 1.65s
    302:	learn: 0.4612985	total: 714ms	remaining: 1.64s
    303:	learn: 0.4609645	total: 716ms	remaining: 1.64s
    304:	learn: 0.4603634	total: 717ms	remaining: 1.63s
    305:	learn: 0.4596479	total: 719ms	remaining: 1.63s
    306:	learn: 0.4591460	total: 721ms	remaining: 1.63s
    307:	learn: 0.4586914	total: 722ms	remaining: 1.62s
    308:	learn: 0.4583886	total: 723ms	remaining: 1.62s
    309:	learn: 0.4579716	total: 725ms	remaining: 1.61s
    310:	learn: 0.4576709	total: 726ms	remaining: 1.61s
    311:	learn: 0.4571110	total: 727ms	remaining: 1.6s
    312:	learn: 0.4567054	total: 728ms	remaining: 1.6s
    313:	learn: 0.4562964	total: 730ms	remaining: 1.59s
    314:	learn: 0.4557273	total: 732ms	remaining: 1.59s
    315:	learn: 0.4553919	total: 733ms	remaining: 1.59s
    316:	learn: 0.4549232	total: 735ms	remaining: 1.58s
    317:	learn: 0.4544991	total: 736ms	remaining: 1.58s
    318:	learn: 0.4540360	total: 737ms	remaining: 1.57s
    319:	learn: 0.4534247	total: 738ms	remaining: 1.57s
    320:	learn: 0.4529385	total: 739ms	remaining: 1.56s
    321:	learn: 0.4525199	total: 741ms	remaining: 1.56s
    322:	learn: 0.4521400	total: 742ms	remaining: 1.55s
    323:	learn: 0.4516770	total: 743ms	remaining: 1.55s
    324:	learn: 0.4511209	total: 745ms	remaining: 1.55s
    325:	learn: 0.4506782	total: 747ms	remaining: 1.54s
    326:	learn: 0.4502335	total: 748ms	remaining: 1.54s
    327:	learn: 0.4498270	total: 749ms	remaining: 1.53s
    328:	learn: 0.4492761	total: 750ms	remaining: 1.53s
    329:	learn: 0.4486022	total: 751ms	remaining: 1.52s
    330:	learn: 0.4481993	total: 753ms	remaining: 1.52s
    331:	learn: 0.4475916	total: 754ms	remaining: 1.52s
    332:	learn: 0.4469475	total: 755ms	remaining: 1.51s
    333:	learn: 0.4465096	total: 756ms	remaining: 1.51s
    334:	learn: 0.4458964	total: 758ms	remaining: 1.5s
    335:	learn: 0.4453308	total: 760ms	remaining: 1.5s
    336:	learn: 0.4448189	total: 761ms	remaining: 1.5s
    337:	learn: 0.4441843	total: 762ms	remaining: 1.49s
    338:	learn: 0.4435656	total: 763ms	remaining: 1.49s
    339:	learn: 0.4431130	total: 764ms	remaining: 1.48s
    340:	learn: 0.4424023	total: 765ms	remaining: 1.48s
    341:	learn: 0.4420524	total: 767ms	remaining: 1.48s
    342:	learn: 0.4414892	total: 768ms	remaining: 1.47s
    343:	learn: 0.4409313	total: 769ms	remaining: 1.47s
    344:	learn: 0.4404261	total: 771ms	remaining: 1.46s
    345:	learn: 0.4400485	total: 772ms	remaining: 1.46s
    346:	learn: 0.4396016	total: 774ms	remaining: 1.46s
    347:	learn: 0.4390671	total: 775ms	remaining: 1.45s
    348:	learn: 0.4387800	total: 776ms	remaining: 1.45s
    349:	learn: 0.4384767	total: 777ms	remaining: 1.44s
    350:	learn: 0.4381728	total: 778ms	remaining: 1.44s
    351:	learn: 0.4379399	total: 780ms	remaining: 1.44s
    352:	learn: 0.4376068	total: 781ms	remaining: 1.43s
    353:	learn: 0.4370865	total: 783ms	remaining: 1.43s
    354:	learn: 0.4366078	total: 784ms	remaining: 1.42s
    355:	learn: 0.4361617	total: 785ms	remaining: 1.42s
    356:	learn: 0.4358538	total: 787ms	remaining: 1.42s
    357:	learn: 0.4355243	total: 788ms	remaining: 1.41s
    358:	learn: 0.4350539	total: 789ms	remaining: 1.41s
    359:	learn: 0.4345864	total: 790ms	remaining: 1.4s
    360:	learn: 0.4340925	total: 791ms	remaining: 1.4s
    361:	learn: 0.4337524	total: 792ms	remaining: 1.4s
    362:	learn: 0.4332822	total: 794ms	remaining: 1.39s
    363:	learn: 0.4329155	total: 795ms	remaining: 1.39s
    364:	learn: 0.4324423	total: 796ms	remaining: 1.38s
    365:	learn: 0.4320703	total: 798ms	remaining: 1.38s
    366:	learn: 0.4315321	total: 799ms	remaining: 1.38s
    367:	learn: 0.4311592	total: 800ms	remaining: 1.37s
    368:	learn: 0.4305721	total: 801ms	remaining: 1.37s
    369:	learn: 0.4302706	total: 803ms	remaining: 1.37s
    370:	learn: 0.4298455	total: 804ms	remaining: 1.36s
    371:	learn: 0.4295163	total: 805ms	remaining: 1.36s
    372:	learn: 0.4289848	total: 806ms	remaining: 1.35s
    373:	learn: 0.4282655	total: 807ms	remaining: 1.35s
    374:	learn: 0.4277896	total: 808ms	remaining: 1.35s
    375:	learn: 0.4273579	total: 810ms	remaining: 1.34s
    376:	learn: 0.4267384	total: 811ms	remaining: 1.34s
    377:	learn: 0.4261097	total: 813ms	remaining: 1.34s
    378:	learn: 0.4256745	total: 814ms	remaining: 1.33s
    379:	learn: 0.4252608	total: 815ms	remaining: 1.33s
    380:	learn: 0.4248515	total: 816ms	remaining: 1.32s
    381:	learn: 0.4244838	total: 817ms	remaining: 1.32s
    382:	learn: 0.4243126	total: 818ms	remaining: 1.32s
    383:	learn: 0.4239668	total: 819ms	remaining: 1.31s
    384:	learn: 0.4233470	total: 820ms	remaining: 1.31s
    385:	learn: 0.4229491	total: 821ms	remaining: 1.31s
    386:	learn: 0.4225574	total: 823ms	remaining: 1.3s
    387:	learn: 0.4221927	total: 825ms	remaining: 1.3s
    388:	learn: 0.4217105	total: 826ms	remaining: 1.3s
    389:	learn: 0.4212659	total: 827ms	remaining: 1.29s
    390:	learn: 0.4206719	total: 836ms	remaining: 1.3s
    391:	learn: 0.4202688	total: 838ms	remaining: 1.3s
    392:	learn: 0.4199662	total: 840ms	remaining: 1.3s
    393:	learn: 0.4196230	total: 841ms	remaining: 1.29s
    394:	learn: 0.4192292	total: 842ms	remaining: 1.29s
    395:	learn: 0.4188960	total: 843ms	remaining: 1.28s
    396:	learn: 0.4183157	total: 844ms	remaining: 1.28s
    397:	learn: 0.4177423	total: 845ms	remaining: 1.28s
    398:	learn: 0.4173232	total: 847ms	remaining: 1.27s
    399:	learn: 0.4169872	total: 848ms	remaining: 1.27s
    400:	learn: 0.4165300	total: 850ms	remaining: 1.27s
    401:	learn: 0.4161933	total: 851ms	remaining: 1.27s
    402:	learn: 0.4158733	total: 853ms	remaining: 1.26s
    403:	learn: 0.4155185	total: 854ms	remaining: 1.26s
    404:	learn: 0.4150895	total: 856ms	remaining: 1.26s
    405:	learn: 0.4146444	total: 857ms	remaining: 1.25s
    406:	learn: 0.4143439	total: 859ms	remaining: 1.25s
    407:	learn: 0.4139090	total: 860ms	remaining: 1.25s
    408:	learn: 0.4135823	total: 861ms	remaining: 1.24s
    409:	learn: 0.4131873	total: 863ms	remaining: 1.24s
    410:	learn: 0.4128178	total: 864ms	remaining: 1.24s
    411:	learn: 0.4123896	total: 866ms	remaining: 1.24s
    412:	learn: 0.4119503	total: 867ms	remaining: 1.23s
    413:	learn: 0.4116167	total: 869ms	remaining: 1.23s
    414:	learn: 0.4113682	total: 871ms	remaining: 1.23s
    415:	learn: 0.4110871	total: 872ms	remaining: 1.22s
    416:	learn: 0.4108131	total: 873ms	remaining: 1.22s
    417:	learn: 0.4103981	total: 874ms	remaining: 1.22s
    418:	learn: 0.4099065	total: 876ms	remaining: 1.21s
    419:	learn: 0.4095592	total: 877ms	remaining: 1.21s
    420:	learn: 0.4093085	total: 879ms	remaining: 1.21s
    421:	learn: 0.4087403	total: 880ms	remaining: 1.21s
    422:	learn: 0.4085358	total: 882ms	remaining: 1.2s
    423:	learn: 0.4082140	total: 883ms	remaining: 1.2s
    424:	learn: 0.4076625	total: 884ms	remaining: 1.2s
    425:	learn: 0.4073838	total: 885ms	remaining: 1.19s
    426:	learn: 0.4069956	total: 886ms	remaining: 1.19s
    427:	learn: 0.4064779	total: 887ms	remaining: 1.19s
    428:	learn: 0.4062312	total: 889ms	remaining: 1.18s
    429:	learn: 0.4059226	total: 890ms	remaining: 1.18s
    430:	learn: 0.4056839	total: 891ms	remaining: 1.18s
    431:	learn: 0.4052678	total: 893ms	remaining: 1.17s
    432:	learn: 0.4048708	total: 894ms	remaining: 1.17s
    433:	learn: 0.4043585	total: 895ms	remaining: 1.17s
    434:	learn: 0.4040660	total: 897ms	remaining: 1.16s
    435:	learn: 0.4038167	total: 898ms	remaining: 1.16s
    436:	learn: 0.4033260	total: 899ms	remaining: 1.16s
    437:	learn: 0.4028786	total: 900ms	remaining: 1.15s
    438:	learn: 0.4025291	total: 901ms	remaining: 1.15s
    439:	learn: 0.4022020	total: 902ms	remaining: 1.15s
    440:	learn: 0.4016643	total: 908ms	remaining: 1.15s
    441:	learn: 0.4013553	total: 909ms	remaining: 1.15s
    442:	learn: 0.4008144	total: 911ms	remaining: 1.14s
    443:	learn: 0.4005761	total: 912ms	remaining: 1.14s
    444:	learn: 0.4002045	total: 913ms	remaining: 1.14s
    445:	learn: 0.3997710	total: 914ms	remaining: 1.14s
    446:	learn: 0.3993118	total: 916ms	remaining: 1.13s
    447:	learn: 0.3990391	total: 917ms	remaining: 1.13s
    448:	learn: 0.3986102	total: 918ms	remaining: 1.13s
    449:	learn: 0.3982317	total: 920ms	remaining: 1.12s
    450:	learn: 0.3977189	total: 921ms	remaining: 1.12s
    451:	learn: 0.3973824	total: 922ms	remaining: 1.12s
    452:	learn: 0.3969249	total: 924ms	remaining: 1.11s
    453:	learn: 0.3965304	total: 925ms	remaining: 1.11s
    454:	learn: 0.3961990	total: 926ms	remaining: 1.11s
    455:	learn: 0.3958629	total: 927ms	remaining: 1.11s
    456:	learn: 0.3954519	total: 928ms	remaining: 1.1s
    457:	learn: 0.3952265	total: 930ms	remaining: 1.1s
    458:	learn: 0.3948552	total: 931ms	remaining: 1.1s
    459:	learn: 0.3945055	total: 933ms	remaining: 1.09s
    460:	learn: 0.3942388	total: 934ms	remaining: 1.09s
    461:	learn: 0.3938845	total: 935ms	remaining: 1.09s
    462:	learn: 0.3934466	total: 936ms	remaining: 1.09s
    463:	learn: 0.3931510	total: 938ms	remaining: 1.08s
    464:	learn: 0.3928267	total: 939ms	remaining: 1.08s
    465:	learn: 0.3926062	total: 940ms	remaining: 1.08s
    466:	learn: 0.3921506	total: 941ms	remaining: 1.07s
    467:	learn: 0.3917620	total: 943ms	remaining: 1.07s
    468:	learn: 0.3914566	total: 944ms	remaining: 1.07s
    469:	learn: 0.3912414	total: 945ms	remaining: 1.07s
    470:	learn: 0.3908352	total: 947ms	remaining: 1.06s
    471:	learn: 0.3905011	total: 948ms	remaining: 1.06s
    472:	learn: 0.3901716	total: 949ms	remaining: 1.06s
    473:	learn: 0.3899627	total: 950ms	remaining: 1.05s
    474:	learn: 0.3894250	total: 951ms	remaining: 1.05s
    475:	learn: 0.3890944	total: 952ms	remaining: 1.05s
    476:	learn: 0.3886896	total: 954ms	remaining: 1.05s
    477:	learn: 0.3883534	total: 956ms	remaining: 1.04s
    478:	learn: 0.3880367	total: 957ms	remaining: 1.04s
    479:	learn: 0.3878126	total: 958ms	remaining: 1.04s
    480:	learn: 0.3874376	total: 959ms	remaining: 1.03s
    481:	learn: 0.3871304	total: 961ms	remaining: 1.03s
    482:	learn: 0.3868231	total: 968ms	remaining: 1.03s
    483:	learn: 0.3864982	total: 971ms	remaining: 1.03s
    484:	learn: 0.3861254	total: 978ms	remaining: 1.04s
    485:	learn: 0.3858998	total: 981ms	remaining: 1.04s
    486:	learn: 0.3854810	total: 986ms	remaining: 1.04s
    487:	learn: 0.3851311	total: 990ms	remaining: 1.04s
    488:	learn: 0.3848123	total: 995ms	remaining: 1.04s
    489:	learn: 0.3844635	total: 1s	remaining: 1.04s
    490:	learn: 0.3840809	total: 1.01s	remaining: 1.04s
    491:	learn: 0.3836944	total: 1.01s	remaining: 1.04s
    492:	learn: 0.3833551	total: 1.01s	remaining: 1.04s
    493:	learn: 0.3830792	total: 1.01s	remaining: 1.04s
    494:	learn: 0.3825047	total: 1.01s	remaining: 1.03s
    495:	learn: 0.3821551	total: 1.01s	remaining: 1.03s
    496:	learn: 0.3819040	total: 1.01s	remaining: 1.03s
    497:	learn: 0.3815504	total: 1.02s	remaining: 1.02s
    498:	learn: 0.3812619	total: 1.02s	remaining: 1.02s
    499:	learn: 0.3809595	total: 1.02s	remaining: 1.02s
    500:	learn: 0.3805517	total: 1.02s	remaining: 1.02s
    501:	learn: 0.3802126	total: 1.02s	remaining: 1.01s
    502:	learn: 0.3800408	total: 1.02s	remaining: 1.01s
    503:	learn: 0.3795083	total: 1.02s	remaining: 1.01s
    504:	learn: 0.3791381	total: 1.03s	remaining: 1.01s
    505:	learn: 0.3786674	total: 1.03s	remaining: 1s
    506:	learn: 0.3782959	total: 1.03s	remaining: 1s
    507:	learn: 0.3779692	total: 1.03s	remaining: 999ms
    508:	learn: 0.3776246	total: 1.03s	remaining: 996ms
    509:	learn: 0.3772367	total: 1.03s	remaining: 994ms
    510:	learn: 0.3768605	total: 1.03s	remaining: 991ms
    511:	learn: 0.3763271	total: 1.04s	remaining: 988ms
    512:	learn: 0.3761111	total: 1.04s	remaining: 986ms
    513:	learn: 0.3758758	total: 1.04s	remaining: 983ms
    514:	learn: 0.3754354	total: 1.04s	remaining: 980ms
    515:	learn: 0.3749483	total: 1.04s	remaining: 977ms
    516:	learn: 0.3745156	total: 1.04s	remaining: 975ms
    517:	learn: 0.3741987	total: 1.04s	remaining: 973ms
    518:	learn: 0.3739278	total: 1.05s	remaining: 970ms
    519:	learn: 0.3736857	total: 1.05s	remaining: 968ms
    520:	learn: 0.3732556	total: 1.05s	remaining: 970ms
    521:	learn: 0.3730407	total: 1.06s	remaining: 969ms
    522:	learn: 0.3728372	total: 1.06s	remaining: 966ms
    523:	learn: 0.3725236	total: 1.06s	remaining: 963ms
    524:	learn: 0.3721006	total: 1.06s	remaining: 960ms
    525:	learn: 0.3717057	total: 1.06s	remaining: 958ms
    526:	learn: 0.3713627	total: 1.06s	remaining: 955ms
    527:	learn: 0.3710899	total: 1.06s	remaining: 952ms
    528:	learn: 0.3707694	total: 1.07s	remaining: 950ms
    529:	learn: 0.3705163	total: 1.07s	remaining: 948ms
    530:	learn: 0.3702163	total: 1.07s	remaining: 945ms
    531:	learn: 0.3699680	total: 1.07s	remaining: 942ms
    532:	learn: 0.3695804	total: 1.07s	remaining: 939ms
    533:	learn: 0.3692362	total: 1.07s	remaining: 937ms
    534:	learn: 0.3690209	total: 1.07s	remaining: 934ms
    535:	learn: 0.3685981	total: 1.07s	remaining: 931ms
    536:	learn: 0.3682060	total: 1.08s	remaining: 928ms
    537:	learn: 0.3679561	total: 1.08s	remaining: 926ms
    538:	learn: 0.3676233	total: 1.08s	remaining: 924ms
    539:	learn: 0.3673112	total: 1.08s	remaining: 921ms
    540:	learn: 0.3669195	total: 1.08s	remaining: 919ms
    541:	learn: 0.3665953	total: 1.08s	remaining: 916ms
    542:	learn: 0.3662761	total: 1.08s	remaining: 914ms
    543:	learn: 0.3658649	total: 1.09s	remaining: 911ms
    544:	learn: 0.3653294	total: 1.09s	remaining: 908ms
    545:	learn: 0.3650385	total: 1.09s	remaining: 906ms
    546:	learn: 0.3646401	total: 1.09s	remaining: 903ms
    547:	learn: 0.3645019	total: 1.09s	remaining: 900ms
    548:	learn: 0.3641810	total: 1.09s	remaining: 898ms
    549:	learn: 0.3639342	total: 1.09s	remaining: 896ms
    550:	learn: 0.3637084	total: 1.1s	remaining: 894ms
    551:	learn: 0.3634693	total: 1.1s	remaining: 891ms
    552:	learn: 0.3630533	total: 1.1s	remaining: 888ms
    553:	learn: 0.3626798	total: 1.1s	remaining: 886ms
    554:	learn: 0.3625060	total: 1.1s	remaining: 883ms
    555:	learn: 0.3620805	total: 1.1s	remaining: 880ms
    556:	learn: 0.3618519	total: 1.1s	remaining: 877ms
    557:	learn: 0.3615626	total: 1.11s	remaining: 878ms
    558:	learn: 0.3613213	total: 1.12s	remaining: 882ms
    559:	learn: 0.3610301	total: 1.12s	remaining: 883ms
    560:	learn: 0.3608510	total: 1.13s	remaining: 880ms
    561:	learn: 0.3606721	total: 1.13s	remaining: 881ms
    562:	learn: 0.3604439	total: 1.13s	remaining: 878ms
    563:	learn: 0.3600980	total: 1.13s	remaining: 876ms
    564:	learn: 0.3596534	total: 1.14s	remaining: 874ms
    565:	learn: 0.3594761	total: 1.14s	remaining: 871ms
    566:	learn: 0.3593242	total: 1.14s	remaining: 869ms
    567:	learn: 0.3591736	total: 1.14s	remaining: 866ms
    568:	learn: 0.3588585	total: 1.14s	remaining: 864ms
    569:	learn: 0.3586155	total: 1.14s	remaining: 862ms
    570:	learn: 0.3583969	total: 1.14s	remaining: 859ms
    571:	learn: 0.3580985	total: 1.14s	remaining: 856ms
    572:	learn: 0.3578687	total: 1.15s	remaining: 854ms
    573:	learn: 0.3575707	total: 1.15s	remaining: 851ms
    574:	learn: 0.3571789	total: 1.15s	remaining: 849ms
    575:	learn: 0.3568428	total: 1.15s	remaining: 846ms
    576:	learn: 0.3565674	total: 1.15s	remaining: 843ms
    577:	learn: 0.3562385	total: 1.15s	remaining: 841ms
    578:	learn: 0.3560349	total: 1.15s	remaining: 838ms
    579:	learn: 0.3556953	total: 1.15s	remaining: 836ms
    580:	learn: 0.3553258	total: 1.16s	remaining: 834ms
    581:	learn: 0.3548626	total: 1.16s	remaining: 831ms
    582:	learn: 0.3545285	total: 1.16s	remaining: 830ms
    583:	learn: 0.3542536	total: 1.16s	remaining: 827ms
    584:	learn: 0.3539073	total: 1.17s	remaining: 827ms
    585:	learn: 0.3534859	total: 1.17s	remaining: 825ms
    586:	learn: 0.3532509	total: 1.17s	remaining: 822ms
    587:	learn: 0.3530436	total: 1.17s	remaining: 819ms
    588:	learn: 0.3527570	total: 1.17s	remaining: 817ms
    589:	learn: 0.3525101	total: 1.17s	remaining: 814ms
    590:	learn: 0.3522113	total: 1.17s	remaining: 812ms
    591:	learn: 0.3520489	total: 1.17s	remaining: 809ms
    592:	learn: 0.3516889	total: 1.18s	remaining: 807ms
    593:	learn: 0.3514473	total: 1.18s	remaining: 805ms
    594:	learn: 0.3511787	total: 1.18s	remaining: 802ms
    595:	learn: 0.3509142	total: 1.18s	remaining: 800ms
    596:	learn: 0.3507387	total: 1.18s	remaining: 797ms
    597:	learn: 0.3504683	total: 1.18s	remaining: 795ms
    598:	learn: 0.3501645	total: 1.18s	remaining: 792ms
    599:	learn: 0.3498414	total: 1.18s	remaining: 790ms
    600:	learn: 0.3493886	total: 1.19s	remaining: 787ms
    601:	learn: 0.3490806	total: 1.19s	remaining: 785ms
    602:	learn: 0.3488135	total: 1.19s	remaining: 782ms
    603:	learn: 0.3485939	total: 1.19s	remaining: 780ms
    604:	learn: 0.3482250	total: 1.19s	remaining: 778ms
    605:	learn: 0.3479442	total: 1.19s	remaining: 776ms
    606:	learn: 0.3477725	total: 1.19s	remaining: 773ms
    607:	learn: 0.3474272	total: 1.2s	remaining: 771ms
    608:	learn: 0.3470944	total: 1.2s	remaining: 768ms
    609:	learn: 0.3466551	total: 1.2s	remaining: 766ms
    610:	learn: 0.3463616	total: 1.2s	remaining: 763ms
    611:	learn: 0.3461570	total: 1.2s	remaining: 761ms
    612:	learn: 0.3458491	total: 1.2s	remaining: 758ms
    613:	learn: 0.3455455	total: 1.2s	remaining: 756ms
    614:	learn: 0.3453008	total: 1.2s	remaining: 754ms
    615:	learn: 0.3451190	total: 1.21s	remaining: 752ms
    616:	learn: 0.3449249	total: 1.21s	remaining: 749ms
    617:	learn: 0.3446516	total: 1.21s	remaining: 747ms
    618:	learn: 0.3444380	total: 1.21s	remaining: 744ms
    619:	learn: 0.3442321	total: 1.21s	remaining: 742ms
    620:	learn: 0.3440420	total: 1.21s	remaining: 739ms
    621:	learn: 0.3437195	total: 1.21s	remaining: 737ms
    622:	learn: 0.3432327	total: 1.21s	remaining: 735ms
    623:	learn: 0.3430127	total: 1.22s	remaining: 733ms
    624:	learn: 0.3427330	total: 1.22s	remaining: 730ms
    625:	learn: 0.3422502	total: 1.22s	remaining: 728ms
    626:	learn: 0.3420584	total: 1.22s	remaining: 726ms
    627:	learn: 0.3417045	total: 1.22s	remaining: 723ms
    628:	learn: 0.3416501	total: 1.22s	remaining: 720ms
    629:	learn: 0.3413845	total: 1.22s	remaining: 718ms
    630:	learn: 0.3410835	total: 1.22s	remaining: 716ms
    631:	learn: 0.3408639	total: 1.22s	remaining: 713ms
    632:	learn: 0.3403330	total: 1.23s	remaining: 711ms
    633:	learn: 0.3400616	total: 1.23s	remaining: 709ms
    634:	learn: 0.3396417	total: 1.23s	remaining: 707ms
    635:	learn: 0.3393275	total: 1.23s	remaining: 704ms
    636:	learn: 0.3389483	total: 1.23s	remaining: 702ms
    637:	learn: 0.3387498	total: 1.23s	remaining: 700ms
    638:	learn: 0.3384303	total: 1.23s	remaining: 697ms
    639:	learn: 0.3381935	total: 1.24s	remaining: 695ms
    640:	learn: 0.3380160	total: 1.24s	remaining: 693ms
    641:	learn: 0.3378595	total: 1.24s	remaining: 690ms
    642:	learn: 0.3377456	total: 1.24s	remaining: 688ms
    643:	learn: 0.3374696	total: 1.25s	remaining: 691ms
    644:	learn: 0.3370792	total: 1.26s	remaining: 692ms
    645:	learn: 0.3368629	total: 1.26s	remaining: 692ms
    646:	learn: 0.3365676	total: 1.26s	remaining: 690ms
    647:	learn: 0.3363391	total: 1.27s	remaining: 688ms
    648:	learn: 0.3361647	total: 1.27s	remaining: 686ms
    649:	learn: 0.3358873	total: 1.28s	remaining: 687ms
    650:	learn: 0.3355135	total: 1.28s	remaining: 685ms
    651:	learn: 0.3351650	total: 1.28s	remaining: 683ms
    652:	learn: 0.3348108	total: 1.28s	remaining: 680ms
    653:	learn: 0.3345348	total: 1.28s	remaining: 678ms
    654:	learn: 0.3343680	total: 1.28s	remaining: 676ms
    655:	learn: 0.3342893	total: 1.28s	remaining: 673ms
    656:	learn: 0.3341190	total: 1.28s	remaining: 671ms
    657:	learn: 0.3339014	total: 1.28s	remaining: 668ms
    658:	learn: 0.3336664	total: 1.29s	remaining: 666ms
    659:	learn: 0.3335302	total: 1.29s	remaining: 663ms
    660:	learn: 0.3331241	total: 1.29s	remaining: 661ms
    661:	learn: 0.3328024	total: 1.29s	remaining: 659ms
    662:	learn: 0.3325316	total: 1.29s	remaining: 657ms
    663:	learn: 0.3324100	total: 1.29s	remaining: 654ms
    664:	learn: 0.3321541	total: 1.29s	remaining: 652ms
    665:	learn: 0.3319974	total: 1.29s	remaining: 650ms
    666:	learn: 0.3317975	total: 1.3s	remaining: 647ms
    667:	learn: 0.3315524	total: 1.3s	remaining: 645ms
    668:	learn: 0.3313412	total: 1.3s	remaining: 643ms
    669:	learn: 0.3311172	total: 1.3s	remaining: 641ms
    670:	learn: 0.3309319	total: 1.3s	remaining: 639ms
    671:	learn: 0.3307170	total: 1.3s	remaining: 637ms
    672:	learn: 0.3304421	total: 1.3s	remaining: 634ms
    673:	learn: 0.3301300	total: 1.31s	remaining: 632ms
    674:	learn: 0.3298557	total: 1.31s	remaining: 630ms
    675:	learn: 0.3295640	total: 1.31s	remaining: 628ms
    676:	learn: 0.3293214	total: 1.31s	remaining: 626ms
    677:	learn: 0.3289244	total: 1.31s	remaining: 624ms
    678:	learn: 0.3287224	total: 1.31s	remaining: 622ms
    679:	learn: 0.3284095	total: 1.32s	remaining: 619ms
    680:	learn: 0.3282182	total: 1.32s	remaining: 617ms
    681:	learn: 0.3279353	total: 1.32s	remaining: 615ms
    682:	learn: 0.3277441	total: 1.32s	remaining: 612ms
    683:	learn: 0.3274582	total: 1.32s	remaining: 610ms
    684:	learn: 0.3271778	total: 1.32s	remaining: 609ms
    685:	learn: 0.3267508	total: 1.33s	remaining: 607ms
    686:	learn: 0.3264459	total: 1.33s	remaining: 607ms
    687:	learn: 0.3260522	total: 1.33s	remaining: 604ms
    688:	learn: 0.3257209	total: 1.33s	remaining: 602ms
    689:	learn: 0.3253320	total: 1.33s	remaining: 600ms
    690:	learn: 0.3249599	total: 1.34s	remaining: 598ms
    691:	learn: 0.3246715	total: 1.34s	remaining: 595ms
    692:	learn: 0.3246043	total: 1.34s	remaining: 593ms
    693:	learn: 0.3243228	total: 1.34s	remaining: 591ms
    694:	learn: 0.3240322	total: 1.34s	remaining: 588ms
    695:	learn: 0.3238588	total: 1.34s	remaining: 586ms
    696:	learn: 0.3237242	total: 1.34s	remaining: 584ms
    697:	learn: 0.3235334	total: 1.34s	remaining: 582ms
    698:	learn: 0.3234583	total: 1.35s	remaining: 580ms
    699:	learn: 0.3233351	total: 1.35s	remaining: 578ms
    700:	learn: 0.3230621	total: 1.35s	remaining: 575ms
    701:	learn: 0.3228487	total: 1.35s	remaining: 573ms
    702:	learn: 0.3225795	total: 1.35s	remaining: 571ms
    703:	learn: 0.3222725	total: 1.35s	remaining: 569ms
    704:	learn: 0.3220679	total: 1.35s	remaining: 566ms
    705:	learn: 0.3218231	total: 1.35s	remaining: 565ms
    706:	learn: 0.3215532	total: 1.36s	remaining: 562ms
    707:	learn: 0.3213057	total: 1.36s	remaining: 560ms
    708:	learn: 0.3211120	total: 1.36s	remaining: 558ms
    709:	learn: 0.3209608	total: 1.36s	remaining: 556ms
    710:	learn: 0.3206958	total: 1.36s	remaining: 554ms
    711:	learn: 0.3205042	total: 1.36s	remaining: 552ms
    712:	learn: 0.3203639	total: 1.36s	remaining: 549ms
    713:	learn: 0.3202346	total: 1.36s	remaining: 547ms
    714:	learn: 0.3200862	total: 1.37s	remaining: 545ms
    715:	learn: 0.3199268	total: 1.37s	remaining: 543ms
    716:	learn: 0.3196081	total: 1.37s	remaining: 541ms
    717:	learn: 0.3194574	total: 1.37s	remaining: 539ms
    718:	learn: 0.3191280	total: 1.37s	remaining: 537ms
    719:	learn: 0.3188974	total: 1.37s	remaining: 534ms
    720:	learn: 0.3186039	total: 1.38s	remaining: 532ms
    721:	learn: 0.3185274	total: 1.38s	remaining: 530ms
    722:	learn: 0.3184672	total: 1.38s	remaining: 528ms
    723:	learn: 0.3182843	total: 1.38s	remaining: 525ms
    724:	learn: 0.3181445	total: 1.38s	remaining: 523ms
    725:	learn: 0.3180535	total: 1.38s	remaining: 521ms
    726:	learn: 0.3176699	total: 1.38s	remaining: 519ms
    727:	learn: 0.3174515	total: 1.38s	remaining: 517ms
    728:	learn: 0.3172517	total: 1.38s	remaining: 515ms
    729:	learn: 0.3169398	total: 1.39s	remaining: 514ms
    730:	learn: 0.3166433	total: 1.39s	remaining: 512ms
    731:	learn: 0.3162918	total: 1.39s	remaining: 510ms
    732:	learn: 0.3160154	total: 1.39s	remaining: 508ms
    733:	learn: 0.3158345	total: 1.4s	remaining: 506ms
    734:	learn: 0.3156884	total: 1.4s	remaining: 504ms
    735:	learn: 0.3154700	total: 1.4s	remaining: 502ms
    736:	learn: 0.3151725	total: 1.4s	remaining: 499ms
    737:	learn: 0.3149833	total: 1.4s	remaining: 497ms
    738:	learn: 0.3148945	total: 1.4s	remaining: 495ms
    739:	learn: 0.3146413	total: 1.4s	remaining: 493ms
    740:	learn: 0.3145453	total: 1.4s	remaining: 491ms
    741:	learn: 0.3142532	total: 1.41s	remaining: 489ms
    742:	learn: 0.3140808	total: 1.41s	remaining: 487ms
    743:	learn: 0.3138641	total: 1.41s	remaining: 485ms
    744:	learn: 0.3137304	total: 1.41s	remaining: 483ms
    745:	learn: 0.3135155	total: 1.41s	remaining: 481ms
    746:	learn: 0.3133424	total: 1.41s	remaining: 478ms
    747:	learn: 0.3130712	total: 1.41s	remaining: 476ms
    748:	learn: 0.3127004	total: 1.41s	remaining: 474ms
    749:	learn: 0.3124777	total: 1.42s	remaining: 472ms
    750:	learn: 0.3121394	total: 1.42s	remaining: 470ms
    751:	learn: 0.3119436	total: 1.42s	remaining: 468ms
    752:	learn: 0.3116986	total: 1.42s	remaining: 466ms
    753:	learn: 0.3115656	total: 1.42s	remaining: 464ms
    754:	learn: 0.3114501	total: 1.42s	remaining: 462ms
    755:	learn: 0.3111048	total: 1.42s	remaining: 460ms
    756:	learn: 0.3109193	total: 1.43s	remaining: 458ms
    757:	learn: 0.3107448	total: 1.43s	remaining: 455ms
    758:	learn: 0.3105257	total: 1.43s	remaining: 453ms
    759:	learn: 0.3104061	total: 1.43s	remaining: 452ms
    760:	learn: 0.3101747	total: 1.43s	remaining: 450ms
    761:	learn: 0.3099412	total: 1.43s	remaining: 447ms
    762:	learn: 0.3097221	total: 1.43s	remaining: 445ms
    763:	learn: 0.3094487	total: 1.44s	remaining: 443ms
    764:	learn: 0.3091716	total: 1.44s	remaining: 441ms
    765:	learn: 0.3090421	total: 1.44s	remaining: 439ms
    766:	learn: 0.3088640	total: 1.44s	remaining: 437ms
    767:	learn: 0.3085965	total: 1.44s	remaining: 435ms
    768:	learn: 0.3083616	total: 1.44s	remaining: 433ms
    769:	learn: 0.3080985	total: 1.44s	remaining: 431ms
    770:	learn: 0.3079715	total: 1.44s	remaining: 429ms
    771:	learn: 0.3076964	total: 1.44s	remaining: 427ms
    772:	learn: 0.3075994	total: 1.45s	remaining: 425ms
    773:	learn: 0.3074512	total: 1.45s	remaining: 423ms
    774:	learn: 0.3072880	total: 1.45s	remaining: 422ms
    775:	learn: 0.3070059	total: 1.45s	remaining: 420ms
    776:	learn: 0.3065409	total: 1.46s	remaining: 418ms
    777:	learn: 0.3062357	total: 1.46s	remaining: 416ms
    778:	learn: 0.3060494	total: 1.46s	remaining: 414ms
    779:	learn: 0.3057171	total: 1.46s	remaining: 412ms
    780:	learn: 0.3054242	total: 1.46s	remaining: 410ms
    781:	learn: 0.3052925	total: 1.46s	remaining: 408ms
    782:	learn: 0.3051321	total: 1.46s	remaining: 406ms
    783:	learn: 0.3050003	total: 1.47s	remaining: 404ms
    784:	learn: 0.3048260	total: 1.47s	remaining: 402ms
    785:	learn: 0.3046389	total: 1.47s	remaining: 400ms
    786:	learn: 0.3044302	total: 1.47s	remaining: 398ms
    787:	learn: 0.3042453	total: 1.47s	remaining: 395ms
    788:	learn: 0.3040766	total: 1.47s	remaining: 393ms
    789:	learn: 0.3039220	total: 1.47s	remaining: 391ms
    790:	learn: 0.3037328	total: 1.47s	remaining: 389ms
    791:	learn: 0.3035034	total: 1.48s	remaining: 388ms
    792:	learn: 0.3034325	total: 1.48s	remaining: 386ms
    793:	learn: 0.3032128	total: 1.48s	remaining: 384ms
    794:	learn: 0.3030553	total: 1.48s	remaining: 382ms
    795:	learn: 0.3029265	total: 1.48s	remaining: 380ms
    796:	learn: 0.3025759	total: 1.48s	remaining: 378ms
    797:	learn: 0.3025014	total: 1.48s	remaining: 376ms
    798:	learn: 0.3021780	total: 1.49s	remaining: 374ms
    799:	learn: 0.3018760	total: 1.49s	remaining: 372ms
    800:	learn: 0.3017365	total: 1.49s	remaining: 370ms
    801:	learn: 0.3015434	total: 1.49s	remaining: 368ms
    802:	learn: 0.3013035	total: 1.49s	remaining: 366ms
    803:	learn: 0.3011298	total: 1.49s	remaining: 364ms
    804:	learn: 0.3009583	total: 1.49s	remaining: 362ms
    805:	learn: 0.3007937	total: 1.49s	remaining: 360ms
    806:	learn: 0.3005020	total: 1.5s	remaining: 358ms
    807:	learn: 0.3002857	total: 1.5s	remaining: 356ms
    808:	learn: 0.2999542	total: 1.5s	remaining: 354ms
    809:	learn: 0.2998034	total: 1.5s	remaining: 352ms
    810:	learn: 0.2994679	total: 1.5s	remaining: 350ms
    811:	learn: 0.2992922	total: 1.5s	remaining: 349ms
    812:	learn: 0.2989525	total: 1.51s	remaining: 347ms
    813:	learn: 0.2988843	total: 1.51s	remaining: 345ms
    814:	learn: 0.2986333	total: 1.51s	remaining: 343ms
    815:	learn: 0.2984650	total: 1.51s	remaining: 341ms
    816:	learn: 0.2983660	total: 1.51s	remaining: 339ms
    817:	learn: 0.2981047	total: 1.51s	remaining: 337ms
    818:	learn: 0.2978962	total: 1.51s	remaining: 335ms
    819:	learn: 0.2976227	total: 1.51s	remaining: 333ms
    820:	learn: 0.2973651	total: 1.53s	remaining: 334ms
    821:	learn: 0.2970952	total: 1.53s	remaining: 332ms
    822:	learn: 0.2969549	total: 1.53s	remaining: 330ms
    823:	learn: 0.2968257	total: 1.53s	remaining: 328ms
    824:	learn: 0.2965174	total: 1.54s	remaining: 326ms
    825:	learn: 0.2962890	total: 1.54s	remaining: 324ms
    826:	learn: 0.2959075	total: 1.54s	remaining: 322ms
    827:	learn: 0.2956116	total: 1.54s	remaining: 320ms
    828:	learn: 0.2952377	total: 1.54s	remaining: 318ms
    829:	learn: 0.2949474	total: 1.54s	remaining: 316ms
    830:	learn: 0.2948073	total: 1.54s	remaining: 314ms
    831:	learn: 0.2945125	total: 1.55s	remaining: 312ms
    832:	learn: 0.2943166	total: 1.55s	remaining: 310ms
    833:	learn: 0.2940166	total: 1.55s	remaining: 308ms
    834:	learn: 0.2938380	total: 1.55s	remaining: 306ms
    835:	learn: 0.2936978	total: 1.55s	remaining: 305ms
    836:	learn: 0.2935170	total: 1.55s	remaining: 303ms
    837:	learn: 0.2932565	total: 1.55s	remaining: 301ms
    838:	learn: 0.2929052	total: 1.56s	remaining: 299ms
    839:	learn: 0.2927079	total: 1.56s	remaining: 297ms
    840:	learn: 0.2924398	total: 1.56s	remaining: 295ms
    841:	learn: 0.2923313	total: 1.56s	remaining: 293ms
    842:	learn: 0.2921684	total: 1.56s	remaining: 291ms
    843:	learn: 0.2919597	total: 1.56s	remaining: 289ms
    844:	learn: 0.2918318	total: 1.56s	remaining: 287ms
    845:	learn: 0.2916668	total: 1.56s	remaining: 285ms
    846:	learn: 0.2914693	total: 1.57s	remaining: 283ms
    847:	learn: 0.2912804	total: 1.57s	remaining: 281ms
    848:	learn: 0.2911508	total: 1.57s	remaining: 279ms
    849:	learn: 0.2910642	total: 1.57s	remaining: 277ms
    850:	learn: 0.2908617	total: 1.57s	remaining: 275ms
    851:	learn: 0.2906814	total: 1.57s	remaining: 273ms
    852:	learn: 0.2904690	total: 1.57s	remaining: 271ms
    853:	learn: 0.2902362	total: 1.57s	remaining: 269ms
    854:	learn: 0.2900135	total: 1.58s	remaining: 267ms
    855:	learn: 0.2897123	total: 1.58s	remaining: 265ms
    856:	learn: 0.2894698	total: 1.58s	remaining: 264ms
    857:	learn: 0.2893142	total: 1.58s	remaining: 262ms
    858:	learn: 0.2890136	total: 1.58s	remaining: 260ms
    859:	learn: 0.2888601	total: 1.58s	remaining: 258ms
    860:	learn: 0.2886630	total: 1.58s	remaining: 256ms
    861:	learn: 0.2885175	total: 1.58s	remaining: 254ms
    862:	learn: 0.2883170	total: 1.59s	remaining: 252ms
    863:	learn: 0.2880664	total: 1.59s	remaining: 250ms
    864:	learn: 0.2878671	total: 1.59s	remaining: 248ms
    865:	learn: 0.2875972	total: 1.59s	remaining: 246ms
    866:	learn: 0.2874414	total: 1.59s	remaining: 244ms
    867:	learn: 0.2872545	total: 1.59s	remaining: 242ms
    868:	learn: 0.2870708	total: 1.59s	remaining: 240ms
    869:	learn: 0.2868316	total: 1.6s	remaining: 239ms
    870:	learn: 0.2866339	total: 1.6s	remaining: 237ms
    871:	learn: 0.2865923	total: 1.6s	remaining: 235ms
    872:	learn: 0.2864664	total: 1.6s	remaining: 233ms
    873:	learn: 0.2863478	total: 1.6s	remaining: 231ms
    874:	learn: 0.2861626	total: 1.6s	remaining: 229ms
    875:	learn: 0.2859715	total: 1.6s	remaining: 227ms
    876:	learn: 0.2858514	total: 1.6s	remaining: 225ms
    877:	learn: 0.2856724	total: 1.61s	remaining: 223ms
    878:	learn: 0.2855089	total: 1.61s	remaining: 221ms
    879:	learn: 0.2854204	total: 1.61s	remaining: 219ms
    880:	learn: 0.2851937	total: 1.61s	remaining: 218ms
    881:	learn: 0.2850213	total: 1.61s	remaining: 216ms
    882:	learn: 0.2848920	total: 1.61s	remaining: 214ms
    883:	learn: 0.2847252	total: 1.61s	remaining: 212ms
    884:	learn: 0.2845051	total: 1.61s	remaining: 210ms
    885:	learn: 0.2843960	total: 1.62s	remaining: 208ms
    886:	learn: 0.2841953	total: 1.62s	remaining: 206ms
    887:	learn: 0.2839355	total: 1.62s	remaining: 204ms
    888:	learn: 0.2836438	total: 1.62s	remaining: 202ms
    889:	learn: 0.2834348	total: 1.62s	remaining: 200ms
    890:	learn: 0.2832590	total: 1.62s	remaining: 199ms
    891:	learn: 0.2829858	total: 1.62s	remaining: 197ms
    892:	learn: 0.2828478	total: 1.63s	remaining: 195ms
    893:	learn: 0.2825301	total: 1.63s	remaining: 193ms
    894:	learn: 0.2824010	total: 1.63s	remaining: 191ms
    895:	learn: 0.2822710	total: 1.63s	remaining: 189ms
    896:	learn: 0.2819156	total: 1.63s	remaining: 187ms
    897:	learn: 0.2817344	total: 1.63s	remaining: 185ms
    898:	learn: 0.2815895	total: 1.63s	remaining: 184ms
    899:	learn: 0.2814930	total: 1.63s	remaining: 182ms
    900:	learn: 0.2812095	total: 1.64s	remaining: 180ms
    901:	learn: 0.2809294	total: 1.64s	remaining: 178ms
    902:	learn: 0.2807974	total: 1.64s	remaining: 176ms
    903:	learn: 0.2806573	total: 1.64s	remaining: 174ms
    904:	learn: 0.2803645	total: 1.64s	remaining: 172ms
    905:	learn: 0.2802090	total: 1.64s	remaining: 170ms
    906:	learn: 0.2800461	total: 1.64s	remaining: 169ms
    907:	learn: 0.2798569	total: 1.65s	remaining: 167ms
    908:	learn: 0.2797643	total: 1.65s	remaining: 165ms
    909:	learn: 0.2795885	total: 1.65s	remaining: 163ms
    910:	learn: 0.2793944	total: 1.65s	remaining: 161ms
    911:	learn: 0.2791919	total: 1.65s	remaining: 159ms
    912:	learn: 0.2790906	total: 1.65s	remaining: 157ms
    913:	learn: 0.2789060	total: 1.65s	remaining: 156ms
    914:	learn: 0.2786720	total: 1.66s	remaining: 154ms
    915:	learn: 0.2784611	total: 1.66s	remaining: 152ms
    916:	learn: 0.2782113	total: 1.67s	remaining: 151ms
    917:	learn: 0.2780414	total: 1.67s	remaining: 150ms
    918:	learn: 0.2778083	total: 1.67s	remaining: 148ms
    919:	learn: 0.2775713	total: 1.68s	remaining: 146ms
    920:	learn: 0.2774738	total: 1.68s	remaining: 144ms
    921:	learn: 0.2773348	total: 1.68s	remaining: 142ms
    922:	learn: 0.2772183	total: 1.68s	remaining: 140ms
    923:	learn: 0.2770912	total: 1.68s	remaining: 138ms
    924:	learn: 0.2769969	total: 1.68s	remaining: 136ms
    925:	learn: 0.2768031	total: 1.68s	remaining: 135ms
    926:	learn: 0.2766616	total: 1.69s	remaining: 133ms
    927:	learn: 0.2765076	total: 1.69s	remaining: 131ms
    928:	learn: 0.2761540	total: 1.69s	remaining: 129ms
    929:	learn: 0.2759572	total: 1.69s	remaining: 127ms
    930:	learn: 0.2757937	total: 1.69s	remaining: 125ms
    931:	learn: 0.2755387	total: 1.69s	remaining: 123ms
    932:	learn: 0.2753424	total: 1.69s	remaining: 122ms
    933:	learn: 0.2750983	total: 1.69s	remaining: 120ms
    934:	learn: 0.2749394	total: 1.7s	remaining: 118ms
    935:	learn: 0.2748684	total: 1.7s	remaining: 116ms
    936:	learn: 0.2747025	total: 1.7s	remaining: 114ms
    937:	learn: 0.2744501	total: 1.7s	remaining: 112ms
    938:	learn: 0.2741766	total: 1.7s	remaining: 110ms
    939:	learn: 0.2740751	total: 1.7s	remaining: 109ms
    940:	learn: 0.2738601	total: 1.7s	remaining: 107ms
    941:	learn: 0.2736221	total: 1.7s	remaining: 105ms
    942:	learn: 0.2734254	total: 1.71s	remaining: 103ms
    943:	learn: 0.2731311	total: 1.71s	remaining: 101ms
    944:	learn: 0.2730030	total: 1.71s	remaining: 99.4ms
    945:	learn: 0.2728339	total: 1.71s	remaining: 97.6ms
    946:	learn: 0.2726247	total: 1.71s	remaining: 95.8ms
    947:	learn: 0.2725411	total: 1.71s	remaining: 93.9ms
    948:	learn: 0.2723206	total: 1.71s	remaining: 92.1ms
    949:	learn: 0.2721476	total: 1.71s	remaining: 90.2ms
    950:	learn: 0.2718759	total: 1.72s	remaining: 88.4ms
    951:	learn: 0.2717793	total: 1.72s	remaining: 86.6ms
    952:	learn: 0.2716197	total: 1.72s	remaining: 84.7ms
    953:	learn: 0.2715248	total: 1.72s	remaining: 82.9ms
    954:	learn: 0.2714280	total: 1.72s	remaining: 81.1ms
    955:	learn: 0.2713150	total: 1.72s	remaining: 79.3ms
    956:	learn: 0.2711096	total: 1.72s	remaining: 77.5ms
    957:	learn: 0.2709656	total: 1.73s	remaining: 75.6ms
    958:	learn: 0.2708148	total: 1.73s	remaining: 73.8ms
    959:	learn: 0.2706601	total: 1.73s	remaining: 72ms
    960:	learn: 0.2703661	total: 1.73s	remaining: 70.2ms
    961:	learn: 0.2701886	total: 1.73s	remaining: 68.3ms
    962:	learn: 0.2701206	total: 1.73s	remaining: 66.5ms
    963:	learn: 0.2699416	total: 1.73s	remaining: 64.7ms
    964:	learn: 0.2698577	total: 1.73s	remaining: 62.8ms
    965:	learn: 0.2697207	total: 1.73s	remaining: 61ms
    966:	learn: 0.2694234	total: 1.74s	remaining: 59.2ms
    967:	learn: 0.2691837	total: 1.74s	remaining: 57.4ms
    968:	learn: 0.2689275	total: 1.74s	remaining: 55.6ms
    969:	learn: 0.2687682	total: 1.74s	remaining: 53.8ms
    970:	learn: 0.2687316	total: 1.74s	remaining: 52ms
    971:	learn: 0.2685794	total: 1.74s	remaining: 50.2ms
    972:	learn: 0.2683534	total: 1.74s	remaining: 48.4ms
    973:	learn: 0.2681200	total: 1.74s	remaining: 46.6ms
    974:	learn: 0.2679638	total: 1.75s	remaining: 44.8ms
    975:	learn: 0.2678516	total: 1.75s	remaining: 42.9ms
    976:	learn: 0.2676302	total: 1.75s	remaining: 41.1ms
    977:	learn: 0.2675006	total: 1.75s	remaining: 39.4ms
    978:	learn: 0.2673920	total: 1.75s	remaining: 37.6ms
    979:	learn: 0.2673105	total: 1.75s	remaining: 35.8ms
    980:	learn: 0.2671645	total: 1.75s	remaining: 34ms
    981:	learn: 0.2670783	total: 1.75s	remaining: 32.2ms
    982:	learn: 0.2669992	total: 1.75s	remaining: 30.4ms
    983:	learn: 0.2668481	total: 1.76s	remaining: 28.6ms
    984:	learn: 0.2667583	total: 1.76s	remaining: 26.8ms
    985:	learn: 0.2665053	total: 1.76s	remaining: 25ms
    986:	learn: 0.2663741	total: 1.76s	remaining: 23.2ms
    987:	learn: 0.2662453	total: 1.76s	remaining: 21.4ms
    988:	learn: 0.2661092	total: 1.76s	remaining: 19.6ms
    989:	learn: 0.2659775	total: 1.77s	remaining: 17.8ms
    990:	learn: 0.2657491	total: 1.77s	remaining: 16.1ms
    991:	learn: 0.2655746	total: 1.77s	remaining: 14.3ms
    992:	learn: 0.2654548	total: 1.77s	remaining: 12.5ms
    993:	learn: 0.2651612	total: 1.77s	remaining: 10.7ms
    994:	learn: 0.2648038	total: 1.77s	remaining: 8.91ms
    995:	learn: 0.2646414	total: 1.77s	remaining: 7.13ms
    996:	learn: 0.2645150	total: 1.77s	remaining: 5.34ms
    997:	learn: 0.2643620	total: 1.78s	remaining: 3.56ms
    998:	learn: 0.2642650	total: 1.78s	remaining: 1.78ms
    999:	learn: 0.2640218	total: 1.78s	remaining: 0us
    


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

