{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "executionInfo": {
     "elapsed": 980,
     "status": "ok",
     "timestamp": 1641224164593,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "X7mjAAcLSsIG"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Asus\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\least_angle.py:30: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  method='lar', copy_X=True, eps=np.finfo(np.float).eps,\n",
      "C:\\Users\\Asus\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\least_angle.py:167: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  method='lar', copy_X=True, eps=np.finfo(np.float).eps,\n",
      "C:\\Users\\Asus\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\least_angle.py:284: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  eps=np.finfo(np.float).eps, copy_Gram=True, verbose=0,\n",
      "C:\\Users\\Asus\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\least_angle.py:862: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  eps=np.finfo(np.float).eps, copy_X=True, fit_path=True,\n",
      "C:\\Users\\Asus\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\least_angle.py:1101: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  eps=np.finfo(np.float).eps, copy_X=True, fit_path=True,\n",
      "C:\\Users\\Asus\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\least_angle.py:1127: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  eps=np.finfo(np.float).eps, positive=False):\n",
      "C:\\Users\\Asus\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\least_angle.py:1362: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  max_n_alphas=1000, n_jobs=None, eps=np.finfo(np.float).eps,\n",
      "C:\\Users\\Asus\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\least_angle.py:1602: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  max_n_alphas=1000, n_jobs=None, eps=np.finfo(np.float).eps,\n",
      "C:\\Users\\Asus\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\least_angle.py:1738: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  eps=np.finfo(np.float).eps, copy_X=True, positive=False):\n",
      "C:\\Users\\Asus\\Anaconda3\\lib\\site-packages\\sklearn\\decomposition\\online_lda.py:29: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  EPS = np.finfo(np.float).eps\n"
     ]
    }
   ],
   "source": [
    "# Importing the libreries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "%matplotlib inline\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn import tree\n",
    "from sklearn.metrics import mean_squared_error\n",
    "import sklearn\n",
    "import math"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 5534,
     "status": "ok",
     "timestamp": 1641224170120,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "3SGC91ve3vv1",
    "outputId": "561fc605-f75f-42ce-c90a-89583e94f600"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pickle5 in c:\\users\\asus\\anaconda3\\lib\\site-packages (0.0.12)\n"
     ]
    }
   ],
   "source": [
    "!pip install pickle5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "executionInfo": {
     "elapsed": 85,
     "status": "ok",
     "timestamp": 1641224170122,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "YjZPZW073-N2"
   },
   "outputs": [],
   "source": [
    "import pickle5 as pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "executionInfo": {
     "elapsed": 85,
     "status": "ok",
     "timestamp": 1641224170124,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "Sgx1ScVfS6iZ",
    "outputId": "c4b7be52-cb9e-49ad-e07c-7c449c899fe6"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Area</th>\n",
       "      <th>Oxygen</th>\n",
       "      <th>Temperature</th>\n",
       "      <th>Humidity</th>\n",
       "      <th>Fire Occurrence</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>Jharkand</td>\n",
       "      <td>40</td>\n",
       "      <td>45</td>\n",
       "      <td>20</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>Bangalore</td>\n",
       "      <td>50</td>\n",
       "      <td>30</td>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>Ecuador</td>\n",
       "      <td>10</td>\n",
       "      <td>20</td>\n",
       "      <td>70</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>a</td>\n",
       "      <td>60</td>\n",
       "      <td>45</td>\n",
       "      <td>70</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>Bangalore</td>\n",
       "      <td>30</td>\n",
       "      <td>48</td>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Area  Oxygen  Temperature  Humidity  Fire Occurrence\n",
       "0   Jharkand      40           45        20                1\n",
       "1  Bangalore      50           30        10                1\n",
       "2    Ecuador      10           20        70                0\n",
       "3          a      60           45        70                1\n",
       "4  Bangalore      30           48        10                1"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fireForest_data = \"https://raw.githubusercontent.com/nachi-hebbar/Forest-Fire-Prediction-Website/master/Forest_fire.csv\"\n",
    "dataset = pd.read_csv(fireForest_data)\n",
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 84,
     "status": "ok",
     "timestamp": 1641224170126,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "AG_gNhqNTNvn",
    "outputId": "df5e3422-bddd-4f7f-84da-831a396ee701"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(39, 5)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "executionInfo": {
     "elapsed": 83,
     "status": "ok",
     "timestamp": 1641224170128,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "qrvYm7x2Wlxw"
   },
   "outputs": [],
   "source": [
    "dataset = dataset.drop(['Area'],axis =1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 84,
     "status": "ok",
     "timestamp": 1641224170129,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "Pgik97YWTPH2",
    "outputId": "0f80de72-3d58-4e7c-903d-57f9d9342fb8"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Oxygen', 'Temperature', 'Humidity', 'Fire Occurrence'], dtype='object')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 300
    },
    "executionInfo": {
     "elapsed": 83,
     "status": "ok",
     "timestamp": 1641224170130,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "guTa6rRBTbBv",
    "outputId": "07d901d4-d906-4cd7-ae49-e7dc5b68e250"
   },
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "Cannot interpret '<attribute 'dtype' of 'numpy.generic' objects>' as a data type",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-8-3ded4c85834d>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mdataset\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdescribe\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36mdescribe\u001b[1;34m(self, percentiles, include, exclude)\u001b[0m\n\u001b[0;32m  10263\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0minclude\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mand\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mexclude\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m  10264\u001b[0m             \u001b[1;31m# when some numerics are found, keep only numerics\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m> 10265\u001b[1;33m             \u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mselect_dtypes\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0minclude\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mnumber\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m  10266\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m  10267\u001b[0m                 \u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36mselect_dtypes\u001b[1;34m(self, include, exclude)\u001b[0m\n\u001b[0;32m   3425\u001b[0m         \u001b[1;31m# the \"union\" of the logic of case 1 and case 2:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3426\u001b[0m         \u001b[1;31m# we get the included and excluded, and return their logical and\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 3427\u001b[1;33m         \u001b[0minclude_these\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mSeries\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;32mnot\u001b[0m \u001b[0mbool\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0minclude\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mindex\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   3428\u001b[0m         \u001b[0mexclude_these\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mSeries\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;32mnot\u001b[0m \u001b[0mbool\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mexclude\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mindex\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3429\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\series.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, data, index, dtype, name, copy, fastpath)\u001b[0m\n\u001b[0;32m    309\u001b[0m                     \u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mdata\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcopy\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    310\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 311\u001b[1;33m                 \u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0msanitize_array\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mindex\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mraise_cast_failure\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    312\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    313\u001b[0m                 \u001b[0mdata\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mSingleBlockManager\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mindex\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfastpath\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\internals\\construction.py\u001b[0m in \u001b[0;36msanitize_array\u001b[1;34m(data, index, dtype, copy, raise_cast_failure)\u001b[0m\n\u001b[0;32m    710\u001b[0m                 \u001b[0mvalue\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmaybe_cast_to_datetime\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mvalue\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    711\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 712\u001b[1;33m             \u001b[0msubarr\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mconstruct_1d_arraylike_from_scalar\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mvalue\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mindex\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    713\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    714\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\dtypes\\cast.py\u001b[0m in \u001b[0;36mconstruct_1d_arraylike_from_scalar\u001b[1;34m(value, length, dtype)\u001b[0m\n\u001b[0;32m   1231\u001b[0m                 \u001b[0mvalue\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mensure_str\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mvalue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1232\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1233\u001b[1;33m         \u001b[0msubarr\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mempty\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlength\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1234\u001b[0m         \u001b[0msubarr\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfill\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mvalue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1235\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: Cannot interpret '<attribute 'dtype' of 'numpy.generic' objects>' as a data type"
     ]
    }
   ],
   "source": [
    "dataset.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "executionInfo": {
     "elapsed": 82,
     "status": "ok",
     "timestamp": 1641224170131,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "jaTzS8iSTgQP"
   },
   "outputs": [],
   "source": [
    "y = dataset['Fire Occurrence']\n",
    "x = dataset.drop(['Fire Occurrence'],axis =1 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "executionInfo": {
     "elapsed": 82,
     "status": "ok",
     "timestamp": 1641224170131,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "OY6BJVlPUhn3"
   },
   "outputs": [],
   "source": [
    "Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,random_state=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6nVfjd7aYyGf"
   },
   "source": [
    "Linear Regression (Without sklearn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "executionInfo": {
     "elapsed": 82,
     "status": "ok",
     "timestamp": 1641224170132,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "J45fh9GPVWo4"
   },
   "outputs": [],
   "source": [
    "class Multiple_Linear_Regression():   \n",
    "    def __init__ (self):\n",
    "        self.theta=np.zeros(int(np.random.random()),float)[:,np.newaxis]; \n",
    "    \n",
    "    def fit(self, X_train, y_train):\n",
    "        \n",
    "        X_b = np.c_[np.ones(len(X_train)), X_train] \n",
    "        theta_bst = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)\n",
    "        self.theta = theta_bst\n",
    "    \n",
    "    def predict(self, X_test):\n",
    "      \n",
    "        X_test = np.c_[np.ones((len(X_test), 1)), X_test]\n",
    "        y_predict = np.dot(X_test, self.theta)\n",
    "        \n",
    "        return y_predict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "executionInfo": {
     "elapsed": 82,
     "status": "ok",
     "timestamp": 1641224170132,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "WqWRvHEMVqbn"
   },
   "outputs": [],
   "source": [
    "model = Multiple_Linear_Regression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "executionInfo": {
     "elapsed": 82,
     "status": "ok",
     "timestamp": 1641224170133,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "YhUq8pHgVvSJ"
   },
   "outputs": [],
   "source": [
    "model.fit(Xtrain, ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 83,
     "status": "ok",
     "timestamp": 1641224170134,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "TohQ6JfXXVLf",
    "outputId": "9a3b0ba6-c9bf-4b16-a985-651c14c53d7f"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.89115046,  0.07475349, -0.00989328,  0.48595944,  0.9848651 ,\n",
       "        1.28231524,  0.79169641,  0.22295063])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_preds = model.predict(Xtest)\n",
    "y_preds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 80,
     "status": "ok",
     "timestamp": 1641224170135,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "fgHFqxUUcEH_",
    "outputId": "eb13c46d-fe2c-4ec0-9c2c-8476af00f0fa"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4     1\n",
       "28    0\n",
       "29    0\n",
       "33    0\n",
       "34    1\n",
       "25    1\n",
       "10    1\n",
       "22    0\n",
       "Name: Fire Occurrence, dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ytest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "executionInfo": {
     "elapsed": 78,
     "status": "ok",
     "timestamp": 1641224170136,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "Hj60J7JLXk6e"
   },
   "outputs": [],
   "source": [
    "def mse(y_preds, y):\n",
    "\n",
    "   \n",
    "    mse = ((y - y_preds)**2).mean()\n",
    "    return mse\n",
    "\n",
    "def rmse(y_preds, y):\n",
    "\n",
    "    rmse = (((y - y_preds)**2).mean())**(1/2)\n",
    "    return rmse\n",
    "\n",
    "def r2(y_preds, y):\n",
    "   \n",
    "    ssr = sum((y - y_preds) ** 2) \n",
    "    sst = sum((y - y.mean()) ** 2) \n",
    "    return 1 - ssr/sst"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 79,
     "status": "ok",
     "timestamp": 1641224170137,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "BL7qMSvbYCDf",
    "outputId": "9f266f85-1251-4adf-b7c0-578366c2f467"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.05333988639632514, 0.2309542950376224, 0.7866404544146994)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mse(y_preds, ytest), rmse(y_preds, ytest), r2(y_preds ,ytest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "inputt=[int(x) for x in \"45 32 60\".split(' ')]\n",
    "final=[np.array(inputt)]\n",
    "\n",
    "b = model.predict(final)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "l96mjZHrY7HP"
   },
   "source": [
    "Decision Tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 76,
     "status": "ok",
     "timestamp": 1641224170137,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "VyWWCaGwYGA_",
    "outputId": "f117ac06-c156-44ce-8605-fb525ef9d87f"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Asus\\Anaconda3\\lib\\site-packages\\sklearn\\tree\\tree.py:163: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
      "  y_encoded = np.zeros(y.shape, dtype=np.int)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,\n",
       "                       max_features=None, max_leaf_nodes=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=1, min_samples_split=2,\n",
       "                       min_weight_fraction_leaf=0.0, presort=False,\n",
       "                       random_state=0, splitter='best')"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fitting the decision tree to the training set\n",
    "decision_model = DecisionTreeClassifier(criterion = \"gini\", random_state=0)\n",
    "decision_model.fit(Xtrain, ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 73,
     "status": "ok",
     "timestamp": 1641224170138,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "WkcslaDnzhxW",
    "outputId": "721f6bd2-62e3-448b-da2c-c2292bb39343"
   },
   "outputs": [],
   "source": [
    "inputt=[int(x) for x in \"45 32 60\".split(' ')]\n",
    "final=[np.array(inputt)]\n",
    "\n",
    "b = decision_model.predict_proba(final)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "executionInfo": {
     "elapsed": 71,
     "status": "ok",
     "timestamp": 1641224170139,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "pjuoIQUw2B_W"
   },
   "outputs": [],
   "source": [
    "with open('model.pkl', 'wb') as fid:\n",
    "     pickle.dump(decision_model, fid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "executionInfo": {
     "elapsed": 74,
     "status": "ok",
     "timestamp": 1641224170142,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "dRdEEyBc2SFu"
   },
   "outputs": [],
   "source": [
    "# Read the data from the file\n",
    "with open('model.pkl', 'rb') as fid:\n",
    "     data = pickle.load(fid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "executionInfo": {
     "elapsed": 74,
     "status": "ok",
     "timestamp": 1641224170143,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "L7CdSBPIz9Ad"
   },
   "outputs": [],
   "source": [
    "# pickle.dump(decision_model,open('model.pkl','wb'))\n",
    "# model=pickle.load(open('model.pkl','rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 75,
     "status": "ok",
     "timestamp": 1641224170144,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "xjYzpZ6pZVJA",
    "outputId": "7f72c9f7-c756-4189-ff97-973d3e13e9b5"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 0, 1, 1, 1, 1, 0], dtype=int64)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Predicting the test set result\n",
    "ypred = decision_model.predict(Xtest)\n",
    "ypred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 74,
     "status": "ok",
     "timestamp": 1641224170145,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "-esVNY_jZcwH",
    "outputId": "270ad249-398d-4cef-b2bb-e7bfab3900ea"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.3535533905932738\n"
     ]
    }
   ],
   "source": [
    "mse = sklearn.metrics.mean_squared_error(ytest, ypred)\n",
    "rmse = math.sqrt(mse)\n",
    "print(rmse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 72,
     "status": "ok",
     "timestamp": 1641224170146,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "dRuOJ_peZ8n_",
    "outputId": "dc125ef2-3f74-4f28-8058-facdcc39588e"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "87.5"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(ytest, ypred)*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 71,
     "status": "ok",
     "timestamp": 1641224170146,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "iNHMbKnSaAKW",
    "outputId": "2a8db42a-1cb9-4da4-8c30-232f04bcfa58"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[3, 1],\n",
       "       [0, 4]], dtype=int64)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Making confusion matrix\n",
    "from sklearn.metrics import confusion_matrix\n",
    "cm = confusion_matrix(ytest, ypred)\n",
    "cm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 69,
     "status": "ok",
     "timestamp": 1641224170147,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "QrLaxSRWaEpW",
    "outputId": "82e8ffcf-9517-4618-fc36-8888f457077e"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100.0 %\n"
     ]
    }
   ],
   "source": [
    "decision_model_accuracy = round(decision_model.score(Xtrain, ytrain)*100,2)\n",
    "print(round(decision_model_accuracy, 2), '%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 67,
     "status": "ok",
     "timestamp": 1641224170147,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "tHD3fTyfaNm2",
    "outputId": "02f30168-c914-45e4-d85f-a33bca4b2246"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "87.5 %\n"
     ]
    }
   ],
   "source": [
    "decision_model_accuracy = round(decision_model.score(Xtest, ytest)*100,2)\n",
    "print(round(decision_model_accuracy, 2), '%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "EMgEiJdkagW3"
   },
   "source": [
    "Naive Bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 65,
     "status": "ok",
     "timestamp": 1641224170148,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "ILMwDhdYaSVG",
    "outputId": "095f9e64-f76a-470d-dede-b4b9a442b7dc"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GaussianNB(priors=None, var_smoothing=1e-09)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fitting the naive bayes to the training set\n",
    "naive_model = GaussianNB()\n",
    "naive_model.fit(Xtrain, ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 63,
     "status": "ok",
     "timestamp": 1641224170148,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "-ZkiFEJcar7e",
    "outputId": "c9ec3460-a397-4b18-cd6f-e75b0089b285"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 0, 0, 1, 1, 1, 0], dtype=int64)"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Predicting the test set result\n",
    "ypred = naive_model.predict(Xtest)\n",
    "ypred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 61,
     "status": "ok",
     "timestamp": 1641224170149,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "wQLbApzGavLO",
    "outputId": "db05abca-5598-47f3-e4c5-be84c117e8df"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.0\n"
     ]
    }
   ],
   "source": [
    "mse = sklearn.metrics.mean_squared_error(ytest, ypred)\n",
    "rmse = math.sqrt(mse)\n",
    "print(rmse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 59,
     "status": "ok",
     "timestamp": 1641224170149,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "aqgBsjxVaznm",
    "outputId": "bcd4355b-5611-4870-dba7-4878aaf669ae"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[4, 0],\n",
       "       [0, 4]], dtype=int64)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Making confusion matrix\n",
    "from sklearn.metrics import confusion_matrix\n",
    "cm = confusion_matrix(ytest, ypred)\n",
    "cm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 55,
     "status": "ok",
     "timestamp": 1641224170149,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "OmbJjwQna3HX",
    "outputId": "ee9c9a14-6f32-4df0-c5bc-3b9e2931b59f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100.0 %\n"
     ]
    }
   ],
   "source": [
    "naive_model_accuracy = round(naive_model.score(Xtrain, ytrain)*100,2)\n",
    "print(round(naive_model_accuracy, 2), '%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 53,
     "status": "ok",
     "timestamp": 1641224170150,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "nHpQxVeka7aW",
    "outputId": "888fc988-914d-48ea-88fa-81304cbd22d6"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "100.0 %\n"
     ]
    }
   ],
   "source": [
    "naive_model_accuracy = round(naive_model.score(Xtest, ytest)*100,2)\n",
    "print(round(naive_model_accuracy, 2), '%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Id0IDAF1m-2I"
   },
   "source": [
    "Creating Pickle Module"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "executionInfo": {
     "elapsed": 51,
     "status": "ok",
     "timestamp": 1641224170150,
     "user": {
      "displayName": "Faysal Mh",
      "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjNRxFGW1UpX2sxNb6lkS5RMnld_zH55HhZxsMzRQ=s64",
      "userId": "14725302750165398384"
     },
     "user_tz": -360
    },
    "id": "mpLhqB29m9_2"
   },
   "outputs": [],
   "source": [
    "# pickle.dump(model, open('model.pkl','wb'))\n",
    "# model = pickle.load(open(\"model.pkl\",\"rb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "fire-prediction.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
