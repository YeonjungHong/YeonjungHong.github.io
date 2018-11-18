---
layout: post
title: Kernel Principal Component Analysis
description: "Principal Component Analysis"
headline: "Kernel Principal Component Analysis"
categories: MACHINELEARNING
tags: 
  - Machine Learning
  - PCA
  - KPCA
comments: false
mathjax: true
featured: true
published: true
---

## Principal Component Analysis

- PCA는 기본적으로 특징추출 기법이다.
- 정보 손실을 최소화 하면서  M차원 데이터를 m차원으로 차원 축소하는 것
- 실제 단계
	1. 각 feature의 평균이 0, 표준편차가 1이 되도록 feature normalization
	2. feature의 공분산 행렬 구하기
	3. Singular Vector Decomposition을 통해 공분산의 Eigenvectors와 Eigenvalue 구하기 
		- Orthogonal Eigenvectors * Diagonal Eigenvalues * Orthogonal Eigenvectors로 분해됨
	4. Eigenvalue값이 큰 Eigenvector순으로 제 1주성분, 제2주성분,...순서가 매겨짐
        - eigenvalue가 variance를 나타냄
	5. 몇 개의 주성분을 선택할지는 Scree plot을 그려서 판단한다.
		- x축 Eigenvector (PC1, PC2, PC3, ..)
		- y축 각 주성분에 대응하는 Eigenvalue
		- Elbow point 지점까지의 주성분 개수 선택 
	6. 선택된 주성분을 축으로 span하는 공간에 데이터를 projection

## PCA vs Linear Regression
- PCA
	- Unsupervised
	- 모델-데이터 간 수직 거리 최소화

- Linear Regression
	- Supervised
	- 종속변수-모델 간 거리 최소화
	- 최소화하는 대상이 PCA와 다르기 때문에 Least Squared 방법으로 구한 리그레션 모델은 항상 PC1보다 기울기가 낮다. 

## Covariance vs Correlation

- Correlation은 Normalized Covariance와 동일하다. 
- PCA 도입부에 feature normalization하는 이유는 covariacne matrix가 사실상 correlation matrix가 되도록 만들어 주기 위함이다.
- Covariance Matrix에 대한 이해 
    - x,y,z feature set에 대한 3-by-3 covariance matrix A가 있을 때,
        - A(1,1): 모든 데이터를 x축에 projection 시킨 다음에 그 축 상에서의 variance 계산한 값.
        - A(1,2): 모든 데이터를 x축과 y축이 이루는 2D 평면에 projection 시킨 다음에 그 평면 상에서의 covariance를 계산한 값. 
    - 또 다른 생각방식: Covariance란 Normalized data value 간의 cosine similarity.  
- Correlation과 Covariance 모두 데이터의 Gaussian 분포를 가정하고 있다는 사실이 중요하다. 따라서 Gaussian 분포를 이루지 않은 데이터에 대해서 PCA를 하면 무의미한 결과가 나온다. (Gaussian 분포 공식을 생각해보면 covariance 구하는 식이 들어가 있음)


## Kernel Principal Component Analysis
- PCA가 linear space로의 차원축소를 해낸다면, KPCA는 nonlinear space로의 차원축소를 해낸다. 
<pre><code>
	<p>def rbf\_kernel(X, gamma=None):
	    sq\_dists = pdist(X, 'sqeuclidean')
	    mat\_sq\_dists = squareform(sq\_dists)
	    K = exp(-gamma * mat\_sq\_dists)
	    return K
	</p>
</code></pre>
<pre><code>
	<p>def polynomial\_kernel(X, degree=3, gamma=None, coef0=1):
		K = (gamma\*np.dot(X,X) + coef0)**degree
    	return K
	</p>
</code></pre>
<pre><code>
	<p>def sigmoid\_kernel(X, gamma=None, coef0=1):
    	K = gamma\*np.dot(X,X) + coef0
    	K = np.tanh(K,K)
    	return K
	</p>
</code></pre>
<pre><code>
	<p>def kernel\_pca(X, gamma, n\_components, kernel="rbf"):
	    if kernel == "rbf":
	        K = rbf\_kernel()
	    elif kernel == "poly":
	        K = polynomial\_kernel()
	    elif kernel == "sig":
	        K = sigmoid\_kernel()
	    N = K.shape[0]
	    one\_n = np.ones((N,N)) / N
	    K = K - one_n.dot(K) - K.dot(one\_n) + one\_n.dot(K).dot(one\_n)
	    eigvals, eigvecs = eigh(K)
	    X\_pc = np.column\_stack((eigvecs[:, -i]
                            for i in range(1, n\_components + 1)))
    	return X\_pc
    </p>
</code></pre>


<p align="right"> Yeonjung Hong <p>
