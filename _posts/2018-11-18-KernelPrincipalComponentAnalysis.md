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

Kernel PCA를 이해하기 위해서는 먼저 PCA를 이해해야 한다. 

- 목적: 차원 축소를 통한 특징추출
- 원리: 데이터를 설명하는 새로운 벡터공간의 축을 찾을 때, 공분산 정보를 최대한 보존하고자 한다. 따라서 새로운 축은 공분산의 eigenvector중에 eigenvalue가 높은 것들이 선택된다.
- 단계
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
	- 모델-데이터 간 ***수직 거리*** 최소화 --> 공분산 정보 최대 캡쳐

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
- PCA가 linear space로의 차원축소 기법이라면, KPCA는 nonlinear space로의 차원축소 기법이다.
- PCA가 데이터의 공분산으로부터 구한 eigenvector에 데이터를 projection 시켰다면, KPCA는 nonlinear kernel 함수를 거쳐 새로운 공간에 있다고 여겨지는 데이터의 공분산으로부터 동일 작업을 수행한다. 
- 몇가지 수식 전개를 통해 KPCA의 절차는 다음과 같은 단계로 요약된다.
	1. kernel 함수를 선택한다.
	2. kernel 함수로부터 Gram 행렬을 구한다. KPCA에서의 공분산 행렬에 해당한다.
	3. PCA에서와 마찬가지로 eigenvalue가 높은 eigenvector를 선정한다.
	4. 선정된 eigenvector에 데이터를 project한다.

- 본 실습에서는 RBF kernel과 Polynomial kernel, 2가지를 적용시킨 KPCA를 구현하고자 한다.

#### RBF kernel
K(x, y) = exp(-gamma ||x-y||^2)
  
<pre><code>
	<blockquote>	
	<p>def rbf\_kernel(X, gamma=None):
		\# dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
	    sq\_dists = pdist(X, 'sqeuclidean')
	    mat\_sq\_dists = squareform(sq\_dists)
	    K = np.exp(-gamma * mat\_sq\_dists)
	    return K
	</p>
	</blockquote>	
</code></pre>

K(X, Y) = (gamma <X, Y> + coef0)^degree
<pre><code>
	<blockquote>	
	<p>def polynomial\_kernel(X, degree=3, gamma=None, coef0=1):
		K = (gamma\*np.dot(X,X) + coef0)**degree
    	return K
	</p>
	</blockquote>	
</code></pre>

<pre><code>
	<blockquote>	
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
    </blockquote>	
</code></pre>
<pre><code>
	<blockquote>	
	<p>X, y = make\_circles(n\_samples=400, factor=.3, noise=.05)
	
	\# Plot results
	
	plt.figure()
	plt.subplot(1, 2, 1, aspect='equal')
	plt.title("Original space")
	reds = y == 0
	blues = y == 1
	
	plt.scatter(X[reds, 0], X[reds, 1], c="red",
	            s=20, edgecolor='k')
	plt.scatter(X[blues, 0], X[blues, 1], c="blue",
	            s=20, edgecolor='k')
	plt.xlabel("$x_1$")
	plt.ylabel("$x_2$")
	
	X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
	X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
	\# projection on the first principal component (in the phi space)
	Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
	plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')
	
	
	plt.subplot(1, 2, 2, aspect='equal')
	plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red",
	            s=20, edgecolor='k')
	plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue",
	            s=20, edgecolor='k')
	plt.title("Projection by KPCA")
	plt.xlabel("1st principal component in space induced by $\phi$")
	plt.ylabel("2nd component")
	
	
	plt.show()
	</p>
	</blockquote>	
</code></pre>
<p align="right"> Yeonjung Hong <p>