import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv('data1.csv', parse_dates=['Date'])

customerdf = df.groupby('customerID').agg(
    totalspend=('TotalPrice', 'sum'),
    avgordervalue=('TotalPrice', 'mean'),
    frequency=('OrderID', 'nunique'),
    avgquantity=('Quantity', 'mean'),
    uniqueproducts=('Product', 'nunique'),
    lastorder=('Date', 'max')
).reset_index()

today = customerdf['lastorder'].max()
customerdf['recency'] = (today - customerdf['lastorder']).dt.days
customerdf.drop('lastorder', axis=1, inplace=True)

featurecols = ['totalspend', 'avgordervalue', 'frequency', 'avgquantity', 'uniqueproducts', 'recency']
Xraw = customerdf[featurecols].fillna(0).values

mean = np.mean(Xraw, axis=0)
std = np.std(Xraw, axis=0)
std[std == 0] = 1
Xscaled = (Xraw - mean) / std

def pca_scratch(X, ncomp):
    Xcentered = X - np.mean(X, axis=0)
    covmat = np.cov(Xcentered, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(covmat)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    evr = eigvals / np.sum(eigvals)
    Xpca = Xcentered @ eigvecs[:, :ncomp]
    return Xpca, evr, eigvecs

Xpca, evr, eigvecs = pca_scratch(Xscaled, 2)
print(f"Explained variance: {evr[0]:.2%} + {evr[1]:.2%} = {np.sum(evr):.2%}")

def kmeans_scratch(X, k, maxiters=100):
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(maxiters):
        dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(dists, axis=1)
        newcentroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, newcentroids):
            break
        centroids = newcentroids
    wcss = np.sum((X - centroids[labels]) ** 2)
    return labels, centroids, wcss

def silhouette_scratch(X, labels):
    n = len(X)
    s = []
    for i in range(n):
        samecluster = X[labels == labels[i]]
        if len(samecluster) > 1:
            ai = np.mean(np.linalg.norm(X[i] - samecluster, axis=1))
        else:
            ai = 0
        otherclusters = np.unique(labels[labels != labels[i]])
        bi = np.inf
        for c in otherclusters:
            otherpts = X[labels == c]
            dists = np.linalg.norm(X[i] - otherpts, axis=1)
            bc = np.mean(dists)
            if bc < bi:
                bi = bc
        if ai == 0 and bi == np.inf:
            si = 0
        else:
            si = (bi - ai) / max(ai, bi)
        s.append(si)
    return np.mean(s)

Krange = range(2, 11)
wcsslist = []
sillist = []

print("\nEvaluating K from 2 to 10...")
for k in Krange:
    labels, _, wcss = kmeans_scratch(Xpca, k)
    wcsslist.append(wcss)
    sil = silhouette_scratch(Xpca, labels)
    sillist.append(sil)
    print(f"K={k}: WCSS={wcss:.2f}, Silhouette={sil:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.plot(Krange, wcsslist, 'bo-')
ax1.set_xlabel('Number of clusters (K)')
ax1.set_ylabel('WCSS')
ax1.set_title('Elbow Method')
ax1.grid(True)
ax2.plot(Krange, sillist, 'rs-')
ax2.set_xlabel('Number of clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score')
ax2.grid(True)
plt.tight_layout()
plt.savefig('elbow_silhouette.png')
plt.show()

optk = Krange[np.argmax(sillist)]
print(f"\nOptimal K: {optk}")

finallabels, finalcentroids, _ = kmeans_scratch(Xpca, optk)
customerdf['Cluster'] = finallabels

plt.figure(figsize=(8,6))
plt.scatter(Xpca[:,0], Xpca[:,1], c=finallabels, cmap='viridis', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'Customer Segments (k={optk})')
plt.colorbar()
plt.savefig('cluster_visualisation.png')
plt.show()

_, _, eigvecs_full = pca_scratch(Xscaled, Xscaled.shape[1])
projmat = eigvecs_full[:, :2]
centroidsscaled = finalcentroids @ projmat.T
centroidsoriginal = centroidsscaled * std + mean

clusterprofile = pd.DataFrame(centroidsoriginal, columns=featurecols, index=[f'Cluster_{i}' for i in range(optk)])
print("\nCluster profiles (original units, centroids):")
print(clusterprofile)

actualmeans = customerdf.groupby('Cluster')[featurecols].mean()
print("\nActual average values per cluster:")
print(actualmeans)

print("\n" + "="*60)
print("BUSINESS PERSONAS")
print("="*60)

globalmeans = customerdf[featurecols].mean()

for cid in range(optk):
    size = (customerdf['Cluster'] == cid).sum()
    prof = actualmeans.loc[cid]
    print(f"\n--- Cluster {cid} (size: {size} customers) ---")
    deviation = (prof - globalmeans) / globalmeans.abs()
    topfeats = deviation.abs().nlargest(2).index.tolist()
    print("  Key characteristics:")
    for feat in topfeats:
        val = prof[feat]
        gval = globalmeans[feat]
        direction = "higher" if val > gval else "lower"
        print(f"    {feat}: {val:.2f} ({direction} than avg by {abs(val/gval -1):.1%})")
    if prof['totalspend'] > globalmeans['totalspend'] * 1.2:
        spendtype = "High-value"
    elif prof['totalspend'] < globalmeans['totalspend'] * 0.8:
        spendtype = "Low-value"
    else:
        spendtype = "Mid-range"
    if prof['frequency'] > globalmeans['frequency'] * 1.2:
        freqtype = "frequent"
    elif prof['frequency'] < globalmeans['frequency'] * 0.8:
        freqtype = "rare"
    else:
        freqtype = "regular"
    persona = f"{spendtype} & {freqtype} buyers"
    print(f"  Persona: {persona}")
    action = "Premium loyalty program" if spendtype == "High-value" else "Discount campaigns"
    print(f"  Suggested action: {action}")

customerdf.to_csv('customer_segments_output.csv', index=False)
print("\nSaved: customer_segments_output.csv, elbow_silhouette.png, cluster_visualisation.png")
