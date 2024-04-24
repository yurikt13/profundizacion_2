# Importando librerias necesarias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_rcv1
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

rcv1 = fetch_rcv1()
print(rcv1)