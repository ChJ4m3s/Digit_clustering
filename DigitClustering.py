from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# --- READING DATA ---
digits = load_digits().data

# --- BUILDING SIMILARITY MATRIX ---
SMatrix = pairwise_distances(digits, metric='sqeuclidean')
