import json
import sys
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.multiclass import OneVsRestClassifier
sys.path.append(str(Path(__file__).parent.parent.parent))
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import os

from scripts.utils import load_csv_features, plot_pr_curve, plot_top_flop_f1, plot_worst_confusion, plot_umap_features_justification


X_train, Y_train, X_val, Y_val, X_test, Y_test, X_nLabeled, Y_nLabeled, Z_nLabeled = load_csv_features()


plot_umap_features_justification(X_train, Y_train, top_k=6)