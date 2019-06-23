import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

import xarray as xr
# import xgeo # Needs to be imported to use geo extension

import geopandas as gpd
import gdal

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# A list of "random" colors
COLORS = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
]




def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    fig, ax = plt.subplots(figsize = (8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def classic_classifier(method, X, y):
    """
    try:
        raster_dataset = xr.open_rasterio(raster_data_path)
        raster_dataset = raster_dataset.to_dataset(name='image')
        geo_transform = raster_dataset.geo.transform
        proj = raster_dataset.geo.projection
        n_bands = len(raster_dataset.band)
        dim = ['x', 'y']
        rows, cols = [raster_dataset.sizes[xy] for xy in dim]
        
        df_raster = raster_dataset.to_dataframe().unstack("band")
        
    except RuntimeError as e:
        report_and_exit(str(e))

    logger.debug("Process the training data")\
    """
    
    from sklearn.model_selection import train_test_split
    
    # create a dataframe from each pixel and its label
    # df_labeled_pixels = sample_vectors_to_raster(train_data_path, raster_dataset)

    # Create dataframe each pixel of the image 
    # X_predict = dataframe_to_features(df_raster)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    n_class = y_train.unique()

    #
    # Perform classification
    #
    CLASSIFIERS = {
        # http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'random-forest': RandomForestClassifier(n_jobs=4, n_estimators=10, class_weight='balanced'),
        # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        'svm': SVC(class_weight='balanced'),
        'linear-svm': SVC(kernel="linear", C=0.025),
        'svm-gamma': SVC(gamma=2, C=1),
        'nearest-neighbors': KNeighborsClassifier(3),
        'Gaussian-Process': GaussianProcessClassifier(1.0 * RBF(1.0)),
        'Decision-Tree': DecisionTreeClassifier(max_depth=5),
        'random-forest-1feature': RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
        'Neural-Net': MLPClassifier(alpha=1, max_iter=1000),
        'AdaBoost': AdaBoostClassifier(),
        'Naive-Bayes': GaussianNB(),
        'Quad-Discr': QuadraticDiscriminantAnalysis()
    }

    classifier = CLASSIFIERS[method]
    logger.debug("Train the classifier: %s", str(classifier))
    classifier.fit(X_train, y_train)
    
    #
    # Validate the results
    #
    logger.debug("Process the verification (testing) data")
    try:
        df_validation_pixels = sample_vectors_to_raster(validation_data_path, raster_dataset)
        X_test, y_test = dataframe_to_features(df_validation_pixels)

    except OSError.FileNotFoundError as e:
        report_and_exit(str(e))
    
    logger.debug("Classifing...")
    y_predicted = classifier.predict(X_test)

    logger.info("Confussion matrix:\n%s", str(
        metrics.confusion_matrix(y_test, y_predicted)))
    target_names = ['Class %s' % s for s in n_class]
    logger.info("Classification report:\n%s",
                metrics.classification_report(y_test, y_predicted,
                                                target_names=target_names))
    logger.info("Classification accuracy: %f",
                metrics.accuracy_score(y_test, y_predicted))
    
    print("Classification report:\n%s",
                metrics.classification_report(y_test, y_predicted,
                                                target_names=target_names))
    print("Classification accuracy: %f",
                metrics.accuracy_score(y_test, y_predicted))
    plot_confusion_matrix(y_true, y_pred, classes) 
    
    
    return classifier, y_test, y_predicted