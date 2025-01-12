divexplorer
===========

.. image:: https://img.shields.io/pypi/v/divexplorer.svg
    :target: https://pypi.python.org/pypi/divexplorer
    :alt: Latest PyPI version


DivExplorer

Usage
-----
Example in notebooks

>>> from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
>>> from divexplorer.FP_Divergence import FP_Divergence
>>> 
>>> min_sup=0.1
>>> # Input: # discretized dataframe, true class (str - column name), predicted class  (opt) (str - column name) 
>>> #Extract frequent patterns (FP) and compute divergence (default metric of interest: False Positive Rate (FPR), False Negative Rate (FNR), Accuracy)
>>> fp_diver=FP_DivergenceExplorer(X_discretized, "class", "predicted", class_map=class_map, dataset_name=dataset_name)
>>> #Minimum support: frequency threshold for frequent pattern extraction and divergence estimation
>>> FP_fm=fp_diver.getFrequentPatternDivergence(min_support=min_sup)
>>> 
>>> # If input just one class --> positive rate and negative rate as metric of interest (d_posr and d_negr)
>>> min_sup=0.1
>>> fp_diver_1cl=FP_DivergenceExplorer(X_discretized.drop(columns="predicted"),"class", class_map=class_map, dataset_name=dataset_name)
>>> FP_fm_1cl=fp_diver_1cl.getFrequentPatternDivergence(min_support=min_sup, metrics=["d_posr", "d_negr"])


The output is a pandas dataframe. Each row is a FP with classification info (e.g. TP/FP/FN/TN, FPR, FNR etc) and its divergence.

We can then analyze the divergence of FP w.r.t. a metric of interest (e.g. FPR).

>>> fp_divergence_fpr=FP_Divergence(FP_fm, "d_fpr")
>>> #FP sorted for their divergence:
>>> fp_divergence_fpr.getDivergence(th_redundancy=0)

>>> #TopK divergent patterns:
>>> #TopK FP and their divergence (dict)
>>> fp_divergence_fpr.getDivergenceTopK(K=5, th_redundancy=0)
>>> #DF format with all info
>>> fp_divergence_fpr.getDivergenceTopKDf(K=10, th_redundancy=0)

>>> #Compute Shapley values
>>> #Let be itemset a Frequent Pattern of interest
>>> itemset_shap=fp_divergence_fpr.computeShapleyValue(itemset)
>>> #Plot shapley values
>>> fp_divergence_fpr.plotShapleyValue(shapley_values=itemset_shap)
>>> #Alternatives
>>> fp_divergence_fpr.plotShapleyValue(itemset=itemset)
>>> 
>>> #Plot the lattice graph
>>> #Th_divergence: if specified, itemsets of the lattice with divergence greater than specified value are highlighted in magenta/squares
>>> Get lower: if True, corrective patterns are highlighted in light blue/diamonds
>>> fig=fp_divergence_fpr.plotLatticeItemset(itemset, Th_divergence=0.15, sizeDot="small", getLower=True)

>>> #Corrective items
>>> fp_divergence_fpr.getCorrectiveItems()
>>> 
>>> #Compute global shapley value
>>> u_h_fpr=fp_divergence_fpr.computeGlobalShapleyValue()
>>> fp_divergence_fpr.plotShapleyValue(shapley_values=u_h_fpr)

Installation
------------

Requirements
^^^^^^^^^^^^

Compatibility
-------------

Licence
-------

Authors
-------

`divexplorer` was written by `Eliana Pastor <eliana.pastor@polito.it>`_.
