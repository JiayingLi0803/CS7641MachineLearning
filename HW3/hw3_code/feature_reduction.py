import pandas as pd
import statsmodels.api as sm
from typing import List


class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            forward_list: (python list) contains significant features. Each feature
            name is a string
        """ 
        included = []
        best_pval = 0
        while best_pval<=significance_level:
            excluded = list(set(data.columns)-set(included))
            new_pval = pd.Series(index = excluded)
            for new_column in excluded:
                model = sm.OLS(target, sm.add_constant(pd.DataFrame(data[included+[new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < significance_level:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
        return included
        #raise NotImplementedError

    @staticmethod
    def backward_elimination(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            backward_list: (python list) contains significant features. Each feature
            name is a string
        """
        included = list(data.columns)
        
        worst_pval = 100
        while worst_pval > significance_level:
            model = sm.OLS(target, sm.add_constant(pd.DataFrame(data[included]))).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > significance_level:
                worst_feature = included[pvalues.argmax()]
                #print("included", included)
                #print("worst feature", worst_feature)
                included.remove(worst_feature)
        return included
        # raise NotImplementedError