import numpy as np
from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.validation import check_X_y
from sklearn.feature_selection import SelectorMixin, mutual_info_classif
from tqdm import tqdm

class HybridRankingWrapperSelector(BaseEstimator, SelectorMixin):
    """
    A hybrid feature selection method that combines feature ranking, 
    fast wrapper adding, and sequential backward selection (SBS).
    """

    def __init__(self, base_estimator=None, alpha=0.99, max_features_ratio=0.2, 
                 cv=5, scoring='accuracy', max_features_value=0, verbose=False):
        
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.max_features_ratio = max_features_ratio
        self.max_features_value = max_features_value
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        
        # Internal tracking variables
        self.best_subset_ = None
        self.best_value_ = -np.inf
        self.n_wrapper_evaluations_ = 0
        self.support_ = None

    def _log(self, msg):
        """Prints log messages only when verbose=True."""
        if self.verbose:
            print(msg)

    def _evaluate(self, X, y, feature_indices):
        if not feature_indices:
            return -np.inf

        X_subset = X[:, feature_indices]
        estimator = clone(self.base_estimator)

        try:
            # Handle both integer cv and cv objects
            scores = cross_val_score(estimator, X_subset, y, cv=self.cv, scoring=self.scoring)
        except ValueError:
            return -np.inf

        self.n_wrapper_evaluations_ += 1

        score = np.mean(scores)
        if self.verbose:
            self._log(f"  ▸ Evaluated subset {feature_indices} → score={score:.4f}")

        return score

    def _sbs_reduction(self, X, y, current_subset, max_features):
        if len(current_subset) <= max_features:
            return current_subset, self._evaluate(X, y, current_subset)

        best_sbs_subset = list(current_subset)

        n_remove = len(current_subset) - max_features
        self._log(f"\n[SBS] Starting Sequential Backward Selection: removing {n_remove} features...")

        for step in range(n_remove):
            worst_feature_idx_in_list = -1
            max_value_after_removal = -np.inf

            for i in range(len(best_sbs_subset)):
                temp_subset = best_sbs_subset[:i] + best_sbs_subset[i+1:]
                value = self._evaluate(X, y, temp_subset)

                if value >= max_value_after_removal:
                    max_value_after_removal = value
                    worst_feature_idx_in_list = i

            removed_feature = best_sbs_subset[worst_feature_idx_in_list]
            best_sbs_subset.pop(worst_feature_idx_in_list)

            self._log(
                f"  ✖ Removed feature {removed_feature} → New SBS score={max_value_after_removal:.4f}, "
                f"Remaining={len(best_sbs_subset)}"
            )

        return best_sbs_subset, max_value_after_removal

    def fit(self, X, y=None):
        if self.base_estimator is None or not is_classifier(self.base_estimator):
            raise ValueError("base_estimator must be a classifier.")
        
        X, y = check_X_y(X, y)

        self.n_wrapper_evaluations_ = 0
        self.best_value_ = -np.inf
        self.best_subset_ = []

        n_total_features = X.shape[1]
        feature_indices = np.arange(n_total_features)

        # ----------------------------------------------------------------------
        # PHASE 1 — RANKING
        # ----------------------------------------------------------------------
        self._log("\n[PHASE 1] Ranking Features...\n")

        ranking_scores = []
        for i in feature_indices:
            score = self._evaluate(X, y, [i])
            ranking_scores.append(score)

            if self.verbose:
                self._log(f"  ● Feature {i} → score={score:.4f}")

        ranked_features = sorted(zip(ranking_scores, feature_indices), key=lambda x: x[0], reverse=True)
        feature_pool = [idx for _, idx in ranked_features]

        max_feats = int(np.ceil(self.max_features_value)) if self.max_features_value > 0 else int(n_total_features * self.max_features_ratio)
        current_val = -np.inf
        current_subset = []

        # ----------------------------------------------------------------------
        # PHASE 2 — FAST WRAPPER ADDING
        # ----------------------------------------------------------------------
        self._log(f"\n[PHASE 2] Adding Features (MaxFeatures={max_feats})...\n")

        iterator = tqdm(feature_pool, desc="HRW Adding") if self.verbose else feature_pool
        for feature_to_add in iterator:
            current_subset.append(feature_to_add)

            val = self._evaluate(X, y, current_subset)

            if val >= current_val * self.alpha:
                self._log(f"  ✔ Keep feature {feature_to_add} (score={val:.4f})")
                current_val = val
            else:
                current_subset.remove(feature_to_add)
                self._log(f"  ✖ Remove feature {feature_to_add} (score={val:.4f})")

            if current_val > self.best_value_:
                self.best_value_ = current_val
                self.best_subset_ = list(current_subset)
                self._log(f"    ★ New BEST subset! size={len(self.best_subset_)} score={self.best_value_:.4f}")

            if len(current_subset) > max_feats:
                self._log(f"    ⚠ Exceeded MaxFeatures → Rollback to best subset.")
                current_val = self.best_value_
                current_subset = list(self.best_subset_)

        # ----------------------------------------------------------------------
        # PHASE 3 — SBS
        # ----------------------------------------------------------------------
        self._log("\n[PHASE 3] SBS Reduction...\n")

        final_subset, final_value = self._sbs_reduction(X, y, self.best_subset_, max_feats)

        self.best_subset_ = final_subset
        self.best_value_ = final_value

        self.support_ = np.zeros(n_total_features, dtype=bool)
        self.support_[self.best_subset_] = True

        self._log(f"\n✔ Final subset size={len(self.best_subset_)} score={self.best_value_:.4f}")
        self._log(f"✔ Selected indices: {self.best_subset_}")

        return self

    def _get_support_mask(self):
        if self.support_ is None:
            raise ValueError("The selector has not been fitted yet.")
        return self.support_

class MRMRSelector(BaseEstimator, SelectorMixin):
    """
    Minimum Redundancy Maximum Relevance (mRMR) feature selection.
    Uses Mutual Information for relevance and absolute correlation for redundancy.
    """

    def __init__(self, n_features_to_select=20, verbose=False):
        self.n_features_to_select = n_features_to_select
        self.verbose = verbose
        self.support_ = None
        self.selected_indices_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_features = X.shape[1]
        
        if self.n_features_to_select > n_features:
            self.n_features_to_select = n_features

        if self.verbose:
            print(f"[mRMR] Calculating relevance (Mutual Information)...")
        
        # 1. Relevance: MI(X_i, y)
        relevance = mutual_info_classif(X, y, random_state=42)
        
        if self.verbose:
            print(f"[mRMR] Precomputing correlation matrix for redundancy...")
            
        # 2. Redundancy: Absolute correlation
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
            corr_matrix = np.nan_to_num(corr_matrix)
        
        selected_indices = []
        unselected_indices = list(range(n_features))
        
        # Select first feature (highest relevance)
        first_feature = np.argmax(relevance)
        selected_indices.append(first_feature)
        unselected_indices.remove(first_feature)
        
        if self.verbose:
            print(f"  ● Selected first feature: {first_feature} with relevance {relevance[first_feature]:.4f}")

        # Iteratively select features
        iterator = tqdm(range(self.n_features_to_select - 1), desc="mRMR Selecting") if self.verbose else range(self.n_features_to_select - 1)
        
        for _ in iterator:
            best_mrmr = -np.inf
            best_feature = -1
            
            for i in unselected_indices:
                rel = relevance[i]
                
                # Average Redundancy: 1/|S| * sum(|corr(X_i, X_j)| for j in S)
                redundancy = np.mean(corr_matrix[i, selected_indices])
                
                mrmr = rel - redundancy
                if mrmr > best_mrmr:
                    best_mrmr = mrmr
                    best_feature = i
            
            selected_indices.append(best_feature)
            unselected_indices.remove(best_feature)
            
            if self.verbose and not isinstance(iterator, tqdm):
                print(f"  ● Selected feature {best_feature}: mRMR score {best_mrmr:.4f}")

        self.selected_indices_ = selected_indices
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[selected_indices] = True
        return self

    def _get_support_mask(self):
        if self.support_ is None:
            raise ValueError("The selector has not been fitted yet.")
        return self.support_
