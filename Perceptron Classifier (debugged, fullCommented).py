"""
Perceptron Classifier for Mushroom Classification
Implemented from scratch - no sklearn or similar libraries

FIXES APPLIED (see comments tagged #FIX):
  1. one_hot_encode: out-of-range index now encoded as missing ('?') instead
     of silently leaving an all-zero block.
  2. fit(): local random.Random instance instead of mutating the global RNG;
     seeded once per fit, not per epoch.
  3. fit(): best-weights checkpointing so a non-separable dataset cannot make
     the final model worse than an earlier epoch.
  4. split_data(): local RNG; no global side effects.
  5. predict(): iterate with zip(x, weights) to avoid IndexError on length
     mismatch.
  6. main(): final model is retrained on all data (deployment), and the
     printed number is explicitly labelled TRAINING accuracy, not a
     generalization estimate.
  7. Guard for empty training set.
  8. Prediction file ends with a newline.
"""

import random


#Python concept: dict literals and list literals; integer keys used as a sparse array. The order of the list defines the position within each one-hot block. 

""" Use a dict mapping attribute index -> list of legal symbolic values: Index 0 is the class label ('e' edible / 'p' poisonous); indices 1..22 are the 22 mushroom features of the UCI Mushroom dataset. """
ATTRIBUTE_VALUES = {
    0: ['e', 'p'],  # class (target)
    1: ['b', 'c', 'x', 'f', 'k', 's'],  # cap-shape
    2: ['f', 'g', 'y', 's'],  # cap-surface
    3: ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],  # cap-color
    4: ['t', 'f'],  # bruises
    5: ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],  # odor
    6: ['a', 'd', 'f', 'n'],  # gill-attachment
    7: ['c', 'w', 'd'],  # gill-spacing
    8: ['b', 'n'],  # gill-size
    9: ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],  # gill-color
    10: ['e', 't'],  # stalk-shape
    11: ['b', 'c', 'u', 'e', 'z', 'r', '?'],  # stalk-root
    12: ['f', 'y', 'k', 's'],  # stalk-surface-above-ring
    13: ['f', 'y', 'k', 's'],  # stalk-surface-below-ring
    14: ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],  # stalk-color-above-ring
    15: ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],  # stalk-color-below-ring
    16: ['p', 'u'],  # veil-type
    17: ['n', 'o', 'w', 'y'],  # veil-color
    18: ['n', 'o', 't'],  # ring-number
    19: ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],  # ring-type
    20: ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],  # spore-print-color
    21: ['a', 'c', 'n', 's', 'v', 'y'],  # population
    22: ['g', 'l', 'm', 'p', 'u', 'w', 'd'],  # habitat
}

# ------------------------------------------------------------------
"""Calculate total feature dimension for one-hot encodin: Each attribute of length L contributes L slots for its legal values plus 1 extra slot reserved for the '?' missing-value indicator."""
def get_feature_dim():
    
    dim = 0
    for i in range(1, 23):          # skip index 0 (class label)
        dim += len(ATTRIBUTE_VALUES[i]) + 1   # +1 for the '?' slot
    return dim

# ------------------------------------------------------------------
""" Calculate total feature dimension once at import and store it to FEATURE_DIM, which is fixed thereafter. """
FEATURE_DIM = get_feature_dim() #129D here

# Python concept: list multiplication `[0.0] * n` creates n distinct references to the same immutable float — safe because floats are immutable. Never do this with mutable defaults.

# ------------------------------------------------------------------
""" One-hot encode a single mushroom instance.
    parts: list of symbolic tokens (e.g. ['p','x','s','n',...])
    include_class: True -> parts[0] is the class, features start at parts[1]; False -> parts[0] is the first feature, features start at parts[0].
    Returns a list of floats of length FEATURE_DIM with exactly one 1.0 per attribute block (or the '?' slot if the value is missing/unknown).
"""
def one_hot_encode(parts, include_class=False):
    
    encoded = [0.0] * FEATURE_DIM # a list of FEATURE_DIM elements 

    # `offset` walks through the big feature vector as we fill each block.
    offset = 0

    for i in range(1, 23):                       # attributes 1..22
        # #FIX 1: map attribute index -> position in `parts`.
        # With class:   parts[0]=class, parts[1..22]=features  -> idx = i
        # Without class: parts[0..21]=features                -> idx = i-1
        idx = i if include_class else i - 1

        values = ATTRIBUTE_VALUES[i]             # legal values for attribute i
        block_size = len(values) + 1             # +1 for the '?' slot

        # #FIX 1 (continued): out-of-range index means the line was truncated. Treat it as missing rather than leaving an all-zero block, which would silently lie to the model.
        if idx >= len(parts):
            value = '?'
        else:
            value = parts[idx]

        if value == '?':
            # Missing value gets its own dimension at the END of the block.
            encoded[offset + len(values)] = 1.0
        elif value in values:
            # `list.index` is O(L); L is tiny (<= 12), so no performance issue.
            encoded[offset + values.index(value)] = 1.0
        # else: unseen symbol -> leave block all zeros.
        #       (Could also be treated as '?'; left as-is to preserve the
        #        original semantics, with the comment making it explicit.)

        offset += block_size

    return encoded

# ------------------------------------------------------------------
"""load_data(): Load and one-hot encode mushroom data from a CSV-like text file.
    has_class=True  -> returns (list_of_features, list_of_labels in {-1,+1})
    has_class=False -> returns (list_of_features, [])   (no labels)

    Label convention: 'p' (poisonous) -> +1, 'e' (edible) -> -1.
    The +/-1 encoding is required by the perceptron update rule w += lr*y*x.
"""
def load_data(filepath, has_class=True):
    
    data = []
    labels = []

# Python concept: `with open(...) as f` uses the context-manager protocol(__enter__/__exit__) to guarantee the file is closed even on exception.
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()          # remove trailing '\n' and whitespace
            if not line:                 # skip blank lines
                continue
            parts = line.split(',')      # split into 23 tokens
            
            #For each single mushroom instance, append its label ('p' (poisonous) -> +1, 'e' (edible) -> -1) to label list; append its 22-features list(one_hot_encode) to data list
            if has_class:                # parts[0] is the class symbol
                label = 1 if parts[0] == 'p' else -1   
                features = one_hot_encode(parts, include_class=True)
                data.append(features)
                labels.append(label)
            else:                       # parts[0] is NOT the class symbol
                features = one_hot_encode(parts, include_class=False)
                data.append(features)
    return data, labels

# ------------------------------------------------------------------
# ------------------------------------------------------------------
class Perceptron:
    """Perceptron classifier from scratch.
    w: a vector of weights (one number per feature); 
    x: the feature vector
    lr: learning rate, set by convention/experience, controls how big each weight update is
    sign(z) = +1  if z ≥ 0; 
              −1  if z < 0
    
    Model:  f(x) = sign( w·x + b )
    
    Predict: ŷ = sign(w·x + b)
    If correct (ŷ == y): do nothing. If wrong (ŷ ≠ y): update:
    for a misclassified sample (x, y) with y in {-1, +1},
        w <-- w + lr * y * x
        b <-- b + lr * y
    
    Case A: y = +1 (truly poisonous) but model said −1.
    Then w·x + b < 0 — the model is on the wrong side. Update adds +lr·x to w. Since the active xᵢ = 1.0, the active weights increase. Thus the updated w·x will be larger (closer to positive), nudging the prediction toward +1. ✓

    Case B: y = −1 (truly edible) but model said +1.
    Now y = −1, so lr·y·x = −lr·x — we subtract lr·x from w. The active weights decrease, making w·x smaller next time, nudging toward −1. ✓
    
    This is the classic Rosenblatt perceptron. It converges iff the data is
    linearly separable; otherwise the weights can oscillate.
    """
    # ------------------------------------------------------------------
    def __init__(self, learning_rate=0.1, epochs=100): 
        """ Defines the constructor with two optional parameters (defaults 0.1 and 100). self is the object (untrained model) being created. Epoch = one complete pass over the entire training set """
        self.lr = learning_rate     # step size for weight updates
        self.epochs = epochs        # max number of passes over the data
        self.weights = None         # populated in fit()
        self.bias = 0.0
    
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Train the perceptron classifier model on feature matrix X (list of lists: n_mushrooms × 129-dim one-hot vector) and labels y."""

        #Guard against empty training data (X[0] would IndexError).
        if not X:
            raise ValueError("fit() received an empty training set")

        n_features = len(X[0]) #=129
        self.weights = [0.0] * n_features   # start from the origin
        self.bias = 0.0

        # Seed once per fit; reshuffle every epoch.
        rng = random.Random(42)

        # Best-epoch checkpointing: because the perceptron may oscillate on
        # non-separable data, remember the weights that gave the lowest
        # training error so far, and restore them at the end.
        best_weights = self.weights[:]   # `[:]` is a shallow copy (list of floats)
        best_bias = self.bias
        best_errors = float('inf')

        for epoch in range(self.epochs): # runs up to 100 times
            errors = 0

            #`list(range(n))` builds [0,1,...,n-1]; `shuffle` permutes in place.
            indices = list(range(len(X)))
            rng.shuffle(indices)

            for idx in indices:         # visit every training mushroom once per epoch
                x = X[idx]              # one mushroom's 22 features: a 129-dim one-hot vector (list of floats)
                target = y[idx]         # +1 or -1

                # Compute: activation = w·x =sum_i w_i * x_i
                activation = self.bias
                for i in range(n_features): # i run 0 to 129-1
                    activation += self.weights[i] * x[i] 

                # Predict: ŷ = sign(w·x + b)
                # `>= 0` convention: zero activation predicts +1.
                prediction = 1 if activation >= 0 else -1

                # Update: only on misclassification
                if prediction != target:
                    errors += 1
                    # `update` is lr * y, which is +lr for a false-negative (y=+1) and -lr for a false-positive (y=-1).
                    update = self.lr * target # lr * y
                    for i in range(n_features): # i run 0 to 129-1
                        self.weights[i] += update * x[i] # w <-- w + lr * y * x
                    self.bias += update # b <-- b + lr * y

            # Check if this epoch is the best so far.
            if errors < best_errors:
                best_errors = errors
                best_weights = self.weights[:]   # copy, not alias
                best_bias = self.bias

            # Perfect separation -> nothing more to learn.
            if errors == 0:
                print(f"  Converged at epoch {epoch + 1}")
                break

        # Restore the best-epoch parameters (see comment above).
        self.weights = best_weights
        self.bias = best_bias

        return self #  return a trained classifier model

    # ------------------------------------------------------------------
    def predict(self, X):
        """Predict class labels (+1 / -1) for each row of X."""
        predictions = []
        for x in X: # run all mushroom instances, each is a 129D vector
            activation = self.bias
            # use zip(x, weights) that stops at the shorter of the two, to avoid IndexError if a feature vector is malformed.
            for xi, wi in zip(x, self.weights):
                activation += wi * xi 
            predictions.append(1 if activation >= 0 else -1)
        return predictions

    # ------------------------------------------------------------------
    def score(self, X, y):
        """Compute accuracy = correct / total."""
        preds = self.predict(X) # 1-D lists of length n_mushrooms
        # `zip(preds, y)` pairs element-wise; generator expression sums 1s.
        correct = sum(1 for p, t in zip(preds, y) if p == t) 
        return correct / len(y)     # len(y) > 0 guaranteed by fit() guard


# ------------------------------------------------------------------
# ------------------------------------------------------------------
def split_data(data, labels, train_ratio=0.8):
    """Split data into training and testing sets"""
    combined = list(zip(data, labels)) # `list(zip(data, labels))`: pairs each feature vector with its label as a tuple, and store them in a list of tuples
    rng = random.Random(42) # fixed seed
    rng.shuffle(combined)   # shuffle pairs keeps them aligned

    split_idx = int(len(combined) * train_ratio)   # int() truncates toward 0
    train = combined[:split_idx]                   # 0-80% for training
    test = combined[split_idx:]                    # 80-100% for training

    train_data = [x[0] for x in train] # x[0]: feature vector. train_data is a list of feature vectors
    train_labels = [x[1] for x in train] # x[1]: label. train_labels is a list of labels
    test_data = [x[0] for x in test]
    test_labels = [x[1] for x in test]

    return train_data, train_labels, test_data, test_labels


# ------------------------------------------------------------------
# ------------------------------------------------------------------
def main():
    """
    1. Load 8000 training instances.
    2. Split 80/20.
    3. Train / test 80%
    4. Retrain on the full 8000.
    5. Predict 100 unknowns with the best k.
    6. Write predictionResultPER.txt and a small summary file.
    """
    print("=" * 66)
    print("Perceptron Mushroom Classifier")
    print("=" * 66)
    print(f"Feature dimension (one-hot): {FEATURE_DIM}")

    # 1. Load & inspect
    print("\nLoading training data...")
    data, labels = load_data('MushroomData_8000.txt', has_class=True)
    print(f"Loaded {len(data)} instances")
    print(f"Poisonous (1): {labels.count(1)}, Edible (-1): {labels.count(-1)}")

    # 2. Train / test split (80/20, fixed seed for reproducibility)
    train_data, train_labels, test_data, test_labels = split_data(
        data, labels, train_ratio=0.8
    )
    print(f"Training set: {len(train_data)}, Test set: {len(test_data)}")

    # 3. Hyperparameter grid search
    print("\nHyperparameter search:")
    print("-" * 66)
    print(f"{'LR':>6} {'Epochs':>8} {'Train Acc':>10} {'Test Acc':>10}")
    print("-" * 66)

    best_acc = 0.0
    best_config = None
    results = []          # list of (lr, epochs, train_acc, test_acc)

    
    for lr in [0.01, 0.05, 0.1, 0.5]: # a typical "log" spread from "small" to "large".
        """ tries 4 × 3 = 12 combinations, trains a fresh Perceptron for each, records train/test accuracy, and remembers the config with the best test accuracy (best_config). """    
        for epochs in [10, 50, 100]:
            perceptron = Perceptron(learning_rate=lr, epochs=epochs)
            perceptron.fit(train_data, train_labels)
            train_acc = perceptron.score(train_data, train_labels)
            test_acc = perceptron.score(test_data, test_labels)
            results.append((lr, epochs, train_acc, test_acc))
            print(f"{lr:>6.2f} {epochs:>8d} {train_acc:>10.4f} {test_acc:>10.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                best_config = (lr, epochs)

    print(f"\nBest config: LR={best_config[0]}, "
          f"Epochs={best_config[1]}, Test Acc={best_acc:.4f}")

    # 4. Retrain on the FULL 8000 instances (Tuning used only 6400) for final predictions for the unknowns
    print("\nTraining final model on full dataset(8000 instances) for final predictions...")
    full_data, full_labels = load_data('MushroomData_8000.txt', has_class=True)
    
    final_model = Perceptron(learning_rate=best_config[0], epochs=best_config[1]) #  using best_config learning_rate and epochs 
    final_model.fit(full_data, full_labels)
    print(f"Final TRAINING accuracy (not a generalization estimate): "
          f"{final_model.score(full_data, full_labels):.4f}")

    # 5. Predict the unknown file
    print("\nLoading unknown data...")
    unknown_data, _ = load_data('MushroomData_Unknwon_100.txt', has_class=False)
    print(f"Loaded {len(unknown_data)} unknown instances")

    predictions_raw = final_model.predict(unknown_data) # return a list of 1 or -1
    # Map numeric predictions back to the dataset's symbolic labels.
    predictions = ['p' if p == 1 else 'e' for p in predictions_raw]

    # write a trailing newline so the file is POSIX-clean.
    with open('predictionResultPER.txt', 'w') as f:
        f.write('\n'.join(predictions) + '\n')

    print(f"\nPredictions written to predictionResultPER.txt")
    print(f"Edible (e): {predictions.count('e')}, "
          f"Poisonous (p): {predictions.count('p')}")

    # 6. Persist the comparison table for the report (optional) ---
    with open('perceptron_results.txt', 'w') as f:
        f.write(f"Best Config: LR={best_config[0]}, Epochs={best_config[1]}\n")
        f.write(f"Test Accuracy: {best_acc:.4f}\n")
        f.write("\nHyperparameter comparison:\n")
        f.write(f"{'LR':>6} {'Epochs':>8} {'Train Acc':>10} {'Test Acc':>10}\n")
        for lr, epochs, train_acc, test_acc in results:
            f.write(f"{lr:>6.2f} {epochs:>8d} "
                    f"{train_acc:>10.4f} {test_acc:>10.4f}\n")


# Python concept: `if __name__ == '__main__':` ensures main() runs only when
# this file is executed directly, not when imported as a module.
if __name__ == '__main__':
    main()