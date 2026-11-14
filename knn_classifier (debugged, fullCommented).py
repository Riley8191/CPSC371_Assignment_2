"""
KNN Classifier for Mushroom Classification
Implemented from scratch - no sklearn or similar libraries

BUGS FIXED FROM THE ORIGINAL UPLOAD:
  1. encode_instance — training vectors were length 23 (the class was
     included as a feature); unknown vectors were length 21 (cap-shape was
     dropped and every attribute was off by one). Both are now length 22
     and correctly aligned.
  2. hamming_distance — added an assert so a length mismatch crashes loudly
     at the point of failure instead of silently truncating and returning
     a wrong answer.
  3. Removed `import math` — it was never used.
"""

# `import random` — module used for `random.seed(...)` and `random.shuffle(...)`.
# Using `import X` (rather than `from X import Y`) keeps the namespace
# explicit: we always write `random.shuffle(...)`, never bare `shuffle(...)`.
import random

# `from collections import Counter` — Counter is a standard-library dict
# subclass that counts occurrences. It is a generic counting utility, NOT
# a KNN/ML library, so it is allowed under the assignment rules.
# We import it by name so we can write `Counter(...)` instead of
# `collections.Counter(...)`.
from collections import Counter


# ---------------------------------------------------------------------------
# ATTRIBUTE VALUES LOOKUP TABLE
# ---------------------------------------------------------------------------
# A dict mapping attribute position (int) -> list of allowed values (chars)
# Storing 2-D list as dict lets us look up the allowed values for any attribute
# Index 0 is the CLASS (target). Indices 1..22 are the 22 FEATURES.
# with O(1) dict access: ATTRIBUTE_VALUES[attr_index]: list of allowed values of the attr
ATTRIBUTE_VALUES = {
    0: ['e', 'p'],  # class (target) — NOT used as a feature vector entry
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


def encode_attribute(value, attr_index):
    """
    Encode a single categorical value as an integer index.
    Returns -1 for missing ('?') or unknown values.
    """
    # Missing value '?' is a sentinel case, not a real category. Return
    # -1 so it can never collide with a real category index (0..N-1).
    if value == '?':
        return -1

    # Look up the list of allowed values for this attribute.
    # e.g. attr_index=1 gives ['b','c','x','f','k','s'].
    values = ATTRIBUTE_VALUES[attr_index]

    # list.index(x) returns the position of x, or raises ValueError if x
    # is not present. Set Returns -1 to avoid the exception.
    if value in values:
        return values.index(value)  # e.g. 'c' in cap-shape -> 1
    return -1


# ---------------------------------------------------------------------------
# BUG FIX #1: encode_instance
# ---------------------------------------------------------------------------
# ORIGINAL (buggy) version:
#     start = 0 if include_class else 1
#     for i in range(start, len(parts)):
#         encoded.append(encode_attribute(parts[i], i))
#
#   include_class=True  -> loop i=0..22  -> 23 features (class as feature!)
#   include_class=False -> loop i=1..21  -> 21 features (cap-shape dropped,
#                                          every attribute off by one)
#
#   So training vectors were length 23, unknown vectors were length 21,
#   and even the attributes that were present were aligned to the wrong
#   ATTRIBUTE_VALUES entry. Every Hamming distance was garbage, so KNN
#   returned essentially random predictions (~52% accuracy).
#
# FIXED version below: both branches produce length-22 vectors.
# ---------------------------------------------------------------------------
def encode_instance(parts, include_class=True):
    """
    Encode a mushroom row into a length-22 feature vector.

    include_class=True  : parts = [class, attr1, ..., attr22]  (training)
                          skip parts[0]; parts[i] -> ATTRIBUTE_VALUES[i]
    include_class=False : parts = [attr1, ..., attr22]         (unknown)
                          parts[i]    -> ATTRIBUTE_VALUES[i + 1]
    """
    # Accumulator for the encoded integers (one per attribute).
    encoded = []

    if include_class:
        # Training row: parts[0] is the class label (handled separately in
        # load_data into `labels`). Attributes live at parts[1..22], and
        # attribute position i corresponds to ATTRIBUTE_VALUES[i].
        for i in range(1, len(parts)):              # i = 1, 2, ..., 22
            encoded.append(encode_attribute(parts[i], i))
    else:
        # Unknown row: there is no class column. parts[0] is cap-shape,
        # which is attribute 1, so we use attribute index i + 1.
        for i in range(len(parts)):                 # i = 0, 1, ..., 21
            encoded.append(encode_attribute(parts[i], i + 1))

    return encoded  # length 22 in both branches


def load_data(filepath, has_class=True):
    """
    Load a mushroom CSV file.

    Returns (data, labels):
      - data   : list of length-22 encoded feature vectors
      - labels : list of class strings ('e'/'p'); empty when has_class=False
    """
    # Two parallel lists: one for encoded features, one for labels.
    data = []
    labels = []

    # `with open(...)` guarantees f.close() even if an exception is raised
    # inside the block. 'r' = read mode. `as f` binds the file object to `f`.
    with open(filepath, 'r') as f:
        # Iterate line by line
        for line in f:
            # Remove the trailing '\n' and any extra whitespace.
            line = line.strip()

            #  Skip blank lines.
            if not line:
                continue

            # Split "p,x,s,n,..." into ['p','x','s','n',...].
            parts = line.split(',')

            if has_class:
                # Training file: parts[0] is the class label.
                labels.append(parts[0])
                # Encode parts[1..22] into a length-22 vector.
                features = encode_instance(parts, include_class=True)
            else:
                # Unknown file: parts[0..21] are attributes 1..22.
                features = encode_instance(parts, include_class=False)

            # Append the feature vector in both branches.
            data.append(features)

    # Return the two lists; the caller decides what to do with them.
    return data, labels


# ---------------------------------------------------------------------------
# BUG FIX #2: hamming_distance
# ---------------------------------------------------------------------------
# ORIGINAL (silent failure):
#     dist = 0
#     for i in range(len(a)):
#         if a[i] != b[i]:
#             dist += 1
#     return dist
#
#   If len(a) != len(b), the loop stops at len(a) and never compares the
#   extra elements of b. No exception, but a WRONG answer. That silent
#   truncation is what hid the length-23 vs length-21 bug above.
#
# FIXED: assert len(a) == len(b) so mismatch crashes loudly. Then use
#   `sum(... for x, y in zip(a, b) if x != y)` — compact, and zip is safe
#   because the assert guarantees equal lengths.
# ---------------------------------------------------------------------------
def hamming_distance(a, b):
    """
    Return the number of positions where two encoded instances disagree.
    Raises AssertionError if the two vectors have different lengths.
    """
    # Assert-with-message: on failure, raises AssertionError carrying the f-string.
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"

    # sum(...) over a generator: 1 for each pair where x != y, 0 otherwise.
    # `zip` pairs a[i] with b[i] for i = 0..len-1 (lengths already verified).
    return sum(1 for x, y in zip(a, b) if x != y)


def knn_predict(train_data, train_labels, test_instance, k):
    """
    Predict the class of `test_instance` using KNN with the given k.

    Algorithm:
      1. Compute Hamming distance to every training point.
      2. Sort ascending; take the k nearests.
      3. Count the labels (e, p) in the k nearests and Majority vote; ties prefer 'p' for safety.
    """
    # List of (distance, label) tuples — one per training instance.
    distances = []

    # enumerate gives both index (i) and element (train_instance). The index
    # is needed to look up the corresponding label in train_labels.
    for i, train_instance in enumerate(train_data):
        d = hamming_distance(test_instance, train_instance)
        # Store the distance with its label so they stay paired through sort.
        distances.append((d, train_labels[i]))

    # Sort in place, ascending, by the first tuple element (the distance).
    # `lambda x: x[0]` is a small anonymous function returning x's first field.
    distances.sort(key=lambda x: x[0])

    # Slice [:k] returns a new list of the first k tuples (d, train_labels[i]), which are k train_instances with smallest distance. e.g.if k=5, it gives [(0,'e'), (1,'p'), (1,'e'), (2,'p'), (2,'e')]
    k_nearest = distances[:k]

    #[label for _, label in k_nearest] extract just the labels in the k_nearest list: e.g. if k=5, it gives: ['e', 'p', 'e', 'p', 'e']
    #Counter() is a subclass of dict: e.g. Counter(['e', 'p', 'e', 'p', 'e']) gives Counter({'e': 3, 'p': 2}). 
    votes = Counter([label for _, label in k_nearest])

    # Tie-breaking (only possible with even k). 
    # max(votes.values()) finds the highest count of 'e' and 'p'in the votes dict, return a number. e.g. Trace with Counter({'e': 3, 'p': 2}), max_count = 3.
    if len(votes) > 1:
        max_count = max(votes.values()) 
        # Builds a list of every label whose count equals max_count.e.g. Trace with Counter({'e': 3, 'p': 2}) and max_count = 3,tied = ['e']
        tied = [label for label, count in votes.items() if count == max_count]

    # If poisonous is in the tied list, return poisonous — regardless of whether it's the sole winner or tied with edible.
    if 'p' in tied:
        return 'p'

    # if len(votes) == 1, then no tie
    # most_common(1) returns [('label', count)]; 
    # [0][0] extracts the label string.
    return votes.most_common(1)[0][0]


def split_data(data, labels, train_ratio=0.8):
    """
    Shuffle and split features/labels into 80% train + 20% test sets.
    Returns four parallel lists: train_data, train_labels, test_data, test_labels.
    """
    # Pair feature vector (a length-22 list of integers) of each train and test data instance with its label BEFORE shuffling so they can never
    # list(zip(data, labels)) returns a a list of tuples (feature vector, label)
    combined = list(zip(data, labels))

    # Fixed random number generator => same sequence of "random" numbers => same shuffle every run => the same train/test split=> same reproducible results.
    random.seed(42)

    # random.shuffle: reorders the elements in place — modify the original list by reordering  elements, returns None.
    random.shuffle(combined)

    # Index at which to split; int() truncates the float;
    split_idx = int(len(combined) * train_ratio)
    # slicing by [a:] and [:b]
    train = combined[:split_idx]
    test = combined[split_idx:]

    # Unpack each (feature vector, label) tuple back into parallel lists, 4 lists in total.
    train_data = [x[0] for x in train]
    train_labels = [x[1] for x in train]
    test_data = [x[0] for x in test]
    test_labels = [x[1] for x in test]

    return train_data, train_labels, test_data, test_labels


def evaluate_knn(train_data, train_labels, test_data, test_labels, k):
    """
    Return the accuracy of KNN with the given k on the test set.
    Accuracy = (# correctly predicted) / (total test instances).
    """
    correct = 0  # counter for correct predictions

    # Loop over every test instance.
    for i, test_instance in enumerate(test_data):
        pred = knn_predict(train_data, train_labels, test_instance, k)
        # Compare the predicted label to the actual label.
        if pred == test_labels[i]:
            correct += 1
            
    # returns a fraction in [0,1].
    return correct / len(test_data)


def main():
    """
      1. Load 8000 training instances.
      2. Split 80/20.
      3. Tune k on the test half.
      4. Retrain on the full 8000.
      5. Predict 100 unknowns with the best k.
      6. Write predictionResultKNN.txt and a small summary file.
    """
    # Cosmetic header:
    print("=" * 66)
    print("KNN Mushroom Classifier")
    print("=" * 66)

    # --- Load training data ---
    print("\nLoading training data...")
    # Each item in `data` is a length-22 vector; `labels` is 'e' or 'p'. 'load_data()' return two lists: data, labels
    data, labels = load_data('MushroomData_8000.txt', has_class=True)
    print(f"Loaded {len(data)} instances")  # Loaded 8000 instance

    # --- Split into 80/20 train/test ---
    # Tuple unpacking: split_data returns 4 lists; bind each to a name.
    train_data, train_labels, test_data, test_labels = split_data(
        data, labels, train_ratio=0.8
    )
    print(f"Training set: {len(train_data)} instances")  # 6400
    print(f"Test set: {len(test_data)} instances")       # 1600

    # --- Try different K values, keep the best ---
    print("\nEvaluating different K values:")
    print("-" * 44)

    best_k = 1                # placeholder; updated as we find better
    best_acc = 0.0            # worst accuracy; updated as we find better
    results = []              # to print/save the comparison table later

    # All odd k values to avoid top-vote ties (with even k, a
    # 2-2 split of two labels is possible).
    for k in [1, 3, 5, 7, 9, 11, 15, 21]:
        #evaluate_knn() returns a fraction; with k as a tuple to storen in the results list
        acc = evaluate_knn(train_data, train_labels, test_data, test_labels, k) 
        results.append((k, acc))
        # {k:2d} right-aligns k in 2 chars; {acc:.4f} shows 4 decimals;
        # {acc*100:.2f} shows percent with 2 decimals.
        print(f"K={k:2d}: Accuracy = {acc:.4f} ({acc*100:.2f}%)")
        if acc > best_acc:                  # strictly better => update
            best_acc = acc
            best_k = k

    print(f"\nBest K = {best_k} with accuracy = {best_acc:.4f}")

    # --- Retrain on the FULL 8000 instances (Tuning used only 6400) for final predictions for the unknowns
    print("\nTraining on full dataset (8000 instances) for final predictions...")
    full_data, full_labels = load_data('MushroomData_8000.txt', has_class=True)

    # --- Load the 100 unknown mushrooms ---
    print("Loading unknown data...")
    # `_` is an explicit placeholder; unknown files have no labels, so the returned `labels` list is empty.
    unknown_data, _ = load_data('MushroomData_Unknwon_100.txt', has_class=False)
    print(f"Loaded {len(unknown_data)} unknown instances") #Loaded 100 unknown instances

    # --- Predict every unknown instance using the best k ---
    print(f"\nMaking predictions with K={best_k}...")
    predictions = []
    for i, test_instance in enumerate(unknown_data):
        pred = knn_predict(full_data, full_labels, test_instance, best_k)
        predictions.append(pred)

        #  Progress every 20 instances, print out processed ratio so a slow run does not look frozen.
        # `(i + 1) % 20 == 0` is True when i+1 is 20, 40, 60, ...
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(unknown_data)}")

    # --- Write the required prediction file ---
    # Format: exactly 100 lines, each 'e' or 'p', one per unknown mushroom.
    # 'w' mode OVERWRITES an existing file; 'with' ensures it is closed and flushed even if an error occurs.
    with open('predictionResultKNN.txt', 'w') as f:
        for pred in predictions:
            f.write(pred + '\n')

    print(f"\nPredictions written to predictionResultKNN.txt")
    # `.count('e')` and `.count('p')` count occurrences in a list.
    print(f"Edible (e): {predictions.count('e')}, Poisonous (p): {predictions.count('p')}")

    # --- Save a small summary file for the report (optional) ---
    with open('knn_best_k.txt', 'w') as f:
        f.write(f"Best K: {best_k}\n")
        f.write(f"Test Accuracy: {best_acc:.4f}\n")
        f.write("\nK-value comparison:\n")
        # `for k, acc in results` unpacks each (k, acc) tuple.
        for k, acc in results:
            f.write(f"K={k}: {acc:.4f}\n")

if __name__ == '__main__':
    main()