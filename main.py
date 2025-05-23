import csv
import math
import random
from collections import Counter
from typing import Any, Dict, List, Union

def load_data(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        data = list(reader)

    # Konwersja typów
    # Załóżmy, że id, age, gender, cholesterol, gluc, smoke, alco, active, cardio to int
    # height, weight, ap_hi, ap_lo mogą mieć wartości z kropką dziesiętną i też można je sprowadzić do int przez zaokrąglenie
    # lub zostawić jako float.

    int_cols = ['id','age','gender','cholesterol','gluc','smoke','alco','active','cardio']
    float_cols = ['height','weight','ap_hi','ap_lo']

    for row in data:
        for col in int_cols:
            row[col] = int(float(row[col]))
        for col in float_cols:
            row[col] = float(row[col])
    return data


def discretize(data: List[Dict[str, Any]]):
    for row in data:
        row['age'] = row['age'] // 365
        

    def discretize_numeric(value, bins):
        # załóżmy, że bins = [b1, b2, b3, ...]
        # tworzymy kategorie: (-∞, b1], (b1,b2], ... , (bN, ∞)
        for i, b in enumerate(bins):
            if value <= b:
                return i
        return len(bins)

    age_bins = [40, 50, 60]
    height_bins = [155,165,175]
    weight_bins = [60,80,100]
    ap_hi_bins = [120,140,160]  # ciśnienie skurczowe kategorie
    ap_lo_bins = [80,90,100]    # ciśnienie rozkurczowe kategorie

    for row in data:
        row['age'] = discretize_numeric(row['age'], age_bins)
        row['height'] = discretize_numeric(row['height'], height_bins)
        row['weight'] = discretize_numeric(row['weight'], weight_bins)
        row['ap_hi'] = discretize_numeric(row['ap_hi'], ap_hi_bins)
        row['ap_lo'] = discretize_numeric(row['ap_lo'], ap_lo_bins)

    return data

def split_data(data: List[Dict[str, Any]], train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data


def entropy(examples: List[Dict[str,Any]]):
    label_counts = Counter(row['cardio'] for row in examples)
    total = len(examples)
    ent = 0.0
    for label, count in label_counts.items():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def information_gain(examples: List[Dict[str,Any]], attribute: str):
    total_entropy = entropy(examples)
    total = len(examples)
    subsets = {}
    for row in examples:
        val = row[attribute]
        subsets.setdefault(val, []).append(row)
    # OEntropia warunkowa
    cond_ent = 0.0
    for val, subset in subsets.items():
        cond_ent += (len(subset)/total)*entropy(subset)
    return total_entropy - cond_ent

def choose_best_attribute(examples: List[Dict[str,Any]], attributes: List[str]):
    best_attr = None
    best_gain = -1
    for attr in attributes:
        gain = information_gain(examples, attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    return best_attr


class DecisionNode:
    def __init__(self, attribute=None, children=None, is_leaf=False, prediction=None):
        self.attribute = attribute
        self.children = children if children is not None else {}
        self.is_leaf = is_leaf
        self.prediction = prediction

def majority_class(examples: List[Dict[str,Any]]):
    label_counts = Counter(row['cardio'] for row in examples)
    return label_counts.most_common(1)[0][0]

def id3(examples: List[Dict[str,Any]], attributes: List[str], max_depth: int):
    classes = [row['cardio'] for row in examples]
    if len(set(classes)) == 1:
        return DecisionNode(is_leaf=True, prediction=classes[0])

    if len(attributes) == 0 or max_depth == 0:
        return DecisionNode(is_leaf=True, prediction=majority_class(examples))

    best = choose_best_attribute(examples, attributes)
    node = DecisionNode(attribute=best, is_leaf=False)
    subsets = {}
    for row in examples:
        val = row[best]
        subsets.setdefault(val, []).append(row)

    new_attrs = [a for a in attributes if a != best]
    for val, subset in subsets.items():
        if len(subset) == 0:
            # Brak przykładów w tym podzbiorze
            node.children[val] = DecisionNode(is_leaf=True, prediction=majority_class(examples))
        else:
            node.children[val] = id3(subset, new_attrs, max_depth-1)
    return node

def predict(example: Dict[str,Any], tree: DecisionNode):
    if tree.is_leaf:
        return tree.prediction
    val = example[tree.attribute]
    if val in tree.children:
        return predict(example, tree.children[val])
    else:
        # Jeżeli brak gałęzi dla danej wartości atrybutu, zwracamy majority z tego węzła:
        return majority_class([example])

def accuracy(data: List[Dict[str,Any]], tree: DecisionNode):
    correct = 0
    for row in data:
        if predict(row, tree) == row['cardio']:
            correct += 1
    return correct / len(data)


# Główny program
if __name__ == "__main__":
    data = load_data("cardio_train.csv")
    data = discretize(data)
    train_data, val_data, test_data = split_data(data)

    attributes = [col for col in train_data[0].keys() if col not in ['id','cardio']]

    # Strojenie parametru maksymalnej głębokości:
    best_depth = None
    best_val_acc = -1
    for depth in range(1,21):
        tree = id3(train_data, attributes, max_depth=depth)
        val_acc = accuracy(val_data, tree)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_depth = depth

    # Budujemy finalne drzewo z wybraną głębokością
    final_tree = id3(train_data, attributes, max_depth=best_depth)
    test_acc = accuracy(test_data, final_tree)

    print(f"Najlepsza maksymalna głębokość: {best_depth}")
    print(f"Wynik na zbiorze walidacyjnym: {best_val_acc}")
    print(f"Wynik na zbiorze testowym: {test_acc}")
