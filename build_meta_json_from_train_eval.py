#!/usr/bin/env python3
import argparse
import ast
import json
import os
import random
import sys
from collections import Counter


_MISSING = object()


def _literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _literal(node.operand)
        if isinstance(val, (int, float)):
            return -val
        return _MISSING
    if isinstance(node, ast.List):
        items = []
        for elt in node.elts:
            val = _literal(elt)
            if val is _MISSING:
                return _MISSING
            items.append(val)
        return items
    if isinstance(node, ast.Tuple):
        items = []
        for elt in node.elts:
            val = _literal(elt)
            if val is _MISSING:
                return _MISSING
            items.append(val)
        return tuple(items)
    if isinstance(node, ast.Dict):
        data = {}
        for key_node, val_node in zip(node.keys, node.values):
            key = _literal(key_node)
            val = _literal(val_node)
            if key is _MISSING or val is _MISSING:
                return _MISSING
            data[key] = val
        return data
    return _MISSING


def _extract_my_config(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "my_config":
                if isinstance(node.value, ast.Dict):
                    config = {}
                    for key_node, val_node in zip(node.value.keys, node.value.values):
                        key = _literal(key_node)
                        val = _literal(val_node)
                        if isinstance(key, str) and val is not _MISSING:
                            config[key] = val
                    return config
    return {}


def _extract_category_list(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "category_list":
                val = _literal(node.value)
                if isinstance(val, list) and all(isinstance(x, str) for x in val):
                    return val
    return None


def _extract_meta_data_path(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "open":
            continue
        if not node.args:
            continue
        path = _literal(node.args[0])
        if isinstance(path, str) and path.endswith("meta_data.json"):
            return path
    return None


def _resolve_meta_path(meta_path, fallback_path):
    if meta_path and os.path.exists(meta_path):
        return meta_path

    if meta_path:
        parts = meta_path.split("/")
        if "Infinity_v3" in parts:
            idx = parts.index("Infinity_v3")
            rel_path = os.path.join(*parts[idx:])
            if os.path.exists(rel_path):
                return rel_path

    if fallback_path and os.path.exists(fallback_path):
        return fallback_path

    raise FileNotFoundError(
        f"Could not resolve meta_data.json path. Tried: {meta_path!r}, {fallback_path!r}"
    )


def _nearly_uniform_counts(total, classes):
    base = total // classes
    rem = total % classes
    return [base + (1 if i < rem else 0) for i in range(classes)]


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def _ensure_unique_ids(entries, label):
    ids = [e["id"] for e in entries]
    dupes = [item for item, count in Counter(ids).items() if count > 1]
    if dupes:
        sample = ", ".join(dupes[:5])
        raise ValueError(f"{label} has duplicate ids (sample): {sample}")


def main():
    parser = argparse.ArgumentParser(
        description="Build train/eval meta_json files from Infinity_v3/train.py selection logic."
    )
    parser.add_argument("--train-py", default="Infinity_v3/train.py")
    parser.add_argument("--meta-data", default=None)
    parser.add_argument("--meta-data-3class", default="Infinity_v2/meta_data_3class.json")
    parser.add_argument("--train-out", default="Infinity_v3/evaluation/MJHQ30K/meta_data_train.json")
    parser.add_argument("--eval-out", default="Infinity_v3/evaluation/MJHQ30K/meta_data_eval.json")
    parser.add_argument("--test-out", default="Infinity_v2/meta_data_test.json")
    args = parser.parse_args()

    with open(args.train_py, "r", encoding="utf-8") as f:
        train_src = f.read()
    tree = ast.parse(train_src)

    my_config = _extract_my_config(tree)
    category_list = _extract_category_list(tree)
    meta_data_path = _extract_meta_data_path(tree)

    if not category_list:
        raise ValueError("Failed to parse category_list from train.py.")

    required_keys = [
        "training_prompt_pool_size",
        "eval_prompt_pool_size",
        "num_training_classes",
        "prompt_shuffle_seed",
    ]
    for key in required_keys:
        if key not in my_config:
            raise ValueError(f"Missing my_config['{key}'] in train.py.")

    num_classes = int(my_config["num_training_classes"])
    if num_classes != len(category_list):
        raise ValueError(
            f"num_training_classes={num_classes} must match len(category_list)={len(category_list)}."
        )

    train_pool_size = int(my_config["training_prompt_pool_size"])
    eval_pool_size = int(my_config["eval_prompt_pool_size"])
    seed = int(my_config["prompt_shuffle_seed"])

    meta_data_path = _resolve_meta_path(meta_data_path or args.meta_data, "Infinity_v3/evaluation/MJHQ30K/meta_data.json")
    meta_data = _load_json(meta_data_path)

    prompts_by_cat = {c: [] for c in category_list}
    for img_id, data in meta_data.items():
        cats = data.get("category", [])
        if isinstance(cats, str):
            cats_list = [cats]
        else:
            cats_list = list(cats)

        prompt = data.get("prompt", None)
        if not prompt:
            continue

        for c in category_list:
            if c in cats_list:
                prompts_by_cat[c].append({"id": img_id, "prompt": prompt, "category": c})

    rng = random.Random(seed)
    for c in category_list:
        rng.shuffle(prompts_by_cat[c])

    per_class_train = _nearly_uniform_counts(train_pool_size, num_classes)
    per_class_eval = _nearly_uniform_counts(eval_pool_size, num_classes)

    train_entries = []
    eval_entries = []
    for i, c in enumerate(category_list):
        need = per_class_train[i] + per_class_eval[i]
        have = len(prompts_by_cat[c])
        if have < need:
            raise ValueError(
                f"Not enough prompts for category '{c}': need {need}, have {have}."
            )
        train_entries.extend(prompts_by_cat[c][:per_class_train[i]])
        eval_entries.extend(prompts_by_cat[c][per_class_train[i]:per_class_train[i] + per_class_eval[i]])

    if len(train_entries) != train_pool_size:
        raise ValueError(f"Train count mismatch: {len(train_entries)} != {train_pool_size}.")
    if len(eval_entries) != eval_pool_size:
        raise ValueError(f"Eval count mismatch: {len(eval_entries)} != {eval_pool_size}.")

    _ensure_unique_ids(train_entries, "Train set")
    _ensure_unique_ids(eval_entries, "Eval set")

    train_meta = {
        entry["id"]: {"prompt": entry["prompt"], "category": entry["category"]}
        for entry in train_entries
    }
    eval_meta = {
        entry["id"]: {"prompt": entry["prompt"], "category": entry["category"]}
        for entry in eval_entries
    }

    _write_json(args.train_out, train_meta)
    _write_json(args.eval_out, eval_meta)

    train_prompts = [e["prompt"] for e in train_entries]
    eval_prompts = [e["prompt"] for e in eval_entries]
    train_prompt_set = set(train_prompts)
    eval_prompt_set = set(eval_prompts)
    overlap_prompts = train_prompt_set & eval_prompt_set

    meta_3class = _load_json(args.meta_data_3class)
    train_eval_prompt_set = train_prompt_set | eval_prompt_set

    test_meta = {}
    for img_id, data in meta_3class.items():
        prompt = data.get("prompt")
        if prompt in train_eval_prompt_set:
            continue
        test_meta[img_id] = data

    _write_json(args.test_out, test_meta)

    test_prompt_set = {data.get("prompt") for data in test_meta.values() if data.get("prompt")}
    overlap_test = test_prompt_set & train_eval_prompt_set

    train_counts = Counter(e["category"] for e in train_entries)
    eval_counts = Counter(e["category"] for e in eval_entries)

    print("=== Verification ===")
    print(f"meta_data source: {meta_data_path}")
    print(f"train meta_json: {args.train_out} ({len(train_entries)} prompts)")
    print(f"eval meta_json: {args.eval_out} ({len(eval_entries)} prompts)")
    print(f"meta_data_3class source: {args.meta_data_3class}")
    print(f"test meta_json: {args.test_out} ({len(test_meta)} prompts)")
    print("train per-category:", dict(train_counts))
    print("eval per-category:", dict(eval_counts))

    if overlap_prompts:
        print(f"WARNING: train/eval prompt overlap count: {len(overlap_prompts)}", file=sys.stderr)
    if overlap_test:
        raise ValueError(f"test meta_json overlaps train/eval prompts: {len(overlap_test)}")


if __name__ == "__main__":
    main()
