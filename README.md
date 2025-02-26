# Dependencies

install required modules as follows:
```python
pip install -r requirements.txt
```

# How to Run

* step1: train vae model

set `data_dir` as data root dir (`'data/'` as default)
```python
python step1_train_vae.py --epochs 50 --batch_size 16 --data_dir [path_to_your_data]
```

* step2: generate data using vae model

```python
python step2_generate_env_data.py --num_generated 64
```

* step3: train classifier

use default data:
```python
python step3_train_classifier.py --epochs 50 --batch_size 16 --data_dir [path_to_your_data] --num_classes 3
```

use generated data (add `use_generator` augments):
```python
python step3_train_classifier.py --epochs 50 --batch_size 16 --data_dir [path_to_your_data] --num_classes 3 --use_generator
```

# Add new data

* add environment data

1. Put $i^{th}$ data in `data/experimentdata{i}`. Each data should have 32 txt files named from 1.txt to 32.txt.
2. Run `scripts/generate_p.py` to generate `data/p{i}`

* add classification data

1. Put $i^{th}$ data in `data/{i}`. Each txt file should have class name (like cola) in its filename.
2. If you add data belonging to a new class, add it to `class_to_index` in `utils/dataset.py`. For example,

```python
class_to_index = {
    'cola': 0,
    'oli': 1,
    'salt': 2,
    [class_name]: index.
}
```

<<<<<<< HEAD
3. The dataset will be split with a ratio of 8/2 to train/test by directory name.
=======
3. The dataset will be split with a ratio of 8/2 to train/test by directory name.
>>>>>>> 089bb2d (initial)
>>>>>>> 02f811d (merge conflict solved)
