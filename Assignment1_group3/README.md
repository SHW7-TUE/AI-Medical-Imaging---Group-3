# README — Assignment Notebook (NanoGPT on Open-MELON captions)

## Purpose
This notebook trains a small **character-level NanoGPT** (decoder-only Transformer) on captions from the **Open-MELON-VL-2.5K** dataset and contains the required experiments for **Exercises 1–5**. All written answers are in the section:

**“⚠️ Everything below this line must be submitted…”**

All original code is still in the notebook. Extra code and text blocks are added with clear captions

---

## Requirements
Python packages used:
- numpy
- datasets
- torch
- tqdm
- matplotlib

If needed:
pip install numpy datasets torch tqdm matplotlib

If you change the preprocessing you must re-run:
Character-level Tokenizer → Encode/Split → Model initialization → Training loop → Generation


## GPU or CPU
In the notebook are clear instructions to check if you use local GPU or CPU. Instructions are provided how to install CUDA, so you can run it on GPU

## Exercise 1:
Run the code block **"Preprocess the Training Text"** if you want: Join separate captions with `<ENDC>` separator. This helps the model learn boundaries.
Run the code block **"Exercise 1: Training without `<ENDC>` separator"**: Run the cell below **instead of** the `<ENDC>` corpus cell above to train a model without explicit caption boundary tokens.

## Exercise 2:
After a baseline training run, the notebook contains:
- **Exercise 2a**: learning rate sweep
- **Exercise 2b**: batch size sweep
- **Exercise 2c**: evaluation interval sweep

Each sweep:
retrains fresh models, records train/val loss at evaluation steps, plots train (solid) and val (dashed) in one figure. Run only one sweep cell at a time, because each one trains multiple models and can take several minutes.

## Exercise 3:
Choose one of the following cells (do not run both):
-**Exercise 3: Lowercase corpus**: create all lowercase
-**Exercise 3: Uppercase corpus**: creates all uppercase
After choosing one, re-run from Character-level Tokenizer onward.
Run the "Let's print some information about the vocabulary that we have created as well as some examples of encoding words." That belongs to exercise 3 to get no error.

## Exercise 5:
After training the baseline model, run the Exercise 5 cell:
temperatures: [0.3, 0.7, 1.5]
top_k values: [10, 60]
10 examples per configuration

This prints the required 10 generated captions per hyperparameter configuration.