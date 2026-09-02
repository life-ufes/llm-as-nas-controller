# LLM as NAS Controller

Neural Architecture Search (NAS) for multimodal skin lesion classification, using **local LLMs served via [Ollama](https://ollama.com/)** as the search controller. At each step, the LLM receives the search space and the history of results, proposes a new architecture configuration in JSON, the model is trained, and the resulting *Balanced Accuracy* (BACC) is returned to the LLM as a reward to guide the next proposals.

Project developed during the master's program (PPGI/UFES).

## How it works

1. A prompt is built with the **search space** and the **history** of already-evaluated configurations (`full`, `last_k`, or `top_k` modes).
2. The prompt is sent to Ollama (`POST /api/generate`, with `format: json` for compatible models such as `qwen*` and `gpt-oss*`) — see [request_to_llm.py](src/scripts/benchmark/utils/request_to_llm.py).
3. The response is filtered (removal of `<think>`, extraction of the first valid JSON) and validated with **Pydantic** ([pydantic_llm_response_formats.py](src/scripts/benchmark/models/pydantic_llm_response_formats.py)). Invalid or repeated configurations are discarded.
4. The valid configuration instantiates a dynamic multimodal CNN ([dynamicMultimodalmodel.py](src/scripts/benchmark/models/dynamicMultimodalmodel.py)), which is trained with *early stopping* and evaluated on the validation set.
5. The BACC becomes the reward recorded in the history, and the cycle repeats for `SEARCH_STEPS` steps. Everything is logged to **MLflow** and to CSV/JSON in the results folder.

## Search space

| Hyperparameter | Values |
| --- | --- |
| `num_blocks` | 2, 5, 10 |
| `initial_filters` | 16, 32, 64 |
| `kernel_size` | 3, 5 |
| `layers_per_block` | 1, 2 |
| `use_pooling` | true, false |
| `common_dim` | 64, 128, 256, 512 |
| `attention_mechanism` | `no-metadata`, `concatenation`, `crossattention`, `metablock` |
| `num_layers_text_fc` | 1, 2, 3 |
| `neurons_per_layer_size_of_text_fc` | 64, 128, 256, 512 |
| `num_layers_fc_module` | 1, 2 |
| `neurons_per_layer_size_of_fc_module` | 256, 512 |

## Project structure

```text
conf/
  .env                  # environment variables (create from .env.test)
src/scripts/
  benchmark/
    nas/                # search and final training scripts
      optimization_train_process_pad_20_llm-as-controller.py   # NAS with LLM as controller
      optimization_train_process_pad_20_using_random-search.py # baseline: random search
      optimization_train_process_pad_20.py                     # exhaustive/grid search
      train_pad_20_optimized_model.py                          # final training (PAD-UFES-20)
      train_isic_2019_optimized_model.py                       # final training (ISIC-2019)
      train_milk10k_optimized_model.py                         # final training (MILK-10k)
      calculate_flops.py                                       # FLOPs/parameters of the models
      run_script_via_bash.sh                                   # launches the search in the background
    models/             # datasets, dynamic CNN, attention mechanisms (MetaBlock, MetaNet, cross-attention), focal loss, Pydantic schemas
    utils/              # Ollama client, LLM response filtering, metrics, early stopping, experiment logs
    interpretability/   # Grad-CAM, Grad-CAM++, Score-CAM, flip rate, uncertainty analysis
    plots/              # result charts, confusion matrices, GIFs
  data_preprocessing/   # preprocessing (ISIC-2019, PAD-UFES-20), data augmentation, LIME
  aggreation/           # metric aggregation and statistical tests (Wilcoxon)
```

## Requirements and installation

Create the virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision pydantic mlflow scikit-learn pandas numpy requests python-dotenv tqdm Pillow
```

> A GPU with CUDA is recommended for model training.

In addition, **Ollama** must be running locally at `http://localhost:11434`, with the controller model downloaded:

```bash
ollama pull qwen3:0.6b   # or another model (qwen*, gpt-oss* support format=json and thinking)
```

- Dataset (e.g., [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)) with the following structure:

```text
<DATASET_FOLDER_PATH>/
  metadata.csv
  images/
```

## Configuration

Create `conf/.env` (use [conf/.env.test](conf/.env.test) as a template):

```env
NUM_EPOCHS=100
BATCH_SIZE=32
K_FOLDS=5
LIST_NUM_HEADS=[8]
COMMON_DIM=512
DATASET_FOLDER_NAME="PAD-UFES-20"
DATASET_FOLDER_PATH="/path/to/PAD-UFES-20"
RESULTS_FOLDER_PATH="./src/results"
UNFREEZE_WEIGHTS=False
LLM_MODEL_NAME_SEQUENCE_GENERATOR="qwen3:0.6b"   # Ollama model used as controller
HISTORY_MODE="full"                              # full | last_k | top_k
SEARCH_STEPS=500                                 # number of search steps
```
#### Environment variables explained

- `NUM_EPOCHS`: Number of training epochs per fold/model. Higher values usually improve convergence, but increase runtime.
- `BATCH_SIZE`: Number of samples per gradient update. Larger batches use more GPU memory.
- `K_FOLDS`: Number of folds used in cross-validation.
- `LIST_NUM_HEADS`: Number of attention heads in multimodal fusion modules.
- `COMMON_DIM`: Size of the shared latent projection space used to align modalities.
- `DATASET_FOLDER_NAME`: Dataset folder identifier (used in naming/organization).
- `DATASET_FOLDER_PATH`: Full path to the dataset root directory used by training scripts.
- `save_to_disk`: If `True`, saves trained model weights and artifacts.
- `RESULTS_FOLDER_PATH`: Output path where metrics, logs, and saved artifacts are written.

## Running

With the virtual environment activated, run the NAS search with the LLM as controller via the `.sh` script (the process is launched in the background, with logging to `logs/`):

```bash
bash ./src/scripts/benchmark/nas/run_script_via_bash.sh
```

Or run the Python script directly:

```bash
python3 ./src/scripts/benchmark/nas/optimization_train_process_pad_20_llm-as-controller.py
```

Random search baseline, for comparison:

```bash
python3 ./src/scripts/benchmark/nas/optimization_train_process_pad_20_using_random-search.py
```

Final training of the best architecture found:

```bash
python3 ./src/scripts/benchmark/nas/train_pad_20_optimized_model.py
```

## Results

Results are written to `RESULTS_FOLDER_PATH/<HISTORY_MODE>/<thinking>/controller-<llm>/<dataset>/...`, including the search history (JSON/CSV), the best configuration found, and per-step metrics. Experiments are also tracked in MLflow (`mlflow ui` to visualize), with parameters such as `search_space`, `history_mode`, and `final_best_reward`.

## Supplementary material

This Supplementary material section reports the configurations of the top-10 architectures identified by the NAS process, ranked in descending order according to the A-TOPSIS multi-criteria decision-making score. These results provide additional transparency regarding the architectural trade-offs explored during search and support the selection of the final model (ID~21).

| Rank | ID | History | CNN Blocks | Init. Filters | Kernel | Layers/Block | Fusion | Fusion Dim | Classifier MLP |
|------|----|---------|------------|---------------|--------|--------------|--------|------------:|---------------:|
| 1    | 21 | TOP-10-BACC | 10 | 64 | 3 | 2 | MetaBlock | 512 | 2 × 512 |
| 2    | 23 | TOP-10-BACC | 5  | 64 | 3 | 1 | MetaBlock | 256 | 1 × 512 |
| 3    | 11 | LAST-10     | 5  | 32 | 3 | 2 | MetaBlock | 512 | 1 × 512 |
| 4    | 7  | FULL        | 5  | 64 | 3 | 2 | MetaBlock | 256 | 1 × 512 |
| 5    | 5  | FULL        | 10 | 64 | 5 | 1 | MetaBlock | 512 | 2 × 512 |
| 6    | 2  | FULL        | 2  | 32 | 5 | 2 | MetaBlock | 512 | 2 × 512 |
| 7    | 3  | FULL        | 5  | 64 | 3 | 2 | MetaBlock | 512 | 2 × 256 |
| 8    | 19 | TOP-10-BACC | 5  | 32 | 3 | 1 | MetaBlock | 128 | 1 × 512 |
| 9    | 15 | LAST-10     | 5  | 64 | 3 | 2 | Cross-Attention | 512 | 1 × 256 |
| 10   | 18 | TOP-10-BACC | 2  | 16 | 5 | 2 | MetaBlock | 128 | 2 × 512 |

Across the top-10 ranked architectures, MetaBlock emerges as the dominant fusion mechanism, appearing in nearly all high-performing solutions (Appendix~A). In addition, the fusion dimension of 512 is the most frequent configuration among these models, indicating a consistent preference for higher-dimensional shared representations within the explored search space.

# Citation

This work is part of a paper titled "LLM-Driven Neural Architecture Search for Multimodal Skin Lesion Classification under Deployment Constraints," currently submitted to a conference.

If you use this code, please cite the corresponding work/paper.

```bibtex
@inproceedings{rocha2026llmnas,
  author    = {Rocha, Wyctor Fogos da and 
               Bouzon, Pedro H. G. and
               Pacheco, Andr{'e} G. C. and
               Souza~J{'u}nior, Luis Ant{^o}nio de},
  title     = {{LLM-Driven Neural Architecture Search for Multimodal
               Skin Lesion Classification under Deployment Constraints}},
  booktitle = {2026 39th SIBGRAPI Conference on Graphics, Patterns and
               Images (SIBGRAPI)},
  year      = {2026},
  publisher = {IEEE},
  address   = {Goiânia-Goiás, Brazil},
  note      = {In press},
}
```
