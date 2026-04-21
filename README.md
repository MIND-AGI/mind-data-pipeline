## mind-data-pipeline

### Requirements

- Python 3.9+
- Dependencies are listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

### What goes in .env

Only global runtime variables should be in `.env`:

```dotenv
HUGGING_FACE_HUB_TOKEN=your_hf_token
HF_HOME=./hf_home
HF_DATASETS_CACHE=./hf_home/hf_datasets
HUGGINGFACE_HUB_CACHE=./hf_cache
```

### Overview

```
Raw Dataset (Parquet / Arrow)
        |
        ↓
     MDS Shards
        |  (shuffle)
        ↓    
       JSONL
```

### Run 1: HuggingFace dataset -> MDS

Default:

```bash
bash hfdata2mds.sh
```

Override parameters (when switching datasets):

```bash
bash hfdata2mds.sh \
	--data-repo HuggingFaceFW/fineweb-edu \
	--out-root ./data/fineweb-edu-sample-10BT_mds \
	--num-groups 10 \
	--num-process 10 \
	--dataset-name sample-10BT
```

### Run 1b: Parquet file(s) -> MDS

Default:

```bash
bash parquet2mds.sh
```

Support one parquet file or glob pattern:

```bash
bash parquet2mds.sh \
	--input-path ./data/*.parquet \
	--out-root ./data/parquet_mds \
	--num-groups 10 \
	--num-process 10 \
	--text-column text
```

### Run 2: MDS -> JSONL

Default:

```bash
bash mds2jsonl.sh
```

Override mds/jsonl path:

```bash
bash mds2jsonl.sh \
	--local-root ./data/fineweb-edu-sample-10BT_mds \
	--out-root ./data/fineweb-edu-10BT_jsonl
```

### GitHub safety

- `.env` is ignored by `.gitignore`.
- Do not commit real tokens.
