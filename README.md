# Usage

## Prerequisites

### Prepare environment

Run `docker-compose run gpu` (strongly recommended) or install required packages using `pip install -r requirements.txt` in your own environment.

### Fetch XLS-R model weights

The configs for continued pretraining can be found in this repo (e.g. in `models/xls-r_300m/config.json`). So all we need to do is fetch the weights.

```
wget -P models/xls-r_300m/ https://huggingface.co/facebook/wav2vec2-xls-r-300m/resolve/main/pytorch_model.bin
```

### Run `accelerate config`

Run `accelerate config` (inside the Docker container, if applicable) and answer the questions as appropriate, e.g.:

```
In which compute environment are you running?                                                     This machine
Which type of machine are you using?                                                              multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]:          1
Do you wish to optimize your script with torch dynamo?[yes/NO]:                                   NO
Do you want to use DeepSpeed? [yes/NO]:                                                           NO
Do you want to use FullyShardedDataParallel? [yes/NO]:                                            NO
Do you want to use Megatron-LM ? [yes/NO]:                                                        NO
How many GPU(s) should be used for distributed training? [1]:                                     4
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]: all
Do you wish to use FP16 or BF16 (mixed precision)?                                                fp16
```

## Training

Demo training command tested using 4 x 8GB RTX 3070 setup (hence `max_duration_in_seconds="5"` and `dbs_max_batch_length=5` to fit both models and data into 8GB memory).

```bash
accelerate launch run_wav2vec2_pretraining_no_trainer.py \
	--data_dir="data/gos-demo/wav" \
	--validation_split_percentage="1" \
	--model_name_or_path="models/xls-r_300m" \
	--output_dir="models/xls-r_300m-cpt_gos-demo" \
	--max_train_steps="1000" \
	--saving_steps="250" \
	--num_warmup_steps="100" \
	--learning_rate="1e-3" \
	--max_duration_in_seconds="5" \
	--min_duration_in_seconds="1" \
	--gradient_accumulation_steps="2" \
	--dbs_max_batch_length="5" \
	--dbs_batch_ordering="ascending" \
	--dbs_num_buckets="20" \
	--adam_beta1="0.9" \
	--adam_beta2="0.98" \
	--adam_epsilon="1e-06" \
	--weight_decay="0.01" \
	--logging_steps="1" \
	--gradient_checkpointing
```

### Notes

- Note `min_duration_in_seconds` acts as a filter such that any short clips will be removed. By contrast, `max_duration_in_seconds` will keep clips longer than the specified duration but just truncate them to the max allowable duration.
- We're making use of [DynamicBatchSampler](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.dataio.sampler.html#speechbrain.dataio.sampler.DynamicBatchSampler) from [SpeechBrain](https://speechbrain.github.io/).
This lets us set a maximum per-GPU batch size in seconds (e.g. `dbs_max_batch_length=5`), where the number of examples in each batch is variable but the total allowable duration of the examples is fixed (e.g. to 5 seconds).
- Make sure your longest clip(s) will fit into memory! Set `dbs_batch_ordering="descending"` and adjust your `max_duration_in_seconds` to whatever your GPU(s) will allow.
- Optionally, adjust `dbs_num_buckets`, according to the [SpeechBrain tutorial](https://colab.research.google.com/drive/1mypqbHDrusZaIbqPoiEGY-WIbnpMHa2I?usp=sharing).
