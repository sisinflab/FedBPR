# Federated Bayesian Personalized Ranking

**warning**: some code improvements in progress

## Reproduce experiments

### Requirements
To use this Python script, install the requirements with:
```
pip install -r requirements.txt
```

To prepare the datasets, run the bash script ```download_raw_datasets.sh```, so that the datasets are placed in the ```raw_datasets``` directory.

Then, use the Python script ```generate_dataset.py``` to create training, validation and test sets from raw datasets. In detail, use the arguments like in the following example:
```
python generate_dataset.py \
  --datasets Brazil Milan \
  --user_cut 20 \
  --item_cut 0 \
  --test_size 0.2 \
  --validation_size 0.2 \
  --parse_dates
```

### Run the federated recommender
The script in this repository simulates in a single machine a federation of clients coordinated by a central server. As an example:
```
python main.py \
  --datasets MovieLens1M \
  --eval_every 5 \
  -F 50 \
  -lr 0.05 \
  -U 0.3 \
  -T single \
  -E 100
```
All the above-mentioned arguments and more options are completely described in the help ```python main.py -h```.
  
### Visualize results
For each experiment, the results are saved as raw recommendations in the folder ```./results/<dataset>/recs/```.
The federated recommender saves a recommendation file in this folder each ```eval_every``` epochs.
The structure of the files is the following:
```
user_id item_id predicted_rating
```

## Our datasets and results

In the following we present the datasets we experimented with and some results

```
SIGNIFICANCE RESULTS VERY SOON
```
