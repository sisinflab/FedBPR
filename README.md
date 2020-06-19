<html><style>
td {
  font-size: 50px
}
</style></html>

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


### Paired t-test results

| Brazil | Random | Top-Pop | User-kNN | Item-kNN | VAE | BPR-MF | FCF | sFPL | sFPL+ | pFPL | pFPL+ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Random** | | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> |
| **Top-Pop** | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> |
| **User-kNN** | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> |
| **Item-kNN** | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.852</li><li>EPC: 0.066</li><li>EFD: 0.000</li></ul> |
| **VAE** | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> |
| **BPR-MF** | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.079</li><li>Rec: 0.052</li><li>nDCG: 0.258</li><li>EPC: 0.158</li><li>EFD: 0.455</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.123</li><li>Rec: 0.162</li><li>nDCG: 0.886</li><li>EPC: 0.766</li><li>EFD: 0.611</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> |
| **FCF** | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul>| <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> |
| **sFPL** | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.079</li><li>Rec: 0.052</li><li>nDCG: 0.258</li><li>EPC: 0.158</li><li>EFD: 0.455</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.738</li><li>Rec: 0.997</li><li>nDCG: 0.396</li><li>EPC: 0.562</li><li>EFD: 0.970</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> |
| **sFPL+** | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul>| <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.311</li><li>Rec: 0.255</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> |
| **pFPL** | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul>| <ul><li>Prec: 0.123</li><li>Rec: 0.162</li><li>nDCG: 0.886</li><li>EPC: 0.766</li><li>EFD: 0.611</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.739</li><li>Rec: 0.997</li><li>nDCG: 0.396</li><li>EPC: 0.562</li><li>EFD: 0.970</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> |
| **pFPL+** | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.852</li><li>EPC: 0.067</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul>| <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.311</li><li>Rec: 0.255</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | <ul><li>Prec: 0.000</li><li>Rec: 0.000</li><li>nDCG: 0.000</li><li>EPC: 0.000</li><li>EFD: 0.000</li></ul> | |
