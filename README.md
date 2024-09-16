<p align="left">
  <img src="./data/e-ztransformer.png" width="256">
  <br />
</p>

EZTransformer is a Transformer implementation that allows you to train translation models with simple API calls. 

* Dependencies: PyTorch, tqdm

## Quickstart

```python3
from eztr import EZTransformer

# Have data
mydata = pickle.load(open("data/sigmorphon2016spanish.p", "rb")) # Example word inflection data for Spanish

# Data format (whitespace-separated tokens, two-tuples for input/target)
>>> mydata['train'][:5]

[('a b a b o l # N PL', 'a b a b o l e s'),
 ('á b a c o # N SG', 'á b a c o'),
 ('a b a c o r a r # V IND PRS 1 PL IPFV/PFV', 'a b a c o r a m o s'),
 ('a b a c o r a r # V IND PRS 3 PL IPFV/PFV', 'a b a c o r a n'),
 ('a b a d e r n a r # V COND 1 PL', 'a b a d e r n a r í a m o s')]


# Initialize model
trf = EZTransformer(device = 'cuda')  # Change device as needed; 'cuda' (NVIDIA), 'mps' (Apple), or 'cpu'

# Train model
trf.fit(mydata['train'], valid_data = mydata['valid'], print_validation_examples = 2, max_epochs = 100)

# Load back the best model wrt validation set (or skip this step to use final weights with trf directly)
trf = EZTransformer(load_model = "best_model.pt")

# Make Predictions
trf.predict(["c o m p r o m e t e r # V IND PST 3 PL PFV", "h a b l a r V IND PST 1 SG"])

# Evaluate on test set
trf.score(mydata['test_inputs'], mydata['test_targets'])
```
