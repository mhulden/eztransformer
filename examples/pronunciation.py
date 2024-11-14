from eztr import EZTransformer
from random import shuffle

# Load English dictionary with IPA transcriptions
lines = [l. strip() for l in open("newdic-ipa.txt", encoding = "utf8")]

# Grab field 3 (spelling) and 0 (pronunciation)
wordpron = [(l.split('\t')[3], l.split('\t')[0]) for l in lines]
data = [(' '.join(list(one)), ' '.join(list(two))) for one, two in wordpron] # Space between tokens

shuffle(data)

split_index = int(len(data) * 0.9)

# Split into 90% train and 10% validation sets
train_data = data[:split_index]
dev_data = data[split_index:]

# Initialize model
trf = EZTransformer(device = 'cuda')

# Train model
trf.fit(train_data, valid_data = dev_data, print_validation_examples = 2, max_epochs = 100)

# Make pronunciation predictions
trf.predict(["s u p e r c a l i f r a g i l i s t i c e x p i a l i d o c i o u s"])
