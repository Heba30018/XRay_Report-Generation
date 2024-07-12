# -*- coding: utf-8 -*-
"""Untitled38.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MCC3Zu7w4GSrsE3IMnhfuzxNRIiOkjr-
"""

# Load data
val_df = pd.read_csv('val_df.csv')

# Generate captions
predicted_captions = []
for i in tqdm.tqdm(val_df['imgs']):
    img = Image.open(i).convert("RGB")
    caption = tokenizer.decode(model.generate(feature_extractor(img, return_tensors="pt").pixel_values.to("cuda"))[0], skip_special_tokens=True)
    predicted_captions.append(caption)

# Calculate BLEU score
ground_truth_captions = val_df['captions'].values
ground_truth_captions = [[caption.split() for caption in captions] for captions in ground_truth_captions]
generated_captions = [caption.split() for caption in predicted_captions]
smoothie = SmoothingFunction().method4
weights = (0.25, 0.25, 0.25, 0.25)
score = corpus_bleu(ground_truth_captions, generated_captions, weights=weights)
print(f'The BLEU Score is: {score}')