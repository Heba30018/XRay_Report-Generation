import nltk
from nltk.translate.bleu_score import corpus_bleu

def calculate_bleu(reference_texts, generated_texts):
    references = [[ref.split()] for ref in reference_texts]
    candidates = [gen.split() for gen in generated_texts]
    score = corpus_bleu(references, candidates)
    return score

if __name__ == "__main__":
    reference_texts = ["example reference text"]
    generated_texts = ["example generated text"]
    score = calculate_bleu(reference_texts, generated_texts)
    print(f"BLEU score: {score}")
