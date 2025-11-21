from cosine_similarity_eval.metric_cosine import CosineSimilarityScorer


def main():
    print("=== Cosine Similarity Demo ===")

    scorer = CosineSimilarityScorer()

    humans = [
        "A dog playing with a ball",
        "A red apple on a table",
    ]

    models = [
        "A dog is playing fetch with a ball",
        "A red apple sitting on a wooden table",
    ]

    scores, mean_score, std_score = scorer.score_pairs(humans, models)

    print("Scores:", scores)
    print("Mean cosine similarity:", mean_score)
    print("Std dev:", std_score)


if __name__ == "__main__":
    main()
