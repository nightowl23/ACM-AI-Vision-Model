def generate_claims(validations):
    claims = []
    for entity, question, answer in validations:
        if "yes" in answer.lower():
            claims.append(f"{entity}: verified")
        elif "no" in answer.lower():
            claims.append(f"{entity}: not present")
        else:
            claims.append(f"{entity}: {answer}")
    return claims
