"""A collection of distance algorithms"""
from math import sqrt

def pearson(vector_1, vector_2):
    """Pearson correlation for two vectors."""
    sum1 = sum(vector_1)
    sum2 = sum(vector_2)

    sum1_sq = sum(v ** 2 for v in vector_1)
    sum2_sq = sum(v ** 2 for v in vector_2)

    sum_product = sum(vector_1[i] * vector_2[i] for i in range(len(vector_1)))

    num = sum_product - (sum1 * sum2 / len(vector_1))
    density = sqrt(abs(sum1_sq - sum1 ** 2 / len(vector_1)) * abs(sum2_sq - sum2 ** 2 / len(vector_1)))

    if density == 0:
        return 0
    return 1.0 - num / density

def tanimoto(vector_1, vector_2):
    """Tanimoto coefficient of two vectors"""
    c1 = 0
    c2 = 0
    shr = 0

    for i in range(len(vector_1)):
        if vector_1[i] != 0:
            c1 += 1
        if vector_2[i] != 0:
            c2 += 1
        if vector_1[i] != 0 and vector_2[i] != 0:
            shr += 1

    return 1.0 - (float(shr) / (c1 + c2 - shr))
