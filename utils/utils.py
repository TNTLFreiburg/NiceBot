def as_tabular(matrix, fmt='.3f'):
    return " \\\\\n".join([" & ".join(map(format(line, fmt))) for line in matrix])
