def get_entities(seq):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> print(get_entities(seq))
        [('PER', 0, 2), ('LOC', 3, 4)]
    """
    i = 0
    chunks = []
    seq = seq + ['O']  # add sentinel
    types = [tag.split('-')[-1] for tag in seq]
    while i < len(seq):
        if seq[i].startswith('B'):
            for j in range(i+1, len(seq)):
                if seq[j].startswith('I') and types[j] == types[i]:
                    continue
                break
            chunks.append((types[i], i, j))
            i = j
        else:
            i += 1
    return chunks


def f1_score(y_true, y_pred, sequence_lengths):
    """Evaluates f1 score.
    Args:
        y_true (list): true labels.
        y_pred (list): predicted labels.
        sequence_lengths (list): sequence lengths.
    Returns:
        float: f1 score.
    Example:
        >>> y_true = []
        >>> y_pred = []
        >>> sequence_lengths = []
        >>> print(f1_score(y_true, y_pred, sequence_lengths))
        0.8
    """
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for lab, lab_pred, length in zip(y_true, y_pred, sequence_lengths):
        lab = lab[:length]
        lab_pred = lab_pred[:length]

        lab_chunks = set(get_entities(lab))
        lab_pred_chunks = set(get_entities(lab_pred))

        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return f1
