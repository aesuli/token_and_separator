def get_tag(annotation):
    '''
    Given an annotation ('O', 'I-tag_name', 'B-tag_name') returns 'O' or the tag name with the prefix removed
    '''
    if annotation == 'O':
        return annotation
    elif annotation.startswith('B-') or annotation.startswith('I-'):
        return annotation[2:]
    else:
        raise ValueError(f'Unknown annotation value: {annotation}')


def get_type(annotation):
    '''
    Given an annotation ('O', 'I-tag_name', 'B-tag_name') returns 'O' or the type of tag, i.e., "B" or "I"
    '''
    if annotation == 'O':
        return annotation
    elif annotation.startswith('B-'):
        return 'B'
    elif annotation.startswith('I-'):
        return 'I'
    else:
        raise ValueError(f'Unknown annotation type: {annotation}')


def to_token_and_separator(annotation):
    '''
    Converts an IOB/IOB2 annotation to the token and separator model.
    Once converted, the resulting lists from gold and predicted annotations can be compared using any evaluation measure.
    @param annotation the sequence of annotation of tokens in IOB/IOB2 format
    @return the list of tags assigned to each token and separator
    '''

    token_and_separator_annotation = list()
    prev_token_tag = None
    for token_annotation in annotation:
        token_tag = get_tag(token_annotation)

        # if not at the first token, then consider the separator between the two tokens
        if prev_token_tag is not None:
            # determine the separator tag value by comparing the current and previous tag
            separator_tag = 'O'
            if token_tag != 'O':
                if token_tag == prev_token_tag and get_type(token_annotation) != 'B':
                    separator_tag = token_tag

            token_and_separator_annotation.append(separator_tag)

        token_and_separator_annotation.append(token_tag)

        # update the previous token tag to determine the next separator tag
        prev_token_tag = token_tag

    return token_and_separator_annotation


if __name__ == '__main__':
    gold_annotation = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-LOC', 'O']
    predicted_annotation = ['B-PER', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'O']

    print(f'Gold annotation      {gold_annotation}')
    print(f'Predicted annotation {predicted_annotation}')

    gold_tok_and_sep = to_token_and_separator(gold_annotation)
    pred_tok_and_sep = to_token_and_separator(predicted_annotation)

    print(f'Gold annotation as token and separators      {gold_tok_and_sep}')
    print(f'Predicted annotation as token and separators {pred_tok_and_sep}')

    # scikit-learn is imported here to avoid introducing a requirement that is not necessary for the to_token_and_separator function.
    from sklearn.metrics import classification_report, confusion_matrix

    print(classification_report(gold_tok_and_sep, pred_tok_and_sep))

    print(confusion_matrix(gold_tok_and_sep, pred_tok_and_sep))
