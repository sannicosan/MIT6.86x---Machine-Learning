def get_sum_metrics(predictions, metrics=[]):

    for i in range(3):
        metrics.append(lambda x,k=(i): x + k)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)

    return sum_metrics

print(get_sum_metrics(3))