def compare(result, segmentation):
    result = result
    segmentation = segmentation
    result = result.astype('bool')
    segmentation = segmentation.astype('bool')
    intersection = result & segmentation
    union = result | segmentation
    return intersection.sum() / union.sum()
