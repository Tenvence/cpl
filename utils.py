import cpl
import datasets


def get_num_folds(args):
    if args.dataset == 'AF':
        num_folds = 5
    elif args.dataset == 'HC':
        num_folds = 10
    elif args.dataset == 'IA':
        raise NotImplementedError
    elif args.dataset == 'MI':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return num_folds


def get_train_val_test_datasets(args, fold_idx):
    if args.dataset == 'AF':
        train_val_test_datasets = datasets.AdienceFaceDatasets(args.af_root, fold_idx)
    elif args.dataset == 'HC':
        raise NotImplementedError
    elif args.dataset == 'IA':
        raise NotImplementedError
    elif args.dataset == 'MI':
        raise NotImplementedError
    else:
        raise NotImplementedError

    train_dataset = train_val_test_datasets.get_train_dataset()
    val_dataset = train_val_test_datasets.get_val_dataset()
    test_dataset = train_val_test_datasets.get_test_dataset()
    num_ranks = train_val_test_datasets.num_ranks

    return train_dataset, val_dataset, test_dataset, num_ranks


def get_feature_extractor(args):
    if args.feature_extractor == 'V16':
        feature_extractor = cpl.Vgg16(args.feature_dim)
    elif args.feature_extractor == 'R18':
        feature_extractor = cpl.ResNet18(args.feature_dim)
    elif args.feature_extractor == 'R101':
        feature_extractor = cpl.ResNet101(args.feature_dim)
    else:
        raise NotImplementedError
    return feature_extractor


def get_metric_method(args):
    if args.metric_method == 'E':
        metric_method = cpl.EuclideanMetric()
    elif args.metric_method == 'C':
        metric_method = cpl.CosineMetric(args.cosine_scale)
    else:
        raise NotImplementedError
    return metric_method


def get_model_criterion(num_ranks, args):
    feature_extractor = get_feature_extractor(args)
    metric_method = get_metric_method(args)

    if args.constraint == 'UPL':
        proxies_learner = cpl.BaseProxiesLearner(num_ranks, args.feature_dim)
        criterion = cpl.UplLoss()
    elif args.constraint == 'S-P':
        raise NotImplementedError
    elif args.constraint == 'S-B':
        raise NotImplementedError
    elif args.constraint == 'H-L':
        proxies_learner = cpl.LinearProxiesLearner(num_ranks, args.feature_dim)
        criterion = cpl.HardCplLoss()
        metric_method = cpl.EuclideanMetric()
    elif args.constraint == 'H-S':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return cpl.CplModel(feature_extractor, proxies_learner, metric_method), criterion
