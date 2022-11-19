import cpl
import datasets


def get_num_folds(args):
    if args.dataset == 'AdienceFace':
        num_folds = 5
    elif args.dataset == 'HistoricalColor':
        num_folds = 10
    elif args.dataset == 'ImageAesthetics':
        num_folds = 5
    else:
        raise NotImplementedError
    return num_folds


def get_train_val_test_datasets(args, fold_idx):
    if args.dataset == 'AdienceFace':
        train_val_test_datasets = datasets.AdienceFaceDatasets(args.adience_face_root, fold_idx)
    elif args.dataset == 'HistoricalColor':
        train_val_test_datasets = datasets.HistoricalColorDatasets(args.historical_color_root, fold_idx)
    elif args.dataset == 'ImageAesthetics':
        train_val_test_datasets = datasets.ImageAestheticsDatasets(f'{args.image_aesthetics_root}#{args.image_aesthetics_cat}', fold_idx)
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

    if args.constraint == 'S-P':
        proxies_learner = cpl.BaseProxiesLearner(num_ranks, args.feature_dim)
        criterion = cpl.SoftCplPoissonLoss(num_ranks, args.tau, args.loss_lam)
    elif args.constraint == 'S-B':
        proxies_learner = cpl.BaseProxiesLearner(num_ranks, args.feature_dim)
        criterion = cpl.SoftCplBinomialLoss(num_ranks, args.tau, args.loss_lam)
    elif args.constraint == 'H-L':
        proxies_learner = cpl.LinearProxiesLearner(num_ranks, args.feature_dim)
        criterion = cpl.HardCplLoss()
        metric_method = cpl.EuclideanMetric()
    elif args.constraint == 'H-S':
        proxies_learner = cpl.SemicircularProxiesLearner(num_ranks, args.feature_dim)
        criterion = cpl.HardCplLoss()
        metric_method = cpl.CosineMetric(args.cosine_scale)
    else:
        raise NotImplementedError

    model = cpl.CplModel(feature_extractor, proxies_learner, metric_method)

    return model, criterion
