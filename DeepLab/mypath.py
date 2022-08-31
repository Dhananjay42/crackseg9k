class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        if dataset == 'crack':
            return 'datasets/crack/'
#         elif dataset == 'sbd':
#             return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
#         elif dataset == 'cityscapes':
#             return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
#         elif dataset == 'coco':
#             return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
