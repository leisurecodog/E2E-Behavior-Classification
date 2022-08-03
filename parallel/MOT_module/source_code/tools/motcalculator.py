import motmetrics as mm
import numpy
import os




if __name__ == '__main__':
    gt_loc = "/home/fengan/Desktop/Dataset/BDD100K MOT/bdd100k_box_track_20_labels_trainval/bdd100k/labels/box_track_20/txt_folder"
    pred_loc = "/home/fengan/Desktop/Dataset/BDD100K MOT/images20-track-val-1/bdd100k/images/track/preds"
    # mh = mm.metrics.create()
    # print(mh.list_metrics_markdown())
    for g, p in zip(os.listdir(gt_loc),os.listdir(pred_loc)):
        print(g, p)
        # input()
        g_f = os.path.join(gt_loc, g)
        p_f = os.path.join(pred_loc, p)
        metrics = list(mm.metrics.motchallenge_metrics)
        gt_l = mm.io.loadtxt(g_f)
        pred_l = mm.io.loadtxt(p_f)
        # print(metrics)

        acc = mm.utils.compare_to_groundtruth(gt_l, pred_l, 'iou', distth=0.5)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics = metrics)
        print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))