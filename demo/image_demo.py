from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmcv
def hook(module, input, output):
    setattr(module, "_value_hook", output)

def filter(dets):

    det_bboxes=np.vstack(dets)

    adj = np.zeros((len(det_bboxes), len(det_bboxes)), int)
    for i in range(len(det_bboxes)):
        for j in range(i):
            boxA = det_bboxes[i]
            boxB = det_bboxes[j]
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            # compute the area of intersection rectangle
            interArea = max(0, xB - xA ) * max(0, yB - yA )

            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            if (iou>0.02):
                adj[i][j] = adj[j][i] = 1
    print(adj)
    n = len(det_bboxes)
    ConSubSets = []

    for i in range(len(det_bboxes)):
        bset = []
        for j in range(len(det_bboxes)):
            if adj[i][j] == 0:
                bset.append(det_bboxes[j])
        ConSubSets.append(np.asarray(bset))

    alla,alls,fscr=[],[],[]

    for sl in ConSubSets:
        areas = sum([(box[2] - box[0]) * (box[3] - box[1]) for box in sl])/500000.0
        alla.append(areas)
        # print(areas)
        bxnd=np.asarray(sl)
        scores = bxnd[:, -1]
        ms=np.mean(scores)
        # print(ms)
        alls.append(ms)
        fscr.append(np.mean (areas + ms) )
        mmcv.imshow_bboxes('demo/minival/PMC4982048_00021.jpg', bxnd)

    idx=fscr.index(max(fscr))
    mmcv.imshow_bboxes('demo/minival/PMC4982048_00021.jpg', np.asarray(ConSubSets[idx]))

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    hooks = {}
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)


    for name, module in model.named_modules():
        print(name)
        print(module)
        if '1fpn_convs' in name and '.conv' in name:
        # if 'backbone.layer1.2.conv3' in name:
            # print(name)
            # print(module)
            hooks[name] = module.register_forward_hook(hook)

    # test a single image
    result = inference_detector(model, args.img)

    for name, module in model.named_modules():
        if '2fpn_convs' in name and '.conv' in name:
        # if 'backbone.layer1.2.conv3' in name:
            out = module._value_hook[0].detach().cpu()
            # print(out)
            for i in range(128):
                plt.box(False)
                fig, axarr = plt.subplots()

                axarr.axis('off')

                axarr.imshow( out[i, :, :] , cmap='gray', vmin=0, vmax=1)
                # plt.show()
                fig.patch.set_visible(False)

                fig.savefig('demo/nest94/'+name+str(i)+'.png')
                plt.close(fig)

    # show the results
    # res=filter(result)
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
