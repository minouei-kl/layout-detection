from argparse import ArgumentParser
import matplotlib.pyplot as plt

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def hook(module, input, output):
    setattr(module, "_value_hook", output)

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
        if 'fpn_convs' in name and '.conv' in name:
        # if 'backbone.layer1.2.conv3' in name:
            # print(name)
            # print(module)
            hooks[name] = module.register_forward_hook(hook)

    # test a single image
    result = inference_detector(model, args.img)

    for name, module in model.named_modules():
        if 'fpn_convs' in name and '.conv' in name:
        # if 'backbone.layer1.2.conv3' in name:
            out = module._value_hook[0].detach().cpu()
            # print(out)
            for i in range(128):
                fig, axarr = plt.subplots()
                axarr.imshow( out[i, :, :] , cmap='gray', vmin=0, vmax=1)
                # plt.show()
                fig.savefig('demo/out/'+name+str(i)+'.png')
                plt.close(fig)

    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
