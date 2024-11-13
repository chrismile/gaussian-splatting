import os
from tensorboard.backend.event_processing import event_accumulator


def main():
    base_dir = ''
    if os.path.exists('/mnt/data/3DGS'):
        base_dir = '/mnt/data/3DGS/train/'
    elif os.path.exists('/home/neuhauser/datasets/3dgs/nerf/3DGS'):
        base_dir = '/home/neuhauser/datasets/3dgs/nerf/3DGS/train/'
    files = [
        'garden_default/events.out.tfevents.1731144201.TUINI15-CG12.7856.0',
        'garden_x2_ninasr_b1/events.out.tfevents.1731429882.TUINI15-CG14.36893.0',
        'garden_x3_ninasr_b1/events.out.tfevents.1731442291.TUINI15-CG14.50415.0',
        'garden_x4_ninasr_b1/events.out.tfevents.1731446452.TUINI15-CG14.50844.0',
    ]
    for file in files:
        test_file = base_dir + file

        ea = event_accumulator.EventAccumulator(
            test_file,
            size_guidance={  # see below regarding this argument
                event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                event_accumulator.IMAGES: 4,
                event_accumulator.AUDIO: 4,
                event_accumulator.SCALARS: 0,
                event_accumulator.HISTOGRAMS: 1,
            })

        ea.Reload()

        start = ea.Scalars("iter_time")[0].wall_time
        end = ea.Scalars("iter_time")[-1].wall_time
        print(f'{file}:')
        print(end - start)


if __name__ == '__main__':
    main()
