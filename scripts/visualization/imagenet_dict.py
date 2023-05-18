import os


class ImageNetDict:
    def __init__(self, info_file):
        assert os.path.isfile(info_file)
        self.info_file = info_file
        self.dir2cls = {}
        with open(self.info_file, "r") as fp:
            while True:
                aline = fp.readline()
                if not aline:
                    break
                parts = aline.split()
                if parts:
                    self.dir2cls[parts[1]] = int(parts[0])

    def __getitem__(self, item):
        return self.dir2cls[item]


if __name__ == '__main__':
    imagenet_dict = ImageNetDict('scripts/visualization/imagenet.txt')
    print(imagenet_dict['n03633091'])
