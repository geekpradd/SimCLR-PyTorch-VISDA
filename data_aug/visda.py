from PIL import Image
import os
import os.path
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.utils.data as data

from pathlib import Path
class StandardTransform(object):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)

class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self) -> str:
        return ""

class Visda(VisionDataset):
    

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        super(Visda, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.root_dir = root
        self.transform = transform
        self.train = train

        # when train is true, this will return samples from the training set
        if self.train:
            self.root_dir = os.path.join(self.root_dir, "train")
        else:
            self.root_dir = os.path.join(self.root_dir, "validation")

        self.categories = (os.listdir(self.root_dir))
        self.categories.remove("image_list.txt")

        self.counter = []
        current = 0 
        for f in self.categories:
            current += len(os.listdir(os.path.join(self.root_dir, f)))
            self.counter.append(current - 1)

        self.length = current

   

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """


        ind = -1
        for c, val in enumerate(self.counter):
            if index <= val:
                ind = c
                break

        current_count = index
        if ind != 0:
            current_count -= self.counter[ind-1] + 1

        folder_path = os.path.join(self.root_dir, self.categories[ind])
        file_name = os.listdir(folder_path)[current_count]

        image_path = os.path.join(folder_path, file_name)

        image = Image.open(image_path)

        basewidth = 96
        
        wpercent = (basewidth/float(image.size[0]))
        hsize = int((float(image.size[1])*float(wpercent)))
        image = image.resize((basewidth,hsize), Image.ANTIALIAS)

        # print (image[215, 384])
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ind = self.target_transform(ind)

        return image, ind

    def __len__(self) -> int:
        return self.length 



# d = Visda(os.path.join(str(Path(os.getcwd()).parent.absolute()), "visda"))
# print(len(d))
# x, y = d[142797]
# x.show()


