# RoIAlign - New style autograd.Function for PyTorch > 1.3


    pytorch warning:
    "Legacy autograd function with non-static forward method is deprecated and will be removed in 1.3. Please use new-style autograd function with static forward method.

```python
#### Old style
class F(torch.autograd.Function):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, args):
        pass
    def backward(self, args):
        pass

#### New style
class F_new(torch.autograd.Function):
    @staticmethod
    def forward(ctx, args, gamma):
        ctx.gamma = gamma
        pass

    @staticmethod
    def backward(ctx, args):
        pass

# Using your old style Function from your code sample:
F(gamma)(inp)
# Using the new style Function:
F_new.apply(inp, gamma)
```
[reference](https://discuss.pytorch.org/t/difference-between-apply-an-call-for-an-autograd-function/13845/4)

# [RoIAlign for PyTorch](https://github.com/longcw/RoIAlign.pytorch) (original version)

This is a PyTorch version of [RoIAlign](https://arxiv.org/abs/1703.06870).
This implementation is based on `crop_and_resize`
and supports both forward and backward on CPU and GPU.

**NOTE:** Thanks [meikuam](https://github.com/meikuam) for updating 
this repo for ***PyTorch 1.0***. You can find the original version for 
`torch <= 0.4.1` in [pytorch_0.4](https://github.com/longcw/RoIAlign.pytorch/tree/pytorch_0.4)
branch.


## Introduction
The `crop_and_resize` function is ported from [tensorflow](https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize),
and has the same interface with tensorflow version, except the input feature map
should be in `NCHW` order in PyTorch.
They also have the same output value (error < 1e-5) for both forward and backward as we expected,
see the comparision in `test.py`.

**Note:**
Document of `crop_and_resize` can be found [here](https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize).
And `RoIAlign` is a wrap of `crop_and_resize`
that uses boxes with *unnormalized `(x1, y1, x2, y2)`* as input
(while `crop_and_resize` use *normalized `(y1, x1, y2, x2)`* as input).
See more details about the difference of
 `RoIAlign` and `crop_and_resize` in [tensorpack](https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301).

**Warning:**
Currently it only works using the default GPU (index 0)

## Usage
+ Install and test
    ```
    python setup.py install
    ./test.sh
    ```

+ Use RoIAlign or crop_and_resize
    ```python
    from roi_align import RoIAlign      # RoIAlign module
    from roi_align import CropAndResize # crop_and_resize module

    # input data
    image = to_varabile(image_data, requires_grad=True, is_cuda=is_cuda)
    boxes = to_varabile(boxes_data, requires_grad=False, is_cuda=is_cuda)
    box_index = to_varabile(box_index_data, requires_grad=False, is_cuda=is_cuda)

    # RoIAlign layer
    roi_align = RoIAlign(crop_height, crop_width)
    crops = roi_align(image, boxes, box_index)
    ```
