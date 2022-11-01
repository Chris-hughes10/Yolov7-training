import torch
from torch import nn

from yolov7.models.core.layers import ImplicitAdd, ImplicitMultiply


class Yolov7DetectionHead(nn.Module):

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        # TODO: Nasty that not everything is defined here
        self.initialize_module_parameters(ch)

    def initialize_module_parameters(self, ch):
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv

        self.ia = nn.ModuleList(ImplicitAdd(x) for x in ch)
        self.im = nn.ModuleList(ImplicitMultiply(self.no * self.na) for _ in ch)

    def forward(self, x):
        for i in range(self.nl):
            self.layer_forward(x, i)
        return x

    def layer_forward(self, x, i):
        x[i] = self.m[i](self.ia[i](x[i]))  # conv
        x[i] = self.im[i](x[i])
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x[i] = (
            x[i]
            .view(bs, self.na, self.no, ny, nx)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )


class Yolov7DetectionHeadWithAux(Yolov7DetectionHead):
    def initialize_module_parameters(self, ch):
        super().initialize_module_parameters(ch[:self.nl])
        self.m2 = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl :]
        )  # output conv

    def aux_layer_forward(self, x, i):
        bs, _, ny, nx = x[i].shape
        x[i + self.nl] = self.m2[i](x[i + self.nl])
        x[i + self.nl] = (
            x[i + self.nl]
            .view(bs, self.na, self.no, ny, nx)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

    def forward(self, x):
        for i in range(self.nl):
            self.layer_forward(x, i)
            self.aux_layer_forward(x, i)
        return x


#################
# LEGACY ########
#################

class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (
                        y[..., 0:2] * 2.0 - 0.5 + self.grid[i]
                    ) * self.stride[
                        i
                    ]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        else:
            out = (torch.cat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32,
            device=z.device,
        )
        box @= convert_matrix
        return (box, score)


class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv

        self.ia = nn.ModuleList(ImplicitAdd(x) for x in ch)
        self.im = nn.ModuleList(ImplicitMultiply(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[
                    i
                ]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        # yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class IAuxDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IAuxDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch[: self.nl]
        )  # output conv
        self.m2 = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl :]
        )  # output conv

        self.ia = nn.ModuleList(ImplicitAdd(x) for x in ch[: self.nl])
        self.im = nn.ModuleList(ImplicitMultiply(self.no * self.na) for _ in ch[: self.nl])

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            x[i + self.nl] = self.m2[i](x[i + self.nl])
            x[i + self.nl] = (
                x[i + self.nl]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (
                        y[..., 0:2] * 2.0 - 0.5 + self.grid[i]
                    ) * self.stride[
                        i
                    ]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x[: self.nl])

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


