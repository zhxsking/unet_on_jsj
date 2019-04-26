# -*- coding: utf-8 -*-

from pyecharts import options as opts
from pyecharts.charts import Map

from snapshot_selenium import snapshot as driver
from pyecharts.render import make_snapshot


def map_without_label() -> Map:
    c = (
        Map()
        .add("", [['佳木斯市', 1]], "china")
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    return c


# 需要安装 snapshot_selenium
make_snapshot(driver, map_without_label().render(), "map.png")
