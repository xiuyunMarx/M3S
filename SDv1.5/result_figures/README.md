先运行 make_html.py 生成网页，然后打开生成的html文件即可浏览图片
目前只展示了20个prompt生成的图片，想看更多可以自行添加（点开图片文件夹就能看到所有prompt）
可以更改 TARGET_STYLE = "style1_0_int_0.1" 其中的 style1 可以换成 style2 到 style5
如果想看到参考的style图片，将 IMG_FILENAME = "out_transfer---seed_42.png" 改为 IMG_FILENAME = "out_joined---seed_42.png"