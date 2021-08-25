from subprocess import run

FPN_fmt = "dashcam_%d_blackgen_bound_0.5_qp_30_conv_9_app_FPN.mp4*"
C4_fmt = "dashcam_%d_blackgen_bound_0.3_qp_30_conv_9_app_C4.mp4*"
DC5_fmt = "dashcam_%d_blackgen_bound_0.2_qp_30_conv_9.mp4*"
mpeg_fmt = "dashcam_%d_qp_%d.mp4"

vids = [(i + 1) for i in range(7)]
qps = [30, 31, 32, 34, 36, 40, 44, 50]

for vid in vids:

    for qp in qps:
        vname = mpeg_fmt % (vid, qp)
        run(["cp", f"dashcam/{vname}", f"/tank/visdrone/"])

    for fmt in [FPN_fmt, C4_fmt, DC5_fmt]:

        import os

        vname = fmt % vid
        os.system(" ".join(["cp", f"dashcam/{vname}", f"/tank/visdrone/"]))

