import numpy as np

camera_matrix = np.array(
    [
        [2.67126877e03, 0.00000000e00, 9.57917092e02],
        [0.00000000e00, 2.67168557e03, 5.19849867e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

center_coordinates_list = np.array(
    [
        [
            [570.3806112702961, 453.00238777459407],
            [602.1808873720137, 183.5551763367463],
            [547.1986809563067, 949.5375103050288],
            [570.3806112702961, 453.00238777459407],
            [960.1367690782953, 158.16352824578792],
            [960.3856722276741, 348.6182531894014],
            [960.3470588235294, 539.1539215686274],
            [1152.3682132280355, 539.3899308983218],
            [1344.000975609756, 539.3990243902439],
            [1318.5394144144145, 184.00225225225225],
            [1373.1917577796467, 129.72918418839362],
            [1318.4018161180477, 894.9704880817253],
            [1373.235841081995, 949.2831783601015],
        ],
        [
            [771.6197530864198, 514.2703703703704],
            [873.8262032085562, 295.7486631016043],
            [682.1353211009174, 800.0206422018349],
            [771.6197530864198, 514.2703703703704],
            [960.1551362683438, 269.6687631027254],
            [960.2348008385744, 404.50314465408803],
            [960.1721991701245, 539.1597510373444],
            [1059.8174757281554, 539.1825242718446],
            [1166.9876543209878, 539.2257495590829],
            [1258.7410881801127, 259.72232645403375],
            [1067.6677631578948, 237.3371710526316],
            [1258.813909774436, 819.2669172932331],
            [1067.7107438016528, 841.5619834710744],
        ],
        [
            [766.9912126537786, 498.6968365553603],
            [661.716796875, 259.626953125],
            [852.867088607595, 841.5664556962025],
            [766.9912126537786, 498.6968365553603],
            [960.1551362683438, 269.6687631027254],
            [960.2348008385744, 404.50314465408803],
            [960.1721991701245, 539.1597510373444],
            [1052.9345372460496, 539.1580135440181],
            [1139.6097560975609, 539.2439024390244],
            [1046.8954423592493, 295.70777479892763],
            [1238.0, 279.0],
            [1047.0666666666666, 783.1973333333333],
            [1238.0613636363637, 799.925],
        ],
        [
            [679.6788899900891, 509.02279484638257],
            [678.6498054474708, 242.72178988326849],
            [697.9977011494253, 815.7701149425287],
            [679.6788899900891, 509.02279484638257],
            [960.234458259325, 334.01598579040854],
            [960.0992217898832, 440.44747081712063],
            [960.1496881496881, 539.1476091476092],
            [1095.8604166666667, 539.1875],
            [1231.4815573770493, 539.4938524590164],
            [1241.5442307692308, 242.82115384615383],
            [1264.1216, 432.528],
            [1205.381074168798, 625.4450127877238],
            [1222.1788990825687, 815.7454128440367],
        ],
        [
            [792.8423817863397, 424.90192644483363],
            [883.8986013986014, 231.87762237762237],
            [741.2145593869732, 665.2796934865901],
            [792.8423817863397, 424.90192644483363],
            [960.1842105263158, 350.2602339181287],
            [960.1987577639752, 446.8074534161491],
            [960.23, 538.9466666666667],
            [1040.6006006006005, 585.1231231231232],
            [1124.9333333333334, 633.5972222222222],
            [1213.2745098039215, 393.94677871148457],
            [1049.2850122850123, 488.031941031941],
            [1189.9134948096885, 758.9134948096886],
            [1040.630094043887, 861.6583072100314],
        ],
        [
            [811.7450142450142, 603.011396011396],
            [707.2439024390244, 393.8455284552846],
            [880.307210031348, 861.5203761755486],
            [811.7450142450142, 603.011396011396],
            [960.1842105263158, 350.2602339181287],
            [960.1987577639752, 446.8074534161491],
            [960.23, 538.9466666666667],
            [1036.9050847457627, 495.0508474576271],
            [1109.9263157894736, 453.1438596491228],
            [1036.8776223776224, 231.83566433566435],
            [1200.861635220126, 308.7327044025157],
            [1030.3724696356276, 579.417004048583],
            [1179.597014925373, 665.0671641791045],
        ],
        [
            [687.6242105263158, 571.9894736842106],
            [715.0871794871795, 453.1794871794872],
            [656.3034257748776, 646.4176182707994],
            [687.6242105263158, 571.9894736842106],
            [960.1299019607843, 361.3921568627451],
            [960.1009174311927, 447.13761467889907],
            [960.1496881496881, 539.1476091476092],
            [1095.8814968814968, 539.1725571725572],
            [1231.5030800821355, 539.4804928131417],
            [1205.421875, 453.3229166666667],
            [1222.0846681922196, 263.1487414187643],
            [1241.6252390057361, 836.0458891013384],
            [1264.2255520504732, 646.3375394321766],
        ],
        [
            [816.6550458715597, 642.7192660550459],
            [890.4285714285714, 499.3109243697479],
            [719.6545454545454, 770.160606060606],
            [816.6550458715597, 642.7192660550459],
            [960.2114695340501, 367.9641577060932],
            [960.1655172413793, 451.8689655172414],
            [960.1107491856677, 539.5114006514658],
            [1040.5169230769231, 493.5261538461538],
            [1124.8137535816618, 445.1289398280802],
            [1189.7929824561404, 319.9684210526316],
            [1040.577287066246, 217.34069400630915],
            [1213.243093922652, 684.9696132596686],
            [1049.2843373493977, 590.7927710843373],
        ],
        [
            [787.9327485380117, 467.0043859649123],
            [730.9249146757679, 320.10921501706486],
            [871.6900726392251, 590.8280871670702],
            [787.9327485380117, 467.0043859649123],
            [960.2114695340501, 367.9641577060932],
            [960.1655172413793, 451.8689655172414],
            [960.1107491856677, 539.5114006514658],
            [1036.9066666666668, 583.6466666666666],
            [1109.9197324414715, 625.4448160535117],
            [1030.3418803418804, 499.3290598290598],
            [1179.5907335907336, 413.6061776061776],
            [1036.939929328622, 847.148409893993],
            [1201.0185758513933, 770.1795665634675],
        ],
    ]
)
# print(center_coordinates_list)
for i, center_coordinates in enumerate(center_coordinates_list):
    # i番目のカメラの中心座標のリストを取得
    for j, center_coordinate in enumerate(center_coordinates):
        # j番目のマーカーの中心座標を取得
        print(f"{i}番目のカメラの{j}番目のマーカーの中心座標: {center_coordinate}")

# 正規化8点アルゴリズム（Norm8Point algorithm）
